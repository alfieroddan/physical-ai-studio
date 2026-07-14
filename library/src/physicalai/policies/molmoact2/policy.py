# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 policy implementation."""

from os import PathLike
from pathlib import Path
from typing import Any, Literal

import torch
from physicalai.inference.data import InferenceFeature, InferenceFeatureDtype, InferenceFeatureType
from physicalai.inference.manifest import ComponentSpec
from torch import Tensor

from physicalai.data.constants import IMAGE_MASKS, STATE, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.data.dataset import Dataset
from physicalai.data.observation import ACTION, IMAGES, TASK, Feature, FeatureType, NormalizationParameters, Observation
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.export.backends import ExportParameters, OpenVINOExportParameters, TorchExportParameters
from physicalai.policies.base import Policy
from physicalai.train.schedulers import cosine_decay_with_warmup_scheduler

from .config import MolmoAct2Config
from .from_hf import build_config_from_hf_config, load_hf_pretrained_container
from .model import MolmoAct2Model
from .processors import MolmoAct2Postprocessor, MolmoAct2Preprocessor, make_molmoact2_preprocessors


def _coerce_dataset_feature(feature: Feature) -> Feature:
    normalization_data = feature.normalization_data
    copied_normalization: NormalizationParameters | None = None
    if normalization_data is not None:
        copied_normalization = NormalizationParameters(
            mean=normalization_data.mean,
            std=normalization_data.std,
            q01=normalization_data.q01,
            q99=normalization_data.q99,
            mask=normalization_data.mask,
        )

    shape = tuple(int(dim) for dim in feature.shape) if feature.shape is not None else ()
    return Feature(
        name=str(feature.name),
        ftype=FeatureType(feature.ftype),
        shape=shape,
        normalization_data=copied_normalization,
    )


def make_molmoact2_config(
    *,
    input_features: list[Feature] | None,
    output_features: list[Feature] | None,
    n_obs_steps: int,
    n_action_steps: int,
    action_mode: Literal["continuous", "discrete", "both"] = "continuous",
    torch_compile: bool = False,
) -> MolmoAct2Config:
    """Create the explicit model config for MolmoAct2.

    This function is the non-policy home for model-definition defaults.

    Args:
        input_features: List of input features the model consumes.
        output_features: List of output features the model produces.
        n_obs_steps: Number of observation steps.
        n_action_steps: Number of action steps.
        action_mode: Action supervision mode.
        torch_compile: Whether to mark the config for optimized inference.

    Returns:
        A fully populated :class:`MolmoAct2Config`.
    """
    return MolmoAct2Config(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        action_mode=action_mode,
        compile_model=torch_compile,
    )


def _as_float_list(value: object) -> list[float]:
    if torch.is_tensor(value):
        return [float(x) for x in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    if isinstance(value, (int, float)):
        return [float(value)]
    msg = f"Unsupported normalization value type: {type(value)}"
    raise TypeError(msg)


def _feature_normalization_mode(feature: Feature) -> str:
    if feature.ftype == FeatureType.VISUAL:
        return "identity"
    norm = feature.normalization_data
    if norm is None:
        return "identity"
    if norm.q01 is not None and norm.q99 is not None:
        return "quantiles"
    if norm.min is not None and norm.max is not None:
        return "min_max"
    if norm.mean is not None and norm.std is not None:
        return "mean_std"
    return "identity"


def _feature_normalization_stats(feature: Feature, mode: str) -> dict[str, list[float]]:
    norm = feature.normalization_data
    if norm is None or mode == "identity":
        return {}
    if mode == "quantiles":
        stats = {"q01": _as_float_list(norm.q01), "q99": _as_float_list(norm.q99)}
    elif mode == "min_max":
        stats = {"min": _as_float_list(norm.min), "max": _as_float_list(norm.max)}
    else:
        stats = {"mean": _as_float_list(norm.mean), "std": _as_float_list(norm.std)}
    if norm.mask is not None:
        stats["mask"] = _as_float_list(norm.mask)
    return stats


class MolmoAct2(ExportablePolicyMixin, Policy):  # pyright: ignore[reportIncompatibleMethodOverride,reportIncompatibleVariableOverride]
    """MolmoAct2 Policy."""

    def __init__(
        self,
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        repo_id: str | Path | None = "allenai/MolmoAct2",
        norm_tag: str | None = None,
        n_obs_steps: int = 30,
        n_action_steps: int = 30,
        action_mode: Literal["continuous", "discrete", "both"] = "continuous",
        *,
        adapt_to_so101: bool = False,
        torch_compile: bool = False,
    ) -> None:
        """Initialize a MolmoAct2 policy wrapper.

        Args:
            input_features: Optional observation feature schema.
            output_features: Optional action feature schema.
            repo_id: Optional pretrained checkpoint identifier or path.
            norm_tag: Optional normalization tag for pretrained checkpoints.
            n_obs_steps: Number of observation steps.
            n_action_steps: Number of predicted action steps.
            action_mode: Training/inference action mode.
            adapt_to_so101: Apply the SO-100/101 joint frame transform to joint
                observations/actions (needed for zero-shot and fine-tuning from the
                pre-#777 LeRobot calibration checkpoint).
            torch_compile: Whether to enable compile-oriented config flags.

        Raises:
            ValueError: If only one of input_features/output_features is provided.
            RuntimeError: If pretrained checkpoint metadata cannot be resolved.
        """
        super().__init__(n_action_steps=n_action_steps)

        # TODO(alfieroddan): enable more than just continous action mode
        if action_mode != "continuous":
            msg = "Only continous action mode is currently supported."
            raise ValueError(msg)

        # check both exist, raise error if not
        if bool(input_features) != bool(output_features):
            msg = f"Need both input and output features: input: {input_features} - output: {output_features}"
            raise ValueError(msg)

        # if pretrained find hf container
        self.hf_container = None
        if repo_id is not None:
            self.hf_container = load_hf_pretrained_container(repo_id)
            if self.hf_container is None:
                msg = "Failed to resolve pretrained MolmoAct2 checkpoint metadata."
                raise RuntimeError(msg)

        # if self.hf_container exists - we should resolve the config
        if self.hf_container:
            self.config = build_config_from_hf_config(
                self.hf_container.hf_config,
                norm_stats=self.hf_container.norm_stats,
                input_features=input_features,
                output_features=output_features,
                checkpoint_path=self.hf_container.checkpoint_location,
                processor_config=self.hf_container.processor_config,
                n_obs_steps=n_obs_steps,
                norm_tag=norm_tag,
                n_action_steps=n_action_steps,
                action_mode=action_mode,
                torch_compile=torch_compile,
            )
        else:
            self.config = make_molmoact2_config(
                input_features=input_features,
                output_features=output_features,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                action_mode=action_mode,
                torch_compile=torch_compile,
            )

        self._checkpoint_location: str | None = (
            self.hf_container.checkpoint_location if self.hf_container is not None else None
        )

        # SO-100/101 joint calibration correction (must be set before processors build).
        self.config.adapt_to_so101 = adapt_to_so101

        # Keep repo_id in checkpoint hparams so load_from_checkpoint reconstructs
        # the same pretrained source during inference adapter reload.
        self.save_hyperparameters(ignore=["config"])

        self.model: MolmoAct2Model | None = None
        self._preprocessor: MolmoAct2Preprocessor | None = None
        self._postprocessor: MolmoAct2Postprocessor | None = None

        # eagerly load
        if (input_features and output_features) or (repo_id and norm_tag):
            self._initialize_model()

    @staticmethod
    def _dataset_features(train_dataset: Dataset) -> tuple[list[Feature], list[Feature]]:
        input_features = [_coerce_dataset_feature(feature) for feature in train_dataset.observation_features.values()]
        output_features = [_coerce_dataset_feature(feature) for feature in train_dataset.action_features.values()]
        return input_features, output_features

    def _initialize_model(self) -> None:
        """Initialize the model architecture, preprocessors, and pretrained weights.

        Model construction and weight loading are kept as explicit sequential
        steps so each concern is visible and testable independently:

        1. Build preprocessor/postprocessor from config.
        2. Construct the :class:`MolmoAct2Model` (architecture only, no weights).
        3. Load pretrained weights if a checkpoint path is present in the config.
        """
        # make pre, post and model from config
        self._preprocessor, self._postprocessor = make_molmoact2_preprocessors(config=self.config)
        self.model = MolmoAct2Model(self.config)
        if self._checkpoint_location is not None:
            self.model.load_pretrained_weights(self._checkpoint_location)

        # parameter setting based on config
        if self.config.train_action_expert_only:
            self._freeze_non_action_expert_parameters()

    def setup(self, stage: str) -> None:
        """Set up model from datamodule (lazy or fine-tuning path).

        Args:
            stage: Lightning stage identifier (unused; required by the interface).

        Raises:
            TypeError: If the attached train dataset is not a physicalai Dataset.
        """
        del stage
        if self.model is not None:
            return

        if not self.config.input_features or not self.config.output_features:
            datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
            train_dataset = datamodule.train_dataset
            if not isinstance(train_dataset, Dataset):
                msg = f"Expected physicalai Dataset, got {type(train_dataset)}"
                raise TypeError(msg)

            dataset_input_features, dataset_output_features = self._dataset_features(train_dataset)
            if not self.config.input_features:
                self.config.input_features = dataset_input_features
                self.hparams["input_features"] = self.config.input_features
            if not self.config.output_features:
                self.config.output_features = dataset_output_features
                self.hparams["output_features"] = self.config.output_features

        if self.model is None:
            self._initialize_model()

    def _backbone(self) -> torch.nn.Module:
        if self.model is None:
            msg = "Model is not initialized"
            raise RuntimeError(msg)
        return self.model.backbone.model

    def _freeze_non_action_expert_parameters(self) -> None:
        if self.model is None:
            msg = "Model is not initialized"
            raise RuntimeError(msg)
        trainable_params = 0
        for name, param in self.model.named_parameters():
            param.requires_grad = "action_expert" in name
            if param.requires_grad:
                trainable_params += param.numel()
        if trainable_params == 0:
            msg = "train_action_expert_only=true, but no action_expert parameters were found."
            raise RuntimeError(msg)

    @torch.no_grad()
    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Predict an action chunk from an observation batch.

        Args:
            batch: Observation batch to run inference on.

        Returns:
            Predicted action tensor of shape
            ``(batch_size, action_horizon, action_dim)``.

        Raises:
            ValueError: If the model or processors ave not been initialized.
        """
        if self.model is None:
            msg = "Model is not initialized. Call setup() first."
            raise ValueError(msg)
        if self._preprocessor is None or self._postprocessor is None:
            msg = "Processors are not initialized. Call setup() first."
            raise ValueError(msg)

        processed_batch = self._preprocessor(batch.to_dict())
        actions = self.model.predict_action_chunk(processed_batch)
        return self._postprocessor({ACTION: actions})[ACTION]

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Run training or inference forward pass.

        Args:
            batch: Input observation batch.

        Returns:
            Training: tuple of loss tensor and metrics dict.
            Inference: predicted action chunk tensor.

        Raises:
            ValueError: If model or preprocessors are not initialized in training mode.
        """
        if self.training:
            if self.model is None or self._preprocessor is None:
                msg = "Model is not initialized"
                raise ValueError(msg)
            processed_batch = self._preprocessor(batch.to_dict())
            return self.model(processed_batch)
        return self.predict_action_chunk(batch)

    def training_step(self, batch: Observation, batch_idx: int) -> torch.Tensor:
        """Lightning training step.

        Returns:
            Training loss tensor.
        """
        del batch_idx
        loss, loss_dict = self(batch)
        self.log("train/loss", loss_dict["loss"], prog_bar=True)
        return loss

    def compute_val_loss(self, batch: Observation) -> tuple[Tensor, dict[str, float]]:
        """Compute validation loss and metrics.

        Args:
            batch: Input observation batch.

        Returns:
            Validation loss tensor and metrics dictionary.

        Raises:
            ValueError: If model or preprocessors are not initialized.
        """
        if self.model is None or self._preprocessor is None:
            msg = "Model is not initialized"
            raise ValueError(msg)

        processed_batch = self._preprocessor(batch.to_dict())
        return self.model.compute_val_loss(processed_batch)

    def get_optim_params(self) -> list[dict[str, Any]]:
        """Group trainable parameters by component with per-component learning rates.

        Returns:
            AdamW parameter groups for the VLM, ViT, connector and action expert.

        Raises:
            RuntimeError: If the model has not been initialized.
        """
        if self.model is None:
            msg = "Model is not initialized"
            raise RuntimeError(msg)

        grouped: dict[str, list[torch.nn.Parameter]] = {
            "vlm": [],
            "vit": [],
            "connector": [],
            "action_expert": [],
        }
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "action_expert" in name:
                grouped["action_expert"].append(param)
            elif "image_pooling_2d" in name or "image_projector" in name:
                grouped["connector"].append(param)
            elif "vision" in name:
                grouped["vit"].append(param)
            else:
                grouped["vlm"].append(param)

        learning_rates = {
            "vlm": self.config.optimizer_lr,
            "vit": self.config.optimizer_vit_lr,
            "connector": self.config.optimizer_connector_lr,
            "action_expert": self.config.optimizer_action_expert_lr,
        }
        return [{"params": params, "lr": learning_rates[name]} for name, params in grouped.items() if params]

    def configure_optimizers(self) -> dict[str, Any]:
        """Build the AdamW optimizer with grouped learning rates and an LR schedule.

        Returns:
            The Lightning optimizer/scheduler configuration.
        """
        if self.model is not None and self.config.train_action_expert_only:
            self.model.freeze_to_action_expert()

        optimizer = torch.optim.AdamW(
            self.get_optim_params(),
            lr=self.config.optimizer_lr,
            weight_decay=self.config.optimizer_weight_decay,
            betas=self.config.optimizer_betas,
            eps=self.config.optimizer_eps,
        )

        num_training_steps = int(self.trainer.estimated_stepping_batches)
        num_decay_steps = self.config.scheduler_decay_steps
        if num_decay_steps is None:
            num_decay_steps = num_training_steps

        scheduler = cosine_decay_with_warmup_scheduler(
            optimizer,
            peak_lr=self.config.optimizer_lr,
            decay_lr=self.config.scheduler_decay_lr,
            num_warmup_steps=self.config.scheduler_warmup_steps,
            num_decay_steps=int(num_decay_steps),
            num_training_steps=num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Clip gradients using the norm configured on the policy."""
        del gradient_clip_algorithm
        clip_val = gradient_clip_val if gradient_clip_val is not None else self.config.optimizer_grad_clip_norm
        if clip_val and clip_val > 0:
            self.clip_gradients(optimizer, gradient_clip_val=clip_val, gradient_clip_algorithm="norm")

    @property
    def input_features(self) -> list[Feature]:
        """Explicit input feature contract.

        Raises:
            ValueError: If the model has not been initialized with input features.
        """
        if self.config.input_features is None:
            msg = "Model has not been initialized, no input features exist yet."
            raise ValueError(msg)
        return self.config.input_features

    @property
    def output_features(self) -> list[Feature]:
        """Explicit output feature contract.

        Raises:
            ValueError: If the model has not been initialized with output features.
        """
        if self.config.output_features is None:
            msg = "Model has not been initialized, no output features exist yet."
            raise ValueError(msg)
        return self.config.output_features

    @property
    def inputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the policy's expected model inputs for export tracing.

        Derived directly from :attr:`config.input_features`. Returns ``None``
        if the model has not yet been initialized.

        Returns:
            A list of :class:`InferenceFeature` descriptors, or ``None``.
        """
        if self.model is None or self.input_features is None:
            return None

        schema: list[InferenceFeature] = []
        for feature in self.input_features:
            if feature.ftype == FeatureType.VISUAL:
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.VISUAL,
                        shape=tuple(feature.shape),
                        name=f"{IMAGES}.{feature.name}",
                        dtype=InferenceFeatureDtype.FLOAT32,
                    ),
                )
            elif feature.ftype == FeatureType.STATE:
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.STATE,
                        shape=tuple(feature.shape),
                        name=str(feature.name),
                        dtype=InferenceFeatureDtype.FLOAT32,
                    ),
                )
            schema.append(
                InferenceFeature(
                    ftype=InferenceFeatureType.LANGUAGE,
                    shape=(),
                    name=TASK,
                    dtype=InferenceFeatureDtype.STRING,
                ),
            )
        return schema

    @property
    def outputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the policy's model output for export.

        Derived directly from :attr:`config.output_features`. Returns ``None``
        if the model has not yet been initialized.

        Returns:
            A list of :class:`InferenceFeature` descriptors, or ``None``.
        """
        if self.model is None or self.output_features is None:
            return None

        return [
            InferenceFeature(
                ftype=InferenceFeatureType.ACTION,
                shape=(self.config.n_action_steps, *tuple(feature.shape)),
                name=ACTION,
                dtype=InferenceFeatureDtype.FLOAT32,
            )
            for feature in self.output_features
        ]

    @property
    def extra_export_args(self) -> dict[str, ExportParameters]:
        """Extra backend export args for inference-time pre/post graph components."""
        normalize_by_mode: dict[str, dict[str, dict[str, list[float]]]] = {}
        for feature in self.config.input_features or []:
            if not feature.name:
                continue
            mode = _feature_normalization_mode(feature)
            normalize_by_mode.setdefault(mode, {})[feature.name] = _feature_normalization_stats(feature, mode)

        action_feature = next((f for f in (self.config.output_features or []) if f.ftype == FeatureType.ACTION), None)
        denorm_by_mode: dict[str, dict[str, dict[str, list[float]]]] = {}
        if action_feature is not None:
            action_mode = _feature_normalization_mode(action_feature)
            action_stats = _feature_normalization_stats(action_feature, action_mode)
            denorm_by_mode.setdefault(action_mode, {})[ACTION] = action_stats

        output_names = [feature.name for feature in (self.outputs_schema or [])]

        preproc_specs = [
            *[
                ComponentSpec(type="normalize", mode=mode, stats=stats)
                for mode, stats in normalize_by_mode.items()
                if stats
            ],
            ComponentSpec(
                type="molmoact2_pre",
                tokenizer_name_or_path=self.config.tokenizer_name_or_path,
                num_state_tokens=self.config.num_state_tokens,
                setup_type=self.config.setup_type,
                control_mode=self.config.control_mode,
                add_setup_tokens=self.config.add_setup_tokens,
                add_control_tokens=self.config.add_control_tokens,
                image_processor_config=self._export_image_processor_config(),
                model_input_config=self._export_model_input_config(),
            ),
        ]

        postproc_specs = [
            ComponentSpec(type="molmoact2_post"),
            *[
                ComponentSpec(type="denormalize", mode=mode, stats=stats)
                for mode, stats in denorm_by_mode.items()
                if stats
            ],
        ]

        return {
            "openvino": OpenVINOExportParameters(
                outputs=output_names,
                preprocessors_specs=preproc_specs,
                postprocessors_specs=postproc_specs,
            ),
            "torch": TorchExportParameters(
                input_names=[TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK, IMAGES, IMAGE_MASKS, STATE],
                output_names=output_names,
            ),
        }

    def _export_image_processor_config(self) -> dict[str, Any] | None:
        """Serialize the image patchify config for the inference-side preprocessor.

        Returns:
            A JSON-serializable image-processor config, or ``None`` when the
            processor config is unavailable.
        """
        processor_config = self.config.processor_config
        if processor_config is None:
            return None
        image_processor = processor_config.image_processor
        return {
            "size": dict(image_processor.size),
            "image_mean": list(image_processor.image_mean),
            "image_std": list(image_processor.image_std),
            "patch_size": int(image_processor.patch_size),
            "pooling_size": list(image_processor.pooling_size),
            "crop_mode": str(image_processor.crop_mode),
        }

    def _export_model_input_config(self) -> dict[str, Any] | None:
        """Serialize token ids and layout flags for the inference-side preprocessor.

        Returns:
            A JSON-serializable model-input config, or ``None`` when the
            processor config is unavailable.
        """
        processor_config = self.config.processor_config
        if processor_config is None:
            return None
        action_feature = next(
            (f for f in (self.config.output_features or []) if f.ftype == FeatureType.ACTION),
            None,
        )
        env_action_dim = int(action_feature.shape[-1]) if action_feature and action_feature.shape else 0
        return {
            "image_placeholder_token_id": self.config.image_placeholder_token_id,
            "image_patch_id": self.config.image_patch_id,
            "image_start_token_id": self.config.image_start_token_id,
            "image_end_token_id": self.config.image_end_token_id,
            "image_col_id": self.config.image_col_id,
            "low_res_image_start_token_id": self.config.low_res_image_start_token_id,
            "frame_start_token_id": self.config.frame_start_token_id,
            "frame_end_token_id": self.config.frame_end_token_id,
            "image_low_res_id": self.config.image_low_res_id,
            "image_use_col_tokens": bool(processor_config.image_use_col_tokens),
            "use_single_crop_col_tokens": processor_config.use_single_crop_col_tokens,
            "use_single_crop_start_token": bool(processor_config.use_single_crop_start_token),
            "max_action_dim": int(self.config.max_action_dim),
            "env_action_dim": env_action_dim,
        }

    @torch.no_grad()
    def to_openvino(
        self,
        output_path: PathLike | str,
        input_sample: dict[str, torch.Tensor] | None = None,
        **export_kwargs: dict,
    ) -> None:
        """Export to OpenVINO, then canonicalize the model's input tensor names.

        The base export names inputs from the traced graph. MolmoAct2 passes
        ``images`` and ``action_dim_is_pad`` straight through to reused/aliased
        ops, so their canonical ``any_name`` can resolve to a numeric alias even
        though the friendly name is present. That breaks name-based input
        matching in ``InferenceModel``. This override re-pins each input tensor to
        its friendly name after the base export completes.

        Args:
            output_path: Directory or file path for the exported ``.xml``.
            input_sample: Optional sample used for tracing; defaults to the
                policy's preprocessed sample input.
            **export_kwargs: Forwarded to the base OpenVINO export.
        """
        super().to_openvino(output_path, input_sample=input_sample, **export_kwargs)
        self._canonicalize_openvino_input_names(output_path, input_sample)

    def _canonicalize_openvino_input_names(
        self,
        output_path: PathLike | str,
        input_sample: dict[str, torch.Tensor] | None,
    ) -> None:
        """Re-pin exported OpenVINO input tensors to their friendly names.

        The renamed model is written to sibling temp files and then atomically
        moved over the originals. Saving directly back over the just-read
        ``.xml``/``.bin`` would overwrite the memory-mapped weights while they are
        still mapped, which crashes the process with ``SIGBUS`` and leaves an
        empty ``.xml``.

        Args:
            output_path: The export path passed to :meth:`to_openvino`.
            input_sample: The tracing sample, or ``None`` to reuse the default.
        """
        import openvino  # noqa: PLC0415

        sample = input_sample if input_sample is not None else self._get_default_export_input_sample()
        if not sample:
            return

        expected_names = list(sample.keys())
        model_path = self._prepare_export_path(output_path, ".xml")
        ov_model = openvino.Core().read_model(str(model_path))

        renamed = False
        for model_input in ov_model.inputs:
            current_names = set(model_input.get_names())
            match = next((name for name in expected_names if name in current_names), None)
            if match is not None and model_input.get_any_name() != match:
                model_input.tensor.set_names({match})
                renamed = True

        if not renamed:
            return

        tmp_xml = model_path.parent / f"{model_path.stem}_named.xml"
        tmp_bin = tmp_xml.with_suffix(".bin")
        openvino.save_model(ov_model, str(tmp_xml))
        # Release the mmap on the original weights before replacing them.
        del ov_model
        tmp_xml.replace(model_path)
        tmp_bin.replace(model_path.with_suffix(".bin"))

    @staticmethod
    def get_supported_export_backends() -> list[str | ExportBackend]:
        """Get a list of export backends supported by policy.

        This method returns a list of supported export backends as strings.

        Returns:
            list[str | ExportBackend]: A list of supported export backends.
        """
        return [ExportBackend.TORCH, ExportBackend.OPENVINO]
