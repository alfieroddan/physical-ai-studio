# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 policy implementation."""

import dataclasses
from pathlib import Path
from typing import IO, Any

import torch
from physicalai.inference.data import InferenceFeature, InferenceFeatureDtype, InferenceFeatureType
from physicalai.inference.manifest import ComponentSpec
from torch import Tensor

from physicalai.data.dataset import Dataset
from physicalai.data.observation import ACTION, IMAGES, TASK, Feature, FeatureType, NormalizationParameters, Observation
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.export.backends import ExportParameters, TorchExportParameters
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
    **overrides: object,
) -> MolmoAct2Config:
    """Create a flat config from defaults and explicit overrides.

    Args:
        input_features: Optional observation schema. Populated lazily from the
            training dataset when omitted.
        output_features: Optional action schema. Populated lazily from the
            training dataset when omitted.
        **overrides: Named :class:`MolmoAct2Config` values. ``None`` values do
            not override the dataclass defaults.

    Returns:
        The resolved flat :class:`MolmoAct2Config`.

    Raises:
        TypeError: If an override is not a config field.
    """
    valid_fields = {field.name for field in dataclasses.fields(MolmoAct2Config)}
    unknown = set(overrides) - valid_fields
    if unknown:
        msg = f"Unknown MolmoAct2 override(s): {sorted(unknown)}"
        raise TypeError(msg)
    config = MolmoAct2Config(
        input_features=input_features,
        output_features=output_features,
    )
    for name, value in overrides.items():
        if value is not None:
            setattr(config, name, value)
    config.__post_init__()
    return config


class MolmoAct2(ExportablePolicyMixin, Policy):
    """MolmoAct2 Policy."""

    def __init__(
        self,
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        *,
        repo_id: str | Path | None = None,
        norm_tag: str | None = None,
        adapt_to_so101: bool | None = None,
        compile_model: bool | None = None,
        load_weights: bool = True,
        **overrides: object,
    ) -> None:
        """Initialize a MolmoAct2 policy wrapper.

        Args:
            input_features: Optional observation feature schema.
            output_features: Optional action feature schema.
            input_features: Optional observation schema. Supply both feature
                lists for eager initialization, or omit both to infer schemas
                from the training dataset during ``setup``.
            output_features: Optional action schema. Must be supplied together
                with ``input_features``.
            repo_id: Optional local checkpoint directory or Hugging Face repo.
                Its checkpoint config is the base when supplied; otherwise
                :class:`MolmoAct2Config` defaults are the base.
            norm_tag: Optional normalization metadata tag to select schemas and
                prompt conditioning from a pretrained checkpoint.
            adapt_to_so101: Apply the SO-100/101 joint frame transform to joint
                observations and actions for pre-#777 LeRobot calibration.
            compile_model: Explicit override for
                :attr:`MolmoAct2Config.compile_model`; enables compiled model
                forward and inference paths.
            load_weights: Whether to load base checkpoint weights after model
                construction when a checkpoint source is available.
            **overrides: Any other named :class:`MolmoAct2Config` field. A
                non-``None`` value overrides the selected base config; ``None``
                preserves the pretrained value or dataclass default.

        Raises:
            ValueError: If only one of input_features/output_features is provided.
            RuntimeError: If pretrained checkpoint metadata cannot be resolved.
        """
        # check both either exist or both don't exit, raise error if not
        if bool(input_features) != bool(output_features):
            msg = f"Need both input and output features: input: {input_features} - output: {output_features}"
            raise ValueError(msg)

        self.hf_container = None
        if compile_model is not None:
            overrides["compile_model"] = compile_model

        # if pretrained find hf container
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
                repo_id=self.hf_container.repo_id,
                tokenizer_config=self.hf_container.tokenizer_config,
                processor_config=self.hf_container.processor_config,
                norm_tag=norm_tag,
                **overrides,
            )
        else:
            self.config = make_molmoact2_config(
                input_features=input_features,
                output_features=output_features,
                norm_tag=norm_tag,
                **overrides,
            )

        if self.config.action_mode != "continuous":
            msg = "Only continous action mode is currently supported."
            raise ValueError(msg)
        super().__init__(n_action_steps=self.config.n_action_steps)

        self._checkpoint_location = self.hf_container.checkpoint_location if self.hf_container is not None else None

        # SO-101 joint calibration correction (must be set before processors build).
        # Applied uniformly regardless of which branch above built the config.
        # https://huggingface.co/docs/lerobot/v0.6.0/en/molmoact2#joint-frame-transform-so-100101-zero-shot
        if adapt_to_so101 is not None:
            self.config.adapt_to_so101 = adapt_to_so101

        # Explicit setup_type/control_mode always win over whatever a
        # norm_tag lookup produced (or the "" default when there was no
        # norm_tag), so any dataset can supply this prompt-conditioning text
        # without needing a matching entry in a pretrained checkpoint's
        # norm_stats.json. Must be set before processors build.
        # Keep repo_id in checkpoint hparams so load_from_checkpoint reconstructs
        # the same pretrained source during inference adapter reload.
        self.save_hyperparameters(ignore=["config", "load_weights"])

        self.model: MolmoAct2Model | None = None  # pyrefly: ignore[bad-override-mutable-attribute]
        self._preprocessor: MolmoAct2Preprocessor | None = None  # pyrefly: ignore[bad-override-mutable-attribute]
        self._postprocessor: MolmoAct2Postprocessor | None = None
        self._load_weights = load_weights

        # eagerly load
        if (input_features and output_features) or (repo_id and norm_tag):
            self._initialize_model()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | IO[bytes],
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> "MolmoAct2":
        """Reload a policy from a Lightning checkpoint.

        A Lightning checkpoint already carries its own trained state dict,
        which is applied on top of the freshly constructed module right
        after ``__init__`` returns. Eagerly loading pretrained weights
        during that ``__init__`` call is therefore pure overhead (network
        or disk I/O for weights that are about to be discarded), so
        ``load_weights`` defaults to ``False`` here unless the caller
        explicitly overrides it.

        Args:
            checkpoint_path: Path (or file-like) to the Lightning checkpoint.
            *args: Forwarded to ``Policy.load_from_checkpoint``.
            **kwargs: Forwarded to ``Policy.load_from_checkpoint``. May
                include an explicit ``load_weights=True`` to force the
                pretrained-weight load anyway.

        Returns:
            The reconstructed :class:`MolmoAct2` policy with the
            checkpoint's state dict applied.
        """
        kwargs.setdefault("load_weights", False)
        return super().load_from_checkpoint(checkpoint_path, *args, **kwargs)

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
        if self._checkpoint_location is not None and self._load_weights:
            self.model.load_pretrained_weights(self._checkpoint_location)

        # Apply LoRA adapters after weight loading so pretrained parameters are
        # preserved and only the low-rank updates are trainable.
        if self.config.use_lora:
            self.model.apply_lora_adapters()

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

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, Tensor | float]]:
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

    def configure_optimizers(self) -> dict[str, Any]:  # pyrefly: ignore[bad-override]
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

        Raises:
            ValueError: If any input feature lacks a concrete shape.
        """
        if self.model is None or self.input_features is None:
            return None

        schema: list[InferenceFeature] = []
        for feature in self.input_features:
            if feature.shape is None:
                msg = "input feature missing concrete shape for export"
                raise ValueError(msg)
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

        Raises:
            ValueError: If any output feature lacks a concrete shape.
        """
        if self.model is None or self.output_features is None:
            return None

        outputs: list[InferenceFeature] = []
        for feature in self.output_features:
            if feature.shape is None:  # pragma: no cover - export requires concrete shapes
                msg = "output feature missing concrete shape for export"
                raise ValueError(msg)
            outputs.append(
                InferenceFeature(
                    ftype=InferenceFeatureType.ACTION,
                    shape=(self.config.n_action_steps, *tuple(feature.shape)),
                    name=ACTION,
                    dtype=InferenceFeatureDtype.FLOAT32,
                ),
            )
        return outputs

    @property
    def extra_export_args(self) -> dict[str, ExportParameters]:
        """Extra backend export args for inference-time pre/post graph components."""
        return {
            "torch": TorchExportParameters(
                preprocessors_specs=[ComponentSpec(type="to_float_tensor")],
            ),
        }

    @staticmethod
    def get_supported_export_backends() -> list[str | ExportBackend]:
        """Get a list of export backends supported by policy.

        This method returns a list of supported export backends as strings.

        Returns:
            list[str | ExportBackend]: A list of supported export backends.
        """
        return [ExportBackend.TORCH]
