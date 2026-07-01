# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 policy implementation."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from physicalai.data.dataset import Dataset
from physicalai.inference.data import InferenceFeature, InferenceFeatureDtype, InferenceFeatureType
from physicalai.inference.manifest import ComponentSpec
from physicalai.train.schedulers import cosine_decay_with_warmup_scheduler

from physicalai.data.observation import ACTION, IMAGES, TASK, Feature, FeatureType, NormalizationParameters, Observation
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.export.backends import ONNXExportParameters, OpenVINOExportParameters, TorchExportParameters
from physicalai.policies.base import Policy

from .config import MolmoAct2Config
from .from_hf import build_config_from_hf_config, load_hf_pretrained_container
from .model import MolmoAct2Model
from .processors import make_molmoact2_preprocessors

if TYPE_CHECKING:
    from .processors import MolmoAct2Postprocessor, MolmoAct2Preprocessor


def make_molmoact2_config(
    *,
    input_features: list[Feature],
    output_features: list[Feature],
    n_obs_steps: int,
    n_action_steps: int,
) -> MolmoAct2Config:
    """Create the explicit model config for MolmoAct2.

    This function is the non-policy home for model-definition defaults.

    Args:
        input_features: List of input features the model consumes.
        output_features: List of output features the model produces.
        n_obs_steps: Number of observation steps.
        n_action_steps: Number of action steps.

    Returns:
        A fully populated :class:`MolmoAct2Config`.
    """
    return MolmoAct2Config(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
    )


def _identity_normalization_for_feature(feature: Feature) -> Feature:
    if feature.normalization_data is not None or feature.ftype not in {FeatureType.STATE, FeatureType.ACTION}:
        return feature

    feature_shape = tuple(feature.shape or ())
    dim = int(feature_shape[0]) if feature_shape else 1
    return Feature(
        name=feature.name,
        ftype=feature.ftype,
        shape=feature.shape,
        normalization_data=NormalizationParameters(
            mean=[0.0] * dim,
            std=[1.0] * dim,
            q01=[-1.0] * dim,
            q99=[1.0] * dim,
        ),
    )


def attach_identity_normalization(features: list[Feature] | None) -> list[Feature]:
    """Attach identity-like normalization to state/action features missing stats.

    Returns:
        Feature list with identity-like normalization attached where needed.
    """
    return [_identity_normalization_for_feature(feature) for feature in (features or [])]


class MolmoAct2(ExportablePolicyMixin, Policy):
    """MolmoAct2 Policy."""

    def __init__(
        self,
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        repo_id: str | Path | None = "allenai/MolmoAct2",
        norm_tag: str | None = "so100_so101_molmoact2",
        n_obs_steps: int = 30,
        n_action_steps: int = 30,
    ) -> None:
        """Initialize MolmoAct2 policy.

        When ``repo_id`` is provided the config is built
        from the HuggingFace checkpoint and ``norm_tag`` is required to resolve
        normalisation statistics. Otherwise a fresh config is built from the
        provided features and step counts.

        Args:
            input_features: Input features the policy consumes. Required.
            output_features: Output features the policy produces. Required.
            repo_id: HuggingFace repo ID or local path to
                a pretrained checkpoint. When given, weights are loaded during
                :meth:`_initialize_model`.
            norm_tag: Tag used to select normalisation statistics from
                ``norm_stats.json`` from repo_id.
            n_obs_steps: Number of observation steps.
            n_action_steps: Number of action steps.

        Raises:
            ValueError: If ``input_features`` or ``output_features`` are not
                provided and cannot be resolved from pretrained metadata or the
                attached dataset.
        """
        super().__init__(n_action_steps=n_action_steps)

        self.hf_container = None
        self._repo_id = repo_id
        self._norm_tag = norm_tag
        self._n_obs_steps = n_obs_steps
        self._n_action_steps = n_action_steps

        eager_input_features = attach_identity_normalization(input_features)
        eager_output_features = attach_identity_normalization(output_features)

        if repo_id is not None:
            self.hf_container = load_hf_pretrained_container(repo_id)

        self.config = make_molmoact2_config(
            input_features=eager_input_features,
            output_features=eager_output_features,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
        )

        can_initialize_eagerly = bool(eager_input_features and eager_output_features)
        if repo_id is not None and norm_tag is not None:
            hf_container = self.hf_container
            if hf_container is None:
                msg = "Failed to resolve pretrained MolmoAct2 checkpoint metadata."
                raise RuntimeError(msg)
            self.config = build_config_from_hf_config(
                hf_container.hf_config,
                norm_stats=hf_container.norm_stats,
                input_features=eager_input_features or None,
                output_features=eager_output_features or None,
                n_obs_steps=n_obs_steps,
                norm_tag=norm_tag,
                n_action_steps=n_action_steps,
                checkpoint_path=hf_container.checkpoint_location,
                processor_config=hf_container.processor_config,
            )
            can_initialize_eagerly = True
        elif repo_id is not None and can_initialize_eagerly:
            hf_container = self.hf_container
            if hf_container is None:
                msg = "Failed to resolve pretrained MolmoAct2 checkpoint metadata."
                raise RuntimeError(msg)
            self.config = build_config_from_hf_config(
                hf_container.hf_config,
                norm_stats=hf_container.norm_stats,
                input_features=eager_input_features,
                output_features=eager_output_features,
                n_obs_steps=n_obs_steps,
                norm_tag=None,
                n_action_steps=n_action_steps,
                checkpoint_path=hf_container.checkpoint_location,
                processor_config=hf_container.processor_config,
            )
        else:
            can_initialize_eagerly = can_initialize_eagerly and repo_id is None

        if repo_id is None and not can_initialize_eagerly:
            msg = "Model requires input and output features when repo_id is not provided."
            raise ValueError(msg)

        self._checkpoint_location: str | None = (
            self.hf_container.checkpoint_location if self.hf_container is not None else None
        )

        # Keep repo_id in checkpoint hparams so load_from_checkpoint reconstructs
        # the same pretrained source during inference adapter reload.
        self.save_hyperparameters(ignore=["config"])

        self.model: MolmoAct2Model | None = None
        self._preprocessor: MolmoAct2Preprocessor | None = None
        self._postprocessor: MolmoAct2Postprocessor | None = None

        if can_initialize_eagerly:
            # Build module tree eagerly so Lightning load_from_checkpoint can restore
            # state_dict without requiring an explicit setup() call.
            self._initialize_model()

    def _attach_features(
        self,
        *,
        input_features: list[Feature],
        output_features: list[Feature],
    ) -> None:
        attached_input_features = attach_identity_normalization(input_features)
        attached_output_features = attach_identity_normalization(output_features)

        if self.hf_container is not None:
            self.config = build_config_from_hf_config(
                self.hf_container.hf_config,
                norm_stats=self.hf_container.norm_stats,
                input_features=attached_input_features,
                output_features=attached_output_features,
                n_obs_steps=self._n_obs_steps,
                norm_tag=self._norm_tag,
                n_action_steps=self._n_action_steps,
                checkpoint_path=self.hf_container.checkpoint_location,
                processor_config=self.hf_container.processor_config,
            )
            return

        self.config = make_molmoact2_config(
            input_features=attached_input_features,
            output_features=attached_output_features,
            n_obs_steps=self._n_obs_steps,
            n_action_steps=self._n_action_steps,
        )

    @staticmethod
    def _dataset_features(train_dataset: Dataset) -> tuple[list[Feature], list[Feature]]:
        return list(train_dataset.observation_features.values()), list(train_dataset.action_features.values())

    def _initialize_model(self) -> None:
        """Initialize the model architecture, preprocessors, and pretrained weights.

        Model construction and weight loading are kept as explicit sequential
        steps so each concern is visible and testable independently:

        1. Build preprocessor/postprocessor from config.
        2. Construct the :class:`MolmoAct2Model` (architecture only, no weights).
        3. Load pretrained weights if a checkpoint path is present in the config.
        """
        self._preprocessor, self._postprocessor = make_molmoact2_preprocessors(
            config=self.config,
        )

        self.model = MolmoAct2Model(self.config)

        if self._checkpoint_location is not None:
            self.model.load_pretrained_weights(self._checkpoint_location)

        if self.config.freeze_embedding:
            self._freeze_input_embeddings()
        if self.config.train_action_expert_only:
            self._freeze_non_action_expert_parameters()
        if self.config.gradient_checkpointing:
            self._enable_gradient_checkpointing()

    def _backbone(self):
        if self.model is None:
            msg = "Model is not initialized"
            raise RuntimeError(msg)
        return self.model.backbone.model

    def _enable_gradient_checkpointing(self) -> None:
        backbone = self._backbone()
        transformer = getattr(backbone, "transformer", None)
        if transformer is None:
            msg = "gradient_checkpointing=true, but MolmoAct2 exposes no text transformer."
            raise RuntimeError(msg)
        transformer.gradient_checkpointing = True
        vision_backbone = getattr(backbone, "vision_backbone", None)
        if vision_backbone is not None:
            vision_backbone.gradient_checkpointing = True

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

    def _freeze_input_embeddings(self) -> None:
        backbone = self._backbone()
        embedding_modules: list[torch.nn.Module] = []
        seen_module_ids: set[int] = set()
        for module in (self.model.backbone, backbone):
            get_input_embeddings = getattr(module, "get_input_embeddings", None)
            if not callable(get_input_embeddings):
                continue
            embeddings = get_input_embeddings()
            if embeddings is None or id(embeddings) in seen_module_ids:
                continue
            embedding_modules.append(embeddings)
            seen_module_ids.add(id(embeddings))

        if not embedding_modules:
            msg = "freeze_embedding=true, but MolmoAct2 checkpoint exposes no input embeddings."
            raise RuntimeError(msg)

        lm_head = getattr(self.model.backbone, "lm_head", None)
        lm_head_params = {id(param) for param in lm_head.parameters()} if lm_head is not None else set()
        embedding_params = [param for embeddings in embedding_modules for param in embeddings.parameters()]
        if any(id(param) in lm_head_params for param in embedding_params):
            msg = (
                "freeze_embedding=true would also freeze lm_head because input embeddings and lm_head "
                "share parameters in this checkpoint."
            )
            raise RuntimeError(msg)
        for param in embedding_params:
            param.requires_grad = False

    def get_optim_params(self) -> list[dict[str, Any]]:
        if self.model is None:
            msg = "Model is not initialized"
            raise RuntimeError(msg)

        vit_params: list[Tensor] = []
        connector_params: list[Tensor] = []
        action_expert_params: list[Tensor] = []
        vlm_params: list[Tensor] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "action_expert" in name:
                action_expert_params.append(param)
            elif any(part in name for part in ("image_pooling_2d", "image_projector")):
                connector_params.append(param)
            elif any(part in name for part in ("vision", "image_encoder", "vit")):
                vit_params.append(param)
            elif any(part in name for part in ("multi_modal_projector", "connector", "mm_projector")):
                connector_params.append(param)
            else:
                vlm_params.append(param)

        groups: list[dict[str, Any]] = []
        if vlm_params:
            groups.append({"params": vlm_params, "lr": self.config.optimizer_lr})
        if vit_params:
            groups.append({"params": vit_params, "lr": self.config.optimizer_vit_lr})
        if connector_params:
            groups.append({"params": connector_params, "lr": self.config.optimizer_connector_lr})
        if action_expert_params:
            groups.append({"params": action_expert_params, "lr": self.config.optimizer_action_expert_lr})
        return groups

    def setup(self, stage: str) -> None:
        """Set up model from datamodule (lazy or fine-tuning path).

        Args:
            stage: Lightning stage identifier (unused; required by the interface).

        Raises:
            TypeError: If the attached train dataset is not a physicalai Dataset.
        """
        del stage
        if self.model is not None and self._preprocessor is not None and self._postprocessor is not None:
            return

        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
        train_dataset = datamodule.train_dataset
        if not isinstance(train_dataset, Dataset):
            msg = f"Expected physicalai Dataset, got {type(train_dataset)}"
            raise TypeError(msg)

        input_features, output_features = self._dataset_features(train_dataset)
        self._attach_features(
            input_features=input_features,
            output_features=output_features,
        )
        if self.model is None or self._preprocessor is None or self._postprocessor is None:
            self._initialize_model()

    @torch.no_grad()
    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Predict an action chunk from an observation batch.

        Args:
            batch: Observation batch to run inference on.

        Returns:
            Predicted action tensor of shape
            ``(batch_size, action_horizon, action_dim)``.

        Raises:
            ValueError: If the model or processors have not been initialized.
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Call setup() first.")
        if self._preprocessor is None or self._postprocessor is None:
            raise ValueError("Processors are not initialized. Call setup() first.")

        processed_batch = self._preprocessor(batch.to_dict(flatten=True))
        actions = self.model.predict_action_chunk(processed_batch)
        return self._postprocessor(actions)

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Forward pass through the policy.

        Args:
            batch: Observation batch.

        Returns:
            Predicted action tensor, or a tuple of (loss, metrics) during training.
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
        """Compute validation loss on a batch.

        Delegates to the model's ``compute_val_loss`` without toggling train mode.
        """
        if self.model is None or self._preprocessor is None:
            msg = "Model is not initialized"
            raise ValueError(msg)

        processed_batch = self._preprocessor(batch.to_dict())
        return self.model.compute_val_loss(processed_batch)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure MolmoAct2 fine-tuning optimizer and scheduler.

        Uses LeRobot's MolmoAct2 parameter-group learning rates and a cosine
        decay schedule with linear warmup as the local best-guess preset.
        """
        optimizer = torch.optim.AdamW(
            self.get_optim_params(),
            lr=self.config.optimizer_lr,
            weight_decay=self.config.optimizer_weight_decay,
            betas=self.config.optimizer_betas,
            eps=self.config.optimizer_eps,
        )

        num_training_steps = self.trainer.estimated_stepping_batches
        num_decay_steps = self.config.scheduler_decay_steps
        if num_decay_steps is None:
            num_decay_steps = num_training_steps

        scheduler = cosine_decay_with_warmup_scheduler(
            optimizer,
            peak_lr=self.config.optimizer_lr,
            decay_lr=self.config.scheduler_decay_lr,
            num_warmup_steps=self.config.scheduler_warmup_steps,
            num_decay_steps=num_decay_steps,
            num_training_steps=num_training_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Configure gradient clipping from policy config."""
        clip_val = gradient_clip_val if gradient_clip_val is not None else self.config.optimizer_grad_clip_norm
        if clip_val and clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm or "norm",
            )

    @property
    def input_features(self) -> list[Feature]:
        """Return the explicit input feature contract."""
        return self.config.input_features

    @property
    def output_features(self) -> list[Feature]:
        """Return the explicit output feature contract."""
        return self.config.output_features

    @property
    def inputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the policy's expected model inputs for export tracing.

        Derived directly from :attr:`config.input_features`. Returns ``None``
        if the model has not yet been initialized.

        Returns:
            A list of :class:`InferenceFeature` descriptors, or ``None``.
        """
        if self.model is None:
            return None

        schema: list[InferenceFeature] = []
        for feature in self.config.input_features:
            if feature.ftype == FeatureType.VISUAL:
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.VISUAL,
                        shape=feature.shape,
                        name=f"{IMAGES}.{feature.name}",
                        dtype=InferenceFeatureDtype.FLOAT32,
                    ),
                )
            elif feature.ftype == FeatureType.STATE:
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.STATE,
                        shape=feature.shape,
                        name=feature.name,
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
        if self.model is None:
            return None

        return [
            InferenceFeature(
                ftype=InferenceFeatureType.ACTION,
                shape=(self.config.n_action_steps, *feature.shape),
                name=feature.name,
                dtype=InferenceFeatureDtype.FLOAT32,
            )
            for feature in self.config.output_features
        ]

    @property
    def extra_export_args(self) -> dict[str, object]:
        """Extra backend export args for inference-time pre/post graph components."""

        def _as_float_list(value: object) -> list[float]:
            if torch.is_tensor(value):
                return [float(x) for x in value.detach().cpu().reshape(-1).tolist()]
            if isinstance(value, (list, tuple)):
                return [float(x) for x in value]
            if isinstance(value, (int, float)):
                return [float(value)]
            msg = f"Unsupported normalization value type: {type(value)}"
            raise TypeError(msg)

        state_feature = next((f for f in self.config.input_features if f.ftype == FeatureType.STATE), None)
        action_feature = next((f for f in self.config.output_features if f.ftype == FeatureType.ACTION), None)
        visual_features = [f for f in self.config.input_features if f.ftype == FeatureType.VISUAL and f.name]

        state_stats: dict[str, list[float]] | None = None
        if state_feature is not None and state_feature.normalization_data is not None:
            state_stats = {
                "q01": _as_float_list(state_feature.normalization_data.q01),
                "q99": _as_float_list(state_feature.normalization_data.q99),
            }
            if state_feature.normalization_data.mask is not None:
                state_stats["mask"] = _as_float_list(state_feature.normalization_data.mask)

        action_stats: dict[str, list[float]] | None = None
        if action_feature is not None and action_feature.normalization_data is not None:
            action_stats = {
                "q01": _as_float_list(action_feature.normalization_data.q01),
                "q99": _as_float_list(action_feature.normalization_data.q99),
            }
            if action_feature.normalization_data.mask is not None:
                action_stats["mask"] = _as_float_list(action_feature.normalization_data.mask)

        image_keys = [feature.name for feature in visual_features if feature.name]
        env_action_dim = int(action_feature.shape[0]) if action_feature is not None and action_feature.shape else int(
            self.config.max_action_dim,
        )

        molmoact2_pre = ComponentSpec.model_validate(
            {
                "type": "molmoact2_pre",
                "tokenizer_name_or_path": str(self.config.tokenizer_name_or_path),
                "num_state_tokens": int(self.config.num_state_tokens),
                "setup_type": str(self.config.setup_type or ""),
                "control_mode": str(self.config.control_mode or ""),
                "add_setup_tokens": bool(self.config.add_setup_tokens),
                "add_control_tokens": bool(self.config.add_control_tokens),
                "state_stats": state_stats,
                "image_keys": image_keys,
            },
        )
        molmoact2_post = ComponentSpec.model_validate(
            {
                "type": "molmoact2_post",
                "action_key": ACTION,
                "env_action_dim": env_action_dim,
                "action_stats": action_stats,
            },
        )

        output_names = [feature.name for feature in (self.outputs_schema or [])]

        return {
            "onnx": ONNXExportParameters(
                exporter_kwargs={
                    "output_names": output_names,
                },
                preprocessors_specs=[molmoact2_pre],
                postprocessors_specs=[molmoact2_post],
                export_tokenizer=False,
            ),
            "openvino": OpenVINOExportParameters(
                outputs=output_names,
                preprocessors_specs=[molmoact2_pre],
                postprocessors_specs=[molmoact2_post],
                export_tokenizer=False,
            ),
            "torch": TorchExportParameters(),
        }

    @staticmethod
    def get_supported_export_backends() -> list[str | ExportBackend]:
        """Get a list of export backends supported by policy.

        This method returns a list of supported export backends as strings.

        Returns:
            list[str | ExportBackend]: A list of supported export backends.
        """
        return [ExportBackend.TORCH, ExportBackend.OPENVINO]
