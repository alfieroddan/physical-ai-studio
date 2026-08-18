# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 policy implementation."""

import logging
import warnings
from pathlib import Path
from typing import IO, Any, Literal

import torch
from physicalai.inference.data import InferenceFeature, InferenceFeatureDtype, InferenceFeatureType
from physicalai.inference.manifest import ComponentSpec
from torch import Tensor

from physicalai.data.dataset import Dataset
from physicalai.data.observation import ACTION, IMAGES, TASK, Feature, FeatureType, NormalizationParameters, Observation
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.export.backends import ExportParameters, OpenVINOExportParameters, TorchExportParameters
from physicalai.policies.base import Policy
from physicalai.policies.utils.features import get_feature_by_type
from physicalai.train.schedulers import cosine_decay_with_warmup_scheduler

from .config import MolmoAct2Config
from .from_hf import build_config_from_hf_config, load_hf_pretrained_container, resolve_tokenizer_assets
from .model import MolmoAct2Model
from .processors import MolmoAct2Postprocessor, MolmoAct2Preprocessor, make_molmoact2_preprocessors

logger = logging.getLogger(__name__)

_NormalizationStats = dict[str, float | list[float] | list[bool] | None]


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


def _normalization_stats(feature: Feature | None) -> _NormalizationStats:
    if feature is None or feature.normalization_data is None:
        return {}

    normalization = feature.normalization_data
    stats: _NormalizationStats = {
        "q01": normalization.q01,
        "q99": normalization.q99,
    }
    if normalization.mask is not None:
        stats["mask"] = normalization.mask
    return stats


def make_molmoact2_config(  # noqa: PLR0913
    *,
    input_features: list[Feature] | None,
    output_features: list[Feature] | None,
    norm_tag: str | None = None,
    n_obs_steps: int | None = None,
    chunk_size: int | None = None,
    n_action_steps: int | None = None,
    use_random_input_noise: bool | None = None,
    setup_type: str | None = None,
    control_mode: str | None = None,
    compile_model: bool | None = None,
    openvino_compress_to_fp16: bool | None = None,
    model_dtype: Literal["float32", "bfloat16", "float16"] | None = None,
    train_action_expert_only: bool | None = None,
    gradient_checkpointing: bool | None = None,
    use_lora: bool | None = None,
    enable_lora_action_expert: bool | None = None,
    lora_rank: int | None = None,
    lora_alpha: int | None = None,
    lora_dropout: float | None = None,
    lora_bias: Literal["all", "lora_only", "none"] | None = None,
    action_mode: Literal["continuous"] = "continuous",
) -> MolmoAct2Config:
    """Create a MolmoAct2 config from explicit policy arguments.

    Returns:
        A :class:`MolmoAct2Config` with the supplied arguments applied.
    """
    return MolmoAct2Config(
        input_features=input_features,
        output_features=output_features,
        norm_tag=norm_tag,
        n_obs_steps=n_obs_steps if n_obs_steps is not None else 1,
        chunk_size=chunk_size if chunk_size is not None else 30,
        n_action_steps=n_action_steps if n_action_steps is not None else 30,
        use_random_input_noise=use_random_input_noise if use_random_input_noise is not None else False,
        setup_type=setup_type if setup_type is not None else "",
        control_mode=control_mode if control_mode is not None else "",
        compile_model=compile_model if compile_model is not None else False,
        openvino_compress_to_fp16=(openvino_compress_to_fp16 if openvino_compress_to_fp16 is not None else False),
        model_dtype=model_dtype if model_dtype is not None else "bfloat16",
        train_action_expert_only=train_action_expert_only if train_action_expert_only is not None else False,
        gradient_checkpointing=gradient_checkpointing if gradient_checkpointing is not None else False,
        use_lora=use_lora if use_lora is not None else False,
        enable_lora_action_expert=(enable_lora_action_expert if enable_lora_action_expert is not None else False),
        lora_rank=lora_rank if lora_rank is not None else 64,
        lora_alpha=lora_alpha if lora_alpha is not None else 16,
        lora_dropout=lora_dropout if lora_dropout is not None else 0.05,
        lora_bias=lora_bias if lora_bias is not None else "none",
        action_mode=action_mode,
    )


class MolmoAct2(ExportablePolicyMixin, Policy):
    """MolmoAct2 Policy."""

    def __init__(  # noqa: PLR0913
        self,
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        *,
        # molmo pretrained args
        repo_id: str | Path | None = None,
        norm_tag: str | None = None,
        adapt_to_so101: bool | None = None,
        # model export / compilation args
        compile_model: bool | None = None,
        openvino_compress_to_fp16: bool | None = None,
        model_dtype: Literal["float32", "bfloat16", "float16"] | None = None,
        # training config
        train_action_expert_only: bool | None = None,
        gradient_checkpointing: bool | None = None,
        use_lora: bool | None = None,
        enable_lora_action_expert: bool | None = None,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
        lora_dropout: float | None = None,
        lora_bias: Literal["all", "lora_only", "none"] | None = None,
        n_obs_steps: int | None = None,
        chunk_size: int | None = None,
        n_action_steps: int | None = None,
        use_random_input_noise: bool | None = None,
        setup_type: str | None = None,
        control_mode: str | None = None,
        optimizer_lr: float = 1e-5,
        optimizer_vit_lr: float = 5e-6,
        optimizer_connector_lr: float = 5e-6,
        optimizer_action_expert_lr: float = 5e-5,
        optimizer_betas: tuple[float, float] = (0.9, 0.95),
        optimizer_eps: float = 1e-6,
        optimizer_weight_decay: float = 0.0,
        optimizer_grad_clip_norm: float = 1.0,
        scheduler_warmup_steps: int = 200,
        scheduler_decay_steps: int | None = 100_000,
        scheduler_decay_lr: float = 1e-6,
        # other overrides
        load_weights: bool = True,
        action_mode: Literal["continuous"] = "continuous",
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
            openvino_compress_to_fp16: Explicit override for OpenVINO export compression.
            model_dtype: Storage and forward dtype for model parameters.
            train_action_expert_only: Explicit action-expert fine-tuning override.
            gradient_checkpointing: Explicit gradient-checkpointing override.
            use_lora: Explicit LoRA enablement override.
            enable_lora_action_expert: Whether LoRA also targets the action expert.
            lora_rank: LoRA rank override.
            lora_alpha: LoRA scaling override.
            lora_dropout: LoRA dropout override.
            lora_bias: LoRA bias-training mode override.
            n_obs_steps: Number of observation steps.
            chunk_size: Number of actions predicted per chunk.
            n_action_steps: Number of actions returned per policy invocation.
            use_random_input_noise: Whether flow matching starts from random noise.
            setup_type: Robot/environment setup prompt text.
            control_mode: Action control-mode prompt text.
            optimizer_lr: Learning rate for VLM parameters.
            optimizer_vit_lr: Learning rate for vision parameters.
            optimizer_connector_lr: Learning rate for connector parameters.
            optimizer_action_expert_lr: Learning rate for action-expert parameters.
            optimizer_betas: AdamW beta coefficients.
            optimizer_eps: AdamW epsilon.
            optimizer_weight_decay: AdamW weight decay.
            optimizer_grad_clip_norm: Default gradient clipping norm.
            scheduler_warmup_steps: Number of scheduler warmup steps.
            scheduler_decay_steps: Number of scheduler decay steps, or the
                full training length when ``None``.
            scheduler_decay_lr: Final scheduler learning rate.
            load_weights: Whether to load base checkpoint weights after model
                construction when a checkpoint source is available.
            action_mode: Action mode to use for the policy. Currently only "continuous" is supported.

        Raises:
            ValueError: If only one of input_features/output_features is provided.
            RuntimeError: If pretrained checkpoint metadata cannot be resolved.
        """
        # check both either exist or both don't exit, raise error if not
        if bool(input_features) != bool(output_features):
            msg = f"Need both input and output features: input: {input_features} - output: {output_features}"
            raise ValueError(msg)

        # if pretrained repo_id exists find hf container
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
                repo_id=self.hf_container.repo_id,
                tokenizer_revision=self.hf_container.tokenizer_revision,
                tokenizer_config=self.hf_container.tokenizer_config,
                processor_config=self.hf_container.processor_config,
                norm_tag=norm_tag,
                n_obs_steps=n_obs_steps,
                chunk_size=chunk_size,
                n_action_steps=n_action_steps,
                use_random_input_noise=use_random_input_noise,
                setup_type=setup_type,
                control_mode=control_mode,
                compile_model=compile_model,
                openvino_compress_to_fp16=openvino_compress_to_fp16,
                model_dtype=model_dtype,
                train_action_expert_only=train_action_expert_only,
                gradient_checkpointing=gradient_checkpointing,
                use_lora=use_lora,
                enable_lora_action_expert=enable_lora_action_expert,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_bias=lora_bias,
                action_mode=action_mode,
            )
        else:
            self.config = make_molmoact2_config(
                input_features=input_features,
                output_features=output_features,
                norm_tag=norm_tag,
                n_obs_steps=n_obs_steps,
                chunk_size=chunk_size,
                n_action_steps=n_action_steps,
                use_random_input_noise=use_random_input_noise,
                setup_type=setup_type,
                control_mode=control_mode,
                compile_model=compile_model,
                openvino_compress_to_fp16=openvino_compress_to_fp16,
                model_dtype=model_dtype,
                train_action_expert_only=train_action_expert_only,
                gradient_checkpointing=gradient_checkpointing,
                use_lora=use_lora,
                enable_lora_action_expert=enable_lora_action_expert,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_bias=lora_bias,
                action_mode=action_mode,
            )

        super().__init__(n_action_steps=self.config.n_action_steps)

        # training config
        self.optimizer_lr = optimizer_lr
        self.optimizer_vit_lr = optimizer_vit_lr
        self.optimizer_connector_lr = optimizer_connector_lr
        self.optimizer_action_expert_lr = optimizer_action_expert_lr
        self.optimizer_betas = optimizer_betas
        self.optimizer_eps = optimizer_eps
        self.optimizer_weight_decay = optimizer_weight_decay
        self.optimizer_grad_clip_norm = optimizer_grad_clip_norm
        self.scheduler_warmup_steps = scheduler_warmup_steps
        self.scheduler_decay_steps = scheduler_decay_steps
        self.scheduler_decay_lr = scheduler_decay_lr

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
        self.save_hyperparameters(ignore=["config", "compile_model", "load_weights"])

        self.model: MolmoAct2Model | None = None  # pyrefly: ignore[bad-override-mutable-attribute]
        self._preprocessor: MolmoAct2Preprocessor | None = None  # pyrefly: ignore[bad-override-mutable-attribute]
        self._postprocessor: MolmoAct2Postprocessor | None = None
        self._load_weights = load_weights

        # Eagerly initialize when config resolution produced a complete schema.
        if self.config.input_features and self.config.output_features:
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
        self._ensure_tokenizer_assets()
        self._preprocessor, self._postprocessor = make_molmoact2_preprocessors(config=self.config)
        self.model = MolmoAct2Model(self.config)
        if self._checkpoint_location is not None and self._load_weights:
            self.model.load_pretrained_weights(self._checkpoint_location)

        # Apply LoRA adapters after weight loading so pretrained parameters are
        # preserved and only the low-rank updates are trainable.
        if self.config.use_lora:
            self.model.apply_lora_adapters()

    def _ensure_tokenizer_assets(self) -> None:
        """Resolve tokenizer-only assets when no model snapshot was supplied."""
        tokenizer_dir = Path(self.config.tokenizer_name_or_path)
        if (tokenizer_dir / "tokenizer.json").is_file() and self.config.tokenizer_config is not None:
            return
        resolved_dir, tokenizer_config = resolve_tokenizer_assets(self.config.tokenizer_name_or_path)
        self.config.tokenizer_name_or_path = resolved_dir
        self.config.tokenizer_config = tokenizer_config

        # parameter setting based on config
        if self.config.train_action_expert_only:
            self._freeze_non_action_expert_parameters()

    def setup(self, stage: str) -> None:
        """Set up model from datamodule (lazy or fine-tuning path).

        Args:
            stage: Lightning stage identifier (unused; required by the interface).

        Raises:
            TypeError: If the attached train dataset is not a physicalai Dataset.
            ValueError: If the training dataset features do not match the initialized feature contract.
        """
        del stage
        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
        train_dataset = datamodule.train_dataset
        if not isinstance(train_dataset, Dataset):
            msg = f"Expected physicalai Dataset, got {type(train_dataset)}"
            raise TypeError(msg)

        dataset_input_features, dataset_output_features = self._dataset_features(train_dataset)

        if self.model is not None:
            if (
                self.config.input_features != dataset_input_features
                or self.config.output_features != dataset_output_features
            ):
                msg = "Training dataset features do not match the initialized MolmoAct2 feature contract."
                raise ValueError(msg)
            return

        if self.config.input_features != dataset_input_features:
            if self.config.input_features is not None:
                warnings.warn(
                    "Configured input features do not match the training dataset; using the dataset features.",
                    UserWarning,
                    stacklevel=2,
                )
            self.config.input_features = dataset_input_features
            self.hparams["input_features"] = dataset_input_features

        if self.config.output_features != dataset_output_features:
            if self.config.output_features is not None:
                warnings.warn(
                    "Configured output features do not match the training dataset; using the dataset features.",
                    UserWarning,
                    stacklevel=2,
                )
            self.config.output_features = dataset_output_features
            self.hparams["output_features"] = dataset_output_features

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

    def compute_val_loss(self, batch: Observation) -> tuple[Tensor, dict[str, Tensor | float]]:
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
        loss, metrics = self.model.compute_val_loss(processed_batch)
        return loss, {name: float(value) for name, value in metrics.items()}

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
            "vlm": self.optimizer_lr,
            "vit": self.optimizer_vit_lr,
            "connector": self.optimizer_connector_lr,
            "action_expert": self.optimizer_action_expert_lr,
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
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
        )

        num_training_steps = int(self.trainer.estimated_stepping_batches)
        num_decay_steps = self.scheduler_decay_steps
        if num_decay_steps is None:
            num_decay_steps = num_training_steps

        scheduler = cosine_decay_with_warmup_scheduler(
            optimizer,
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
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
        clip_val = gradient_clip_val if gradient_clip_val is not None else self.optimizer_grad_clip_norm
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

    def _openvino_token_ids(self) -> tuple[int, int, list[int]]:
        required_token_ids = {
            "image_start_token_id": self.config.image_start_token_id,
            "image_end_token_id": self.config.image_end_token_id,
            "image_patch_id": self.config.image_patch_id,
        }
        missing_token_ids = [name for name, value in required_token_ids.items() if value is None]
        if missing_token_ids:
            msg = f"MolmoAct2 OpenVINO export requires token IDs: {', '.join(missing_token_ids)}"
            raise ValueError(msg)

        if self._preprocessor is None:
            msg = "MolmoAct2 preprocessor must be initialized before export."
            raise ValueError(msg)
        tokenizer = self._preprocessor.tokenizer
        bos_token_id = tokenizer.bos_token_id
        if not isinstance(bos_token_id, int):
            bos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id
        if not isinstance(bos_token_id, int) or not isinstance(pad_token_id, int):
            msg = "MolmoAct2 tokenizer must define integer BOS/EOS and padding token IDs"
            raise TypeError(msg)

        image_token_ids = [
            token_id
            for token_id in (
                self.config.image_patch_id,
                self.config.image_col_id,
                self.config.image_start_token_id,
                self.config.low_res_image_start_token_id,
                self.config.frame_start_token_id,
                self.config.image_end_token_id,
                self.config.frame_end_token_id,
                self.config.image_low_res_id,
            )
            if token_id is not None
        ]
        return bos_token_id, pad_token_id, image_token_ids

    @property
    def extra_export_args(self) -> dict[str, ExportParameters]:
        """Build backend export arguments for the policy.

        Raises:
            ValueError: If the model has not been initialized and input/output features are unavailable.
            ValueError: If any required token IDs are missing.
            ValueError: If the preprocessor has not been initialized.
        """
        # Ensure input/output features are available for export; they are required to construct the export parameters.
        if self.input_features is None or self.output_features is None:
            msg = "Model has not been initialized, input/output features are unavailable for export."
            raise ValueError(msg)

        # features
        input_features = list(self.input_features)
        output_features = list(self.output_features)
        state_feature = get_feature_by_type(input_features, FeatureType.STATE)
        action_feature = get_feature_by_type(output_features, FeatureType.ACTION)

        state_stats = _normalization_stats(state_feature)
        action_stats = _normalization_stats(action_feature)
        bos_token_id, pad_token_id, image_token_ids = self._openvino_token_ids()

        image_size = (
            int(self.config.image_processor_size["height"]),
            int(self.config.image_processor_size["width"]),
        )
        image_keys = [
            str(feature.name) for feature in input_features if feature.ftype == FeatureType.VISUAL and feature.name
        ]
        if action_feature is None or not action_feature.shape:
            msg = "MolmoAct2 OpenVINO export requires an action output feature with a defined shape."
            raise ValueError(msg)
        output_names = [feature.name for feature in (self.outputs_schema or [])]

        openvino_preprocessors = [
            ComponentSpec(
                type="molmoact2",
                image_keys=image_keys,
                state_stats=state_stats,
                image_size=image_size,
                num_state_tokens=self.config.num_state_tokens,
                setup_type=self.config.setup_type,
                control_mode=self.config.control_mode,
                add_setup_tokens=self.config.add_setup_tokens,
                add_control_tokens=self.config.add_control_tokens,
                adapt_to_so101=self.config.adapt_to_so101,
                joint_signs=self.config.joint_signs,
                joint_offsets=self.config.joint_offsets,
            ),
            ComponentSpec(
                type="ov_tokenizer",
                artifact="tokenizer.xml",
            ),
            ComponentSpec(
                type="molmoact2_inputs",
                max_action_dim=self.config.max_action_dim,
                action_dim=int(action_feature.shape[-1]),
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                image_placeholder_token_id=self.config.image_placeholder_token_id,
                image_start_token_id=self.config.image_start_token_id,
                image_end_token_id=self.config.image_end_token_id,
                image_patch_id=self.config.image_patch_id,
                image_col_id=self.config.image_col_id,
                low_res_image_start_token_id=self.config.low_res_image_start_token_id,
                image_size=image_size,
                patch_size=self.config.image_processor_patch_size,
                pooling_size=tuple(self.config.image_processor_pooling_size),
                image_mean=self.config.image_processor_mean,
                image_std=self.config.image_processor_std,
                image_use_col_tokens=self.config.image_use_col_tokens,
                use_single_crop_col_tokens=bool(self.config.use_single_crop_col_tokens),
                use_single_crop_start_token=self.config.use_single_crop_start_token,
                image_token_ids=image_token_ids,
            ),
        ]
        openvino_postprocessors = [
            ComponentSpec(
                type="molmoact2_postprocess",
                action_stats=action_stats,
                adapt_to_so101=self.config.adapt_to_so101,
                joint_signs=self.config.joint_signs,
                joint_offsets=self.config.joint_offsets,
            ),
        ]
        return {
            "torch": TorchExportParameters(
                preprocessors_specs=[ComponentSpec(type="to_float_tensor")],
            ),
            "openvino": OpenVINOExportParameters(
                outputs=output_names,
                export_tokenizer=True,
                compress_to_fp16=self.config.openvino_compress_to_fp16,
                via_onnx=False,
                exporter_kwargs={},
                preprocessors_specs=openvino_preprocessors,
                postprocessors_specs=openvino_postprocessors,
            ),
        }

    @staticmethod
    def get_supported_export_backends() -> list[str | ExportBackend]:
        """Get a list of export backends supported by policy.

        This method returns a list of supported export backends as strings.

        Returns:
            list[str | ExportBackend]: A list of supported export backends.
        """
        return [ExportBackend.TORCH, ExportBackend.OPENVINO]
