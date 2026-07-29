# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 policy implementation."""

import dataclasses
from pathlib import Path
from typing import IO, Any, Literal

import torch
from physicalai.inference.data import InferenceFeature, InferenceFeatureDtype, InferenceFeatureType
from physicalai.inference.manifest import ComponentSpec
from torch import Tensor

from physicalai.data.constants import IMAGE_MASKS, STATE, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
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


def make_molmoact2_config(  # noqa: PLR0913
    *,
    input_features: list[Feature] | None,
    output_features: list[Feature] | None,
    n_obs_steps: int,
    n_action_steps: int,
    chunk_size: int = 30,
    action_mode: Literal["continuous", "discrete", "both"] = "continuous",
    use_random_input_noise: bool = False,
    torch_compile: bool = False,
    use_lora: bool = False,
    enable_lora_action_expert: bool = False,
    lora_rank: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_bias: Literal["all", "lora_only", "none"] = "none",
    gradient_checkpointing: bool = False,
    train_action_expert_only: bool = False,
    num_flow_timesteps: int | None = None,
    checkpoint_path: str | None = None,
) -> MolmoAct2Config:
    """Create the explicit model config for MolmoAct2.

    This function is the non-policy home for model-definition defaults.

    Args:
        input_features: List of input features the model consumes.
        output_features: List of output features the model produces.
        n_obs_steps: Number of observation steps.
        n_action_steps: Number of action steps.
        chunk_size: Action chunk size (must be >= n_action_steps).
        action_mode: Action supervision mode.
        use_random_input_noise: Start flow matching from sampled Gaussian noise
            instead of zeros. Kept off by default so the exported graph stays
            deterministic and RNG-free.
        torch_compile: Whether to mark the config for optimized inference.
        use_lora: Whether to apply LoRA adapters to the VLM.
        enable_lora_action_expert: Whether to also adapt the action expert.
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: LoRA dropout rate.
        lora_bias: Which biases to train ('none', 'all', 'lora_only').
        gradient_checkpointing: Whether to enable gradient checkpointing.
        train_action_expert_only: Freeze the VLM entirely and only train the
            action expert. Incompatible with ``use_lora``.
        num_flow_timesteps: Number of independent (timestep, noise) samples
            drawn per training example and averaged in the flow-matching
            loss (variance reduction). ``None`` (default) uses
            :class:`MolmoAct2Config`'s own default (``8``, matching the
            reference MolmoAct2 training recipe).
        checkpoint_path: Optional local path to a pretrained checkpoint snapshot
            directory. Forwarded to :class:`MolmoAct2Config` so a config built
            outside the HF flow can still locate its pretrained weights (e.g.
            for :meth:`MolmoAct2.from_config`).

    Returns:
        A fully populated :class:`MolmoAct2Config`.
    """
    config_kwargs: dict[str, Any] = {}
    if num_flow_timesteps is not None:
        config_kwargs["num_flow_timesteps"] = num_flow_timesteps
    return MolmoAct2Config(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        chunk_size=chunk_size,
        action_mode=action_mode,
        use_random_input_noise=use_random_input_noise,
        compile_model=torch_compile,
        use_lora=use_lora,
        enable_lora_action_expert=enable_lora_action_expert,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_bias=lora_bias,
        gradient_checkpointing=gradient_checkpointing,
        train_action_expert_only=train_action_expert_only,
        checkpoint_path=checkpoint_path,
        **config_kwargs,
    )


class MolmoAct2(ExportablePolicyMixin, Policy):
    """MolmoAct2 Policy."""

    def __init__(  # noqa: PLR0913
        self,
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        repo_id: str | Path | None = "allenai/MolmoAct2",
        norm_tag: str | None = None,
        n_obs_steps: int = 30,
        n_action_steps: int = 30,
        chunk_size: int = 30,
        action_mode: Literal["continuous", "discrete", "both"] = "continuous",
        *,
        config: MolmoAct2Config | None = None,
        use_random_input_noise: bool = False,
        adapt_to_so101: bool = False,
        torch_compile: bool = False,
        load_weights: bool = True,
        use_lora: bool = False,
        enable_lora_action_expert: bool = False,
        lora_rank: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_bias: Literal["all", "lora_only", "none"] = "none",
        gradient_checkpointing: bool = False,
        train_action_expert_only: bool = False,
        num_flow_timesteps: int | None = None,
        setup_type: str | None = None,
        control_mode: str | None = None,
    ) -> None:
        """Initialize a MolmoAct2 policy wrapper.

        Args:
            input_features: Optional observation feature schema.
            output_features: Optional action feature schema.
            repo_id: Optional pretrained checkpoint identifier or path. Ignored
                if ``config`` is provided.
            norm_tag: Optional normalization tag for pretrained checkpoints.
            n_obs_steps: Number of observation steps.
            n_action_steps: Number of predicted action steps.
            chunk_size: Action chunk size (must be >= n_action_steps).
            action_mode: Training/inference action mode.
            config: A fully-built :class:`MolmoAct2Config` to use as-is,
                bypassing HF-container resolution and ad-hoc config
                construction entirely. Prefer
                :meth:`from_config` over passing this directly.
            use_random_input_noise: Start flow matching from sampled Gaussian
                noise instead of zeros. Kept off by default so the exported
                graph stays deterministic and RNG-free.
            adapt_to_so101: Apply the SO-100/101 joint frame transform to joint
                observations/actions (needed for zero-shot and fine-tuning from the
                pre-#777 LeRobot calibration checkpoint).
            torch_compile: Whether to enable compile-oriented config flags.
            load_weights: Whether to eagerly load pretrained weights when a
                checkpoint is available. Set to ``False`` to skip the
                (potentially expensive) pretrained-weight load, e.g. when a
                caller is about to overwrite the state dict anyway (this is
                done automatically by :meth:`load_from_checkpoint`).
            use_lora: Apply LoRA adapters to the VLM (text transformer +
                vision backbone) linears for parameter-efficient
                fine-tuning. Incompatible with ``train_action_expert_only``.
            enable_lora_action_expert: Extend LoRA targets to the action
                expert linears. Requires ``use_lora=True``.
            lora_rank: LoRA rank (dimension of the low-rank update).
            lora_alpha: LoRA alpha scaling factor.
            lora_dropout: Dropout rate applied to the LoRA layers.
            lora_bias: Which biases to train ('none', 'all', 'lora_only').
            gradient_checkpointing: Enable gradient checkpointing on the text
                transformer, vision backbone and action expert to trade
                compute for memory during training.
            train_action_expert_only: Freeze the VLM entirely and only train
                the action expert. Incompatible with ``use_lora``.
            num_flow_timesteps: Number of independent (timestep, noise)
                samples drawn per training example and averaged in the
                flow-matching loss (variance reduction). ``None`` (default)
                uses the pretrained checkpoint's HF config value when
                resolvable, otherwise :class:`MolmoAct2Config`'s own default
                (``8``, matching the reference MolmoAct2 training recipe).
            setup_type: Text describing the robot/scene, inserted into the
                model prompt (e.g. ``"single franka robotic arm in libero"``).
                MolmoAct2 was pretrained to condition on this text. It is
                normally only populated from a pretrained checkpoint's
                ``norm_stats.json`` when ``norm_tag`` matches one of its
                ``metadata_by_tag`` entries; pass this explicitly to supply
                the same conditioning for *any* dataset, including ones with
                no matching (or no) ``norm_tag``. Always wins over any
                norm_tag-derived value when provided.
            control_mode: Text describing the action space, inserted into the
                model prompt (e.g. ``"delta end-effector pose"`` for relative
                end-effector deltas, or ``"absolute joint pose"`` for target
                joint angles). Same override semantics as ``setup_type``.

        Raises:
            ValueError: If only one of input_features/output_features is provided.
            RuntimeError: If pretrained checkpoint metadata cannot be resolved.
        """
        super().__init__(n_action_steps=n_action_steps)

        # Currently continuous mode is only supported
        if action_mode != "continuous":
            msg = "Only continous action mode is currently supported."
            raise ValueError(msg)

        # check both either exist or both don't exit, raise error if not
        if bool(input_features) != bool(output_features):
            msg = f"Need both input and output features: input: {input_features} - output: {output_features}"
            raise ValueError(msg)

        self.hf_container = None

        if config is not None:
            # Config was already built (e.g. via from_config, or reconstructed
            # from a saved/serialized config). Skip HF-container resolution
            # and ad-hoc config construction entirely.
            self.config = config
            # NOTE: assumes MolmoAct2Config exposes a `checkpoint_path`
            # attribute (mirrors the kwarg accepted by
            # build_config_from_hf_config below). Adjust the attribute name
            # here if it differs in your actual config class.
            self._checkpoint_location: str | None = getattr(config, "checkpoint_path", None)
        else:
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
                    n_obs_steps=n_obs_steps,
                    norm_tag=norm_tag,
                    n_action_steps=n_action_steps,
                    chunk_size=chunk_size,
                    action_mode=action_mode,
                    use_random_input_noise=use_random_input_noise,
                    torch_compile=torch_compile,
                    use_lora=use_lora,
                    enable_lora_action_expert=enable_lora_action_expert,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    lora_bias=lora_bias,
                    gradient_checkpointing=gradient_checkpointing,
                    train_action_expert_only=train_action_expert_only,
                    num_flow_timesteps=num_flow_timesteps,
                )
            else:
                self.config = make_molmoact2_config(
                    input_features=input_features,
                    output_features=output_features,
                    n_obs_steps=n_obs_steps,
                    n_action_steps=n_action_steps,
                    chunk_size=chunk_size,
                    action_mode=action_mode,
                    use_random_input_noise=use_random_input_noise,
                    torch_compile=torch_compile,
                    use_lora=use_lora,
                    enable_lora_action_expert=enable_lora_action_expert,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    lora_bias=lora_bias,
                    gradient_checkpointing=gradient_checkpointing,
                    train_action_expert_only=train_action_expert_only,
                    num_flow_timesteps=num_flow_timesteps,
                )

            self._checkpoint_location = self.hf_container.checkpoint_location if self.hf_container is not None else None

        # SO-101 joint calibration correction (must be set before processors build).
        # Applied uniformly regardless of which branch above built the config.
        # https://huggingface.co/docs/lerobot/v0.6.0/en/molmoact2#joint-frame-transform-so-100101-zero-shot
        self.config.adapt_to_so101 = adapt_to_so101

        # Explicit setup_type/control_mode always win over whatever a
        # norm_tag lookup produced (or the "" default when there was no
        # norm_tag), so any dataset can supply this prompt-conditioning text
        # without needing a matching entry in a pretrained checkpoint's
        # norm_stats.json. Must be set before processors build.
        if setup_type is not None:
            self.config.setup_type = setup_type
        if control_mode is not None:
            self.config.control_mode = control_mode

        # Keep repo_id in checkpoint hparams so load_from_checkpoint reconstructs
        # the same pretrained source during inference adapter reload.
        self.save_hyperparameters(ignore=["config", "load_weights"])

        self.model: MolmoAct2Model | None = None  # pyrefly: ignore[bad-override-mutable-attribute]
        self._preprocessor: MolmoAct2Preprocessor | None = None  # pyrefly: ignore[bad-override-mutable-attribute]
        self._postprocessor: MolmoAct2Postprocessor | None = None
        self._load_weights = load_weights

        # eagerly load
        if (input_features and output_features) or (repo_id and norm_tag) or config is not None:
            self._initialize_model()

    @classmethod
    def from_config(
        cls,
        config: MolmoAct2Config,
        *,
        load_weights: bool = True,
        adapt_to_so101: bool = False,
        **overrides: Any,  # noqa: ANN401
    ) -> "MolmoAct2":
        """Build a policy directly from an already-constructed config.

        Skips HF-container resolution entirely (no network/metadata lookup),
        which makes this the cheapest way to reconstruct a policy whose
        config has already been resolved once (e.g. loaded from disk,
        produced by a previous ``build_config_from_hf_config`` call, or
        hand-built via :func:`make_molmoact2_config`).

        Args:
            config: The base config to use. Not mutated; overrides are
                applied to a copy.
            load_weights: Whether to eagerly load pretrained weights from
                ``config.checkpoint_path`` (if present). Set to ``False`` to
                skip the weight load, e.g. before manually loading a
                fine-tuned state dict.
            adapt_to_so101: Apply the SO-100/101 joint frame transform. See
                :meth:`__init__` for details.
            **overrides: Field overrides applied to ``config`` before
                construction, via ``dataclasses.replace``. Keys must match
                existing fields on ``config``.

        Returns:
            A constructed :class:`MolmoAct2` policy.

        Raises:
            TypeError: If ``config`` is not a dataclass instance (required
                for ``dataclasses.replace``), or if ``overrides`` contains
                keys that are not fields on ``config``.
        """
        if overrides:
            if not dataclasses.is_dataclass(config):
                msg = (
                    "from_config overrides require MolmoAct2Config to be a dataclass; "
                    "got a non-dataclass config instead."
                )
                raise TypeError(msg)
            valid_fields = {f.name for f in dataclasses.fields(config)}
            unknown = set(overrides) - valid_fields
            if unknown:
                msg = f"Unknown config override(s): {sorted(unknown)}"
                raise TypeError(msg)
            config = dataclasses.replace(config, **overrides)

        return cls(
            input_features=config.input_features,
            output_features=config.output_features,
            repo_id=None,
            config=config,
            load_weights=load_weights,
            adapt_to_so101=adapt_to_so101,
        )

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

        training_config = self.config.training_config
        learning_rates = {
            "vlm": training_config.optimizer_lr,
            "vit": training_config.optimizer_vit_lr,
            "connector": training_config.optimizer_connector_lr,
            "action_expert": training_config.optimizer_action_expert_lr,
        }
        return [{"params": params, "lr": learning_rates[name]} for name, params in grouped.items() if params]

    def configure_optimizers(self) -> dict[str, Any]:  # pyrefly: ignore[bad-override]
        """Build the AdamW optimizer with grouped learning rates and an LR schedule.

        Returns:
            The Lightning optimizer/scheduler configuration.
        """
        if self.model is not None and self.config.train_action_expert_only:
            self.model.freeze_to_action_expert()

        training_config = self.config.training_config
        optimizer = torch.optim.AdamW(
            self.get_optim_params(),
            lr=training_config.optimizer_lr,
            weight_decay=training_config.optimizer_weight_decay,
            betas=training_config.optimizer_betas,
            eps=training_config.optimizer_eps,
        )

        num_training_steps = int(self.trainer.estimated_stepping_batches)
        num_decay_steps = training_config.scheduler_decay_steps
        if num_decay_steps is None:
            num_decay_steps = num_training_steps

        scheduler = cosine_decay_with_warmup_scheduler(
            optimizer,
            peak_lr=training_config.optimizer_lr,
            decay_lr=training_config.scheduler_decay_lr,
            num_warmup_steps=training_config.scheduler_warmup_steps,
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
        clip_val = (
            gradient_clip_val if gradient_clip_val is not None else self.config.training_config.optimizer_grad_clip_norm
        )
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
            if feature.shape is None:  # pragma: no cover - export requires concrete shapes
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
        output_names = [feature.name for feature in (self.outputs_schema or [])]

        return {
            "torch": TorchExportParameters(
                input_names=[TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK, IMAGES, IMAGE_MASKS, STATE],
                output_names=output_names,
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
