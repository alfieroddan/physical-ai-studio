# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 Gaoyue Zhou
# Authors: Gaoyue Zhou, Zichen Jeff Cui
# SPDX-License-Identifier: MIT

"""Lightning policy for Patch Policy."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import IO, TYPE_CHECKING, Any

import torch
from torch import Tensor

from physicalai.data import Dataset, Feature, FeatureType, NormalizationParameters, Observation
from physicalai.policies.base import Model, Policy

from .config import (
    ActionHeadName,
    EncoderOutput,
    FeatureEncoderName,
    PatchPolicyConfig,
    resolve_action_dim,
)
from .model import PatchPolicyModel
from .processors import (
    PatchPolicyPostprocessor,
    PatchPolicyPreprocessor,
    make_policy_processors,
    resolve_camera_name,
)

if TYPE_CHECKING:
    from pathlib import Path


def _feature_key(feature: Feature) -> tuple[FeatureType | None, str | None]:
    name = resolve_camera_name(feature) if feature.ftype is FeatureType.VISUAL else feature.name
    return feature.ftype, name


def _copy_normalization_data(features: list[Feature], sources: list[Feature]) -> list[Feature]:
    """Replace configured statistics with matching training dataset statistics."""
    source_features = {_feature_key(feature): feature for feature in sources}
    resolved = []
    for feature in features:
        source = source_features.get(_feature_key(feature))
        if source is None:
            resolved.append(feature)
            continue
        if feature.shape != source.shape:
            raise ValueError(
                f"Feature {feature.name!r} shape {feature.shape} does not match dataset feature shape {source.shape}.",
            )
        if feature.normalization_data is not None and feature.normalization_data != source.normalization_data:
            warnings.warn(
                f"Replacing normalization statistics for feature {feature.name!r} with training dataset statistics.",
                stacklevel=2,
            )
        resolved.append(replace(feature, normalization_data=source.normalization_data))
    return resolved


def _attach_encoder_normalization(
    features: list[Feature],
    mean: Tensor,
    std: Tensor,
) -> list[Feature]:
    """Attach the selected timm encoder's input statistics to visual features."""
    normalization = NormalizationParameters(
        mean=mean.flatten().detach().cpu().tolist(),
        std=std.flatten().detach().cpu().tolist(),
    )
    resolved = []
    for feature in features:
        if feature.ftype is not FeatureType.VISUAL:
            resolved.append(feature)
            continue
        if feature.normalization_data is not None and feature.normalization_data != normalization:
            warnings.warn(
                f"Replacing normalization statistics for visual feature {feature.name!r} with encoder statistics.",
                stacklevel=2,
            )
        resolved.append(replace(feature, normalization_data=normalization))
    return resolved


class PatchPolicy(Policy):
    """Train a timm visual encoder followed by a diffusion or Behavior Transformer head.

    Args:
        input_features: Optional observation schema. Resolved from training data when omitted.
        output_features: Optional action schema. Resolved from training data when omitted.
        encoder_name: Name of the timm visual encoder architecture.
        encoder_output: Visual representation extracted from the encoder.
        encoder_pretrained: Whether to initialize the encoder with pretrained weights.
        freeze_encoder: Whether to prevent updates to encoder parameters.
        action_head: Action-head architecture used to predict action chunks.
        n_obs_steps: Number of observation timesteps used as context.
        chunk_size: Number of action timesteps predicted per chunk.
        n_action_steps: Number of predicted actions executed per invocation.
        hidden_dim: Hidden width of the action head.
        num_layers: Number of transformer layers in the action head.
        num_heads: Number of attention heads in each transformer layer.
        dropout: Dropout probability used by transformer layers.
        num_train_timesteps: Number of diffusion timesteps used during training.
        num_inference_steps: Number of denoising steps used during inference.
        eta: DDIM stochasticity coefficient used during diffusion inference.
        use_random_input_noise: Whether diffusion sampling starts from random noise.
        optimizer_lr: AdamW learning rate.
        optimizer_weight_decay: AdamW weight-decay coefficient.
    """

    model: Model | None
    config: PatchPolicyConfig | None
    encoder_name: FeatureEncoderName
    encoder_output: EncoderOutput
    action_head: ActionHeadName
    _preprocessor: PatchPolicyPreprocessor | None
    _postprocessor: PatchPolicyPostprocessor | None

    def __init__(
        self,
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        *,
        encoder_name: FeatureEncoderName = "webssl",
        encoder_output: EncoderOutput = "patches",
        encoder_pretrained: bool = True,
        freeze_encoder: bool = True,
        action_head: ActionHeadName = "behavior_transformer",
        n_obs_steps: int = 1,
        chunk_size: int = 16,
        n_action_steps: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 100,
        eta: float = 1.0,
        use_random_input_noise: bool = True,
        optimizer_lr: float = 1e-4,
        optimizer_weight_decay: float = 0.0,
    ) -> None:
        super().__init__(n_action_steps=n_action_steps)

        self.model = None
        self.config = None

        self.input_features = input_features
        self.output_features = output_features

        self._preprocessor = None
        self._postprocessor = None

        # Action-related parameters
        self.n_obs_steps = n_obs_steps
        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps

        # Model params
        self.encoder_name = encoder_name
        self.encoder_output = encoder_output
        self.encoder_pretrained = encoder_pretrained
        self.freeze_encoder = freeze_encoder
        self.action_head = action_head
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        self.use_random_input_noise = use_random_input_noise

        # Extra params for training
        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay

        # Don't save possibly unresolved input / output features
        self.save_hyperparameters(ignore=["input_features", "output_features"])

        if self.input_features is not None and self.output_features is not None:
            self.configure_model()

    def setup(self, stage: str) -> None:
        # only setup on fit stage
        if stage != "fit":
            return

        # Get the training features from the datamodule
        train_dataset = self.trainer.datamodule.train_dataset  # type: ignore[attr-defined]
        if not isinstance(train_dataset, Dataset):
            raise TypeError(f"Expected physicalai Dataset, got {type(train_dataset)}.")

        dataset_input_features = list(train_dataset.observation_features.values())
        dataset_output_features = list(train_dataset.action_features.values())

        # Validate that the configured input and output features match the dataset features
        if self.input_features is not None and [
            (_feature_key(feature), feature.shape) for feature in self.input_features
        ] != [(_feature_key(feature), feature.shape) for feature in dataset_input_features]:
            raise ValueError("Configured input_features do not match the training dataset features.")
        if self.output_features is not None and [
            (_feature_key(feature), feature.shape) for feature in self.output_features
        ] != [(_feature_key(feature), feature.shape) for feature in dataset_output_features]:
            raise ValueError("Configured output_features do not match the training dataset features.")

        # Copy normalization data from the dataset to the configured features for new training
        input_features = dataset_input_features if self.input_features is None else self.input_features
        output_features = _copy_normalization_data(
            dataset_output_features if self.output_features is None else self.output_features,
            dataset_output_features,
        )
        self.input_features = input_features
        self.output_features = output_features

    @classmethod
    def from_config(
        cls,
        config: PatchPolicyConfig,
        *,
        optimizer_lr: float = 1e-4,
        optimizer_weight_decay: float = 0.0,
    ) -> PatchPolicy:
        policy = cls(
            input_features=None,
            output_features=None,
            encoder_name=config.encoder_name,
            encoder_output=config.encoder_output,
            encoder_pretrained=config.encoder_pretrained,
            freeze_encoder=config.freeze_encoder,
            action_head=config.action_head,
            n_obs_steps=config.n_obs_steps,
            chunk_size=config.chunk_size,
            n_action_steps=config.n_action_steps,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            num_train_timesteps=config.num_train_timesteps,
            num_inference_steps=config.num_inference_steps,
            eta=config.eta,
            use_random_input_noise=config.use_random_input_noise,
            optimizer_lr=optimizer_lr,
            optimizer_weight_decay=optimizer_weight_decay,
        )
        policy._initialize_from_checkpoint_config(config)
        return policy

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | IO[bytes],
        map_location: torch.device | str | int | Callable | dict | None = None,
        hparams_file: str | Path | None = None,
        strict: bool | None = None,  # noqa: FBT001
        weights_only: bool | None = None,  # noqa: FBT001
        **kwargs: Any,  # noqa: ANN401
    ) -> PatchPolicy:
        return super().load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=strict,
            weights_only=weights_only,
            **kwargs,
        )

    def configure_model(self) -> None:
        """Create the model once and synchronize feature normalization and processors."""
        # config requires input and output features
        if self.input_features is None or self.output_features is None:
            return

        # if no config, construct (for the first time)
        if self.config is None:
            # construct config
            self.config = PatchPolicyConfig(
                input_features=list(self.input_features),
                output_features=list(self.output_features),
                action_dim=resolve_action_dim(self.output_features),
                encoder_name=self.encoder_name,
                encoder_output=self.encoder_output,
                encoder_pretrained=self.encoder_pretrained,
                freeze_encoder=self.freeze_encoder,
                action_head=self.action_head,
                n_obs_steps=self.n_obs_steps,
                chunk_size=self.chunk_size,
                n_action_steps=self.n_action_steps,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                dropout=self.dropout,
                num_train_timesteps=self.num_train_timesteps,
                num_inference_steps=self.num_inference_steps,
                eta=self.eta,
                use_random_input_noise=self.use_random_input_noise,
            )

        # Init model from local config
        config = self.config

        self.encoder_name = config.encoder_name
        self.encoder_output = config.encoder_output
        self.encoder_pretrained = config.encoder_pretrained
        self.freeze_encoder = config.freeze_encoder
        self.action_head = config.action_head
        self.n_obs_steps = config.n_obs_steps
        self.chunk_size = config.chunk_size
        self.n_action_steps = config.n_action_steps
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        self.num_train_timesteps = config.num_train_timesteps
        self.num_inference_steps = config.num_inference_steps
        self.eta = config.eta
        self.use_random_input_noise = config.use_random_input_noise
        self.hparams.update(
            encoder_name=config.encoder_name,
            encoder_output=config.encoder_output,
            encoder_pretrained=config.encoder_pretrained,
            freeze_encoder=config.freeze_encoder,
            action_head=config.action_head,
            n_obs_steps=config.n_obs_steps,
            chunk_size=config.chunk_size,
            n_action_steps=config.n_action_steps,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            num_train_timesteps=config.num_train_timesteps,
            num_inference_steps=config.num_inference_steps,
            eta=config.eta,
            use_random_input_noise=config.use_random_input_noise,
        )

        model = self.model
        if model is None:
            model = PatchPolicyModel.from_config(config)
        elif not isinstance(model, PatchPolicyModel):
            raise TypeError("Patch Policy requires a PatchPolicyModel instance.")

        # Replace input features with normalized versions based on encoder statistics
        # Normalization is only used by pre / post processor
        config = replace(
            config,
            input_features=_attach_encoder_normalization(
                self.input_features,
                model.encoder.mean,
                model.encoder.std,
            ),
            output_features=self.output_features,
        )

        self.input_features = config.input_features
        self.output_features = config.output_features
        if (
            self.model is model
            and self.config == config
            and self._preprocessor is not None
            and self._postprocessor is not None
        ):
            return

        # Store the final configuration and model instance in Policy
        self.config = config
        self.input_features = config.input_features
        self.model = model
        self._preprocessor, self._postprocessor = make_policy_processors(config, model.encoder.image_size)

    def _initialize_from_checkpoint_config(self, config: PatchPolicyConfig) -> None:
        if self.model is not None:
            if (
                self.config is None
                or replace(
                    config,
                    input_features=self.config.input_features,
                    output_features=self.config.output_features,
                )
                != self.config
                or [(_feature_key(feature), feature.shape) for feature in config.input_features]
                != [(_feature_key(feature), feature.shape) for feature in self.config.input_features]
                or [(_feature_key(feature), feature.shape) for feature in config.output_features]
                != [(_feature_key(feature), feature.shape) for feature in self.config.output_features]
            ):
                raise ValueError("Checkpoint policy config does not match the initialized policy.")
        else:
            self.config = config
        self.input_features = config.input_features
        self.output_features = config.output_features
        self.configure_model()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.config is None:
            raise RuntimeError("Cannot save a Patch Policy checkpoint before the policy is configured.")
        checkpoint["policy_config"] = self.config.to_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        config_data = checkpoint.get("policy_config")
        if not isinstance(config_data, Mapping):
            raise TypeError("Patch Policy checkpoint is missing a valid policy_config.")
        self._initialize_from_checkpoint_config(PatchPolicyConfig.from_dict(config_data))

    def reset(self) -> None:
        if self.model is not None:
            if not isinstance(self.model, PatchPolicyModel):
                raise TypeError("Patch Policy requires a PatchPolicyModel instance.")
            self.model.reset()
        # Call the reset method of the parent class
        super().reset()

    def _prepare_batch(
        self,
        batch: Observation,
        *,
        require_actions: bool,
    ) -> dict[str, Any]:
        if self.config is None or self._preprocessor is None:
            raise RuntimeError("Patch Policy is not initialized.")
        values = batch.to(self.device).to_dict()
        if require_actions and not isinstance(values.get("action"), Tensor):
            raise TypeError("Training requires tensor actions.")
        return self._preprocessor(values)

    def forward(self, batch: Observation) -> Tensor | tuple[Tensor, dict[str, Tensor | float]]:
        if self.model is None:
            raise RuntimeError("Patch Policy is not initialized.")
        if self.training:
            return self.model(self._prepare_batch(batch, require_actions=True))
        return self.predict_action_chunk(batch)

    @torch.no_grad()
    def predict_action_chunk(self, batch: Observation) -> Tensor:
        if not isinstance(self.model, PatchPolicyModel) or self._postprocessor is None:
            raise RuntimeError("Patch Policy is not initialized.")
        prepared = self._prepare_batch(batch, require_actions=False)
        return self._postprocessor(self.model.predict_action_chunk(prepared))

    def compute_val_loss(self, batch: Observation) -> tuple[Tensor, dict[str, Tensor | float]]:
        if self.model is None:
            raise RuntimeError("Patch Policy is not initialized.")
        return self.model.compute_val_loss(self._prepare_batch(batch, require_actions=True))

    def training_step(self, batch: Observation, batch_idx: int) -> Tensor:
        del batch_idx
        result = self(batch)
        if not isinstance(result, tuple):
            raise TypeError("Training forward must return loss and metrics.")
        loss, metrics = result
        self.log_dict({f"train/{name}": value for name, value in metrics.items()}, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.model is None:
            raise RuntimeError("Cannot configure optimizers before Patch Policy features and model are resolved.")
        return torch.optim.AdamW(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )
