# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning policy for Patch Policy."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import IO, TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from physicalai.data import Dataset, Feature, Observation
from physicalai.policies.base import Model, Policy

from .config import (
    ActionHeadName,
    EncoderOutput,
    FeatureEncoderName,
    PatchPolicyConfig,
    resolve_action_dim,
)
from .model import PatchPolicyModel
from .processors import PatchPolicyPostprocessor, PatchPolicyPreprocessor, make_policy_processors

if TYPE_CHECKING:
    from pathlib import Path


class PatchPolicy(Policy):
    """Train a timm visual encoder followed by a diffusion or Behavior Transformer head.

    Export is intentionally not enabled until the stochastic heads have dedicated export contracts.
    """

    model: Model | None

    def __init__(  # noqa: PLR0913
        self,
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        *,
        encoder_name: FeatureEncoderName = "dinov2",
        encoder_output: EncoderOutput = "patches",
        encoder_pretrained: bool = True,
        freeze_encoder: bool = True,
        image_size: tuple[int, int] = (224, 224),
        action_head: ActionHeadName = "diffusion",
        n_obs_steps: int = 1,
        chunk_size: int = 16,
        n_action_steps: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 100,
        use_random_input_noise: bool = True,
        optimizer_lr: float = 1e-4,
        optimizer_weight_decay: float = 0.0,
    ) -> None:
        super().__init__(n_action_steps=n_action_steps)
        self.save_hyperparameters(ignore=["input_features", "output_features"])
        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay
        self._preprocessor: PatchPolicyPreprocessor | None = None
        self._postprocessor: PatchPolicyPostprocessor | None = None
        self._input_features = input_features
        self._output_features = output_features
        self._config: PatchPolicyConfig | None = None
        self._encoder_name = encoder_name
        self._encoder_output = encoder_output
        self._encoder_pretrained = encoder_pretrained
        self._freeze_encoder = freeze_encoder
        self._image_size = image_size
        self._action_head = action_head
        self._n_obs_steps = n_obs_steps
        self._chunk_size = chunk_size
        self._n_action_steps = n_action_steps
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout = dropout
        self._num_train_timesteps = num_train_timesteps
        self._num_inference_steps = num_inference_steps
        self._use_random_input_noise = use_random_input_noise
        if input_features is not None and output_features is not None:
            self.configure_model()

    @classmethod
    def from_config(
        cls,
        config: PatchPolicyConfig,
        *,
        optimizer_lr: float = 1e-4,
        optimizer_weight_decay: float = 0.0,
    ) -> PatchPolicy:
        """Construct a policy from a complete serialized config."""
        policy = cls(
            input_features=None,
            output_features=None,
            encoder_name=config.encoder_name,
            encoder_output=config.encoder_output,
            encoder_pretrained=config.encoder_pretrained,
            freeze_encoder=config.freeze_encoder,
            image_size=config.image_size,
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
            use_random_input_noise=config.use_random_input_noise,
            optimizer_lr=optimizer_lr,
            optimizer_weight_decay=optimizer_weight_decay,
        )
        policy._initialize_from_config(config)
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
        """Load a checkpoint whose saved config reconstructs the model before tensor restoration.

        Returns:
            The restored Patch Policy.
        """
        return super().load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=strict,
            weights_only=weights_only,
            **kwargs,
        )

    @property
    def config(self) -> PatchPolicyConfig:
        if self._config is None:
            raise RuntimeError("Patch Policy config is not resolved yet.")
        return self._config

    def _initialize_from_config(self, config: PatchPolicyConfig) -> None:
        if self.model is not None:
            if self.config != config:
                raise ValueError("Checkpoint policy config does not match the initialized policy.")
            return
        self._config = config
        self._input_features = config.input_features
        self._output_features = config.output_features
        self.configure_model()

    def configure_model(self) -> None:
        """Create the model and processors exactly once."""
        if self.model is not None:
            return
        if self._config is None:
            if self._input_features is None or self._output_features is None:
                return
            self._config = PatchPolicyConfig(
                input_features=list(self._input_features),
                output_features=list(self._output_features),
                action_dim=resolve_action_dim(self._output_features),
                encoder_name=self._encoder_name,
                encoder_output=self._encoder_output,
                encoder_pretrained=self._encoder_pretrained,
                freeze_encoder=self._freeze_encoder,
                image_size=self._image_size,
                action_head=self._action_head,
                n_obs_steps=self._n_obs_steps,
                chunk_size=self._chunk_size,
                n_action_steps=self._n_action_steps,
                hidden_dim=self._hidden_dim,
                num_layers=self._num_layers,
                num_heads=self._num_heads,
                dropout=self._dropout,
                num_train_timesteps=self._num_train_timesteps,
                num_inference_steps=self._num_inference_steps,
                use_random_input_noise=self._use_random_input_noise,
            )
        config = self._config
        self.model = PatchPolicyModel.from_config(config)
        self._preprocessor, self._postprocessor = make_policy_processors(config)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Save the complete resolved config needed to reconstruct the model."""
        checkpoint["policy_config"] = self.config.to_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Reconstruct the model before Lightning restores checkpoint tensors."""
        config_data = checkpoint.get("policy_config")
        if not isinstance(config_data, Mapping):
            raise TypeError("Patch Policy checkpoint is missing a valid policy_config.")
        self._initialize_from_config(PatchPolicyConfig.from_dict(config_data))

    def _prepare_batch(self, batch: Observation, *, require_actions: bool) -> dict[str, Tensor]:
        if self._preprocessor is None:
            raise RuntimeError("Patch Policy is not initialized.")
        values = batch.to_dict(flatten=False)
        if isinstance(values.get("action"), Mapping):
            action_map = values["action"]
            values["action"] = torch.cat(
                [action_map[feature.name] for feature in self.config.output_features if feature.name is not None],
                dim=-1,
            )
        return self._preprocessor(values, require_actions=require_actions)

    def forward(self, batch: Observation) -> Tensor | tuple[Tensor, dict[str, Tensor | float]]:
        if self.model is None:
            raise RuntimeError("Patch Policy is not initialized.")
        if self.training:
            return self.model(self._prepare_batch(batch, require_actions=True))
        return self.predict_action_chunk(batch)

    def predict_action_chunk(self, batch: Observation) -> Tensor:
        if self.model is None or self._postprocessor is None:
            raise RuntimeError("Patch Policy is not initialized.")
        model = cast("PatchPolicyModel", self.model)
        return self._postprocessor(model.predict_action_chunk(self._prepare_batch(batch, require_actions=False)))

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

    def setup(self, stage: str) -> None:
        """Resolve feature metadata from the training dataset when needed."""
        if stage != "fit" or self.model is not None:
            return
        datamodule = self.trainer.datamodule  # pyrefly: ignore[missing-attribute]
        dataset = datamodule.train_dataset
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Expected physicalai Dataset, got {type(dataset)}.")
        self._input_features = list(dataset.observation_features.values())
        self._output_features = list(dataset.action_features.values())
        self.configure_model()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )
