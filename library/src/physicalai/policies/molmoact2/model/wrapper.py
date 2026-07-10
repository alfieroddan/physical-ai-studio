# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 physicalai ``Model`` frontend.

Owns the backbone and preprocessing, and exposes the physicalai inference API.
Weight loading targets the backbone directly so checkpoint keys are preserved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

import torch
import torch.nn.functional as F  # noqa: N812
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor

from physicalai.data.constants import ACTION
from physicalai.data.observation import FeatureType
from physicalai.policies.base import Model

from .backbone import MolmoAct2ForConditionalGeneration

if TYPE_CHECKING:
    from physicalai.policies.molmoact2.config import MolmoAct2Config

_SAFE_WEIGHTS_NAME = "model.safetensors"
_SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"


def _env_action_dim(config: MolmoAct2Config) -> int:
    """Return the environment action dimension from the output features."""
    for feature in config.output_features or []:
        if feature.ftype == FeatureType.ACTION and feature.shape:
            return int(feature.shape[0])
    return 0

# Keys of the fully-prepared, traceable tensors the backbone consumes. All
# value-dependent host prep runs in the preprocessor, so the model graph only
# sees these fixed-shape tensors (keeping it exportable).
_MODEL_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "images",
    "token_pooling",
    "action_dim_is_pad",
)


def _masked_flow_loss(
    predicted: Tensor,
    target: Tensor,
    *,
    action_horizon_is_pad: Tensor | None,
    action_dim_is_pad: Tensor | None,
) -> Tensor:
    """Mean squared error over valid action steps and dimensions only."""
    loss = F.mse_loss(predicted, target, reduction="none")
    mask = torch.ones_like(loss, dtype=torch.bool)
    if action_horizon_is_pad is not None:
        mask = mask & (~action_horizon_is_pad.to(device=loss.device, dtype=torch.bool))[:, :, None]
    if action_dim_is_pad is not None:
        mask = mask & (~action_dim_is_pad.to(device=loss.device, dtype=torch.bool))[:, None, :]
    valid = mask.to(loss.dtype)
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def _strict_load_safetensors_weights(model: torch.nn.Module, checkpoint_location: str) -> None:
    """Load safetensors weights into ``model``, verifying exact key correspondence.

    Raises:
        RuntimeError: If the checkpoint keys do not exactly match the model.
        FileNotFoundError: If no safetensors checkpoint is found.
    """
    checkpoint_dir = Path(checkpoint_location)
    index_path = checkpoint_dir / _SAFE_WEIGHTS_INDEX_NAME
    single_file_path = checkpoint_dir / _SAFE_WEIGHTS_NAME

    if index_path.is_file():
        with index_path.open(encoding="utf-8") as index_file:
            weight_map = json.load(index_file)["weight_map"]
        missing = sorted(set(model.state_dict()) - set(weight_map))
        unexpected = sorted(set(weight_map) - set(model.state_dict()))
        if missing or unexpected:
            msg = f"MolmoAct2 checkpoint keys mismatch. Missing: {missing[:6]} Unexpected: {unexpected[:6]}"
            raise RuntimeError(msg)
        for shard in sorted(set(weight_map.values())):
            state_dict = load_safetensors_file(str(checkpoint_dir / shard), device="cpu")
            model.load_state_dict(state_dict, strict=False)
        return

    if single_file_path.is_file():
        model.load_state_dict(load_safetensors_file(str(single_file_path), device="cpu"), strict=True)
        return

    msg = f"No safetensors checkpoint found at {checkpoint_location!r}."
    raise FileNotFoundError(msg)


class MolmoAct2Model(Model):
    """Inference frontend wrapping :class:`MolmoAct2ForConditionalGeneration`."""

    def __init__(self, config: MolmoAct2Config) -> None:
        """Build the backbone."""
        super().__init__()
        self.config = config
        self.backbone = MolmoAct2ForConditionalGeneration(config)

    def load_pretrained_weights(self, checkpoint_location: str) -> None:
        """Load safetensors weights into the backbone (strict key match)."""
        _strict_load_safetensors_weights(self.backbone, checkpoint_location)

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Any],
        *,
        sample_noise: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Generate an action chunk from a preprocessed inference batch.

        Args:
            batch: Preprocessed inference batch of backbone-ready model inputs.
            sample_noise: Start flow matching from sampled Gaussian noise instead
                of zeros. Defaults to ``config.sample_noise``.
            generator: Optional RNG used when ``sample_noise`` is enabled.

        Returns:
            Actions of shape ``(batch, n_action_steps, env_action_dim)``.
        """
        model_inputs = {key: batch[key] for key in _MODEL_INPUT_KEYS if key in batch}
        actions = self.backbone.model.generate_actions_from_inputs(
            **model_inputs,
            action_horizon=int(self.config.n_action_steps),
            sample_noise=bool(self.config.sample_noise) if sample_noise is None else sample_noise,
            generator=generator,
        )
        env_action_dim = _env_action_dim(self.config) or actions.shape[-1]
        return actions[:, : int(self.config.n_action_steps), :env_action_dim].to(torch.float32)

    def forward(self, batch: dict[str, Any]) -> Tensor | tuple[Tensor, dict[str, float]]:
        """Run the model.

        Returns:
            Training mode: ``(loss, metrics)``. Inference mode: action chunk.
        """
        if self.training:
            return self.compute_loss(batch)
        return self.predict_action_chunk(batch)

    @override
    def compute_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        """Continuous flow-matching training loss.

        Returns:
            The scalar loss and a metrics dict.
        """
        predicted, target = self.backbone.model.predict_flow_velocity(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            token_type_ids=batch.get("token_type_ids"),
            images=batch.get("images"),
            token_pooling=batch.get("token_pooling"),
            actions=batch[ACTION],
            action_dim_is_pad=batch.get("action_dim_is_pad"),
            freeze_encoder=bool(self.config.train_action_expert_only),
        )
        loss = _masked_flow_loss(
            predicted,
            target,
            action_horizon_is_pad=batch.get("action_horizon_is_pad"),
            action_dim_is_pad=batch.get("action_dim_is_pad") if self.config.mask_action_dim_padding else None,
        )
        value = float(loss.detach().float())
        return loss, {"action_flow_loss": value, "loss": value}

    @override
    @torch.no_grad()
    def compute_val_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        """Validation MSE between predicted and ground-truth action chunks.

        Returns:
            The scalar loss and a metrics dict.
        """
        gt_actions = batch[ACTION]
        predicted = self.predict_action_chunk(batch)
        horizon = min(int(gt_actions.shape[1]), int(predicted.shape[1]))
        action_dim = min(int(gt_actions.shape[2]), int(predicted.shape[2]))
        target = gt_actions[:, :horizon, :action_dim].to(device=predicted.device, dtype=predicted.dtype)
        loss = F.mse_loss(predicted[:, :horizon, :action_dim], target)
        return loss, {"loss": float(loss.detach().float())}

    def freeze_to_action_expert(self) -> None:
        """Freeze every parameter except the action expert (memory-lean fine-tuning)."""
        trainable = 0
        for name, param in self.named_parameters():
            is_action_expert = "action_expert" in name
            param.requires_grad_(is_action_expert)
            trainable += param.numel() if is_action_expert else 0
        if trainable == 0:
            msg = "train_action_expert_only=True, but no action_expert parameters were found."
            raise RuntimeError(msg)

    # Unused delta-index properties.

    @property
    def reward_delta_indices(self) -> list[int] | None:
        """Reward delta indices (unused)."""
        return None

    @property
    def action_delta_indices(self) -> list[int] | None:
        """Action delta indices (unused)."""
        return None

    @property
    def observation_delta_indices(self) -> list[int] | None:
        """Observation delta indices (unused)."""
        return None
