# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Serializable configuration for Patch Policy."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Literal

from physicalai.config import Config

from physicalai.data import Feature, FeatureType

type FeatureEncoderName = Literal["dinov2", "webssl", "resnet18"]
type EncoderOutput = Literal["patches", "cls", "average"]
type ActionHeadName = Literal["diffusion", "behavior_transformer"]


def resolve_action_dim(output_features: list[Feature]) -> int:
    """Return the flattened width of the configured action features."""
    if not output_features:
        raise ValueError("Patch Policy requires at least one output feature.")
    width = 0
    for feature in output_features:
        if feature.ftype is not FeatureType.ACTION or not feature.shape:
            raise ValueError(f"Output feature {feature.name!r} must be an action with a non-empty shape.")
        width += prod(feature.shape)
    return width


@dataclass(frozen=True)
class PatchPolicyConfig(Config):
    """Feature, architecture, and horizon contract for Patch Policy."""

    input_features: list[Feature]
    output_features: list[Feature]
    action_dim: int

    n_obs_steps: int = 1
    chunk_size: int = 16
    n_action_steps: int = 16
    image_size: tuple[int, int] = (224, 224)

    encoder_name: FeatureEncoderName = "dinov2"
    encoder_output: EncoderOutput = "patches"
    encoder_pretrained: bool = True
    freeze_encoder: bool = True
    action_head: ActionHeadName = "diffusion"

    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1

    num_train_timesteps: int = 100
    num_inference_steps: int = 100
    beta_schedule: Literal["linear", "scaled_linear", "squaredcos_cap_v2"] = "squaredcos_cap_v2"
    prediction_type: Literal["epsilon", "sample"] = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    eta: float = 1.0

    vq_latent_dim: int = 128
    vq_num_embeddings: int = 16
    vq_num_groups: int = 2
    vq_reconstruction_weight: float = 1.0
    vq_commitment_weight: float = 0.25
    offset_loss_weight: float = 10.0
    secondary_code_weight: float = 0.5
    focal_loss_gamma: float = 2.0

    def __post_init__(self) -> None:
        """Validate values that affect model shapes."""
        if not any(feature.ftype is FeatureType.VISUAL for feature in self.input_features):
            raise ValueError("Patch Policy requires at least one visual input feature.")
        if self.action_dim != resolve_action_dim(self.output_features):
            raise ValueError("action_dim does not match the configured output features.")
        if self.n_obs_steps <= 0 or self.chunk_size <= 0:
            raise ValueError("n_obs_steps and chunk_size must be positive.")
        if not 0 < self.n_action_steps <= self.chunk_size:
            raise ValueError("n_action_steps must be in [1, chunk_size].")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if self.num_layers <= 0 or self.num_heads <= 0:
            raise ValueError("num_layers and num_heads must be positive.")
        if self.vq_num_groups <= 0 or self.vq_num_embeddings <= 0:
            raise ValueError("VQ group and embedding counts must be positive.")
