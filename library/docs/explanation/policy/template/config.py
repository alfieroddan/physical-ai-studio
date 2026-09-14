# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Serializable model configuration for the native policy design."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod

from physicalai.config import Config
from physicalai.data import Feature


@dataclass
class NewPolicyModelConfig(Config):
    """Resolved config owned by the policy."""
    input_features: list[Feature]
    output_features: list[Feature]
    action_dim: int
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    vocab_size: int = 256_000
    chunk_size: int = 32
    n_action_steps: int = 32
    image_size: tuple[int, int] = (224, 224)
    tokenizer_max_length: int = 48
    lora_config: dict[str, int | float] = field(
        default_factory=lambda: {"rank": 64, "alpha": 16, "dropout": 0.05}
    )


def resolve_action_dim(output_features: list[Feature]) -> int:
    """Compute the flattened action width implied by a set of output features."""
    if not output_features:
        raise ValueError("At least one output feature is required")

    action_dim = 0
    for feature in output_features:
        if feature.shape is None or not feature.shape:
            raise ValueError(f"Output feature {feature.name!r} must define a non-empty shape")
        action_dim += prod(feature.shape)
    return action_dim

