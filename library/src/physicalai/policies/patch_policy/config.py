# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Patch Policy Config."""

from dataclasses import dataclass

from physicalai.config import Config


@dataclass(frozen=True)
class PatchPolicyConfig(Config):
    """Configuration for the Patch Policy."""

    input_features: list[str] | None = None
    output_features: list[str] | None = None
    n_action_steps: int = 50
    n_obs_steps: int = 10
    chunk_size: int = 50

    # Image encoder arguments
    encoder_name: str = "webssl"

    # Goal args
    use_goal_image: bool = False

    # Action head arguments
    action_head_name: str = "vqbet"

    # Transformer trunk arguments
    n_layer: int = 8
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0

    # Action quantizer arguments
    vqvae_latent_dim: int = 512
    vqvae_n_embed: int = 16
    vqvae_groups: int = 2
    vqvae_lr: float = 1e-3
    vqvae_iters: int = 1000
    vqvae_batch_size: int = 1024

    # Loss weighting
    offset_loss_multiplier: float = 1.0e3
    secondary_code_multiplier: float = 0.5
    focal_loss_gamma: float = 2.0

    def __post_init__(self) -> None:
        """Validate action chunking configuration."""
        if self.chunk_size < self.n_action_steps:
            msg = (
                f"chunk_size must be >= n_action_steps. Got chunk_size={self.chunk_size} "
                f"and n_action_steps={self.n_action_steps}."
            )
            raise ValueError(msg)
