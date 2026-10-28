# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 Gaoyue Zhou
# Authors: Gaoyue Zhou, Zichen Jeff Cui
# SPDX-License-Identifier: MIT

"""Patch Policy Behavior Transformer action head."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from physicalai.policies.components.action_heads import ActionHead, Context
from physicalai.policies.patch_policy.components.gpt import PatchGPT


class _MLP(nn.Sequential):
    """Upstream Behavior Transformer prediction MLP."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(1024, output_dim),
            nn.Dropout(0.0),
        )


class _ActionMLP(nn.Module):
    """MLP used by the upstream action VQ-VAE encoder and decoder."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.fc = nn.Linear(128, output_dim)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, values: Tensor) -> Tensor:
        """Project flattened action windows.

        Returns:
            Projected values with the configured output width.
        """
        return self.fc(self.encoder(values))


class ResidualActionQuantizer(nn.Module):
    """Residual action VQ-VAE with the upstream encoder/decoder equations."""

    def __init__(
        self,
        action_dim: int,
        chunk_size: int,
        latent_dim: int,
        num_embeddings: int,
        groups: int,
        *,
        action_scale: float,
    ) -> None:
        """Initialize action projections and the residual codebook."""
        super().__init__()
        flat_dim = action_dim * chunk_size
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.groups = groups
        self.action_scale = action_scale
        self.encoder = _ActionMLP(flat_dim, latent_dim)
        self.decoder = _ActionMLP(latent_dim, flat_dim)
        codebook = torch.empty(groups, num_embeddings, latent_dim)
        nn.init.kaiming_uniform_(codebook)
        self.codebook = nn.Parameter(codebook)

    def encode(self, actions: Tensor) -> Tensor:
        """Return residual code indices for action windows."""
        latent = self.encoder((actions / self.action_scale).flatten(-2))
        residual = latent
        codes = []
        for codebook in self.codebook:
            distances = torch.cdist(residual[:, None], codebook[None]).squeeze(1)
            code = distances.argmin(-1)
            residual = torch.sub(residual, codebook[code])
            codes.append(code)
        return torch.stack(codes, dim=-1)

    def decode_codes(self, codes: Tensor) -> Tensor:
        """Decode residual code indices into action centers.

        Returns:
            Decoded actions shaped ``[B, chunk_size, action_dim]``.
        """
        groups = torch.arange(self.groups, device=codes.device)
        latent = self.codebook[groups[:, None], codes.transpose(0, 1)].sum(0)
        decoded = self.decoder(latent) * self.action_scale
        return decoded.reshape(-1, self.chunk_size, self.action_dim)


class PatchBehaviorTransformerHead(ActionHead):
    """Upstream-compatible patch-aware Behavior Transformer head."""

    def __init__(  # noqa: PLR0913
        self,
        context_dim: int,
        action_dim: int,
        chunk_size: int,
        *,
        n_obs_steps: int,
        n_patches: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        latent_dim: int = 512,
        num_embeddings: int = 16,
        groups: int = 2,
        action_scale: float = 1.0,
        offset_loss_weight: float = 10.0,
        secondary_code_weight: float = 0.5,
        focal_loss_gamma: float = 2.0,
    ) -> None:
        """Initialize the transformer, action predictors, and quantizer."""
        super().__init__(chunk_size, action_dim)
        self.n_obs_steps = n_obs_steps
        self.n_patches = n_patches
        self.groups = groups
        self.num_embeddings = num_embeddings
        self.offset_loss_weight = offset_loss_weight
        self.secondary_code_weight = secondary_code_weight
        self.focal_loss_gamma = focal_loss_gamma
        self.trunk = PatchGPT(
            input_dim=context_dim,
            output_dim=256,
            n_patches=n_patches,
            n_obs_steps=n_obs_steps + chunk_size,
            n_layer=num_layers,
            n_head=num_heads,
            n_embd=hidden_dim,
            dropout=dropout,
        )
        self.code_head = _MLP(256, groups * num_embeddings)
        self.offset_head = _MLP(256, groups * num_embeddings * chunk_size * action_dim)
        self.quantizer = ResidualActionQuantizer(
            action_dim,
            chunk_size,
            latent_dim,
            num_embeddings,
            groups,
            action_scale=action_scale,
        )

    def _embeddings(self, context: Context) -> Tensor:
        tokens = context["tokens"]
        expected = self.n_obs_steps * self.n_patches
        if tokens.shape[1] != expected:
            message = f"Expected {expected} context tokens, got {tokens.shape[1]}."
            raise ValueError(message)
        return self.trunk(tokens.reshape(tokens.shape[0], self.n_obs_steps, self.n_patches, tokens.shape[-1]))

    def _predictions(self, context: Context) -> tuple[Tensor, Tensor]:
        embeddings = self._embeddings(context)
        logits = self.code_head(embeddings).reshape(
            embeddings.shape[0],
            self.n_obs_steps,
            self.groups,
            self.num_embeddings,
        )
        offsets = self.offset_head(embeddings).reshape(
            embeddings.shape[0],
            self.n_obs_steps,
            self.groups,
            self.num_embeddings,
            self.chunk_size,
            self.action_dim,
        )
        return logits, offsets

    def _actions(self, logits: Tensor, offsets: Tensor, codes: Tensor | None = None) -> Tensor:
        if codes is None:
            probabilities = logits.softmax(-1)
            codes = torch.multinomial(probabilities.flatten(0, -2), 1).reshape(*logits.shape[:-1])
        flat_codes = codes.flatten(0, 1)
        centers = self.quantizer.decode_codes(flat_codes).reshape(
            logits.shape[0],
            logits.shape[1],
            self.chunk_size,
            self.action_dim,
        )
        batch = torch.arange(logits.shape[0], device=logits.device)[:, None, None]
        time = torch.arange(logits.shape[1], device=logits.device)[None, :, None]
        groups = torch.arange(self.groups, device=logits.device)[None, None, :]
        selected_offsets = offsets[batch, time, groups, codes].sum(2)
        return centers.detach() + selected_offsets

    def sample(self, context: Context, *, noise: Tensor | None = None, num_steps: int | None = None) -> Tensor:
        """Sample the final observation timestep's action chunk.

        Returns:
            Predicted actions shaped ``[B, chunk_size, action_dim]``.
        """
        del noise, num_steps
        logits, offsets = self._predictions(context)
        return self._actions(logits, offsets)[:, -1]

    def compute_loss(self, actions: Tensor, context: Context, *, valid_steps: Tensor | None = None) -> Tensor:
        """Compute the upstream focal-code and continuous-offset loss.

        Returns:
            Combined code-prediction and action-offset loss.

        Raises:
            ValueError: If actions do not cover every overlapping target window.
        """
        del valid_steps
        expected = self.n_obs_steps + self.chunk_size - 1
        if actions.shape[1] != expected:
            message = f"Expected {expected} action steps, got {actions.shape[1]}."
            raise ValueError(message)
        windows = actions.unfold(1, self.chunk_size, 1).transpose(-1, -2)
        logits, offsets = self._predictions(context)
        target_codes = self.quantizer.encode(windows.flatten(0, 1)).reshape(
            actions.shape[0],
            self.n_obs_steps,
            self.groups,
        )
        predicted = self._actions(logits, offsets)

        probabilities = logits.softmax(-1)
        target_probability = probabilities.gather(-1, target_codes[..., None]).squeeze(-1).clamp_min(1e-8)
        focal = -((1 - target_probability) ** self.focal_loss_gamma) * target_probability.log()
        code_loss = 5 * focal[..., 0].mean()
        if self.groups > 1:
            code_loss = torch.add(code_loss, self.secondary_code_weight * focal[..., 1:].mean())
        offset_loss = F.l1_loss(predicted, windows)
        return code_loss + self.offset_loss_weight * offset_loss
