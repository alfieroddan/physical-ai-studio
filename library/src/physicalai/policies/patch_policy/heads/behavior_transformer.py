# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Patch Policy Behavior Transformer action head."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from physicalai.policies.components.action_heads import ActionHead, Context
from physicalai.policies.patch_policy.gpt import PatchGPT


def _mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))


class ResidualActionQuantizer(nn.Module):
    """Compact residual vector quantizer for flattened action chunks."""

    def __init__(self, action_dim: int, chunk_size: int, latent_dim: int, num_embeddings: int, groups: int) -> None:
        super().__init__()
        flat_dim = action_dim * chunk_size
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.groups = groups
        self.encoder = _mlp(flat_dim, latent_dim, 128)
        self.decoder = _mlp(latent_dim, flat_dim, 128)
        self.codebook = nn.Parameter(torch.randn(groups, num_embeddings, latent_dim) * 0.02)

    def encode(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return quantized latent, residual code indices, and commitment loss."""
        latent = self.encoder(actions.flatten(-2))
        residual = latent
        quantized = torch.zeros_like(latent)
        codes = []
        for group in range(self.groups):
            distances = (residual[:, None] - self.codebook[group][None]).square().sum(-1)
            code = distances.argmin(-1)
            selected = self.codebook[group][code]
            quantized += selected
            residual -= selected.detach()
            codes.append(code)
        commitment = (latent - quantized.detach()).square().mean(-1)
        quantized = latent + (quantized - latent).detach()
        return quantized, torch.stack(codes, dim=-1), commitment

    def decode_codes(self, codes: Tensor) -> Tensor:
        """Decode residual code indices to action chunks."""
        latent = torch.stack(
            [self.codebook[group][codes[:, group]] for group in range(self.groups)],
        ).sum(0)
        return self.decode(latent)

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent).reshape(-1, self.chunk_size, self.action_dim)


class PatchBehaviorTransformerHead(ActionHead):
    """Predict residual-VQ action centers and continuous offsets from patch tokens."""

    def __init__(
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
        latent_dim: int = 128,
        num_embeddings: int = 16,
        groups: int = 2,
        reconstruction_weight: float = 1.0,
        commitment_weight: float = 0.25,
        offset_loss_weight: float = 10.0,
        secondary_code_weight: float = 0.5,
        focal_loss_gamma: float = 2.0,
    ) -> None:
        super().__init__(chunk_size, action_dim)
        self.n_obs_steps = n_obs_steps
        self.n_patches = n_patches
        self.groups = groups
        self.num_embeddings = num_embeddings
        self.reconstruction_weight = reconstruction_weight
        self.commitment_weight = commitment_weight
        self.offset_loss_weight = offset_loss_weight
        self.secondary_code_weight = secondary_code_weight
        self.focal_loss_gamma = focal_loss_gamma
        self.trunk = PatchGPT(
            input_dim=context_dim,
            output_dim=hidden_dim,
            n_patches=n_patches,
            n_obs_steps=n_obs_steps,
            n_layer=num_layers,
            n_head=num_heads,
            n_embd=hidden_dim,
            dropout=dropout,
        )
        self.code_head = _mlp(hidden_dim, groups * num_embeddings, hidden_dim)
        self.offset_head = _mlp(hidden_dim, groups * num_embeddings * chunk_size * action_dim, hidden_dim)
        self.quantizer = ResidualActionQuantizer(action_dim, chunk_size, latent_dim, num_embeddings, groups)

    def _embedding(self, context: Context) -> Tensor:
        tokens = context["tokens"]
        expected = self.n_obs_steps * self.n_patches
        if tokens.shape[1] != expected:
            raise ValueError(f"Expected {expected} context tokens, got {tokens.shape[1]}.")
        return self.trunk(tokens.reshape(tokens.shape[0], self.n_obs_steps, self.n_patches, tokens.shape[-1]))[:, -1]

    def _predictions(self, context: Context) -> tuple[Tensor, Tensor]:
        embedding = self._embedding(context)
        logits = self.code_head(embedding).reshape(-1, self.groups, self.num_embeddings)
        offsets = self.offset_head(embedding).reshape(
            -1,
            self.groups,
            self.num_embeddings,
            self.chunk_size,
            self.action_dim,
        )
        return logits, offsets

    def _actions(self, logits: Tensor, offsets: Tensor, codes: Tensor | None = None) -> Tensor:
        if codes is None:
            codes = logits.argmax(-1)
        centers = self.quantizer.decode_codes(codes)
        batch = torch.arange(logits.shape[0], device=logits.device)[:, None]
        groups = torch.arange(self.groups, device=logits.device)[None]
        selected_offsets = offsets[batch, groups, codes].sum(1)
        return centers + selected_offsets

    def sample(self, context: Context, *, noise: Tensor | None = None, num_steps: int | None = None) -> Tensor:
        """Return the most likely action chunk; iterative arguments are ignored."""
        del noise, num_steps
        logits, offsets = self._predictions(context)
        return self._actions(logits, offsets)

    def compute_loss(self, actions: Tensor, context: Context) -> Tensor:
        """Return combined focal-code, offset, reconstruction, and commitment losses."""
        logits, offsets = self._predictions(context)
        latent, target_codes, commitment = self.quantizer.encode(actions)
        reconstructed = self.quantizer.decode(latent)
        predicted = self._actions(logits, offsets, target_codes)

        probabilities = logits.softmax(-1)
        target_probability = probabilities.gather(-1, target_codes[..., None]).squeeze(-1).clamp_min(1e-8)
        focal = -((1 - target_probability) ** self.focal_loss_gamma) * target_probability.log()
        weights = focal.new_ones(self.groups)
        if self.groups > 1:
            weights[1:] = self.secondary_code_weight
        code_loss = (focal * weights).mean(-1)
        reconstruction = F.mse_loss(reconstructed, actions, reduction="none").mean((-2, -1))
        global_loss = code_loss + self.reconstruction_weight * reconstruction + self.commitment_weight * commitment
        offset = F.l1_loss(predicted, actions, reduction="none") * self.offset_loss_weight
        return offset + global_loss[:, None, None]
