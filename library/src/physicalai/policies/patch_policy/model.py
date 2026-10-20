# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Encoder-to-head Patch Policy model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from physicalai.policies.base import Model
from physicalai.policies.utils import in_episode_bound, reduce_losses

from .encoder import TimmFeatureEncoder
from .heads import PatchBehaviorTransformerHead, PatchDiffusionHead

if TYPE_CHECKING:
    from .config import ActionHeadName, EncoderOutput, FeatureEncoderName, PatchPolicyConfig


class PatchPolicyModel(Model):
    """A visual feature encoder followed by an interchangeable action head."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        action_dim: int,
        num_views: int,
        n_obs_steps: int = 1,
        chunk_size: int = 16,
        image_size: tuple[int, int] = (224, 224),
        encoder_name: FeatureEncoderName = "dinov2",
        encoder_output: EncoderOutput = "patches",
        encoder_pretrained: bool = True,
        freeze_encoder: bool = True,
        action_head: ActionHeadName = "diffusion",
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        eta: float = 1.0,
        vq_latent_dim: int = 128,
        vq_num_embeddings: int = 16,
        vq_num_groups: int = 2,
        vq_reconstruction_weight: float = 1.0,
        vq_commitment_weight: float = 0.25,
        offset_loss_weight: float = 10.0,
        secondary_code_weight: float = 0.5,
        focal_loss_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self._n_obs_steps = n_obs_steps
        self._chunk_size = chunk_size
        self.encoder = TimmFeatureEncoder(
            encoder_name,
            encoder_output,
            pretrained=encoder_pretrained,
            frozen=freeze_encoder,
            image_size=image_size,
        )
        if action_head == "diffusion":
            self.action_head = PatchDiffusionHead(
                self.encoder.output_dim,
                action_dim,
                chunk_size,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                num_train_timesteps=num_train_timesteps,
                num_inference_steps=num_inference_steps,
                beta_schedule=beta_schedule,  # type: ignore[arg-type]
                prediction_type=prediction_type,  # type: ignore[arg-type]
                clip_sample=clip_sample,
                clip_sample_range=clip_sample_range,
                eta=eta,
            )
        else:
            self.action_head = PatchBehaviorTransformerHead(
                self.encoder.output_dim,
                action_dim,
                chunk_size,
                n_obs_steps=n_obs_steps,
                n_patches=self.encoder.num_tokens * num_views,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                latent_dim=vq_latent_dim,
                num_embeddings=vq_num_embeddings,
                groups=vq_num_groups,
                reconstruction_weight=vq_reconstruction_weight,
                commitment_weight=vq_commitment_weight,
                offset_loss_weight=offset_loss_weight,
                secondary_code_weight=secondary_code_weight,
                focal_loss_gamma=focal_loss_gamma,
            )

    @classmethod
    def from_config(cls, config: PatchPolicyConfig) -> PatchPolicyModel:
        """Build the complete model architecture from one serialized config.

        Returns:
            A model containing the encoder and configured action head.
        """
        num_views = sum(
            feature.ftype is not None and feature.ftype.value == "VISUAL" for feature in config.input_features
        )
        return cls(
            action_dim=config.action_dim,
            num_views=num_views,
            n_obs_steps=config.n_obs_steps,
            chunk_size=config.chunk_size,
            image_size=config.image_size,
            encoder_name=config.encoder_name,
            encoder_output=config.encoder_output,
            encoder_pretrained=config.encoder_pretrained,
            freeze_encoder=config.freeze_encoder,
            action_head=config.action_head,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            num_train_timesteps=config.num_train_timesteps,
            num_inference_steps=config.num_inference_steps,
            beta_schedule=config.beta_schedule,
            prediction_type=config.prediction_type,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            eta=config.eta,
            vq_latent_dim=config.vq_latent_dim,
            vq_num_embeddings=config.vq_num_embeddings,
            vq_num_groups=config.vq_num_groups,
            vq_reconstruction_weight=config.vq_reconstruction_weight,
            vq_commitment_weight=config.vq_commitment_weight,
            offset_loss_weight=config.offset_loss_weight,
            secondary_code_weight=config.secondary_code_weight,
            focal_loss_gamma=config.focal_loss_gamma,
        )

    def _context(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        images = batch["images"]
        tokens = self.encoder(images)
        return {"tokens": tokens.flatten(1, 3)}

    def forward(self, batch: dict[str, Any]) -> Tensor | tuple[Tensor, dict[str, Tensor | float]]:
        if self.training:
            return self.compute_loss(batch)
        return self.predict_action_chunk(batch)

    def compute_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]]:
        losses = self.action_head.compute_loss(batch["action"], self._context(batch))
        loss = reduce_losses(losses, in_episode_bound(batch))
        return loss, {"loss": loss.detach()}

    @torch.no_grad()
    def compute_val_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]]:
        actions = self.action_head.sample(self._context(batch))
        loss = reduce_losses(F.mse_loss(actions, batch["action"], reduction="none"), in_episode_bound(batch))
        return loss, {"loss": loss.detach(), "action_mse": loss.detach()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any]) -> Tensor:
        return self.action_head.sample(self._context(batch))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self._chunk_size))

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(1 - self._n_obs_steps, 1))

    @property
    def reward_delta_indices(self) -> None:
        return None
