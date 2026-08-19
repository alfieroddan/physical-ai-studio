# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 Gaoyue Zhou
# Authors: Gaoyue Zhou, Zichen Jeff Cui
# SPDX-License-Identifier: MIT

from typing import Any

import torch
from torch import Tensor

from physicalai.data.observation import GOAL_IMAGE, IMAGES, Observation
from physicalai.policies.base import Model

from .config import PatchPolicyConfig
from .image_encoder import resolve_image_encoder


def _stack_views(observation: Any, field: str) -> Tensor | None:
    """Stack a flat or nested observation image field into a ``[B, V, C, H, W]`` tensor.

    Returns:
        The stacked views, or ``None`` if the field is absent.
    """
    keys = Observation.get_flattened_keys(observation, field)
    if not keys:
        return None

    value = observation[keys[0]] if keys == [field] else None
    views = list(value.values()) if isinstance(value, dict) else [observation[key] for key in keys]
    views = [view for view in views if view is not None]
    if not views:
        return None
    return torch.stack(views, dim=1)


class PatchPolicyModel(Model):
    """Patch Policy Model class."""

    def __init__(
        self,
        input_features: list,
        output_features: list,
        n_action_steps: int = 50,
        encoder_name: str = "webssl",
        use_goal_image: bool = False,
    ) -> None:
        """Initialize Patch Policy Model."""
        super().__init__()
        # build config
        self.config = PatchPolicyConfig(
            input_features=input_features,
            output_features=output_features,
            n_action_steps=n_action_steps,
            encoder_name=encoder_name,
            use_goal_image=use_goal_image,
        )

        # build image encoder
        self.image_encoder = resolve_image_encoder(self.config.encoder_name)

        # build transformer

        # build action head

    def compute_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]]:
        device = next(self.image_encoder.parameters()).device
        loss = torch.zeros(1, device=device)
        return loss, {"loss": loss.detach().clone()}

    def predict_action_chunk(self, observation: Any) -> Tensor:
        """Predict action chunk."""

        # call image encoder on images
        images = _stack_views(observation, IMAGES)
        if images is None:
            msg = f"Observation does not contain images: {observation}"
            raise ValueError(msg)

        # optional goal image
        goal_vector = _stack_views(observation, GOAL_IMAGE)

        if self.config.use_goal_image and goal_vector is None:
            msg = f"Observation does not contain goal image: {observation}"
            raise ValueError(msg)

        if goal_vector is not None and goal_vector.shape[2:] != images.shape[2:]:
            msg = f"Goal image shape {goal_vector.shape} does not match image shape {images.shape}"
            raise ValueError(msg)

        # at this point Observation is no longer needed, del for potential memory savings
        del observation

        # call image encoder
        image_features = self.image_encoder(images, goal_vector)

        # transformer

        return image_features

    @property
    def action_delta_indices(self) -> list | None:
        return None

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        return None
