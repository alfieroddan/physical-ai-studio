# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 Gaoyue Zhou
# Authors: Gaoyue Zhou, Zichen Jeff Cui
# SPDX-License-Identifier: MIT

from typing import Any

import torch
from torch import Tensor

from physicalai.data.observation import ACTION, GOAL_IMAGE, IMAGES, Feature, FeatureType, Observation
from physicalai.policies.base import Model

from .action_head import resolve_action_head
from .config import PatchPolicyConfig
from .image_encoder import resolve_image_encoder

ACTION_CHUNK_NDIM = 3


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


def _select_features(features: list[Feature] | None, ftype: FeatureType) -> list[Feature]:
    """Select the features of a given type.

    Returns:
        The matching features, or an empty list.
    """
    return [feature for feature in features or [] if feature.ftype == ftype]


def _action_dim(output_features: list[Feature] | None) -> int:
    """Read the action dimension from the output features.

    Returns:
        Size of the last action axis.

    Raises:
        ValueError: If no action feature with a known shape is present.
    """
    for feature in _select_features(output_features, FeatureType.ACTION):
        if feature.shape:
            return feature.shape[-1]
    msg = "Output features must contain an ACTION feature with a shape."
    raise ValueError(msg)


class PatchPolicyModel(Model):
    """Patch Policy Model class."""

    def __init__(  # noqa: PLR0913
        self,
        input_features: list,
        output_features: list,
        n_action_steps: int = 50,
        encoder_name: str = "webssl",
        use_goal_image: bool = False,
        action_head_name: str = "vqbet",
        n_obs_steps: int = 10,
        chunk_size: int = 50,
    ) -> None:
        """Initialize Patch Policy Model."""
        super().__init__()
        self.config = PatchPolicyConfig(
            input_features=input_features,
            output_features=output_features,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            chunk_size=chunk_size,
            encoder_name=encoder_name,
            use_goal_image=use_goal_image,
            action_head_name=action_head_name,
        )

        # build image encoder
        self.image_encoder = resolve_image_encoder(self.config.encoder_name)

        # build transformer and action head
        n_views = len(_select_features(input_features, FeatureType.VISUAL)) or 1
        self.action_head = resolve_action_head(
            self.config.action_head_name,
            config=self.config,
            token_dim=self.image_encoder.output_dim * (2 if use_goal_image else 1),
            act_dim=_action_dim(output_features),
            n_patches=self.image_encoder.n_patches * n_views,
        )

    def encode_observation(self, observation: Any) -> Tensor:
        """Encode an observation into ``[B, T, P, E]`` patch tokens.

        Returns:
            Patch tokens with views folded into the patch axis.

        Raises:
            ValueError: If images are missing, or a required/mismatched goal image is present.
        """
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

        tokens = self.image_encoder(images, goal_vector)
        batch_size, n_views, n_patches, token_dim = tokens.shape
        tokens = tokens.reshape(batch_size, 1, n_views * n_patches, token_dim)
        if self.config.n_obs_steps > 1:
            tokens = tokens.expand(batch_size, self.config.n_obs_steps, n_views * n_patches, token_dim)
        return tokens

    def compute_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Compute the training loss.

        Returns:
            Tuple of (loss, loss dict).
        """
        tokens = self.encode_observation(batch)
        actions = batch[ACTION]
        if actions.ndim == ACTION_CHUNK_NDIM:
            actions = actions.unsqueeze(1)
        if actions.shape[1] > self.config.chunk_size:
            actions = actions[:, : self.config.chunk_size]
        return self.action_head.compute_loss(tokens, actions)

    def predict_action_chunk(self, observation: Any) -> Tensor:
        """Predict a ``[B, n_action_steps, A]`` action chunk.

        The model emits the full chunk for the latest observation timestep and trims it to the
        configured number of executed steps.
        """
        tokens = self.encode_observation(observation)
        del observation
        predicted = self.action_head.predict(tokens)
        if predicted.ndim == 4:
            predicted = predicted[:, -1]
        return predicted[:, : self.config.n_action_steps]

    @property
    def action_delta_indices(self) -> list | None:
        return None

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        return None
