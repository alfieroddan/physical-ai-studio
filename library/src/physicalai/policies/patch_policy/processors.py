# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Input and output processing for Patch Policy."""

from __future__ import annotations

from collections.abc import Mapping
from math import prod
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from physicalai.data import Feature, FeatureType

if TYPE_CHECKING:
    from .config import PatchPolicyConfig


def _statistics(feature: Feature, reference: Tensor) -> tuple[Tensor, Tensor]:
    """Materialize a feature's normalization statistics like a reference tensor.

    Args:
        feature: Feature containing optional mean and standard-deviation statistics.
        reference: Tensor supplying the target device and dtype.

    Returns:
        The mean and standard deviation as tensors.
    """
    stats = feature.normalization_data
    mean = 0.0 if stats is None or stats.mean is None else stats.mean
    std = 1.0 if stats is None or stats.std is None else stats.std
    return (
        torch.as_tensor(mean, device=reference.device, dtype=reference.dtype),
        torch.as_tensor(std, device=reference.device, dtype=reference.dtype),
    )


class PatchPolicyPreprocessor(nn.Module):
    """Order camera views, resize images, and normalize action targets.

    Args:
        input_features: Observation features used to determine camera order.
        output_features: Action features and their normalization statistics.
        image_size: Target image height and width.
    """

    def __init__(
        self,
        input_features: list[Feature],
        output_features: list[Feature],
        image_size: tuple[int, int],
    ) -> None:
        super().__init__()
        self.visual_features = [feature for feature in input_features if feature.ftype is FeatureType.VISUAL]
        self.output_features = output_features
        self.image_size = image_size

    @staticmethod
    def _image(images: Mapping[str, Any] | Tensor, feature: Feature) -> Tensor:
        """Resolve one configured camera image from an observation.

        Args:
            images: A shared image tensor or mapping of camera names to tensors.
            feature: Visual feature identifying the requested camera.

        Returns:
            The selected image tensor.

        Raises:
            KeyError: If the configured camera is absent from an image mapping.
            ValueError: If a shared image tensor has too few dimensions.
        """
        if isinstance(images, Tensor):
            if len(images.shape) < 4:
                raise ValueError("Visual input must include batch, channel, height, and width dimensions.")
            return images
        names = [feature.name or "", (feature.name or "").removeprefix("observation.images.")]
        for name in names:
            if name in images and isinstance(images[name], Tensor):
                return images[name]
        raise KeyError(f"Missing visual feature {feature.name!r} in Observation.images.")

    def normalize_actions(self, actions: Tensor) -> Tensor:
        """Normalize each action feature with its configured statistics.

        Args:
            actions: Concatenated action tensor.

        Returns:
            The normalized action tensor.

        Raises:
            ValueError: If an output shape is missing or the action width is invalid.
        """
        pieces = []
        offset = 0
        for feature in self.output_features:
            if not feature.shape:
                raise ValueError(f"Output feature {feature.name!r} has no shape.")
            width = prod(feature.shape)
            value = actions[..., offset : offset + width]
            mean, std = _statistics(feature, value)
            pieces.append((value - mean) / std.clamp_min(1e-8))
            offset += width
        if actions.shape[-1] != offset:
            raise ValueError(f"Action width {actions.shape[-1]} does not match configured width {offset}.")
        return torch.cat(pieces, dim=-1)

    def forward(self, batch: Mapping[str, Any], *, require_actions: bool = False) -> dict[str, Tensor]:
        """Convert an observation mapping into model inputs.

        Args:
            batch: Observation values containing images and optional actions.
            require_actions: Whether actions must be present and normalized.

        Returns:
            Model inputs containing stacked images and optional action metadata.

        Raises:
            TypeError: If images or required actions are not tensors in a supported layout.
            ValueError: If an image has an unsupported shape.
        """
        raw_images = batch.get("images")
        if not isinstance(raw_images, (Mapping, Tensor)):
            raise TypeError("Observation.images must be a tensor or camera dictionary.")
        views = []
        for feature in self.visual_features:
            image = self._image(raw_images, feature).float()
            if image.ndim == 4:
                image = image[:, None]
            if image.ndim != 5:
                raise ValueError(f"Expected image shape [B, T, C, H, W], got {tuple(image.shape)}.")
            flat = image.flatten(0, 1)
            if flat.shape[-2:] != self.image_size:
                flat = F.interpolate(flat, size=self.image_size, mode="bilinear", align_corners=False)
            views.append(flat.reshape(*image.shape[:-2], *self.image_size))
        processed = {"images": torch.stack(views, dim=2)}

        action = batch.get("action")
        if require_actions:
            if not isinstance(action, Tensor):
                raise TypeError("Training requires tensor actions.")
            processed["action"] = self.normalize_actions(action)
        extra = batch.get("extra")
        if isinstance(extra, Mapping) and isinstance(extra.get("action_is_pad"), Tensor):
            processed["extra.action_is_pad"] = extra["action_is_pad"]
        return processed


class PatchPolicyPostprocessor(nn.Module):
    """Denormalize actions and apply the execution horizon.

    Args:
        output_features: Action features and their normalization statistics.
        n_action_steps: Number of action timesteps to retain.
    """

    def __init__(self, output_features: list[Feature], n_action_steps: int) -> None:
        super().__init__()
        self.output_features = output_features
        self.n_action_steps = n_action_steps

    def forward(self, actions: Tensor) -> Tensor:
        """Denormalize action features and truncate the predicted horizon.

        Args:
            actions: Normalized action chunks.

        Returns:
            Denormalized chunks limited to the execution horizon.

        Raises:
            ValueError: If an output shape is missing or the action width is invalid.
        """
        pieces = []
        offset = 0
        for feature in self.output_features:
            if not feature.shape:
                raise ValueError(f"Output feature {feature.name!r} has no shape.")
            width = prod(feature.shape)
            value = actions[..., offset : offset + width]
            mean, std = _statistics(feature, value)
            pieces.append(value * std + mean)
            offset += width
        if actions.shape[-1] != offset:
            raise ValueError(f"Action width {actions.shape[-1]} does not match configured width {offset}.")
        return torch.cat(pieces, dim=-1)[:, : self.n_action_steps]


def make_policy_processors(config: PatchPolicyConfig) -> tuple[PatchPolicyPreprocessor, PatchPolicyPostprocessor]:
    """Create matching Patch Policy processors.

    Args:
        config: Resolved policy configuration.

    Returns:
        The input preprocessor and action postprocessor.
    """
    return (
        PatchPolicyPreprocessor(config.input_features, config.output_features, config.image_size),
        PatchPolicyPostprocessor(config.output_features, config.n_action_steps),
    )
