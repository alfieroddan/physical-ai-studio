# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 Gaoyue Zhou
# Authors: Gaoyue Zhou, Zichen Jeff Cui
# SPDX-License-Identifier: MIT

"""Input and output processing for Patch Policy."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from itertools import starmap
from math import prod
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn
from torchvision.transforms.v2 import functional as tvf

from physicalai.data import Feature, FeatureType
from physicalai.data.observation import ACTION, IMAGES, Observation
from physicalai.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType

if TYPE_CHECKING:
    from .config import PatchPolicyConfig


def resolve_camera_name(feature: Feature) -> str:
    """Return the canonical camera name for a visual feature.

    Raises:
        ValueError: If the feature has no usable name.
    """
    name = (feature.name or "").removeprefix("observation.images.").removeprefix("images.")
    if not name:
        raise ValueError("Patch Policy visual features must have non-empty camera names.")
    return name


def _normalizer(feature: Feature, key: str, *, inverse: bool = False) -> FeatureNormalizeTransform:
    """Build a mean/std normalizer when this policy has complete feature stats."""
    stats = feature.normalization_data
    has_mean_std = stats is not None and stats.mean is not None and stats.std is not None
    mode = (
        NormalizationType.MEAN_STD
        if has_mean_std and feature.ftype in {FeatureType.VISUAL, FeatureType.ACTION}
        else NormalizationType.IDENTITY
    )
    feature_type = feature.ftype
    features = {key: feature} if mode is not NormalizationType.IDENTITY else {}
    norm_map = {feature_type: mode} if feature_type is not None else {}
    return FeatureNormalizeTransform(features, norm_map, inverse=inverse)


def _warn_missing_action_normalization(features: list[Feature], operation: str) -> None:
    missing_features = []
    for index, feature in enumerate(features):
        stats = feature.normalization_data
        if stats is None or stats.mean is None or stats.std is None:
            missing_features.append(feature.name or f"index {index}")
    if missing_features:
        warnings.warn(
            f"Missing action mean/std normalization data for {operation} on: {', '.join(missing_features)}. "
            "Identity normalization will be used.",
            stacklevel=3,
        )


def _normalize_features(
    values: Tensor,
    features: list[Feature],
    normalizers: nn.ModuleList,
) -> Tensor:
    """Apply shared feature normalizers to slices of a concatenated tensor.

    Raises:
        ValueError: If a feature has no shape or the configured total width differs from the tensor.
    """
    normalized = values.clone()
    offset = 0
    for index, (feature, normalizer) in enumerate(zip(features, normalizers, strict=True)):
        if not feature.shape:
            raise ValueError(f"Feature {feature.name!r} has no shape.")
        end = offset + prod(feature.shape)
        key = f"feature_{index}"
        piece = {key: normalized[..., offset:end]}
        normalizer(piece)
        normalized[..., offset:end] = piece[key]
        offset = end
    if values.shape[-1] != offset:
        raise ValueError(f"Feature width {values.shape[-1]} does not match configured width {offset}.")
    return normalized


def _gather_flattened_or_unflattened_key(batch: dict[str, Any], field: str) -> Mapping[str, Any]:
    """Gather a nested or flattened observation field into a keyed mapping."""
    if field in batch and isinstance(batch[field], Mapping):
        return batch[field]
    prefixed_keys = Observation.get_flattened_keys(batch, field)
    prefix = f"{field}."
    return {key.removeprefix(prefix): batch[key] for key in prefixed_keys}


class PatchPolicyPreprocessor(nn.Module):
    """Order camera views, resize and normalize images, and normalize action targets.

    Args:
        input_features: Observation features used to determine camera order.
        output_features: Action features and their normalization statistics.
        image_size: Target image height and width.
        n_obs_steps: Required number of temporal observation frames.
    """

    def __init__(
        self,
        input_features: list[Feature],
        output_features: list[Feature],
        image_size: tuple[int, int],
        n_obs_steps: int = 1,
    ) -> None:
        super().__init__()
        self.visual_features = [feature for feature in input_features if feature.ftype is FeatureType.VISUAL]
        self.output_features = output_features
        _warn_missing_action_normalization(output_features, "normalization")
        self.image_size = image_size
        self.n_obs_steps = n_obs_steps
        camera_names = [resolve_camera_name(feature) for feature in self.visual_features]
        if len(camera_names) != len(set(camera_names)):
            raise ValueError(f"Patch Policy camera names must be unique, got {camera_names!r}.")
        self._image_normalizers = nn.ModuleList(
            starmap(_normalizer, zip(self.visual_features, camera_names, strict=True)),
        )
        self._action_normalizers = nn.ModuleList(
            _normalizer(feature, f"feature_{index}") for index, feature in enumerate(output_features)
        )

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
        canonical_name = resolve_camera_name(feature)
        names = [feature.name or "", canonical_name, f"images.{canonical_name}"]
        for name in names:
            if name in images and isinstance(images[name], Tensor):
                return images[name]
        raise KeyError(f"Missing visual feature {feature.name!r} in Observation.images.")

    @staticmethod
    def _with_time_axis(image: Tensor, feature: Feature, n_obs_steps: int) -> Tensor:
        if image.ndim == 4:
            return image[:, None]
        if image.ndim == 5 and image.shape[1] == n_obs_steps:
            return image
        if image.ndim == 5:
            raise ValueError(
                f"Expected {n_obs_steps} observation timestep(s), got {image.shape[1]} for {feature.name!r}.",
            )
        raise ValueError(
            f"Expected image shape [B, C, H, W] or [B, T, C, H, W] for key {feature.name}, got {tuple(image.shape)}.",
        )

    @staticmethod
    def _actions(batch: dict[str, Any]) -> Tensor | None:
        if ACTION in batch:
            actions = batch[ACTION]
        elif Observation.get_flattened_keys(batch, ACTION):
            raise TypeError("Expected actions as a tensor, not flattened action fields.")
        else:
            return None
        if not isinstance(actions, Tensor):
            raise TypeError(f"Expected actions to be a tensor or None, got {type(actions)}.")
        if actions.ndim != 3:
            raise ValueError(f"Expected action shape [B, CHUNK, DIM], got {tuple(actions.shape)}.")
        return actions

    def normalize_actions(self, actions: Tensor) -> Tensor:
        """Normalize each action feature with its configured statistics.

        Args:
            actions: Concatenated action tensor.

        Returns:
            The normalized action tensor.

        """
        return _normalize_features(actions, self.output_features, self._action_normalizers)

    def forward(
        self,
        batch: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Convert an observation mapping into model inputs.

        Args:
            batch: Observation values containing images and optional actions.

        Returns:
            Model inputs containing per-camera images and optional action metadata.

        Raises:
            TypeError: If images or required actions are not tensors in a supported layout.
        """
        # Gather camera fields from either flattened or nested observations.
        batch = dict(batch)
        images = _gather_flattened_or_unflattened_key(batch, IMAGES)
        if not images:
            raise TypeError("Observation must contain at least one image tensor.")
        if len(images) == 1 and IMAGES in images:
            images = images[IMAGES]
        if images is None:
            raise TypeError("Observation.images cannot be None.")

        processed: dict[str, Any] = {"images": {}}
        for feature, normalizer in zip(self.visual_features, self._image_normalizers, strict=True):
            camera_name = resolve_camera_name(feature)
            image = self._with_time_axis(self._image(images, feature), feature, self.n_obs_steps)
            image = tvf.to_dtype(image, torch.float32, scale=True)
            processed["images"][camera_name] = tvf.resize(
                image,
                list(self.image_size),
                antialias=False,
            )
            normalizer(processed["images"])

        # Keep actions as [B, CHUNK, DIM] while normalizing configured features.
        actions = self._actions(batch)
        if isinstance(actions, Tensor):
            processed["action"] = self.normalize_actions(actions)
        action_is_pad = batch.get("extra.action_is_pad")
        extra = batch.get("extra")
        if action_is_pad is None and isinstance(extra, Mapping):
            action_is_pad = extra.get("action_is_pad")
        if isinstance(action_is_pad, Tensor):
            processed["extra.action_is_pad"] = action_is_pad
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
        _warn_missing_action_normalization(output_features, "de-normalization")
        self.n_action_steps = n_action_steps
        self._action_denormalizers = nn.ModuleList(
            _normalizer(feature, f"feature_{index}", inverse=True) for index, feature in enumerate(output_features)
        )

    def forward(self, actions: Tensor) -> Tensor:
        """Denormalize action features and truncate the predicted horizon.

        Args:
            actions: Normalized action chunks.

        Returns:
            Denormalized chunks limited to the execution horizon.

        """
        return _normalize_features(actions, self.output_features, self._action_denormalizers)[:, : self.n_action_steps]


def make_policy_processors(
    config: PatchPolicyConfig,
    image_size: tuple[int, int],
) -> tuple[PatchPolicyPreprocessor, PatchPolicyPostprocessor]:
    """Create matching Patch Policy processors.

    Args:
        config: Resolved policy configuration.
        image_size: Input image height and width resolved by the encoder.

    Returns:
        The input preprocessor and action postprocessor.
    """
    return (
        PatchPolicyPreprocessor(config.input_features, config.output_features, image_size, config.n_obs_steps),
        PatchPolicyPostprocessor(config.output_features, config.n_action_steps),
    )
