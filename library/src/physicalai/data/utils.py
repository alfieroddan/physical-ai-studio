# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for data."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

import torch

from .observation import FeatureType, Observation

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


def infer_batch_size(batch: dict[str, Any] | Observation) -> int:
    """Infer the batch size from the first tensor in the batch.

    This function scans the values of the input batch dictionary for important keys and returns
    the size of the first dimension of the first `torch.Tensor` it finds. It
    assumes that all tensors in the batch have the same batch dimension.

    Args:
        batch (dict[str, Any] | Observation): A dictionary where values may include tensors.

    Returns:
        int: The inferred batch size.

    Raises:
        ValueError: If no tensor is found in the batch.
    """
    data = batch.__dict__ if isinstance(batch, Observation) else batch

    priority_keys = ("action", "state", "images")

    # first scan observation keys we expect
    for key in priority_keys:
        if key in data:
            value = data[key]
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            if isinstance(value, dict):
                for item in value.values():
                    if isinstance(item, torch.Tensor):
                        return item.shape[0]

    # fallback, scan all the values looking for a tensor
    for value in data.values():
        if isinstance(value, torch.Tensor):
            return value.shape[0]
        if isinstance(value, dict):
            for item in value.values():
                if isinstance(item, torch.Tensor):
                    return item.shape[0]

    msg = "Could not infer batch size — no tensors found."
    raise ValueError(msg)


def is_visual_stat(stat: dict[str, Any]) -> bool:
    """Return whether a dataset stat entry represents a visual feature.

    Args:
        stat: Dataset stats entry metadata.

    Returns:
        True if the entry type is visual, otherwise False.
    """
    return stat.get("type") == FeatureType.VISUAL.value


def to_observation_image_key(feature_name: str, *, images_field_name: str = "images") -> str:
    """Normalize an image feature name into observation-format key form.

    Args:
        feature_name: Input feature name from user config or dataset metadata.
        images_field_name: Observation field name used for image containers.

    Returns:
        Observation-format key, such as ``observation.image`` or
        ``observation.images.camera1``.
    """
    if feature_name.startswith("observation."):
        return feature_name
    if feature_name in {"image", images_field_name}:
        return "observation.image"
    if feature_name.startswith("image") and not feature_name.startswith(images_field_name):
        return f"observation.{feature_name}"
    if feature_name.startswith(f"{images_field_name}."):
        return f"observation.{feature_name}"
    return f"observation.{images_field_name}.{feature_name}"


def to_stat_name_from_observation_key(observation_key: str) -> str:
    """Convert an observation-format key into the stored stat name.

    Args:
        observation_key: Key in observation format (``observation.*``).

    Returns:
        Dataset stat name without the ``observation.`` prefix.
    """
    return observation_key.removeprefix("observation.")


def to_runtime_image_batch_key(observation_key: str, *, images_field_name: str = "images") -> str:
    """Convert an observation-format visual key into runtime batch image key form.

    Args:
        observation_key: Observation-format key (``observation.*``).
        images_field_name: Observation field name used for runtime image tensors.

    Returns:
        Runtime image key used by flattened observation batches, such as
        ``images`` or ``images.camera1``.
    """
    key = observation_key.removeprefix("observation.")
    if key == "image":
        return images_field_name
    if key.startswith("image") and not key.startswith(f"{images_field_name}."):
        return (
            f"{images_field_name}.{key.removeprefix('image.')}"
            if key.startswith("image.")
            else f"{images_field_name}.{key}"
        )
    if key.startswith("image."):
        return f"{images_field_name}.{key.removeprefix('image.')}"
    return key


def ordered_config_image_features(dataset_stats: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    """Return visual dataset stat keys as config-level image feature names.

    Args:
        dataset_stats: Dataset statistics dictionary.

    Returns:
        Ordered tuple of config image feature names, e.g.
        ``("image", "image2")`` or ``("images.left", "images.right")``.
    """
    keys: list[str] = []
    for key, stat in dataset_stats.items():
        if not is_visual_stat(stat):
            continue
        keys.append(key.removeprefix("observation."))
    return tuple(keys)


def ordered_observation_image_keys(
    dataset_stats: dict[str, dict[str, Any]],
    *,
    images_field_name: str = "images",
) -> tuple[str, ...]:
    """Return visual dataset stat keys as ordered runtime observation image keys.

    Args:
        dataset_stats: Dataset statistics dictionary.
        images_field_name: Observation field name used for runtime image tensors.

    Returns:
        Ordered tuple of runtime image keys, such as
        ``("images", "images.image2")``.
    """
    keys: list[str] = []
    for key, stat in dataset_stats.items():
        if not is_visual_stat(stat):
            continue
        keys.append(to_runtime_image_batch_key(key, images_field_name=images_field_name))
    return tuple(keys)


def _default_mean_std(shape: tuple[Any, ...]) -> tuple[list[float], list[float]]:
    channel_dim = int(shape[0]) if shape else 1
    return [0.0] * channel_dim, [1.0] * channel_dim


def _normalize_requested_image_features(
    image_features: Iterable[str],
    *,
    images_field_name: str,
) -> list[str]:
    requested = [to_observation_image_key(name, images_field_name=images_field_name) for name in image_features]
    if len(requested) != len(set(requested)):
        msg = f"Duplicate image feature names are not allowed: {requested}"
        raise ValueError(msg)
    return requested


def _rebuild_dataset_stats_with_requested_visuals(
    dataset_stats: dict[str, dict[str, Any]],
    requested: list[str],
) -> dict[str, dict[str, Any]]:
    resolved: dict[str, dict[str, Any]] = {}
    visuals_inserted = False
    for key, stat in dataset_stats.items():
        if is_visual_stat(stat):
            if visuals_inserted:
                continue
            for image_feature in requested:
                visual_stat = deepcopy(dataset_stats[image_feature])
                visual_stat["type"] = FeatureType.VISUAL.value
                visual_stat["name"] = to_stat_name_from_observation_key(image_feature)
                resolved[image_feature] = visual_stat
            visuals_inserted = True
            continue
        if key in requested:
            msg = f"Resolved image feature key {key} collides with an existing non-visual key."
            raise ValueError(msg)
        resolved[key] = deepcopy(stat)
    return resolved


def _resolve_visual_dataset_stats_strict(
    dataset_stats: dict[str, dict[str, Any]],
    requested: list[str],
    ds_visual_keys: list[str],
) -> dict[str, dict[str, Any]]:
    if len(requested) > len(ds_visual_keys):
        msg = (
            "image_features and dataset_stats visual features must match in count when "
            "dataset_stats is provided. "
            f"image_features={len(requested)} dataset_stats={len(ds_visual_keys)}"
        )
        raise ValueError(msg)

    missing_visual_keys = [key for key in requested if key not in ds_visual_keys]
    if missing_visual_keys:
        msg = (
            "image_features contains visual keys that do not exist in dataset_stats "
            "when dataset_stats is provided. "
            f"missing={missing_visual_keys} ds_visual_keys={ds_visual_keys}"
        )
        raise ValueError(msg)

    expected_prefix = ds_visual_keys[: len(requested)]
    if requested != expected_prefix:
        logger.warning(
            "Reordering visual dataset_stats to match image_features order. requested=%s dataset_stats_prefix=%s",
            requested,
            expected_prefix,
        )

    dropped_visual_keys = [key for key in ds_visual_keys if key not in requested]
    if dropped_visual_keys:
        logger.warning(
            "Pruning %d visual dataset_stats entries not present in image_features: %s",
            len(dropped_visual_keys),
            dropped_visual_keys,
        )

    return _rebuild_dataset_stats_with_requested_visuals(dataset_stats, requested)


def _resolve_visual_dataset_stats_with_defaults(
    dataset_stats: dict[str, dict[str, Any]],
    requested: list[str],
    ds_visual_keys: list[str],
) -> dict[str, dict[str, Any]]:
    if not ds_visual_keys and requested:
        msg = "Cannot resolve image_features because dataset_stats has no reference visual feature."
        raise ValueError(msg)

    reference_visual_key = ds_visual_keys[0] if ds_visual_keys else None
    reference_visual_stat = deepcopy(dataset_stats[reference_visual_key]) if reference_visual_key is not None else None

    dropped_visual_keys = ds_visual_keys[len(requested) :]
    if dropped_visual_keys:
        logger.warning(
            "Pruning %d visual dataset_stats entries not present in image_features: %s",
            len(dropped_visual_keys),
            dropped_visual_keys,
        )

    resolved_with_defaults: dict[str, dict[str, Any]] = {}
    visuals_inserted = False
    for key, stat in dataset_stats.items():
        if is_visual_stat(stat):
            if visuals_inserted:
                continue

            for idx, image_feature in enumerate(requested):
                if idx < len(ds_visual_keys):
                    visual_stat = deepcopy(dataset_stats[ds_visual_keys[idx]])
                else:
                    if reference_visual_stat is None:
                        msg = "Missing reference visual feature while expanding image_features."
                        raise ValueError(msg)
                    visual_stat = deepcopy(reference_visual_stat)
                    mean, std = _default_mean_std(cast("tuple[Any, ...]", visual_stat.get("shape", (1,))))
                    visual_stat["mean"] = mean
                    visual_stat["std"] = std

                visual_stat["type"] = FeatureType.VISUAL.value
                visual_stat["name"] = to_stat_name_from_observation_key(image_feature)
                resolved_with_defaults[image_feature] = visual_stat

            visuals_inserted = True
            continue

        if key in requested:
            msg = f"Resolved image feature key {key} collides with an existing non-visual key."
            raise ValueError(msg)
        resolved_with_defaults[key] = deepcopy(stat)

    return resolved_with_defaults


def resolve_visual_dataset_stats(
    dataset_stats: dict[str, dict[str, Any]],
    image_features: Iterable[str] | None,
    *,
    allow_missing_visual_defaults: bool = False,
    images_field_name: str = "images",
) -> dict[str, dict[str, Any]]:
    """Resolve visual dataset stats against user-provided image feature names.

    Args:
        dataset_stats: Dataset statistics dictionary with visual and non-visual entries.
        image_features: Requested image feature names. If None, returns input unchanged.
        allow_missing_visual_defaults: When True, synthesizes default stats for
            requested visual features that are not present in ``dataset_stats``.
        images_field_name: Observation field name used for image containers.

    Returns:
        Dataset stats with visual entries reordered, pruned, or expanded to match
        requested image features.
    """
    if image_features is None:
        return dataset_stats

    requested = _normalize_requested_image_features(image_features, images_field_name=images_field_name)
    ds_visual_keys = [key for key, stat in dataset_stats.items() if is_visual_stat(stat)]
    if allow_missing_visual_defaults:
        return _resolve_visual_dataset_stats_with_defaults(dataset_stats, requested, ds_visual_keys)
    return _resolve_visual_dataset_stats_strict(dataset_stats, requested, ds_visual_keys)


__all__ = [
    "infer_batch_size",
    "is_visual_stat",
    "ordered_config_image_features",
    "ordered_observation_image_keys",
    "resolve_visual_dataset_stats",
    "to_observation_image_key",
    "to_runtime_image_batch_key",
    "to_stat_name_from_observation_key",
]
