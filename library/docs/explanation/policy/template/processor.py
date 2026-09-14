# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Observation/action pre- and post-processing for the native policy design."""

from __future__ import annotations

from collections.abc import Mapping
from math import prod
from typing import Any

import torch
from torch import Tensor, nn

from physicalai.data import Feature

from .config import NewPolicyModelConfig


class NewPolicyPreprocessor(nn.Module):
    """Preprocessor - handles normlaization from feature and defaults if None"""
    def __init__(self, input_features: list[Feature], output_features: list[Feature]) -> None:
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

    @staticmethod
    def _normalize(value: Tensor, feature: Feature) -> Tensor:
        normalization = feature.normalization_data
        mean = 0.0 if normalization is None or normalization.mean is None else normalization.mean
        std = 1.0 if normalization is None or normalization.std is None else normalization.std
        mean_tensor = torch.as_tensor(mean, dtype=value.dtype, device=value.device)
        std_tensor = torch.as_tensor(std, dtype=value.dtype, device=value.device)
        return (value - mean_tensor) / (std_tensor + 1e-8)

    def normalize_actions(self, actions: Tensor) -> Tensor:
        normalized_actions = []
        offset = 0
        for feature in self.output_features:
            if feature.shape is None:
                raise ValueError(f"Output feature {feature.name!r} must define a shape")

            feature_size = prod(feature.shape)
            feature_actions = actions[..., offset : offset + feature_size]
            normalized_actions.append(self._normalize(feature_actions, feature))
            offset += feature_size

        if actions.shape[-1] != offset:
            raise ValueError(
                f"Action width {actions.shape[-1]} does not match configured output width {offset}"
            )
        return torch.cat(normalized_actions, dim=-1)

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        task = batch.get("task")
        if not isinstance(task, Tensor):
            raise TypeError("Expected Observation.task to contain token IDs")

        processed = {"input_ids": task.long()}
        for feature in self.input_features:
            if feature.name is None:
                continue
            value = batch.get(feature.name)
            if isinstance(value, Tensor) and value.is_floating_point():
                processed[feature.name] = self._normalize(value, feature)
        return processed


class NewPolicyPostprocessor(nn.Module):
    """Postprocessor - handles denormalization from feature and defaults if None"""
    def __init__(self, output_features: list[Feature], n_action_steps: int) -> None:
        super().__init__()
        self.output_features = output_features
        self.n_action_steps = n_action_steps

    def forward(self, actions: Tensor) -> Tensor:
        processed_actions = []
        offset = 0
        for feature in self.output_features:
            if feature.shape is None:
                raise ValueError(f"Output feature {feature.name!r} must define a shape")

            feature_size = prod(feature.shape)
            feature_actions = actions[..., offset : offset + feature_size]
            normalization = feature.normalization_data
            mean = 0.0 if normalization is None or normalization.mean is None else normalization.mean
            std = 1.0 if normalization is None or normalization.std is None else normalization.std
            mean_tensor = torch.as_tensor(mean, dtype=actions.dtype, device=actions.device)
            std_tensor = torch.as_tensor(std, dtype=actions.dtype, device=actions.device)
            processed_actions.append(feature_actions * std_tensor + mean_tensor)
            offset += feature_size

        if actions.shape[-1] != offset:
            raise ValueError(
                f"Action width {actions.shape[-1]} does not match configured output width {offset}"
            )
        return torch.cat(processed_actions, dim=-1)[:, : self.n_action_steps]


def make_policy_processors(
    config: NewPolicyModelConfig,
) -> tuple[NewPolicyPreprocessor, NewPolicyPostprocessor]:
    return (
        NewPolicyPreprocessor(config.input_features, config.output_features),
        NewPolicyPostprocessor(config.output_features, config.n_action_steps),
    )
