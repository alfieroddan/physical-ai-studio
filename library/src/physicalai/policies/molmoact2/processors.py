# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: ANN,COM812,D,E501,EM,FBT001,I001,PLR2004,S105,SIM108,TCH001,TRY
# pylint: disable=all

"""Processors for MolmoAct2."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from physicalai.data.observation import ACTION, Feature, FeatureType
from physicalai.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType

if TYPE_CHECKING:
    from .config import MolmoAct2Config


def _feature_by_type(features: list[Feature], feature_type: FeatureType) -> Feature | None:
    for feature in features:
        if feature.ftype == feature_type:
            return feature
    return None


class MolmoAct2Preprocessor(torch.nn.Module):
    """Pack Observation objects into MolmoAct2 model-ready tensors."""

    def __init__(self, config: MolmoAct2Config) -> None:
        super().__init__()
        self.config = config

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(batch, dict):
            msg = f"MolmoAct2Preprocessor.forward expects dict[str, Any], got {type(batch)}"
            raise TypeError(msg)

        packed_batch = batch
        return packed_batch


class MolmoAct2Postprocessor(torch.nn.Module):
    """Convert normalized MolmoAct2 outputs back into action space.

    This module expects normalized action predictions in ``[-1, 1]``, truncates any
    padded action dimensions to the environment action size, clamps values to the
    normalized range, and applies inverse normalization using resolved
    ``output_features`` metadata.
    """

    def __init__(self, *, output_features: list[Feature], max_action_dim: int) -> None:
        """Initialize the MolmoAct2 action postprocessor.

        Args:
            output_features: Resolved output feature definitions that include
                normalization metadata.
            max_action_dim: Fallback maximum action dimension when no ACTION
                feature shape is available.
        """
        super().__init__()
        action_feature = _feature_by_type(output_features, FeatureType.ACTION)
        self.env_action_dim = (
            int(action_feature.shape[0]) if action_feature and action_feature.shape else int(max_action_dim)
        )
        self.action_name = action_feature.name if action_feature else ACTION
        output_features_by_name = {f.name: f for f in output_features if f.name}
        self._denormalizer = FeatureNormalizeTransform(
            output_features_by_name,
            {FeatureType.ACTION: NormalizationType.QUANTILES},
            inverse=True,
        )

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        """Postprocess normalized action predictions.

        Args:
            batch: Model output batch containing an ``ACTION`` tensor.

        Returns:
            torch.Tensor: Action tensor in environment scale after slicing,
            clamping, and inverse normalization.

        Raises:
            TypeError: If ``batch`` is not a dictionary.
            ValueError: If ``ACTION`` is not present in ``batch``.
        """
        if not isinstance(batch, dict):
            msg = f"MolmoAct2Postprocessor.forward expects dict[str, Any], got {type(batch)}"
            raise TypeError(msg)

        if ACTION in batch:
            action = torch.as_tensor(batch[ACTION])
        else:
            msg = "MolmoAct2 postprocessor expected an action tensor in outputs."
            raise ValueError(msg)

        action = action[..., : self.env_action_dim]
        action = action.clamp(-1.0, 1.0)
        return self._denormalizer.to(action.device)({self.action_name: action})[self.action_name]


def make_molmoact2_preprocessors(config: MolmoAct2Config) -> tuple[MolmoAct2Preprocessor, MolmoAct2Postprocessor]:
    preprocessor = MolmoAct2Preprocessor(config=config)
    postprocessor = MolmoAct2Postprocessor(
        output_features=config.output_features,
        max_action_dim=config.max_action_dim,
    )
    return preprocessor, postprocessor
