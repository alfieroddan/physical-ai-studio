# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 postprocessor."""

from __future__ import annotations

import torch

from physicalai.data.observation import ACTION, Feature, FeatureType
from physicalai.policies.utils.features import feature_by_type
from physicalai.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType


class MolmoAct2Postprocessor(torch.nn.Module):
    """Convert normalized MolmoAct2 outputs back into action space."""

    def __init__(
        self,
        *,
        output_features: list[Feature],
        normalization_mode: str = "QUANTILES",
    ) -> None:
        """Initialize MolmoAct2 postprocessor.

        Args:
            output_features: Output feature definitions.
            normalization_mode: Normalization mode for action denormalization.
        """
        super().__init__()
        action_feature = feature_by_type(output_features, FeatureType.ACTION)
        self.action_name = action_feature.name if action_feature else ACTION
        features_map = {f.name: f for f in output_features if f.name}
        action_norm = (
            NormalizationType(normalization_mode)
            if action_feature is not None and action_feature.normalization_data is not None
            else NormalizationType.IDENTITY
        )
        self._denormalizer = FeatureNormalizeTransform(
            features_map,
            {FeatureType.ACTION: action_norm},
            inverse=True,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Denormalize and clamp actions.

        Args:
            batch: Batch containing ACTION or "actions" tensor.

        Returns:
            Batch with ACTION denormalized and clamped.

        Raises:
            ValueError: If no action tensor is present in the batch.
        """
        batch = dict(batch)
        action = batch.get(ACTION, batch.get("actions"))
        if action is None:
            msg = "MolmoAct2 postprocessor expected an action tensor in outputs."
            raise ValueError(msg)

        action = action.clamp(-1.0, 1.0)
        action = self._denormalizer.to(action.device)({self.action_name: action})[self.action_name]
        batch[ACTION] = action
        return batch
