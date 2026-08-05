# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 postprocessor."""

from __future__ import annotations

import torch

from physicalai.data.observation import ACTION, Feature, FeatureType
from physicalai.policies.utils.features import get_feature_by_type
from physicalai.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType

from .joint_transform import JointFrameTransform


class MolmoAct2Postprocessor(torch.nn.Module):
    """Convert normalized MolmoAct2 outputs back into action space."""

    def __init__(
        self,
        *,
        output_features: list[Feature],
        normalization_mode: str = "QUANTILES",
        adapt_to_so101: bool = False,
        joint_signs: list[float] | None = None,
        joint_offsets: list[float] | None = None,
    ) -> None:
        """Initialize MolmoAct2 postprocessor.

        Args:
            output_features: Output feature definitions.
            normalization_mode: Normalization mode for action denormalization.
            adapt_to_so101: Map actions from the checkpoint frame back to the SO-101
                robot frame after denormalization.
            joint_signs: Per-joint signs for the frame transform.
            joint_offsets: Per-joint offsets for the frame transform.
        """
        super().__init__()
        action_feature = get_feature_by_type(output_features, FeatureType.ACTION)
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
        self._adapt_to_so101 = adapt_to_so101
        self._joint_transform = JointFrameTransform(joint_signs or [], joint_offsets or []) if adapt_to_so101 else None

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Denormalize, clamp and (optionally) map actions back to the robot frame.

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
        if self._joint_transform is not None:
            action = self._joint_transform.to_robot(action)
        batch[ACTION] = action
        return batch
