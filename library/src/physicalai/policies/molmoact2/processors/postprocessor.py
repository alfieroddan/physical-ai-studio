# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 postprocessor."""

from __future__ import annotations

from typing import Any

import torch

from physicalai.data.observation import ACTION, FeatureType
from physicalai.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType

from .common import feature_by_type


class MolmoAct2Postprocessor(torch.nn.Module):
    """Convert normalized MolmoAct2 outputs back into action space."""

    def __init__(self, *, config: Any) -> None:
        super().__init__()
        action_feature = feature_by_type(config.output_features, FeatureType.ACTION)
        self.env_action_dim = int(action_feature.shape[0]) if action_feature and action_feature.shape else int(config.max_action_dim)
        self.action_name = action_feature.name if action_feature else ACTION
        output_features = {f.name: f for f in config.output_features if f.name}
        action_norm = (
            NormalizationType.QUANTILES
            if action_feature is not None and action_feature.normalization_data is not None
            else NormalizationType.IDENTITY
        )
        self._denormalizer = FeatureNormalizeTransform(
            output_features,
            {FeatureType.ACTION: action_norm},
            inverse=True,
        )

    def forward(self, outputs: torch.Tensor | dict[str, Any]) -> torch.Tensor:
        if isinstance(outputs, dict):
            if ACTION in outputs:
                action = torch.as_tensor(outputs[ACTION])
            elif "actions" in outputs:
                action = torch.as_tensor(outputs["actions"])
            else:
                raise ValueError("MolmoAct2 postprocessor expected an action tensor in outputs.")
        else:
            action = torch.as_tensor(outputs)

        action = action[..., : self.env_action_dim]
        action = action.clamp(-1.0, 1.0)
        return self._denormalizer.to(action.device)({self.action_name: action})[self.action_name]
