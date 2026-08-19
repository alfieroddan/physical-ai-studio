# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 Gaoyue Zhou
# Authors: Gaoyue Zhou, Zichen Jeff Cui
# SPDX-License-Identifier: MIT

from typing import Any

import torch
from torch import Tensor

from physicalai.policies.base import Model

from .config import PatchPolicyConfig


class PatchPolicyModel(Model):
    """Patch Policy Model class."""

    def __init__(
        self,
        input_features: list,
        output_features: list,
        n_action_steps: int = 50,
    ):
        """Initialize Patch Policy Model."""
        super().__init__()
        self.config = PatchPolicyConfig(
            input_features=input_features,
            output_features=output_features,
            n_action_steps=n_action_steps,
        )

    def compute_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]]:
        return torch.randn(1, 1), {"loss": torch.tensor(0.0)}

    def predict_action_chunk(self, observation: Any) -> Tensor:
        return torch.randn(self.config.n_action_steps, self.config.output_features[0].shape[-1])

    @property
    def action_delta_indices(self) -> list | None:
        pass

    @property
    def observation_delta_indices(self) -> list | None:
        pass

    @property
    def reward_delta_indices(self) -> list | None:
        pass
