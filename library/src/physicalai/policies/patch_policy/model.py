# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 Gaoyue Zhou
# Authors: Gaoyue Zhou, Zichen Jeff Cui
# SPDX-License-Identifier: MIT

from typing import Any

from physicalai.policies.base import Model
import torch
import torch
from torch._tensor import Tensor


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
        self.input_features = input_features
        self.output_features = output_features
        self.n_action_steps = n_action_steps

    def compute_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]]:
        return torch.randn(1, 1), {"loss": torch.tensor(0.0)}

    @property
    def action_delta_indices(self) -> list | None:
        pass

    @property
    def observation_delta_indices(self) -> list | None:
        pass

    @property
    def reward_delta_indices(self) -> list | None:
        pass
