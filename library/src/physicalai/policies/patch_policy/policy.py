# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning module for Patch Policy."""

from typing import Any

import torch

from physicalai.data import Feature, Observation
from physicalai.export.mixin_policy import ExportablePolicyMixin
from physicalai.policies.base import Policy

from .model import PatchPolicyModel


class PatchPolicy(ExportablePolicyMixin, Policy):
    """Patch Policy class."""

    def __init__(
            self,
            input_features: list[Feature] | None = None,
            output_features: list[Feature] | None = None,
            *,
            n_action_steps: int = 50,
        ):
        """Initialize Patch Policy."""
        # init temp args
        self.input_features = input_features
        self.output_features = output_features
        self.n_action_steps = n_action_steps
        # super init
        super().__init__(n_action_steps=self.n_action_steps)
        # save hyperparameters and initialize model
        self.save_hyperparameters()
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model."""
        self.model = PatchPolicyModel(
            input_features=self.input_features,
            output_features=self.output_features,
            n_action_steps=self.n_action_steps,
        )

    def forward(self, batch: Observation) -> Any:
        """Perform forward pass of the policy."""
        pass

    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Predict a chunk of actions from observation."""
        # temp random
        return torch.randn(1, self.output_features[0].shape[1])