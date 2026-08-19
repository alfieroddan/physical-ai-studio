# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning module for Patch Policy."""

from torch import Tensor

from physicalai.data import Feature, Observation
from physicalai.export.mixin_policy import ExportablePolicyMixin
from physicalai.policies.base import Policy

from .model import PatchPolicyModel
from .processors import make_policy_processors


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
        # super init
        super().__init__(n_action_steps=n_action_steps)

        # init temp args
        self._input_features = input_features
        self._output_features = output_features
        self._n_action_steps = n_action_steps

        # processors
        self._preprocessor = None
        self._postprocessor = None

        # save hyperparameters and initialize model
        self.save_hyperparameters()

        # eager init
        if self._input_features is not None and self._output_features is not None:
            self._initialize_model()

    def _initialize_model(self):
        """Initialize the model."""

        # init model
        self.model = PatchPolicyModel(
            input_features=self._input_features,
            output_features=self._output_features,
            n_action_steps=self._n_action_steps,
        )

        # init pre and post processors here
        self._preprocessor, self._postprocessor = make_policy_processors(self.model.config)

    def forward(self, batch: Observation) -> tuple[Tensor, dict[str, Tensor]]:
        if self.model is None or self._preprocessor is None:
            raise RuntimeError("Policy is not initialized")
        return self.model(self._preprocessor(batch.to_dict()))

    def predict_action_chunk(self, batch: Observation) -> Tensor:
        if self.model is None or self._preprocessor is None or self._postprocessor is None:
            raise RuntimeError("Policy is not initialized")
        actions = self.model.predict_action_chunk(self._preprocessor(batch.to_dict()))
        return self._postprocessor(actions)
