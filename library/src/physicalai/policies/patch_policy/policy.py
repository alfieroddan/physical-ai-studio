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
        n_obs_steps: int = 10,
        chunk_size: int = 50,
        # image encoder args
        encoder_name: str = "webssl",
        # goal args
        use_goal_image: bool = False,
    ) -> None:
        """Initialize Patch Policy."""
        # super init
        super().__init__(n_action_steps=n_action_steps)

        # init args
        self._input_features = input_features
        self._output_features = output_features
        self._n_action_steps = n_action_steps
        self._n_obs_steps = n_obs_steps
        self._chunk_size = chunk_size
        self._encoder_name = encoder_name
        self._use_goal_image = use_goal_image

        # processors
        self._preprocessor = None
        self._postprocessor = None

        # model
        self.model: PatchPolicyModel | None = None

        # save hyperparameters and initialize model
        self.save_hyperparameters()

        # eager init
        if self._input_features is not None and self._output_features is not None:
            self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the model."""
        # init model
        self.model = PatchPolicyModel(
            input_features=self._input_features,
            output_features=self._output_features,
            n_action_steps=self._n_action_steps,
            n_obs_steps=self._n_obs_steps,
            chunk_size=self._chunk_size,
            encoder_name=self._encoder_name,
            use_goal_image=self._use_goal_image,
        )

        # init pre and post processors here
        self._preprocessor, self._postprocessor = make_policy_processors(self.model.config)

    def forward(self, batch: Observation):
        if self.training:
            # During training, return loss information for backpropagation
            if self.model is None:
                msg = "Model is not initialized."
                raise RuntimeError(msg)
            processed_batch = self._preprocessor(batch.to_dict())
            return self.model(processed_batch)

        # During evaluation, return action chunk predictions
        return self.predict_action_chunk(batch)

    def predict_action_chunk(self, batch: Observation) -> Tensor:
        if self.model is None or self._preprocessor is None or self._postprocessor is None:
            raise RuntimeError("Policy is not initialized")
        actions = self.model.predict_action_chunk(self._preprocessor(batch.to_dict()))
        return self._postprocessor(actions)
