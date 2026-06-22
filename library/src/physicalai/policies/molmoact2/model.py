# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 model implementation."""

# isort: skip_file
# ruff: noqa: D,I001

from typing import Any

from torch import Tensor, nn

from lerobot.configs import FeatureType as LRFeatureType, PolicyFeature
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config as LeroBotMolmoAct2Config
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy as LeroBotMolmoAct2Policy

from physicalai.data import FeatureType as PAFeatureType
from physicalai.data.lerobot import FormatConverter
from physicalai.policies.base import Model
from physicalai.utils.hf_utils import HuggingfacePolicyContainer

from .backbones import MolmoAct2Backbone
from .config import MolmoAct2Config


class _MolmoAct2Model(nn.Module):
    """Hidden classs for molmoact2 model.

    Splits up forward pass into these modules:
        - MolmoAct2Backbone
        - language modelling head
    """
    def __init__(self) -> None:
        super().__init__()
        self.base_model = MolmoAct2Backbone


class MolmoAct2Model(Model):
    """Underlying model for PyTorch model for MolmoAct2.

    The model splits up the process of a forward pass into these modules:
        - _MolmoAct2Model
    """

    def __init__(self, config: MolmoAct2Config, hf_container: HuggingfacePolicyContainer | None) -> None:
        """Initialize the MolmoAct2 model wrapper.

        Raises:
            ValueError: If the resolved feature set is empty and cannot build the LeRobot bridge.
        """
        super().__init__()
        self.config = config
        self.hf_container = hf_container

        if self.hf_container:
            lr_input_features: dict[str, PolicyFeature] = {}
            for feature in self.config.input_features:
                if feature.name is None or feature.shape is None or feature.ftype is None:
                    continue

                if feature.ftype == PAFeatureType.VISUAL:
                    key = feature.name
                    if not key.startswith("observation.images."):
                        key = f"observation.images.{key}"
                    lr_input_features[key] = PolicyFeature(type=LRFeatureType.VISUAL, shape=feature.shape)
                elif feature.ftype == PAFeatureType.STATE:
                    key = feature.name
                    if not key.startswith("observation."):
                        key = f"observation.{key}"
                    lr_input_features[key] = PolicyFeature(type=LRFeatureType.STATE, shape=feature.shape)

            lr_output_features: dict[str, PolicyFeature] = {}
            for feature in self.config.output_features:
                if feature.name is None or feature.shape is None or feature.ftype is None:
                    continue

                if feature.ftype == PAFeatureType.ACTION:
                    key = feature.name
                    if key != "action" and not key.startswith("action."):
                        key = f"action.{key}"
                    lr_output_features[key] = PolicyFeature(type=LRFeatureType.ACTION, shape=feature.shape)

            if not lr_input_features or not lr_output_features:
                msg = "MolmoAct2Model requires non-empty input/output features to build lerobot bridge config."
                raise ValueError(msg)

            lr_config = LeroBotMolmoAct2Config(
                checkpoint_path=str(self.hf_container.checkpoint_location),
                norm_tag=self.config.norm_tag,
                inference_action_mode="continuous",
                enable_inference_cuda_graph=False,
                input_features=lr_input_features,
                output_features=lr_output_features,
            )

            self._model = LeroBotMolmoAct2Policy(config=lr_config)

    # physicalai
    @property
    def action_delta_indices(self) -> list | None:
        """Return action delta indices if this wrapper defines them."""
        return None

    def compute_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        """Compute the supervised training loss.

        Raises:
            NotImplementedError: This bridge only implements inference.
        """
        raise NotImplementedError

    @property
    def observation_delta_indices(self) -> list | None:
        """Return observation delta indices if this wrapper defines them."""
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        """Return reward delta indices if this wrapper defines them."""
        return None

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]] | Tensor:
        """Run training loss or inference depending on module mode.

        Returns:
            A training loss tuple when in training mode, otherwise the predicted action chunk.
        """
        if self.training:
            return self.compute_loss(batch)
        return self.predict_action_chunk(batch)

    def predict_action_chunk(self, batch: dict[str, Any]) -> Tensor:
        """Convert a processed batch into a predicted action chunk.

        Returns:
            The predicted action tensor produced by the wrapped LeRobot policy.
        """
        # If the preprocessor already emitted lerobot-ready keys (input_ids, pixel_values, …)
        # skip FormatConverter — it would drop those keys.
        if isinstance(batch, dict) and any(k in batch for k in ("input_ids", "attention_mask", "pixel_values")):
            lerobot_batch = batch
        else:
            lerobot_batch = FormatConverter.to_lerobot_dict(batch)
        model_device = next(self._model.parameters()).device
        lerobot_batch = {
            k: (v.to(device=model_device) if isinstance(v, Tensor) else v)
            for k, v in lerobot_batch.items()
        }
        return self._model.predict_action_chunk(lerobot_batch)
