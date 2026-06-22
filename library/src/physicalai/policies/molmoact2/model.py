# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 model implementation."""

from typing import Any

from torch import Tensor, nn

from physicalai.data.lerobot import FormatConverter
from physicalai.policies.base import Model

from .backbones import MolmoAct2Backbone


class _MolmoAct2Model(nn.Module):
    """Hidden classs for molmoact2 model.

    Splits up forward pass into these modules:
        - MolmoAct2Backbone
        - language modelling head
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.base_model = MolmoAct2Backbone


class MolmoAct2Model(Model):
    """Underlying model for PyTorch model for MolmoAct2.

    The model splits up the process of a forward pass into these modules:
        - _MolmoAct2Model
    """

    def __init__(self, config, hf_container):
        super().__init__()
        self.config = config
        self.hf_container = hf_container

        if self.hf_container:
            from lerobot.configs import FeatureType as LRFeatureType
            from lerobot.configs import PolicyFeature
            from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config as LeroBotMolmoAct2Config
            from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy as LeroBotMolmoAct2Policy

            lr_config = LeroBotMolmoAct2Config(
                checkpoint_path=str(self.hf_container.checkpoint_location),
                norm_tag="libero",
                inference_action_mode="continuous",
                enable_inference_cuda_graph=False,
                input_features={
                    "observation.images.image": PolicyFeature(type=LRFeatureType.VISUAL, shape=(3, 224, 224)),
                    "observation.images.image2": PolicyFeature(type=LRFeatureType.VISUAL, shape=(3, 224, 224)),
                    "observation.state": PolicyFeature(type=LRFeatureType.STATE, shape=(8,)),
                },
                output_features={
                    "action": PolicyFeature(type=LRFeatureType.ACTION, shape=(7,)),
                },
            )

            self._model = LeroBotMolmoAct2Policy(config=lr_config)

    # physicalai
    @property
    def action_delta_indices(self) -> list | None:
        pass

    def compute_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        pass

    @property
    def observation_delta_indices(self) -> list | None:
        pass

    @property
    def reward_delta_indices(self) -> list | None:
        pass

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]] | Tensor:
        if self.training:
            return self.compute_loss(batch)
        return self.predict_action_chunk(batch)

    def predict_action_chunk(self, batch):
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
        return self._model.predict_action_chunk(lerobot_batch, inference_action_mode="continuous")
