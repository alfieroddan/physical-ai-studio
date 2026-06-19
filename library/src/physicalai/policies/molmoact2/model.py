# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 model implementation."""

from typing import Any

from torch import Tensor, nn

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
        self.config = config
        self.hf_container = hf_container

        from lerobot.configs import FeatureType as FeatureType
        from lerobot.configs import PolicyFeature
        from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config as LeroBotMolmoAct2Config
        from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy as LeroBotMolmoAct2Policy
        # Build a clean lerobot config — mirror what the working named-wrapper path does.
        # Use checkpoint_path pointing to the HF repo (or local dir) so that lerobot's
        # internal _load_hf_model() and norm-stats loading both resolve correctly.
        # All model/norm specifics (setup_type, control_mode, chunk_size, etc.) are
        # populated automatically by apply_norm_tag_metadata() inside MolmoAct2Policy.__init__.
        config = LeroBotMolmoAct2Config(
            checkpoint_path=str(self.hf_container.checkpoint_location),
            norm_tag="libero",
            inference_action_mode="continuous",
            enable_inference_cuda_graph=False,
            input_features={
                "observation.images.image": PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, 224, 224),
                ),
                "observation.images.image2": PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, 224, 224),
                ),
                "observation.state": PolicyFeature(
                    type=FeatureType.STATE,
                    shape=(8,),
                ),
            },
            output_features={
                "action": PolicyFeature(
                    type=FeatureType.ACTION,
                    shape=(7,),
                ),
            },
        )

        # Instantiate the lerobot policy — this internally calls:
        #   apply_norm_tag_metadata()  → sets chunk_size, n_action_steps, setup_type, control_mode
        #   _load_hf_model()           → loads HF weights from checkpoint_path
        self.model = LeroBotMolmoAct2Policy(
            config=config,
        )

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
        lerobot_batch = FormatConverter.to_lerobot_dict(batch)
        return self._model.predict_action_chunk(lerobot_batch, inference_action_mode="continuous")

    # model
