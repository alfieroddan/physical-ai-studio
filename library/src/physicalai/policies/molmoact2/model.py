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

    def __init__(self):
        pass

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

    # model
