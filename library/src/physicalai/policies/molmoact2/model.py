# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 policy model stub."""

from __future__ import annotations

import torch

from physicalai.policies.base import Model


class MolmoAct2Model(Model):
    """Placeholder MolmoAct2 model stub."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize the MolmoAct2 model stub."""
        msg = "MolmoAct2Model is not implemented yet."
        raise NotImplementedError(msg)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Run the model forward pass."""
        msg = "MolmoAct2Model.forward is not implemented yet."
        raise NotImplementedError(msg)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the training loss."""
        msg = "MolmoAct2Model.compute_loss is not implemented yet."
        raise NotImplementedError(msg)

    @torch.no_grad()
    def compute_val_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the validation loss."""
        msg = "MolmoAct2Model.compute_val_loss is not implemented yet."
        raise NotImplementedError(msg)

    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict an action chunk."""
        msg = "MolmoAct2Model.predict_action_chunk is not implemented yet."
        raise NotImplementedError(msg)

    @property
    def reward_delta_indices(self) -> list[int] | None:
        """Reward delta indices."""
        msg = "MolmoAct2Model.reward_delta_indices is not implemented yet."
        raise NotImplementedError(msg)

    @property
    def action_delta_indices(self) -> list[int] | None:
        """Action delta indices."""
        msg = "MolmoAct2Model.action_delta_indices is not implemented yet."
        raise NotImplementedError(msg)

    @property
    def observation_delta_indices(self) -> list[int] | None:
        """Observation delta indices."""
        msg = "MolmoAct2Model.observation_delta_indices is not implemented yet."
        raise NotImplementedError(msg)
