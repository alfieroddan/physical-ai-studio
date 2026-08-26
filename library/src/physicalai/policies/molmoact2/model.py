# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Minimal MolmoAct2 model surface used by policy initialization wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from .config import MolmoAct2Config


class MolmoAct2Model:
    """Placeholder model class with the constructor hooks used by the policy."""

    def __init__(self, config: MolmoAct2Config) -> None:
        """Store config required for model construction."""
        self.config = config

    @classmethod
    def from_config(cls, config: MolmoAct2Config) -> MolmoAct2Model:
        """Build a model instance from a config object.

        Returns:
            MolmoAct2Model: Constructed model instance.
        """
        return cls(config)

    def load_weights(self, weights_path: str | Path) -> None:
        """Load model weights from the specified path.

        Args:
            weights_path: Path to the file containing the model weights.
        """
        del weights_path

    def use_lora(self) -> None:
        """Enable LoRA (Low-Rank Adaptation) for the model."""

    def enable_lora(self) -> None:
        """Compatibility alias matching the policy hook name."""
        self.use_lora()

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for the model."""

    def gradient_checkpointing_enable(self) -> None:
        """Compatibility alias matching the policy hook name."""
        self.enable_gradient_checkpointing()

    def freeze_vlm(self) -> None:
        """Freeze the Vision-Language Model (VLM) parameters to prevent them from being updated during training."""
