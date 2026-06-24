# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 model implementation."""

# isort: skip_file
# ruff: noqa: D,I001

import json
import os
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor
from tqdm import tqdm

from physicalai.policies.base import Model

from .backbones import MolmoAct2ForConditionalGeneration
from .config import MolmoAct2Config

SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"


def _strict_load_safetensors_weights(model: torch.nn.Module, checkpoint_location: str) -> None:
    index_path = os.path.join(checkpoint_location, SAFE_WEIGHTS_INDEX_NAME)
    single_file_path = os.path.join(checkpoint_location, SAFE_WEIGHTS_NAME)
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as file_obj:
            index = json.load(file_obj)
        weight_map = index["weight_map"]
        loaded_keys = set(weight_map)
        model_keys = set(model.state_dict())
        missing_keys = sorted(model_keys - loaded_keys)
        unexpected_keys = sorted(loaded_keys - model_keys)
        if missing_keys or unexpected_keys:
            message = ["MolmoAct2 safetensors do not match the local model implementation."]
            if missing_keys:
                message.append(f"Missing keys: {missing_keys[:8]}")
            if unexpected_keys:
                message.append(f"Unexpected keys: {unexpected_keys[:8]}")
            raise RuntimeError(" ".join(message))
        shards = sorted(set(weight_map.values()))
        for shard_file in tqdm(shards, desc="Loading MolmoAct2 weights", unit="shard"):
            state_dict = load_safetensors_file(os.path.join(checkpoint_location, shard_file), device="cpu")
            model.load_state_dict(state_dict, strict=False)
            del state_dict
        return
    if os.path.isfile(single_file_path):
        print(f"Loading MolmoAct2 weights from {single_file_path} ...")
        state_dict = load_safetensors_file(single_file_path, device="cpu")
        model.load_state_dict(state_dict, strict=True)
        print("MolmoAct2 weights loaded.")
        return

    msg = (
        f"MolmoAct2 checkpoint at {checkpoint_location} must contain {SAFE_WEIGHTS_NAME} "
        f"or {SAFE_WEIGHTS_INDEX_NAME}."
    )
    raise FileNotFoundError(msg)


class MolmoAct2Model(Model):
    """Wrapper for MolmoAct2ForConditionalGeneration using physicalai config.

    This model handles both training and inference modes:
    - Training: Computes supervised losses using the backbone
    - Inference: Generates predicted action chunks
    """

    def __init__(self, config: MolmoAct2Config) -> None:
        """Initialize the MolmoAct2 model wrapper.

        Args:
            config: MolmoAct2Config instance with all model components defined.
        """
        super().__init__()
        self.config = config

        # Initialize the backbone model using the provided config
        # Use add_module so it's properly registered for _apply() device/dtype handling
        backbone = MolmoAct2ForConditionalGeneration(config)
        self.add_module("_backbone", backbone)
        self._maybe_load_pretrained_checkpoint()

    def _resolve_checkpoint_location(self) -> str | None:
        candidates = [
            self.config.processor_assets_path,
            self.config.tokenizer_name_or_path,
        ]
        for candidate in candidates:
            if candidate and os.path.isdir(candidate):
                return candidate
        return None

    def _maybe_load_pretrained_checkpoint(self) -> None:
        checkpoint_location = self._resolve_checkpoint_location()
        if checkpoint_location is None:
            return
        if not (
            os.path.isfile(os.path.join(checkpoint_location, SAFE_WEIGHTS_NAME))
            or os.path.isfile(os.path.join(checkpoint_location, SAFE_WEIGHTS_INDEX_NAME))
        ):
            return
        _strict_load_safetensors_weights(self._backbone, checkpoint_location)

    # ============ physicalai.policies.base.Model interface ============

    @property
    def action_delta_indices(self) -> list | None:
        """Return action delta indices if this wrapper defines them."""
        return None

    @property
    def observation_delta_indices(self) -> list | None:
        """Return observation delta indices if this wrapper defines them."""
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        """Return reward delta indices if this wrapper defines them."""
        return None

    # ============ Training and Inference ============

    def compute_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        """Compute the supervised training loss.

        Args:
            batch: Input batch with model inputs and action targets.

        Returns:
            Tuple of (loss_tensor, metrics_dict).

        Raises:
            NotImplementedError: Training not yet fully implemented.
        """
        msg = "Training loss computation not yet implemented."
        raise NotImplementedError(msg)

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]] | Tensor:
        """Run forward pass in training or inference mode.

        Args:
            batch: Input batch dictionary.

        Returns:
            - In training mode: Tuple of (loss, metrics_dict)
            - In inference mode: Predicted action tensor
        """
        if self.training:
            return self.compute_loss(batch)
        return self.predict_action_chunk(batch)

    def predict_action_chunk(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """Convert a processed batch into a predicted action chunk.

        Args:
            batch: Input batch with encoded observations and prompts.

        Returns:
            Dictionary with "actions" key containing the predicted action tensor 
            of shape (batch_size, action_horizon, action_dim).
        """
        model_inputs: dict[str, Any] = {}
        for key in (
            "input_ids",
            "pixel_values",
            "image_token_pooling",
            "image_grids",
            "image_num_crops",
            "pixel_values_videos",
            "video_token_pooling",
            "video_grids",
            "attention_mask",
            "token_type_ids",
            "action_dim_is_pad",
        ):
            if key in batch and batch[key] is not None:
                model_inputs[key] = batch[key]

        # Pass through backbone action-generation path from the inner model.
        with torch.no_grad():
            actions = self._backbone.model.generate_actions_from_inputs(**model_inputs)
        return {"actions": actions}
