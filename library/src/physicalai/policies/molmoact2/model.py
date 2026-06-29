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

from physicalai.data.observation import FeatureType
from physicalai.policies.base import Model

from .backbones import MolmoAct2ForConditionalGeneration
from .config import MolmoAct2Config

SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"


def _strict_load_safetensors_weights(model: torch.nn.Module, checkpoint_location: str) -> None:
    """Load safetensors weights into a model, strictly verifying key correspondence.

    Supports both sharded (index JSON) and single-file checkpoints. Raises
    clearly if the checkpoint does not exist or does not match the model.

    Args:
        model: The PyTorch module to load weights into.
        checkpoint_location: Directory containing the safetensors checkpoint.

    Raises:
        FileNotFoundError: If neither a sharded index nor a single weights file
            is found at ``checkpoint_location``.
        RuntimeError: If the checkpoint keys do not exactly match the model keys.
    """
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

    raise FileNotFoundError(
        f"No safetensors checkpoint found at '{checkpoint_location}'. "
        f"Expected '{SAFE_WEIGHTS_NAME}' or '{SAFE_WEIGHTS_INDEX_NAME}'."
    )


class MolmoAct2Model(Model):
    """Wrapper for MolmoAct2ForConditionalGeneration using physicalai config.

    This model handles both training and inference modes:

    - Training: Computes supervised losses using the backbone.
    - Inference: Generates predicted action chunks.

    Weight loading is intentionally separated from construction. Call
    :meth:`load_pretrained_weights` explicitly after instantiation when a
    pretrained checkpoint is available.
    """

    def __init__(self, config: MolmoAct2Config) -> None:
        """Initialize the MolmoAct2 model wrapper.

        Constructs the backbone architecture from ``config`` but does **not**
        load any weights. Call :meth:`load_pretrained_weights` separately to
        load a pretrained checkpoint.

        Args:
            config: MolmoAct2Config instance with all model components defined.
        """
        super().__init__()
        self.config = config
        self.backbone = MolmoAct2ForConditionalGeneration(config)
        self.max_sequence_length = _text_max_positions(config)
        self.max_action_dim = int(config.max_action_dim)
        self.env_action_dim = _env_action_dim(config)

    def load_pretrained_weights(self, checkpoint_location: str) -> None:
        """Load pretrained safetensors weights from a checkpoint directory.

        Args:
            checkpoint_location: Path to a directory containing either
                ``model.safetensors`` or ``model.safetensors.index.json``.
        """
        _strict_load_safetensors_weights(self.backbone, checkpoint_location)

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

    def compute_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        """Compute the supervised training loss.

        Args:
            batch: Input batch with model inputs and action targets.

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
            In training mode, a tuple of (loss_tensor, metrics_dict).
            In inference mode, the predicted action tensor.
        """
        if self.training:
            return self.compute_loss(batch)
        return self.predict_action_chunk(batch)

    @staticmethod
    def _validate_local_pooling_indices(inputs: dict[str, Any]) -> None:
        if "pixel_values" in inputs and "image_token_pooling" in inputs:
            pixel_values = inputs["pixel_values"]
            image_token_pooling = inputs["image_token_pooling"]
            if torch.is_tensor(pixel_values) and pixel_values.ndim == 3 and torch.is_tensor(image_token_pooling):
                n_patches = int(pixel_values.shape[1])
                valid = image_token_pooling >= 0
                if torch.any(valid):
                    max_idx = int(image_token_pooling[valid].max().item())
                    if max_idx >= n_patches:
                        raise ValueError(
                            "image_token_pooling contains out-of-range indices for per-image local patch IDs: "
                            f"max_idx={max_idx}, n_patches={n_patches}."
                        )

        if "pixel_values_videos" in inputs and "video_token_pooling" in inputs:
            pixel_values_videos = inputs["pixel_values_videos"]
            video_token_pooling = inputs["video_token_pooling"]
            if (
                torch.is_tensor(pixel_values_videos)
                and pixel_values_videos.ndim == 3
                and torch.is_tensor(video_token_pooling)
            ):
                n_frame_patches_total = int(pixel_values_videos.shape[0] * pixel_values_videos.shape[1])
                valid = video_token_pooling >= 0
                if torch.any(valid):
                    max_idx = int(video_token_pooling[valid].max().item())
                    if max_idx >= n_frame_patches_total:
                        raise ValueError(
                            "video_token_pooling contains out-of-range indices for local frame patch IDs: "
                            f"max_idx={max_idx}, total_patches={n_frame_patches_total}."
                        )

    def _default_action_dim_is_pad(self, *, batch_size: int, device: torch.device) -> torch.Tensor:
        action_dim_is_pad = torch.ones((batch_size, self.max_action_dim), dtype=torch.bool, device=device)
        if self.env_action_dim > 0:
            action_dim_is_pad[:, : self.env_action_dim] = False
        return action_dim_is_pad

    def prepare_graph_inputs(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare tensor-only model inputs from processor output.

        This stage keeps graph-friendly tensor logic inside the model boundary:
        sequence length validation, local pooling-index validation, action mask
        defaults, and tensor device placement.
        """
        if "input_ids" not in batch or not torch.is_tensor(batch["input_ids"]):
            raise ValueError("MolmoAct2 model expects tensor input_ids from preprocessor output.")

        if int(batch["input_ids"].shape[1]) > self.max_sequence_length:
            raise ValueError(
                f"MolmoAct2 sequence length {int(batch['input_ids'].shape[1])} exceeds max_sequence_length={self.max_sequence_length}.",
            )

        self._validate_local_pooling_indices(batch)

        batch_size = int(batch["input_ids"].shape[0])
        action_dim_is_pad = batch.get("action_dim_is_pad")
        if not torch.is_tensor(action_dim_is_pad):
            action_dim_is_pad = self._default_action_dim_is_pad(batch_size=batch_size, device=batch["input_ids"].device)
        else:
            action_dim_is_pad = action_dim_is_pad.to(dtype=torch.bool)

        model_inputs: dict[str, Any] = {
            key: batch[key]
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
            )
            if key in batch and batch[key] is not None
        }
        model_inputs["action_dim_is_pad"] = action_dim_is_pad

        target_device = next(self.backbone.parameters()).device
        for key, value in list(model_inputs.items()):
            if torch.is_tensor(value):
                model_inputs[key] = value.to(device=target_device)
        return model_inputs

    def predict_action_chunk(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """Convert a processed batch into a predicted action chunk.

        Args:
            batch: Input batch with encoded observations and prompts.

        Returns:
            Dictionary with an ``"actions"`` key containing the predicted
            action tensor of shape ``(batch_size, action_horizon, action_dim)``.
        """
        model_inputs = self.prepare_graph_inputs(batch)

        with torch.no_grad():
            actions = self.backbone.model.generate_actions_from_inputs(**model_inputs)
        return {"actions": actions}


def _text_max_positions(config: Any, *, default: int = 4096) -> int:
    text_config = getattr(config, "text_config", None)
    if isinstance(text_config, dict):
        return int(text_config.get("max_position_embeddings", default))
    return int(getattr(text_config, "max_position_embeddings", default))


def _env_action_dim(config: MolmoAct2Config) -> int:
    action_feature = next((f for f in config.output_features if f.ftype == FeatureType.ACTION), None)
    if action_feature is None or action_feature.shape is None:
        return 0
    return int(action_feature.shape[0])
