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
from .image import MolmoAct2ImageProcessor
from .video import MolmoAct2VideoProcessor
from ..config import MolmoAct2Config

SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"


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


class MolmoAct2TorchFrontend(torch.nn.Module):
    """Torch-only frontend that builds model-ready tensors from token and vision inputs."""

    def __init__(self, config: MolmoAct2Config) -> None:
        super().__init__()
        self.config = config
        self.max_sequence_length = _text_max_positions(config)
        self.max_action_dim = int(config.max_action_dim)
        self.env_action_dim = _env_action_dim(config)

        processor_config = config.processor_config
        if processor_config is None:
            raise ValueError("MolmoAct2Config.processor_config must be set for torch frontend creation.")

        image_cfg = processor_config.image_processor
        video_cfg = processor_config.video_processor

        self.image_processor = MolmoAct2ImageProcessor(
            size=image_cfg.size,
            image_mean=image_cfg.image_mean,
            image_std=image_cfg.image_std,
            do_convert_rgb=image_cfg.do_convert_rgb,
            max_crops=image_cfg.max_crops,
            overlap_margins=image_cfg.overlap_margins,
            crop_mode=image_cfg.crop_mode,
            patch_size=image_cfg.patch_size,
            pooling_size=image_cfg.pooling_size,
        )
        self.video_processor = MolmoAct2VideoProcessor(
            size=video_cfg.size,
            image_mean=video_cfg.image_mean,
            image_std=video_cfg.image_std,
            do_convert_rgb=video_cfg.do_convert_rgb,
            patch_size=video_cfg.patch_size,
            pooling_size=video_cfg.pooling_size,
            do_sample_frames=video_cfg.do_sample_frames,
            frame_sample_mode=video_cfg.frame_sample_mode,
            max_fps=int(video_cfg.max_fps),
            sampling_fps=video_cfg.sampling_fps,
        )
        self.image_use_col_tokens = bool(processor_config.image_use_col_tokens)
        self.use_single_crop_col_tokens = processor_config.use_single_crop_col_tokens
        self.use_single_crop_start_token = bool(processor_config.use_single_crop_start_token)

    def _default_action_dim_is_pad(self, *, batch_size: int, device: torch.device) -> torch.Tensor:
        action_dim_is_pad = torch.ones((batch_size, self.max_action_dim), dtype=torch.bool, device=device)
        if self.env_action_dim > 0:
            action_dim_is_pad[:, : self.env_action_dim] = False
        return action_dim_is_pad

    @staticmethod
    def _ensure_tensor_or_none(name: str, value: torch.Tensor | None) -> None:
        if value is not None and not torch.is_tensor(value):
            raise TypeError(f"MolmoAct2 torch frontend expected tensor for '{name}', got {type(value)}")

    @staticmethod
    def _as_int_or_none(value: int | torch.Tensor | None, name: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if torch.is_tensor(value):
            if value.ndim == 0:
                return int(value.item())
            if value.ndim == 1 and int(value.numel()) == 1:
                return int(value.reshape(()).item())
        raise TypeError(f"MolmoAct2 torch frontend expected int-like value for '{name}', got {type(value)}")

    def _image_token_ids_for_grid(self, grid: torch.Tensor) -> list[int]:
        if grid.ndim != 1 or int(grid.numel()) != 4:
            raise ValueError(f"Expected image grid shape (4,), got {tuple(grid.shape)}")

        if self.config.image_patch_id is None or self.config.image_start_token_id is None or self.config.image_end_token_id is None:
            raise ValueError("MolmoAct2 config must define image_patch_id/image_start_token_id/image_end_token_id.")

        image_patch_id = int(self.config.image_patch_id)
        image_start_token_id = int(self.config.image_start_token_id)
        image_end_token_id = int(self.config.image_end_token_id)
        image_col_id = None if self.config.image_col_id is None else int(self.config.image_col_id)
        low_res_start_id = (
            int(self.config.low_res_image_start_token_id)
            if self.config.low_res_image_start_token_id is not None
            else image_start_token_id
        )

        resized_h, resized_w, height, width = [int(x) for x in grid.tolist()]

        def make_rows(num_rows: int, num_cols: int, *, use_col: bool) -> list[int]:
            row = [image_patch_id] * int(num_cols)
            if use_col and image_col_id is not None:
                row = row + [image_col_id]
            return row * int(num_rows)

        use_single_crop_col_tokens = (
            self.image_use_col_tokens if self.use_single_crop_col_tokens is None else bool(self.use_single_crop_col_tokens)
        )

        if height == 0 or width == 0:
            return [image_start_token_id] + make_rows(resized_h, resized_w, use_col=use_single_crop_col_tokens) + [
                image_end_token_id,
            ]

        high_res = [image_start_token_id] + make_rows(height, width, use_col=self.image_use_col_tokens) + [
            image_end_token_id,
        ]
        low_start = low_res_start_id if self.use_single_crop_start_token else image_start_token_id
        low_res = [low_start] + make_rows(resized_h, resized_w, use_col=use_single_crop_col_tokens) + [image_end_token_id]
        return low_res + high_res

    def _build_token_type_ids(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor | None:
        image_token_ids = [
            self.config.image_patch_id,
            self.config.image_col_id,
            self.config.image_start_token_id,
            self.config.low_res_image_start_token_id,
            self.config.frame_start_token_id,
            self.config.image_end_token_id,
            self.config.frame_end_token_id,
            self.config.image_low_res_id,
        ]
        image_token_ids = [int(x) for x in image_token_ids if x is not None]
        if not image_token_ids:
            return None

        token_set = torch.as_tensor(image_token_ids, device=input_ids.device, dtype=input_ids.dtype)
        token_type_ids = (input_ids.unsqueeze(-1) == token_set.view(1, 1, -1)).any(dim=-1).to(dtype=torch.long)
        token_type_ids = token_type_ids * attention_mask.to(dtype=torch.long)
        return token_type_ids

    def _expand_image_placeholders(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_grids: torch.Tensor,
        image_placeholder_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if int(image_grids.shape[0]) == 0:
            return input_ids, attention_mask, self._build_token_type_ids(input_ids, attention_mask)

        pad_values = input_ids[attention_mask == 0]
        pad_token_id = int(pad_values[0].item()) if int(pad_values.numel()) > 0 else 0

        expanded_per_example: list[list[int]] = []
        grid_idx = 0
        batch_size = int(input_ids.shape[0])

        for b_idx in range(batch_size):
            valid_ids = input_ids[b_idx][attention_mask[b_idx].to(dtype=torch.bool)]
            expanded_ids: list[int] = []
            for token in valid_ids.tolist():
                token_int = int(token)
                if token_int == image_placeholder_token_id:
                    if grid_idx >= int(image_grids.shape[0]):
                        raise ValueError(
                            "Not enough image grids to expand all <|image|> placeholders in input_ids.",
                        )
                    expanded_ids.extend(self._image_token_ids_for_grid(image_grids[grid_idx]))
                    grid_idx += 1
                else:
                    expanded_ids.append(token_int)
            expanded_per_example.append(expanded_ids)

        if grid_idx != int(image_grids.shape[0]):
            raise ValueError(
                "Unconsumed image grids after placeholder expansion. "
                f"consumed={grid_idx}, total={int(image_grids.shape[0])}",
            )

        max_len = max((len(tokens) for tokens in expanded_per_example), default=0)
        max_len = max(max_len, 1)
        out_ids = torch.full(
            (batch_size, max_len),
            fill_value=pad_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        out_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)

        for b_idx, tokens in enumerate(expanded_per_example):
            if not tokens:
                continue
            token_tensor = torch.as_tensor(tokens, dtype=input_ids.dtype, device=input_ids.device)
            out_ids[b_idx, : token_tensor.numel()] = token_tensor
            out_mask[b_idx, : token_tensor.numel()] = 1

        token_type_ids = self._build_token_type_ids(out_ids, out_mask)
        return out_ids, out_mask, token_type_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        image_placeholder_token_id: int | torch.Tensor | None = None,
        images_bchw: torch.Tensor | None = None,
        videos_btchw: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_token_pooling: torch.Tensor | None = None,
        image_grids: torch.Tensor | None = None,
        image_num_crops: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_token_pooling: torch.Tensor | None = None,
        video_grids: torch.Tensor | None = None,
        action_dim_is_pad: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not torch.is_tensor(input_ids):
            raise ValueError("MolmoAct2 torch frontend expects tensor input_ids.")
        self._ensure_tensor_or_none("attention_mask", attention_mask)
        self._ensure_tensor_or_none("token_type_ids", token_type_ids)
        self._ensure_tensor_or_none("images_bchw", images_bchw)
        self._ensure_tensor_or_none("videos_btchw", videos_btchw)
        self._ensure_tensor_or_none("pixel_values", pixel_values)
        self._ensure_tensor_or_none("image_token_pooling", image_token_pooling)
        self._ensure_tensor_or_none("image_grids", image_grids)
        self._ensure_tensor_or_none("image_num_crops", image_num_crops)
        self._ensure_tensor_or_none("pixel_values_videos", pixel_values_videos)
        self._ensure_tensor_or_none("video_token_pooling", video_token_pooling)
        self._ensure_tensor_or_none("video_grids", video_grids)
        self._ensure_tensor_or_none("action_dim_is_pad", action_dim_is_pad)
        image_placeholder_token_id_int = self._as_int_or_none(image_placeholder_token_id, "image_placeholder_token_id")

        if pixel_values is None and images_bchw is not None:
            image_out = self.image_processor(images_bchw, return_tensors="pt")
            pixel_values = image_out["pixel_values"]
            image_token_pooling = image_out["image_token_pooling"]
            image_grids = image_out["image_grids"]
            image_num_crops = image_out["image_num_crops"]

        if pixel_values_videos is None and videos_btchw is not None:
            video_out = self.video_processor(videos_btchw, return_tensors="pt", return_metadata=False)
            pixel_values_videos = video_out["pixel_values_videos"]
            video_token_pooling = video_out["video_token_pooling"]
            video_grids = video_out["video_grids"]

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if image_grids is not None and image_placeholder_token_id_int is not None:
            input_ids, attention_mask, rebuilt_token_type_ids = self._expand_image_placeholders(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grids=image_grids,
                image_placeholder_token_id=image_placeholder_token_id_int,
            )
            token_type_ids = rebuilt_token_type_ids

        if int(input_ids.shape[1]) > self.max_sequence_length:
            raise ValueError(
                f"MolmoAct2 sequence length {int(input_ids.shape[1])} exceeds max_sequence_length={self.max_sequence_length}.",
            )

        if action_dim_is_pad is None:
            action_dim_is_pad = self._default_action_dim_is_pad(batch_size=int(input_ids.shape[0]), device=input_ids.device)
        else:
            action_dim_is_pad = action_dim_is_pad.to(dtype=torch.bool)

        model_inputs: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "action_dim_is_pad": action_dim_is_pad,
        }
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        if image_token_pooling is not None:
            model_inputs["image_token_pooling"] = image_token_pooling
        if image_grids is not None:
            model_inputs["image_grids"] = image_grids
        if image_num_crops is not None:
            model_inputs["image_num_crops"] = image_num_crops
        if pixel_values_videos is not None:
            model_inputs["pixel_values_videos"] = pixel_values_videos
        if video_token_pooling is not None:
            model_inputs["video_token_pooling"] = video_token_pooling
        if video_grids is not None:
            model_inputs["video_grids"] = video_grids

        _validate_local_pooling_indices(model_inputs)
        return model_inputs

    def from_batch(self, batch: dict[str, Any], *, target_device: torch.device | None = None) -> dict[str, torch.Tensor]:
        model_inputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            token_type_ids=batch.get("token_type_ids"),
            image_placeholder_token_id=batch.get("image_placeholder_token_id"),
            images_bchw=batch.get("images_bchw"),
            videos_btchw=batch.get("videos_btchw"),
            pixel_values=batch.get("pixel_values"),
            image_token_pooling=batch.get("image_token_pooling"),
            image_grids=batch.get("image_grids"),
            image_num_crops=batch.get("image_num_crops"),
            pixel_values_videos=batch.get("pixel_values_videos"),
            video_token_pooling=batch.get("video_token_pooling"),
            video_grids=batch.get("video_grids"),
            action_dim_is_pad=batch.get("action_dim_is_pad"),
        )
        if target_device is not None:
            for key, value in list(model_inputs.items()):
                if torch.is_tensor(value):
                    model_inputs[key] = value.to(device=target_device)
        return model_inputs


class MolmoAct2TorchInference(torch.nn.Module):
    """Single torch module boundary: frontend + model action generation."""

    def __init__(self, frontend: MolmoAct2TorchFrontend, backbone: MolmoAct2ForConditionalGeneration) -> None:
        super().__init__()
        self.frontend = frontend
        self.backbone = backbone

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        image_placeholder_token_id: int | torch.Tensor | None = None,
        images_bchw: torch.Tensor | None = None,
        videos_btchw: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_token_pooling: torch.Tensor | None = None,
        image_grids: torch.Tensor | None = None,
        image_num_crops: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_token_pooling: torch.Tensor | None = None,
        video_grids: torch.Tensor | None = None,
        action_dim_is_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model_inputs = self.frontend(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            image_placeholder_token_id=image_placeholder_token_id,
            images_bchw=images_bchw,
            videos_btchw=videos_btchw,
            pixel_values=pixel_values,
            image_token_pooling=image_token_pooling,
            image_grids=image_grids,
            image_num_crops=image_num_crops,
            pixel_values_videos=pixel_values_videos,
            video_token_pooling=video_token_pooling,
            video_grids=video_grids,
            action_dim_is_pad=action_dim_is_pad,
        )
        return self.backbone.model.generate_actions_from_inputs(**model_inputs)

    def from_batch(self, batch: dict[str, Any], *, target_device: torch.device | None = None) -> torch.Tensor:
        model_inputs = self.frontend.from_batch(batch, target_device=target_device)
        return self.backbone.model.generate_actions_from_inputs(**model_inputs)


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
        self.frontend = MolmoAct2TorchFrontend(config)
        self.torch_inference = MolmoAct2TorchInference(self.frontend, self.backbone)

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

    def prepare_graph_inputs(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare tensor-only model inputs from processor output.

        This stage keeps graph-friendly tensor logic inside the model boundary:
        sequence length validation, local pooling-index validation, action mask
        defaults, and tensor device placement.
        """
        target_device = next(self.backbone.parameters()).device
        return self.frontend.from_batch(batch, target_device=target_device)

    @property
    def exported_torch_module(self) -> MolmoAct2TorchInference:
        """Single torch inference boundary for export and runtime parity."""
        return self.torch_inference

    def predict_action_chunk(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """Convert a processed batch into a predicted action chunk.

        Args:
            batch: Input batch with encoded observations and prompts.

        Returns:
            Dictionary with an ``"actions"`` key containing the predicted
            action tensor of shape ``(batch_size, action_horizon, action_dim)``.
        """
        with torch.no_grad():
            actions = self.torch_inference.from_batch(
                batch,
                target_device=next(self.backbone.parameters()).device,
            )
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
