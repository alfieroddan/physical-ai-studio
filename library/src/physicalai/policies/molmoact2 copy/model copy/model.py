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
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor
from tqdm import tqdm
from transformers import Qwen2Tokenizer

from physicalai.data.constants import IMAGE_MASKS, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.data.observation import ACTION, FeatureType
from physicalai.policies.base import Model
from physicalai.policies.molmoact2.action_tokenizer import UniversalActionProcessor

from .backbones import MolmoAct2ForConditionalGeneration
from .image import MolmoAct2ImageProcessor
from .video import MolmoAct2VideoProcessor
from ..config import MolmoAct2Config

SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
IMAGE_PROMPT = "<|image|>"


def _sample_beta_timesteps(
    *,
    batch_size: int,
    device: torch.device,
    cutoff: float,
    time_offset: float,
    time_scale: float,
    alpha: float,
    beta: float,
) -> Tensor:
    if cutoff < time_offset:
        raise ValueError(f"flow-matching cutoff must be >= time_offset, got {cutoff} < {time_offset}")
    if time_scale <= 0:
        raise ValueError(f"flow-matching time_scale must be > 0, got {time_scale}")
    upper = min(cutoff, time_offset + time_scale)
    dist = torch.distributions.Beta(
        torch.tensor(alpha, device=device),
        torch.tensor(beta, device=device),
    )
    samples = dist.sample((batch_size,))
    scale = upper - time_offset
    if scale == 0:
        return torch.full((batch_size,), time_offset, device=device, dtype=samples.dtype)
    return time_offset + scale * samples


def _masked_loss_mean(
    loss: Tensor,
    *,
    action_horizon_is_pad: Tensor | None,
    action_dim_is_pad: Tensor | None,
) -> Tensor:
    mask = torch.ones_like(loss, dtype=torch.bool)

    if action_horizon_is_pad is not None:
        horizon_mask = ~action_horizon_is_pad.to(device=loss.device, dtype=torch.bool)
        horizon_view = horizon_mask.view(
            horizon_mask.shape[0],
            *([1] * max(loss.ndim - 3, 0)),
            horizon_mask.shape[1],
            1,
        )
        mask = mask & horizon_view

    if action_dim_is_pad is not None:
        dim_mask = ~action_dim_is_pad.to(device=loss.device, dtype=torch.bool)
        dim_view = dim_mask.view(
            dim_mask.shape[0],
            *([1] * max(loss.ndim - 2, 0)),
            dim_mask.shape[1],
        )
        mask = mask & dim_view

    valid = mask.to(dtype=loss.dtype)
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


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
        video_cfg = processor_config.video_processor
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
        self._image_placeholder_token_id: int | None = (
            int(config.image_placeholder_token_id) if config.image_placeholder_token_id is not None else None
        )

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

    def _resolved_image_placeholder_token_id(self) -> int | None:
        if self._image_placeholder_token_id is not None:
            return self._image_placeholder_token_id

        tokenizer_name_or_path = str(getattr(self.config, "tokenizer_name_or_path", "") or "").strip()
        if not tokenizer_name_or_path:
            return None

        tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_name_or_path, local_files_only=True)
        token_id = tokenizer.convert_tokens_to_ids(IMAGE_PROMPT)
        self._image_placeholder_token_id = int(token_id) if isinstance(token_id, int) else None
        return self._image_placeholder_token_id

    @staticmethod
    def _flatten_images(
        images: torch.Tensor | None,
        image_masks: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if images is None:
            return None
        if images.ndim == 4:
            images = images.unsqueeze(0)
        if images.ndim != 5:
            raise ValueError(f"MolmoAct2 expected images with shape [N, B, C, H, W], got {tuple(images.shape)}.")
        if int(images.shape[2]) != 3:
            raise ValueError(f"MolmoAct2 expected image channels in axis 2, got shape {tuple(images.shape)}.")

        num_images, batch_size = int(images.shape[0]), int(images.shape[1])
        if image_masks is None:
            image_masks = torch.ones((num_images, batch_size), dtype=torch.bool, device=images.device)
        elif image_masks.ndim == 1:
            image_masks = image_masks.unsqueeze(0)
        elif image_masks.ndim != 2:
            raise ValueError(f"MolmoAct2 expected image_masks with shape [N, B], got {tuple(image_masks.shape)}.")

        if tuple(image_masks.shape) != (num_images, batch_size):
            raise ValueError(
                f"MolmoAct2 image mask shape {tuple(image_masks.shape)} does not match images {(num_images, batch_size)}.",
            )

        flat_images: list[torch.Tensor] = []
        valid_masks = image_masks.to(device=images.device, dtype=torch.bool)
        for batch_idx in range(batch_size):
            for image_idx in range(num_images):
                if bool(valid_masks[image_idx, batch_idx].item()):
                    flat_images.append(images[image_idx, batch_idx])

        if not flat_images:
            return None
        return torch.stack(flat_images, dim=0)

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
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if int(image_grids.shape[0]) == 0:
            return (
                input_ids,
                attention_mask,
                self._build_token_type_ids(input_ids, attention_mask),
                labels,
            )

        pad_values = input_ids[attention_mask == 0]
        pad_token_id = int(pad_values[0].item()) if int(pad_values.numel()) > 0 else 0

        expanded_per_example: list[list[int]] = []
        expanded_labels_per_example: list[list[int]] | None = [] if labels is not None else None
        grid_idx = 0
        batch_size = int(input_ids.shape[0])

        for b_idx in range(batch_size):
            valid_mask_bool = attention_mask[b_idx].to(dtype=torch.bool)
            valid_ids = input_ids[b_idx][valid_mask_bool]
            valid_labels = labels[b_idx][valid_mask_bool] if labels is not None else None
            expanded_ids: list[int] = []
            expanded_labels: list[int] | None = [] if labels is not None else None
            for pos, token in enumerate(valid_ids.tolist()):
                token_int = int(token)
                if token_int == image_placeholder_token_id:
                    if grid_idx >= int(image_grids.shape[0]):
                        raise ValueError(
                            "Not enough image grids to expand all <|image|> placeholders in input_ids.",
                        )
                    image_token_ids = self._image_token_ids_for_grid(image_grids[grid_idx])
                    expanded_ids.extend(image_token_ids)
                    if expanded_labels is not None:
                        # Image patch tokens are never supervised targets.
                        expanded_labels.extend([-100] * len(image_token_ids))
                    grid_idx += 1
                else:
                    expanded_ids.append(token_int)
                    if expanded_labels is not None and valid_labels is not None:
                        expanded_labels.append(int(valid_labels[pos].item()))
            expanded_per_example.append(expanded_ids)
            if expanded_labels_per_example is not None:
                expanded_labels_per_example.append(expanded_labels)

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
        out_labels: torch.Tensor | None = None
        if labels is not None:
            out_labels = torch.full(
                (batch_size, max_len),
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device,
            )

        for b_idx, tokens in enumerate(expanded_per_example):
            if not tokens:
                continue
            token_tensor = torch.as_tensor(tokens, dtype=input_ids.dtype, device=input_ids.device)
            out_ids[b_idx, : token_tensor.numel()] = token_tensor
            out_mask[b_idx, : token_tensor.numel()] = 1
            if out_labels is not None and expanded_labels_per_example is not None:
                label_tensor = torch.as_tensor(
                    expanded_labels_per_example[b_idx],
                    dtype=labels.dtype,
                    device=labels.device,
                )
                out_labels[b_idx, : label_tensor.numel()] = label_tensor

        token_type_ids = self._build_token_type_ids(out_ids, out_mask)
        return out_ids, out_mask, token_type_ids, out_labels

    def forward(
        self,
        tokenized_prompt: torch.Tensor,
        tokenized_prompt_mask: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        image_masks: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        videos_btchw: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_token_pooling: torch.Tensor | None = None,
        image_grids: torch.Tensor | None = None,
        image_num_crops: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_token_pooling: torch.Tensor | None = None,
        video_grids: torch.Tensor | None = None,
        action_dim_is_pad: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        input_ids = tokenized_prompt
        attention_mask = tokenized_prompt_mask
        if not torch.is_tensor(input_ids):
            raise ValueError("MolmoAct2 torch frontend expects tensor tokenized_prompt.")
        self._ensure_tensor_or_none("attention_mask", attention_mask)
        self._ensure_tensor_or_none("images", images)
        self._ensure_tensor_or_none("image_masks", image_masks)
        self._ensure_tensor_or_none("state", state)
        self._ensure_tensor_or_none("token_type_ids", token_type_ids)
        self._ensure_tensor_or_none("videos_btchw", videos_btchw)
        self._ensure_tensor_or_none("pixel_values", pixel_values)
        self._ensure_tensor_or_none("image_token_pooling", image_token_pooling)
        self._ensure_tensor_or_none("image_grids", image_grids)
        self._ensure_tensor_or_none("image_num_crops", image_num_crops)
        self._ensure_tensor_or_none("pixel_values_videos", pixel_values_videos)
        self._ensure_tensor_or_none("video_token_pooling", video_token_pooling)
        self._ensure_tensor_or_none("video_grids", video_grids)
        self._ensure_tensor_or_none("action_dim_is_pad", action_dim_is_pad)
        self._ensure_tensor_or_none("labels", labels)

        if pixel_values is None:
            flat_images = self._flatten_images(images, image_masks)
            if flat_images is not None:
                image_out = self.image_processor(flat_images, return_tensors="pt")
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

        image_placeholder_token_id_int = self._resolved_image_placeholder_token_id()
        if image_grids is not None and image_placeholder_token_id_int is not None:
            input_ids, attention_mask, rebuilt_token_type_ids, labels = self._expand_image_placeholders(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grids=image_grids,
                image_placeholder_token_id=image_placeholder_token_id_int,
                labels=labels,
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
        if labels is not None:
            model_inputs["labels"] = labels

        _validate_local_pooling_indices(model_inputs)
        return model_inputs

    def from_batch(self, batch: dict[str, Any], *, target_device: torch.device | None = None) -> dict[str, torch.Tensor]:
        model_inputs = self.forward(
            tokenized_prompt=batch[TOKENIZED_PROMPT],
            tokenized_prompt_mask=batch.get(TOKENIZED_PROMPT_MASK),
            images=batch.get("images"),
            image_masks=batch.get(IMAGE_MASKS),
            state=batch.get("state"),
            token_type_ids=batch.get("token_type_ids"),
            videos_btchw=batch.get("videos_btchw"),
            pixel_values=batch.get("pixel_values"),
            image_token_pooling=batch.get("image_token_pooling"),
            image_grids=batch.get("image_grids"),
            image_num_crops=batch.get("image_num_crops"),
            pixel_values_videos=batch.get("pixel_values_videos"),
            video_token_pooling=batch.get("video_token_pooling"),
            video_grids=batch.get("video_grids"),
            action_dim_is_pad=batch.get("action_dim_is_pad"),
            labels=batch.get("labels"),
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
        tokenized_prompt: torch.Tensor,
        tokenized_prompt_mask: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        image_masks: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
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
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            images=images,
            image_masks=image_masks,
            state=state,
            token_type_ids=token_type_ids,
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
        model_inputs.pop("labels", None)
        return self.backbone.model.generate_actions_from_inputs(**model_inputs)

    def from_batch(self, batch: dict[str, Any], *, target_device: torch.device | None = None) -> torch.Tensor:
        model_inputs = self.frontend.from_batch(batch, target_device=target_device)
        model_inputs.pop("labels", None)
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
        self._action_tokenizer: UniversalActionProcessor | None = None
        self._optimized_inference_enabled = False

    def enable_optimized_inference(self, *, enabled: bool = True) -> None:
        """Enable torch.compile optimization for inference-only entry points."""
        if not enabled or self._optimized_inference_enabled:
            return

        torch.set_float32_matmul_precision("high")
        compile_mode = "default"
        # TODO(export): Keep compile targets on tensor-heavy inference calls
        # because wrapper-level dict plumbing frequently causes graph breaks.
        self._inner_model.generate_actions_from_inputs = torch.compile(  # type: ignore[method-assign]
            self._inner_model.generate_actions_from_inputs,
            mode=compile_mode,
        )
        self.torch_inference.forward = torch.compile(self.torch_inference.forward, mode=compile_mode)  # type: ignore[method-assign]
        self._optimized_inference_enabled = True

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

    @property
    def _inner_model(self):
        return self.backbone.model

    @property
    def action_tokenizer(self) -> UniversalActionProcessor:
        if self._action_tokenizer is not None:
            return self._action_tokenizer
        tokenizer_path = str(getattr(self.config, "discrete_action_tokenizer", "")).strip()
        if not tokenizer_path:
            raise ValueError("config.discrete_action_tokenizer is required for discrete MolmoAct2 training.")
        self._action_tokenizer = UniversalActionProcessor.from_pretrained_local(tokenizer_path)
        return self._action_tokenizer

    def _action_mode(self) -> str:
        return str(getattr(self.config, "action_mode", "both"))

    def _resolved_action_dim(self, batch: dict[str, Any], gt_actions: Tensor | None = None) -> int:
        action_dim_is_pad = batch.get("action_dim_is_pad")
        if action_dim_is_pad is not None:
            valid_counts = (~action_dim_is_pad.to(dtype=torch.bool)).sum(dim=-1)
            if bool((valid_counts == valid_counts[0]).all()) and int(valid_counts[0]) > 0:
                return int(valid_counts[0])
        config_dim = _env_action_dim(self.config)
        if config_dim > 0:
            return int(config_dim)
        if gt_actions is not None:
            return int(gt_actions.shape[-1])
        return int(self.config.max_action_dim)

    def _discrete_generation_max_steps(self, action_horizon: int) -> int:
        return max(1, int(action_horizon) * 16)

    def _generate_discrete_actions_from_inputs(
        self,
        *,
        model_inputs: dict[str, Tensor],
        action_dim: int,
        action_horizon: int,
    ) -> Tensor:
        backbone_inputs = {
            key: value for key, value in model_inputs.items() if key not in ("action_dim_is_pad", "labels")
        }
        prefill_output = self.backbone(
            **backbone_inputs,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        generated_token_ids = self.backbone._continue_discrete_generation_from_output(
            prefill_output,
            past_key_values=prefill_output.past_key_values,
            attention_mask=backbone_inputs.get("attention_mask"),
            end_token_id=self.backbone._require_eos_token_id(),
            max_steps=self._discrete_generation_max_steps(action_horizon),
        )
        return self.backbone._decode_discrete_action_chunk(
            generated_token_ids,
            action_tokenizer=self.action_tokenizer,
            action_dim=action_dim,
            action_horizon=action_horizon,
        )

    def _compute_discrete_loss(self, model_inputs: dict[str, Tensor], labels: Tensor) -> Tensor:
        backbone_inputs = {
            key: value for key, value in model_inputs.items() if key not in ("action_dim_is_pad", "labels")
        }
        outputs = self.backbone(
            **backbone_inputs,
            labels=labels,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
        )
        if outputs.loss is None:
            raise RuntimeError("MolmoAct2 backbone did not return a discrete training loss.")
        return outputs.loss

    def _encoder_attention_mask_for_action_expert(
        self,
        *,
        input_ids: Tensor | None,
        attention_mask: Tensor | None,
    ) -> Tensor | None:
        get_encoder_attention_mask = getattr(self._inner_model, "_get_encoder_attention_mask", None)
        if callable(get_encoder_attention_mask):
            return get_encoder_attention_mask(input_ids, attention_mask)
        if attention_mask is not None:
            return attention_mask.to(dtype=torch.bool)
        if input_ids is not None:
            return input_ids != -1
        return None

    def _prepare_flow_matching_tensors(
        self,
        *,
        actions: Tensor,
        action_dim_is_pad: Tensor | None,
        timesteps: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        action_expert = self._inner_model._require_action_expert()
        action_dtype = next(action_expert.parameters()).dtype
        actions = actions.to(dtype=action_dtype)
        batch_size = int(actions.shape[0])
        device = actions.device

        if timesteps is None:
            timesteps = _sample_beta_timesteps(
                batch_size=batch_size,
                device=device,
                cutoff=self.config.flow_matching_cutoff,
                time_offset=self.config.flow_matching_time_offset,
                time_scale=self.config.flow_matching_time_scale,
                alpha=self.config.flow_matching_beta_alpha,
                beta=self.config.flow_matching_beta_beta,
            ).to(dtype=action_dtype)
        else:
            timesteps = timesteps.to(device=device, dtype=action_dtype)
            if tuple(timesteps.shape) != (batch_size,):
                raise ValueError(f"flow timesteps must have shape {(batch_size,)}, got {tuple(timesteps.shape)}.")

        if self.config.mask_action_dim_padding:
            actions = self._inner_model._mask_action_dim_tensor(
                actions,
                action_dim_is_pad=action_dim_is_pad,
                enabled=True,
            )

        expected_noise_shape = tuple(actions.shape)
        if noise is None:
            noise = torch.randn(*expected_noise_shape, device=device, dtype=actions.dtype)
        else:
            noise = noise.to(device=device, dtype=actions.dtype)
            if tuple(noise.shape) != expected_noise_shape:
                raise ValueError(
                    f"flow noise must have shape {expected_noise_shape}, got {tuple(noise.shape)}.",
                )

        if self.config.mask_action_dim_padding:
            noise = self._inner_model._mask_action_dim_tensor(
                noise,
                action_dim_is_pad=action_dim_is_pad,
                enabled=True,
            )

        t_broadcast = timesteps.view(batch_size, 1, 1)
        xt = (1.0 - t_broadcast) * noise + t_broadcast * actions
        target_velocity = actions - noise
        return actions, timesteps, xt, target_velocity

    def _prepare_joint_training_backbone_inputs(
        self,
        model_inputs: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        backbone_model = self._inner_model
        input_ids = model_inputs.get("input_ids")
        if input_ids is None:
            raise ValueError("MolmoAct2 training requires input_ids.")

        images, token_pooling = backbone_model.merge_visual_inputs(
            input_ids=input_ids,
            pixel_values=model_inputs.get("pixel_values"),
            image_token_pooling=model_inputs.get("image_token_pooling"),
            image_grids=model_inputs.get("image_grids"),
            image_num_crops=model_inputs.get("image_num_crops"),
            pixel_values_videos=model_inputs.get("pixel_values_videos"),
            video_token_pooling=model_inputs.get("video_token_pooling"),
            video_grids=model_inputs.get("video_grids"),
        )
        inputs_embeds, _image_features = backbone_model.build_input_embeddings(input_ids, images, token_pooling)

        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = model_inputs.get("position_ids")
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask_mapping = backbone_model._build_native_attention_bias(
            inputs_embeds=inputs_embeds,
            attention_mask=model_inputs.get("attention_mask"),
            token_type_ids=model_inputs.get("token_type_ids"),
            past_key_values=None,
        )
        return inputs_embeds, causal_mask_mapping, position_ids, cache_position

    @staticmethod
    def _decoder_layer_kv_outputs(layer_outputs: tuple[Any, ...], *, output_attentions: bool) -> tuple[Tensor, Tensor]:
        output_idx = 2 if output_attentions else 1
        return layer_outputs[output_idx], layer_outputs[output_idx + 1]

    @staticmethod
    def _action_time_conditioning(action_expert: torch.nn.Module, timesteps: Tensor) -> Tensor:
        time_conditioning = getattr(action_expert, "_time_conditioning", None)
        if callable(time_conditioning):
            return time_conditioning(timesteps)
        return action_expert.time_embed(timesteps)

    def _compute_flow_matching_loss_joint_per_layer(
        self,
        *,
        batch: dict[str, Any],
        model_inputs: dict[str, Tensor],
        timesteps: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        backbone_model = self._inner_model
        transformer = backbone_model.transformer
        action_expert = backbone_model._require_action_expert()
        actions = batch.get(ACTION)
        if actions is None:
            raise ValueError("MolmoAct2 training requires padded action targets in the preprocessed batch.")

        actions, timesteps, xt, target_velocity = self._prepare_flow_matching_tensors(
            actions=actions,
            action_dim_is_pad=batch.get("action_dim_is_pad"),
            timesteps=timesteps,
            noise=noise,
        )
        batch_size = int(actions.shape[0])
        device = actions.device

        hidden_states, causal_mask_mapping, position_ids, cache_position = self._prepare_joint_training_backbone_inputs(
            model_inputs,
        )
        if hidden_states.shape[0] != batch_size:
            raise ValueError(
                f"Backbone batch size {hidden_states.shape[0]} does not match action batch size {batch_size}.",
            )

        encoder_attention_mask = self._encoder_attention_mask_for_action_expert(
            input_ids=model_inputs.get("input_ids"),
            attention_mask=model_inputs.get("attention_mask"),
        )
        action_attention_mask = None
        if batch.get("action_horizon_is_pad") is not None:
            action_attention_mask = ~batch["action_horizon_is_pad"].to(device=device, dtype=torch.bool)

        valid_action = None
        if action_attention_mask is not None:
            valid_action = action_attention_mask.to(device=device, dtype=actions.dtype).unsqueeze(-1)

        rope_cache = None
        if len(action_expert.blocks) > 0 and action_expert.blocks[0].self_attn.rope is not None:
            rope_cache = action_expert.blocks[0].self_attn.rope.build_cache(
                seq_len=actions.shape[1],
                device=device,
                dtype=actions.dtype,
            )

        cross_mask = action_expert._build_cross_attention_mask(
            encoder_attention_mask,
            batch_size,
            actions.dtype,
        )
        self_mask = action_expert._build_self_attention_mask(
            action_attention_mask,
            actions.shape[1],
            device,
            actions.dtype,
        )

        conditioning = self._action_time_conditioning(action_expert, timesteps)
        action_hidden = action_expert.action_embed(xt)
        if valid_action is not None:
            action_hidden = action_hidden * valid_action

        if transformer.config.rope_scaling_layers is not None:
            position_embeddings_mapping = {
                "default": transformer.rotary_embs["default"](hidden_states, position_ids),
                "scaling": transformer.rotary_embs["scaling"](hidden_states, position_ids),
            }
        else:
            position_embeddings = transformer.rotary_emb(hidden_states, position_ids)

        for layer_idx in range(int(transformer.config.num_hidden_layers)):
            decoder_block = transformer.blocks[layer_idx]
            action_block = action_expert.blocks[layer_idx]
            if transformer.config.rope_scaling_layers is not None:
                position_embeddings_i = (
                    position_embeddings_mapping["scaling"]
                    if layer_idx in transformer.config.rope_scaling_layers
                    else position_embeddings_mapping["default"]
                )
            else:
                position_embeddings_i = position_embeddings

            layer_outputs = decoder_block(
                hidden_states,
                position_embeddings=position_embeddings_i,
                attention_mask=causal_mask_mapping,
                position_ids=position_ids,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
                collect_layer_kv_states=True,
            )
            hidden_states = layer_outputs[0]
            key_states, value_states = self._decoder_layer_kv_outputs(layer_outputs, output_attentions=False)
            key_states = backbone_model._cache_to_sequence(key_states)
            value_states = backbone_model._cache_to_sequence(value_states)

            k_ctx = action_expert._project_kv_tensor(key_states, action_expert.context_k_proj)
            v_ctx = action_expert._project_kv_tensor(value_states, action_expert.context_v_proj)
            k_norm = action_block.cross_attn.k_norm
            if k_norm is not None:
                k_ctx = k_norm(k_ctx.transpose(1, 2)).transpose(1, 2)

            action_hidden = action_block(
                action_hidden,
                conditioning,
                cross_kv=(k_ctx, v_ctx),
                self_attn_mask=self_mask,
                attn_mask=cross_mask,
                is_causal=action_expert.config.causal_attn,
                modulation=None,
                rope_cache=rope_cache,
            )
            if valid_action is not None:
                action_hidden = action_hidden * valid_action

        hidden_states = transformer.ln_f(hidden_states)
        del hidden_states
        pred_velocity = action_expert.final_layer(action_hidden, conditioning)
        if valid_action is not None:
            pred_velocity = pred_velocity * valid_action

        loss = F.mse_loss(pred_velocity, target_velocity, reduction="none")
        return _masked_loss_mean(
            loss,
            action_horizon_is_pad=batch.get("action_horizon_is_pad"),
            action_dim_is_pad=batch.get("action_dim_is_pad") if self.config.mask_action_dim_padding else None,
        )

    def compute_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        """Compute mode-aware MolmoAct2 training loss."""
        model_inputs = self.prepare_graph_inputs(batch)
        losses: list[Tensor] = []
        metrics: dict[str, float] = {}

        action_mode = self._action_mode()
        if action_mode in {"continuous", "both"}:
            flow_loss = self._compute_flow_matching_loss_joint_per_layer(batch=batch, model_inputs=model_inputs)
            losses.append(flow_loss)
            metrics["action_flow_loss"] = flow_loss.detach().float().item()

        if action_mode in {"discrete", "both"}:
            labels = model_inputs.get("labels")
            if labels is None:
                raise ValueError("MolmoAct2 discrete training requires labels in the preprocessed batch.")
            discrete_loss = self._compute_discrete_loss(model_inputs, labels)
            losses.append(discrete_loss)
            metrics["discrete_ce_loss"] = discrete_loss.detach().float().item()

        if not losses:
            raise ValueError(f"Unsupported MolmoAct2 action_mode={action_mode!r}.")

        loss = torch.stack(losses).sum()
        metrics["loss"] = loss.detach().float().item()
        return loss, metrics

    @torch.no_grad()
    def compute_val_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        """Compute validation MSE between generated and target actions."""
        gt_actions = batch.get(ACTION)
        if gt_actions is None:
            raise ValueError("MolmoAct2 validation requires padded action targets in the preprocessed batch.")

        model_inputs = self.prepare_graph_inputs(batch)
        action_horizon = int(gt_actions.shape[1])
        action_mode = self._action_mode()
        if action_mode == "discrete":
            action_dim = self._resolved_action_dim(batch, gt_actions)
            predicted = self._generate_discrete_actions_from_inputs(
                model_inputs=model_inputs,
                action_dim=action_dim,
                action_horizon=action_horizon,
            )
            gt_actions = gt_actions[..., :action_dim]
            action_dim_mask = None
        else:
            generation_inputs = {
                key: value for key, value in model_inputs.items() if key not in ("labels",)
            }
            predicted = self._inner_model.generate_actions_from_inputs(
                **generation_inputs,
                action_horizon=action_horizon,
            )
            action_dim_mask = batch.get("action_dim_is_pad") if self.config.mask_action_dim_padding else None

        min_horizon = min(int(gt_actions.shape[1]), int(predicted.shape[1]))
        gt_trimmed = gt_actions[:, :min_horizon].to(device=predicted.device, dtype=predicted.dtype)
        pred_trimmed = predicted[:, :min_horizon]
        loss = F.mse_loss(pred_trimmed, gt_trimmed, reduction="none")
        loss = _masked_loss_mean(
            loss,
            action_horizon_is_pad=batch.get("action_horizon_is_pad"),
            action_dim_is_pad=action_dim_mask,
        )
        loss_value = loss.detach().float().item()
        return loss, {"loss": loss_value}

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
        # TODO(export): Keep a single torch inference boundary because split
        # boundaries tend to drift in input contracts between runtime/export.
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
