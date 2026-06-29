# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 multimodal processor contract and pipeline.

This module is the preprocessing boundary before the model-facing torch frontend.

Inputs expected by MolmoAct2Processor.__call__:
- text: str or list[str], with optional placeholders:
    - "<|image|>" for each image to inject image patch tokens.
    - "<|video|>" for each video segment if used by prompting logic.
- images: torch tensor batch from upstream preprocessing, expected as BCHW.
    - Image shape/layout validation and patch extraction are handled by image.py.
- videos: torch tensor batch from upstream preprocessing, expected as TCHW or BTCHW.
    - Video frame validation/sampling and patch extraction are handled by video.py.

Outputs:
- When return_tensors="pt": returns a dict where numeric model inputs are torch tensors.
    Typical keys include:
    - input_ids, attention_mask, token_type_ids (optional)
    - pixel_values, image_token_pooling, image_grids, image_num_crops (if images provided)
    - pixel_values_videos, video_token_pooling, video_grids (if videos provided)
    - video_metadata (optional, controlled by return_metadata)
- When return_tensors is not "pt": returns Python lists/arrays following tokenizer
    defaults plus image/video processor outputs.

Steps performed in this processor:
1. Read options and dispatch image/video tensors to local image/video processors.
2. Expand image placeholders in text using computed image grids.
3. Tokenize text with Qwen2Tokenizer.
4. Ensure BOS placement and build optional multimodal token type ids.
5. Merge text + vision outputs and optionally convert numeric values to torch.

Steps intentionally left for the model/frontend path:
- Final strict tensor contract checks before model forward.
- Construction/validation of pooled index tensors and graph-ready fields.
- Any model-specific normalization, masking, padding, or action-head preparation.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import torch

from .image import MolmoAct2ImageProcessor, MolmoAct2ImagesOptions
from .video import MolmoAct2VideoProcessor, MolmoAct2VideoProcessorOptions

IMAGE_PATCH_TOKEN = "<im_patch>"
IMAGE_LOW_RES_TOKEN = "<im_low>"
IM_START_TOKEN = "<im_start>"
LOW_RES_IMAGE_START_TOKEN = "<low_res_im_start>"
FRAME_START_TOKEN = "<frame_start>"
IM_END_TOKEN = "<im_end>"
FRAME_END_TOKEN = "<frame_end>"
IM_COL_TOKEN = "<im_col>"
IMAGE_PROMPT = "<|image|>"
VIDEO_PROMPT = "<|video|>"

IMAGE_TOKENS = [
    IMAGE_PATCH_TOKEN,
    IM_COL_TOKEN,
    IM_START_TOKEN,
    LOW_RES_IMAGE_START_TOKEN,
    FRAME_START_TOKEN,
    IM_END_TOKEN,
    FRAME_END_TOKEN,
    IMAGE_LOW_RES_TOKEN,
]


class MolmoAct2ProcessorOptions(TypedDict, total=False):
    images_kwargs: MolmoAct2ImagesOptions
    videos_kwargs: MolmoAct2VideoProcessorOptions
    text_kwargs: dict[str, Any]
    return_metadata: bool


class MolmoAct2Processor:
    """MolmoAct2 multimodal processor composed from local tokenizer/image/video processors."""

    def __init__(
        self,
        image_processor: MolmoAct2ImageProcessor,
        video_processor: MolmoAct2VideoProcessor,
        tokenizer: Any,
        chat_template: str | None = None,
        image_use_col_tokens: bool | None = True,
        use_single_crop_col_tokens: bool | None = None,
        use_single_crop_start_token: bool | None = True,
        video_use_col_tokens: bool | None = False,
        use_frame_special_tokens: bool | None = True,
    ) -> None:
        self.image_processor = image_processor
        self.video_processor = video_processor
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.image_use_col_tokens = image_use_col_tokens
        self.use_single_crop_col_tokens = use_single_crop_col_tokens
        self.use_single_crop_start_token = use_single_crop_start_token
        self.video_use_col_tokens = video_use_col_tokens
        self.use_frame_special_tokens = use_frame_special_tokens
        self.image_placeholder_token = IMAGE_PROMPT
        self.video_placeholder_token = VIDEO_PROMPT
        self.image_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in IMAGE_TOKENS]

    def get_image_tokens(self, image_grid: np.ndarray) -> np.ndarray:
        resized_h, resized_w, height, width = image_grid
        if int(height) == 0 or int(width) == 0:
            per_row = np.full(int(resized_w), IMAGE_PATCH_TOKEN)
            use_col = self.image_use_col_tokens if self.use_single_crop_col_tokens is None else self.use_single_crop_col_tokens
            if use_col:
                per_row = np.concatenate([per_row, [IM_COL_TOKEN]], axis=0)
            return np.concatenate([[IM_START_TOKEN], np.tile(per_row, [int(resized_h)]), [IM_END_TOKEN]])

        per_row = np.full(int(width), IMAGE_PATCH_TOKEN)
        if self.image_use_col_tokens:
            per_row = np.concatenate([per_row, [IM_COL_TOKEN]], axis=0)
        high_res = np.concatenate([[IM_START_TOKEN], np.tile(per_row, [int(height)]), [IM_END_TOKEN]])

        low_per_row = np.full(int(resized_w), IMAGE_PATCH_TOKEN)
        use_col = self.image_use_col_tokens if self.use_single_crop_col_tokens is None else self.use_single_crop_col_tokens
        start_token = LOW_RES_IMAGE_START_TOKEN if self.use_single_crop_start_token else IM_START_TOKEN
        if use_col:
            low_per_row = np.concatenate([low_per_row, [IM_COL_TOKEN]], axis=0)
        low_res = np.concatenate([[start_token], np.tile(low_per_row, [int(resized_h)]), [IM_END_TOKEN]])
        return np.concatenate([low_res, high_res])

    def insert_bos(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        bos_token_id: int,
        pad_token_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
            attention_mask = attention_mask[None, :]
            squeeze = True
        else:
            squeeze = False

        batch_size, seq_len = input_ids.shape
        if seq_len == 0:
            out_ids = np.full((batch_size, 1), bos_token_id, dtype=input_ids.dtype)
            out_mask = np.ones((batch_size, 1), dtype=attention_mask.dtype)
            return (out_ids[0], out_mask[0]) if squeeze else (out_ids, out_mask)

        first_valid = (attention_mask == 1).argmax(axis=-1)
        if np.all(input_ids[np.arange(batch_size), first_valid] == bos_token_id):
            return (input_ids[0], attention_mask[0]) if squeeze else (input_ids, attention_mask)

        out_ids = np.full((batch_size, seq_len + 1), pad_token_id, dtype=input_ids.dtype)
        out_mask = np.zeros((batch_size, seq_len + 1), dtype=attention_mask.dtype)

        src = np.tile(np.arange(seq_len), (batch_size, 1))
        valid = src >= first_valid[:, None]
        tgt = src + 1
        batch_idx = np.tile(np.arange(batch_size)[:, None], (1, seq_len))

        out_ids[batch_idx[valid], tgt[valid]] = input_ids[valid]
        out_mask[batch_idx[valid], tgt[valid]] = 1
        out_ids[np.arange(batch_size), first_valid] = bos_token_id
        out_mask[np.arange(batch_size), first_valid] = 1

        return (out_ids[0], out_mask[0]) if squeeze else (out_ids, out_mask)

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: Any = None,
        videos: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        explicit_return_metadata = "return_metadata" in kwargs or (
            isinstance(kwargs.get("videos_kwargs"), dict) and "return_metadata" in kwargs["videos_kwargs"]
        )
        images_kwargs = dict(kwargs.pop("images_kwargs", {}) or {})
        videos_kwargs = {"return_metadata": True}
        videos_kwargs.update(dict(kwargs.pop("videos_kwargs", {}) or {}))

        text_kwargs = {"padding": False, "return_mm_token_type_ids": True}
        text_kwargs.update(dict(kwargs.pop("text_kwargs", {}) or {}))
        text_kwargs.update(kwargs)

        image_inputs = self.image_processor(images, **images_kwargs) if images is not None else {}
        image_grids = image_inputs.get("image_grids")

        videos_inputs = self.video_processor(videos=videos, **videos_kwargs) if videos is not None else {}
        if videos is not None and not explicit_return_metadata and "video_metadata" in videos_inputs:
            videos_inputs.pop("video_metadata")

        if not isinstance(text, list):
            text = [text]
        text = list(text)

        if image_grids is not None:
            idx = 0
            for i in range(len(text)):
                num_images = text[i].count(self.image_placeholder_token)
                for grid in image_grids[idx : idx + num_images]:
                    image_tokens = self.get_image_tokens(np.asarray(grid))
                    text[i] = text[i].replace(self.image_placeholder_token, "".join(image_tokens), 1)
                idx += num_images

        return_tensors = text_kwargs.pop("return_tensors", None)
        return_mm_token_type_ids = bool(text_kwargs.pop("return_mm_token_type_ids", False))
        text_inputs = self.tokenizer(text, **text_kwargs)

        input_ids = np.asarray(text_inputs["input_ids"])
        attention_mask = np.asarray(text_inputs["attention_mask"])
        bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        input_ids, attention_mask = self.insert_bos(input_ids, attention_mask, bos, self.tokenizer.pad_token_id)

        if return_mm_token_type_ids:
            image_tokens = np.asarray(self.image_token_ids, dtype=input_ids.dtype)
            token_type_ids = np.any(input_ids[:, :, None] == image_tokens[None, None, :], axis=-1)
            text_inputs["token_type_ids"] = token_type_ids.tolist()

        text_inputs["input_ids"] = input_ids.tolist()
        text_inputs["attention_mask"] = attention_mask.tolist()

        data = {**text_inputs, **image_inputs, **videos_inputs}
        if return_tensors == "pt":
            return {key: _to_torch(value) for key, value in data.items()}
        return data


def _to_torch(value: Any) -> Any:
    if torch.is_tensor(value):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S", "O"}:
            return value
        return torch.from_numpy(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return value
        first = value[0]
        if isinstance(first, (str, bytes, dict)):
            return value
        if hasattr(first, "__dict__") and not isinstance(first, (int, float, bool, list, tuple, np.ndarray)):
            return value
        try:
            return torch.as_tensor(value)
        except (TypeError, ValueError):
            return value
    return value


__all__ = ["MolmoAct2Processor", "MolmoAct2ProcessorOptions"]
