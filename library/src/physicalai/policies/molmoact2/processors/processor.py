# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 multimodal processor contract and pipeline.

This module is the preprocessing boundary before the model-facing torch frontend.

Inputs expected by MolmoAct2Processor.__call__:
- text: str or list[str], with optional placeholders:
    - "<|image|>" for each image region.
    - "<|video|>" for each video segment if used by prompting logic.
- images: torch tensor batch from upstream preprocessing, expected as BCHW.
    - Passed through as images_bchw for model-side vision preprocessing.
- videos: torch tensor batch from upstream preprocessing, expected as TCHW or BTCHW.
    - Passed through as videos_btchw for model-side vision preprocessing.

Outputs:
- When return_tensors="pt": returns a dict where numeric model inputs are torch tensors.
    Typical keys include:
    - input_ids, attention_mask
    - image_placeholder_token_id (for model-side image token expansion)
    - images_bchw (if images provided)
    - videos_btchw (if videos provided)
- When return_tensors is not "pt": returns Python lists/arrays following tokenizer
    defaults plus raw vision passthrough fields.

Steps performed in this processor:
1. Tokenize text with Qwen2Tokenizer.
2. Ensure BOS placement.
3. Attach raw vision tensors and placeholder token metadata.
4. Optionally convert numeric values to torch.

Steps intentionally left for the model/frontend path:
- Final strict tensor contract checks before model forward.
- Image/video preprocessing and pooled index construction.
- Expansion of image placeholders into patch-token sequences.
- Construction/validation of token_type_ids and graph-ready fields.
- Any model-specific normalization, masking, padding, or action-head preparation.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import torch

IMAGE_PROMPT = "<|image|>"


class MolmoAct2ProcessorOptions(TypedDict, total=False):
    text_kwargs: dict[str, Any]


class MolmoAct2Processor:
    """MolmoAct2 tokenizer-first processor with raw vision tensor passthrough."""

    def __init__(
        self,
        tokenizer: Any,
        chat_template: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.image_placeholder_token = IMAGE_PROMPT

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
        text_kwargs = {"padding": False}
        text_kwargs.update(dict(kwargs.pop("text_kwargs", {}) or {}))
        text_kwargs.update(kwargs)

        if not isinstance(text, list):
            text = [text]
        text = list(text)

        return_tensors = text_kwargs.pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **text_kwargs)

        input_ids = np.asarray(text_inputs["input_ids"])
        attention_mask = np.asarray(text_inputs["attention_mask"])
        bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        input_ids, attention_mask = self.insert_bos(input_ids, attention_mask, bos, self.tokenizer.pad_token_id)

        text_inputs["input_ids"] = input_ids.tolist()
        text_inputs["attention_mask"] = attention_mask.tolist()
        text_inputs["image_placeholder_token_id"] = int(
            self.tokenizer.convert_tokens_to_ids(self.image_placeholder_token),
        )

        data = dict(text_inputs)
        if images is not None:
            data["images_bchw"] = images
        if videos is not None:
            data["videos_btchw"] = videos
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
