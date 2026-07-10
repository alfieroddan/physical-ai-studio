# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: ANN,COM812,D,E501,EM,FBT001,I001,PLR2004,S105,SIM108,TCH001,TRY
# pylint: disable=all

"""Shared helpers for MolmoAct2 processor components."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch

from physicalai.data.observation import Feature, FeatureType

ACTION_OUTPUT_TOKEN = "<action_output>"
SETUP_START_TOKEN = "<setup_start>"
SETUP_END_TOKEN = "<setup_end>"
CONTROL_START_TOKEN = "<control_start>"
CONTROL_END_TOKEN = "<control_end>"
STATE_START_TOKEN = "<state_start>"
STATE_END_TOKEN = "<state_end>"
STATE_TOKEN_PREFIX = "<state_"

_TRAILING_PUNCTUATION = ".,!?;:"
_PREFIX_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"^(?:task|instruction|language[_ ]instruction|goal)\s*[:\-]\s*",
        r"^(?:the\s+task\s+is\s+to|your\s+task\s+is\s+to)\s+",
    )
)


def feature_by_type(features: list[Feature], feature_type: FeatureType) -> Feature | None:
    for feature in features:
        if feature.ftype == feature_type:
            return feature
    return None


def as_tensor(value: torch.Tensor | np.ndarray | None, *, dtype: torch.dtype = torch.float32) -> torch.Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def normalize_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return ""
    for pattern in _PREFIX_PATTERNS:
        normalized = pattern.sub("", normalized, count=1).strip()
    normalized = normalized.rstrip(_TRAILING_PUNCTUATION).strip()
    return normalized.lower()


def wrap_setup_text(setup_type: str, add_setup_tokens: bool) -> str:
    if not setup_type:
        return ""
    if not add_setup_tokens:
        return setup_type
    if setup_type.startswith(SETUP_START_TOKEN) and setup_type.endswith(SETUP_END_TOKEN):
        return setup_type
    return f"{SETUP_START_TOKEN}{setup_type}{SETUP_END_TOKEN}"


def wrap_control_text(control_mode: str, add_control_tokens: bool) -> str:
    if not control_mode:
        return ""
    if not add_control_tokens:
        return control_mode
    if control_mode.startswith(CONTROL_START_TOKEN) and control_mode.endswith(CONTROL_END_TOKEN):
        return control_mode
    return f"{CONTROL_START_TOKEN}{control_mode}{CONTROL_END_TOKEN}"


def build_discrete_state_string(state: np.ndarray, num_state_tokens: int) -> str:
    if num_state_tokens <= 0:
        raise ValueError(f"num_state_tokens must be > 0, got {num_state_tokens}.")
    arr = np.asarray(state, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    arr = np.clip(arr, -1.0, 1.0)
    scaled = (arr + 1.0) / 2.0 * float(num_state_tokens - 1)
    token_ids = np.clip(np.rint(scaled).astype(np.int64), 0, int(num_state_tokens) - 1).reshape(-1)
    return f"{STATE_START_TOKEN}{''.join(f'{STATE_TOKEN_PREFIX}{int(token_id)}>' for token_id in token_ids)}{STATE_END_TOKEN}"


def build_robot_text(
    *,
    task: str,
    discrete_state_string: str,
    setup_type: str,
    control_mode: str,
    add_setup_tokens: bool,
    add_control_tokens: bool,
    num_images: int,
) -> str:
    setup_text = wrap_setup_text(setup_type, add_setup_tokens=add_setup_tokens)
    control_text = wrap_control_text(control_mode, add_control_tokens=add_control_tokens)
    state_clause = f" The current state of the robot is {discrete_state_string}." if discrete_state_string else ""
    prompt = (
        f"The task is to {task}. The setup is {setup_text}.{state_clause} "
        f"The expected control mode is {control_text}. Given these, what action should the robot take to complete the task?"
    )
    if num_images <= 0:
        image_prefix = ""
    elif num_images == 1:
        image_prefix = "<|image|>"
    else:
        image_prefix = "".join(f"Image {idx + 1}<|image|>" for idx in range(num_images))
    return f"{image_prefix}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{ACTION_OUTPUT_TOKEN}"


def to_numpy_image(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape for MolmoAct2: {arr.shape}")

    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype in (np.float16, np.float32, np.float64):
        if arr.size > 0 and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def text_max_positions(config: Any, *, default: int = 4096) -> int:
    text_config = getattr(config, "text_config", None)
    if isinstance(text_config, dict):
        return int(text_config.get("max_position_embeddings", default))
    return int(getattr(text_config, "max_position_embeddings", default))
