# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for MolmoAct2 processor components."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from physicalai.data.observation import Feature, FeatureType


ACTION_OUTPUT_TOKEN = "<action_output>"  # noqa: S105
SETUP_START_TOKEN = "<setup_start>"  # noqa: S105
SETUP_END_TOKEN = "<setup_end>"  # noqa: S105
CONTROL_START_TOKEN = "<control_start>"  # noqa: S105
CONTROL_END_TOKEN = "<control_end>"  # noqa: S105
STATE_START_TOKEN = "<state_start>"  # noqa: S105
STATE_END_TOKEN = "<state_end>"  # noqa: S105
STATE_TOKEN_PREFIX = "<state_"  # noqa: S105

_TRAILING_PUNCTUATION = ".,!?;:"
_PREFIX_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"^(?:task|instruction|language[_ ]instruction|goal)\s*[:\-]\s*",
        r"^(?:the\s+task\s+is\s+to|your\s+task\s+is\s+to)\s+",
    )
)


def feature_by_type(features: list[Feature], feature_type: FeatureType) -> Feature | None:
    """Return the first feature matching the requested type."""
    for feature in features:
        if feature.ftype == feature_type:
            return feature
    return None


def normalize_text(text: str) -> str:
    """Normalize user task text to match training preprocessing style.

    Args:
        text: Raw task text.

    Returns:
        Normalized task text.
    """
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return ""
    for pattern in _PREFIX_PATTERNS:
        normalized = pattern.sub("", normalized, count=1).strip()
    normalized = normalized.rstrip(_TRAILING_PUNCTUATION).strip()
    return normalized.lower()


def build_discrete_state_string(state: np.ndarray, num_state_tokens: int) -> str:
    """Convert normalized state values into discrete state token text.

    Args:
        state: Normalized state vector.
        num_state_tokens: Number of discrete state bins.

    Returns:
        Serialized discrete state token span.

    Raises:
        ValueError: If num_state_tokens is not positive.
    """
    if num_state_tokens <= 0:
        msg = f"num_state_tokens must be > 0, got {num_state_tokens}."
        raise ValueError(msg)

    arr = np.asarray(state, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    arr = np.clip(arr, -1.0, 1.0)
    scaled = (arr + 1.0) / 2.0 * float(num_state_tokens - 1)
    token_ids = np.clip(np.rint(scaled).astype(np.int64), 0, int(num_state_tokens) - 1).reshape(-1)
    payload = "".join(f"{STATE_TOKEN_PREFIX}{int(token_id)}>" for token_id in token_ids)
    return f"{STATE_START_TOKEN}{payload}{STATE_END_TOKEN}"


def wrap_setup_text(setup_type: str, *, add_setup_tokens: bool) -> str:
    """Optionally wrap setup text in dedicated setup markers.

    Args:
        setup_type: Setup text.
        add_setup_tokens: Whether to add setup markers.

    Returns:
        Wrapped or original setup text.
    """
    if not setup_type:
        return ""
    if not add_setup_tokens:
        return setup_type
    if setup_type.startswith(SETUP_START_TOKEN) and setup_type.endswith(SETUP_END_TOKEN):
        return setup_type
    return f"{SETUP_START_TOKEN}{setup_type}{SETUP_END_TOKEN}"


def wrap_control_text(control_mode: str, *, add_control_tokens: bool) -> str:
    """Optionally wrap control-mode text in dedicated control markers.

    Args:
        control_mode: Control mode text.
        add_control_tokens: Whether to add control markers.

    Returns:
        Wrapped or original control text.
    """
    if not control_mode:
        return ""
    if not add_control_tokens:
        return control_mode
    if control_mode.startswith(CONTROL_START_TOKEN) and control_mode.endswith(CONTROL_END_TOKEN):
        return control_mode
    return f"{CONTROL_START_TOKEN}{control_mode}{CONTROL_END_TOKEN}"


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
    """Build the MolmoAct2 prompt text for one training example.

    Returns:
        Fully formatted prompt text.
    """
    setup_text = wrap_setup_text(setup_type, add_setup_tokens=add_setup_tokens)
    control_text = wrap_control_text(control_mode, add_control_tokens=add_control_tokens)
    state_clause = f" The current state of the robot is {discrete_state_string}." if discrete_state_string else ""
    prompt_intro = f"The task is to {task}. The setup is {setup_text}.{state_clause}"
    prompt_expectation = f"The expected control mode is {control_text}."
    prompt_question = "Given these, what action should the robot take to complete the task?"
    prompt = f"{prompt_intro} {prompt_expectation} {prompt_question}"

    if num_images <= 0:
        image_prefix = ""
    elif num_images == 1:
        image_prefix = "<|image|>"
    else:
        image_prefix = "".join(f"Image {index + 1}<|image|>" for index in range(num_images))

    return f"{image_prefix}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{ACTION_OUTPUT_TOKEN}"
