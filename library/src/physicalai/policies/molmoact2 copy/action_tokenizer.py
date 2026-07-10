# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Minimal discrete action tokenizer support for MolmoAct2."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import SupportsFloat, SupportsInt, cast

import numpy as np
from huggingface_hub import snapshot_download
from scipy.fft import dct, idct
from transformers import PreTrainedTokenizerFast

_ACTION_SEQ_NDIM = 2
_ACTION_BATCH_NDIM = 3


def _as_int(value: object, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, (str, bytes)):
        return int(value)
    return int(cast("SupportsInt", value))


def _as_float(value: object, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (str, bytes)):
        return float(value)
    return float(cast("SupportsFloat", value))


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HF_ACCESS_TOKEN")


def _resolve_tokenizer_location(
    tokenizer_path: str,
    *,
    revision: str | None = None,
    force_download: bool = False,
) -> str:
    local_path = Path(str(tokenizer_path)).expanduser()
    if local_path.exists():
        return str(local_path)

    return snapshot_download(
        repo_id=str(tokenizer_path),
        repo_type="model",
        revision=revision,
        force_download=force_download,
        ignore_patterns=["*.py", "*.pyc", "__pycache__/*"],
        token=_hf_token(),
    )


class UniversalActionProcessor:
    """Tokenizer used to discretize continuous action chunks for MolmoAct2."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        *,
        scale: float = 10,
        vocab_size: int = 1024,
        min_token: int = 0,
        action_dim: int | None = None,
        time_horizon: int | None = None,
    ) -> None:
        """Initialize the action tokenizer wrapper and cached decode metadata."""
        self.tokenizer = tokenizer
        self.scale = scale
        self.vocab_size = vocab_size
        self.min_token = min_token
        self.time_horizon = time_horizon
        self.action_dim = action_dim
        self.called_time_horizon = time_horizon
        self.called_action_dim = action_dim

    def __call__(self, action_chunk: np.ndarray) -> list[list[int]]:
        """Encode continuous action chunks into discrete token ids.

        Returns:
            Token ids for each action chunk in the batch.

        Raises:
            ValueError: If the action chunk does not have rank 1, 2, or 3.
        """
        if action_chunk.ndim == _ACTION_SEQ_NDIM:
            action_chunk = action_chunk[None, ...]
        elif action_chunk.ndim == 1:
            action_chunk = action_chunk[None, None, ...]
        elif action_chunk.ndim != _ACTION_BATCH_NDIM:
            msg = "Only [batch, timesteps, action_dim] discrete action inputs are supported."
            raise ValueError(msg)

        self.called_time_horizon = int(action_chunk.shape[-2])
        self.called_action_dim = int(action_chunk.shape[-1])

        dct_coeff = np.asarray(dct(action_chunk, axis=1, norm="ortho"), dtype=np.float32)
        dct_coeff = np.around(dct_coeff * self.scale)
        tokens: list[list[int]] = []
        for elem in dct_coeff:
            token_str = "".join(map(chr, np.maximum(elem.flatten() - self.min_token, 0).astype(int)))
            encoded = self.tokenizer(token_str)
            token_ids = cast("list[int]", encoded["input_ids"])
            tokens.append([int(token_id) for token_id in token_ids])
        return tokens

    def decode(
        self,
        tokens: list[list[int]],
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
    ) -> np.ndarray:
        """Decode discrete token ids back into continuous action chunks.

        Returns:
            Decoded continuous action chunks.

        Raises:
            ValueError: If action shape metadata is unavailable.
        """
        self.time_horizon = time_horizon or self.time_horizon or self.called_time_horizon
        self.action_dim = action_dim or self.action_dim or self.called_action_dim
        self.called_time_horizon = self.time_horizon
        self.called_action_dim = self.action_dim

        if self.time_horizon is None or self.action_dim is None:
            msg = "Action tokenizer requires time_horizon and action_dim before decode."
            raise ValueError(msg)

        decoded_actions = []
        for token_row in tokens:
            decoded_tokens = self.tokenizer.decode(token_row)
            decoded_dct_coeff = np.array(list(map(ord, decoded_tokens))) + self.min_token
            decoded_dct_coeff = decoded_dct_coeff.reshape(-1, self.action_dim)
            if decoded_dct_coeff.shape != (self.time_horizon, self.action_dim):
                decoded_dct_coeff = np.zeros((self.time_horizon, self.action_dim), dtype=np.float32)
            decoded_actions.append(idct(decoded_dct_coeff / self.scale, axis=0, norm="ortho"))
        return np.stack(decoded_actions)

    @classmethod
    def from_pretrained_local(
        cls,
        pretrained_model_name_or_path: str,
        *,
        revision: str | None = None,
        force_download: bool = False,
    ) -> UniversalActionProcessor:
        """Load tokenizer weights and metadata from a local path or Hub repo.

        Returns:
            Configured action tokenizer instance.
        """
        location = Path(
            _resolve_tokenizer_location(
                pretrained_model_name_or_path,
                revision=revision,
                force_download=force_download,
            ),
        )
        processor_config: dict[str, object] = {}
        processor_config_path = location / "processor_config.json"
        if processor_config_path.exists():
            processor_config = json.loads(processor_config_path.read_text())
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(location))
        return cls(
            tokenizer,
            scale=_as_float(processor_config.get("scale"), 10),
            vocab_size=_as_int(processor_config.get("vocab_size"), 1024),
            min_token=_as_int(processor_config.get("min_token"), 0),
            action_dim=(
                _as_int(processor_config.get("action_dim"), 0)
                if processor_config.get("action_dim") is not None
                else None
            ),
            time_horizon=(
                _as_int(processor_config.get("time_horizon"), 0)
                if processor_config.get("time_horizon") is not None
                else None
            ),
        )

