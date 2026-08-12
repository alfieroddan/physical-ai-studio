# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer helpers for MolmoAct2 preprocessing."""

from __future__ import annotations

import re
from copy import copy
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from transformers import Qwen2Tokenizer

_TOKENIZER_JSON_FILENAME = "tokenizer.json"
_MIN_TOKEN_LEN = 2
_OUTPUT_ONLY_TOKEN = re.compile(r"^<(?:action|extra)_\d+>$")


# Note: This removes discrete action tokens which are only used in the model output,
# in discrete mode. The OpenVINO tokenizer view is used for inference,
# where the model does not generate discrete actions.
def _drop_output_only_added_tokens(tokenizer: Qwen2Tokenizer) -> Qwen2Tokenizer:
    """Return a conversion view without decoder-only action and extra tokens."""
    kept_tokens = {
        token_id: token
        for token_id, token in tokenizer.added_tokens_decoder.items()
        if not _OUTPUT_ONLY_TOKEN.match(token.content)
    }
    trimmed = copy(tokenizer)
    trimmed.__class__ = type(
        f"OpenVINO{type(tokenizer).__name__}",
        (type(tokenizer),),
        {"added_tokens_decoder": property(lambda _self: kept_tokens)},
    )
    return trimmed


class MolmoAct2Tokenizers:
    """Class-based tokenizer utilities used by the MolmoAct2 preprocessor."""

    def __init__(
        self,
        *,
        tokenizer_name_or_path: str,
        max_token_len: int = 256,
        padding: Literal["max_length", "longest"] = "max_length",
        tokenizer_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize lazy tokenizer helpers from a local tokenizer directory.

        Args:
            tokenizer_name_or_path: Local directory containing ``tokenizer.json``.
            max_token_len: Maximum tokenized prompt length, including BOS.
            padding: Tokenizer padding strategy.
            tokenizer_config: Checkpoint-derived tokenizer construction options.

        Raises:
            ValueError: If ``max_token_len`` is less than 2.
        """
        self.tokenizer_name_or_path = tokenizer_name_or_path
        if max_token_len < _MIN_TOKEN_LEN:
            msg = "max_token_len must be at least 2 to reserve space for BOS."
            raise ValueError(msg)
        self.max_token_len = max_token_len
        self.padding = padding
        self.tokenizer_config = tokenizer_config or {}
        self._tokenizer: Qwen2Tokenizer | None = None
        self._openvino_tokenizer: Qwen2Tokenizer | None = None
        self._tokenizer_dir = self._resolve_tokenizer_dir()

    def _resolve_tokenizer_dir(self) -> str:
        """Resolve a local directory containing the tokenizer files.

        Returns:
            Path to a directory containing ``tokenizer.json`` (and the
            tokenizer config) that can be passed to
            ``Qwen2Tokenizer.from_pretrained(..., local_files_only=True)``.

        Raises:
            FileNotFoundError: If ``tokenizer.json`` is not present locally.
        """
        local_path = Path(self.tokenizer_name_or_path)
        if not local_path.is_dir() or not (local_path / _TOKENIZER_JSON_FILENAME).is_file():
            msg = f"MolmoAct2 tokenizer directory must contain '{_TOKENIZER_JSON_FILENAME}': {local_path}"
            raise FileNotFoundError(msg)
        return str(local_path)

    def _qwen_tokenizer(self) -> Qwen2Tokenizer:
        """Main text tokenizer loaded lazily.

        ``Qwen2Tokenizer.from_pretrained(..., local_files_only=True)`` is
        called at most once per instance.

        Returns:
            The ``Qwen2Tokenizer`` instance used for prompt tokenization.

        Raises:
            ValueError: If the tokenizer failed to initialize.
        """
        if self._tokenizer is not None:
            return self._tokenizer

        self._tokenizer = Qwen2Tokenizer.from_pretrained(  # nosec: B615
            self._tokenizer_dir,
            local_files_only=True,
            **self.tokenizer_config,
        )
        if self._tokenizer is None:
            msg = "Failed to initialize MolmoAct2 text tokenizer."
            raise ValueError(msg)
        return self._tokenizer

    @property
    def tokenizer(self) -> Qwen2Tokenizer:
        """The main text tokenizer.

        Returns:
            The ``Qwen2Tokenizer`` instance used for prompt tokenization.
        """
        if self._openvino_tokenizer is None:
            self._openvino_tokenizer = _drop_output_only_added_tokens(self._qwen_tokenizer())
        return self._openvino_tokenizer

    @staticmethod
    def _insert_bos(
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

        valid_rows = [input_ids[index][attention_mask[index].astype(bool)] for index in range(batch_size)]
        if all(row.size > 0 and int(row[0]) == bos_token_id for row in valid_rows):
            return (input_ids[0], attention_mask[0]) if squeeze else (input_ids, attention_mask)

        out_ids = np.full((batch_size, seq_len + 1), pad_token_id, dtype=input_ids.dtype)
        out_mask = np.zeros((batch_size, seq_len + 1), dtype=attention_mask.dtype)
        for batch_idx in range(batch_size):
            valid_ids = input_ids[batch_idx][attention_mask[batch_idx].astype(bool)]
            if valid_ids.size == 0 or int(valid_ids[0]) != bos_token_id:
                valid_ids = np.concatenate((np.asarray([bos_token_id], dtype=input_ids.dtype), valid_ids))
            out_ids[batch_idx, : valid_ids.size] = valid_ids
            out_mask[batch_idx, : valid_ids.size] = 1

        return (out_ids[0], out_mask[0]) if squeeze else (out_ids, out_mask)

    def tokenize_prompts(
        self,
        prompt_texts: list[str],
        *,
        padding: Literal["max_length", "longest"] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize prompt text and ensure BOS insertion.

        Args:
            prompt_texts: Prompt text for each batch element.
            padding: Optional padding strategy override.

        Returns:
            Tuple of input ids and attention mask tensors.

        Raises:
            TypeError: If required tokenizer special token ids are missing.
        """
        padding = self.padding if padding is None else padding
        tokenizer = self._qwen_tokenizer()
        text_inputs = tokenizer(
            prompt_texts,
            max_length=self.max_token_len - 1,
            truncation=True,
            padding=padding,
        )

        input_ids = np.asarray(text_inputs["input_ids"])
        attention_mask = np.asarray(text_inputs["attention_mask"])

        bos_token_id = tokenizer.bos_token_id
        if not isinstance(bos_token_id, int):
            eos_token_id = tokenizer.eos_token_id
            if not isinstance(eos_token_id, int):
                msg = "Tokenizer must define bos_token_id or eos_token_id."
                raise TypeError(msg)
            bos_token_id = eos_token_id

        pad_token_id = tokenizer.pad_token_id
        if not isinstance(pad_token_id, int):
            msg = "Tokenizer must define pad_token_id."
            raise TypeError(msg)

        input_ids, attention_mask = self._insert_bos(
            input_ids,
            attention_mask,
            int(bos_token_id),
            int(pad_token_id),
        )

        pad_width = self.max_token_len - input_ids.shape[-1]
        if padding == "max_length" and pad_width > 0:
            input_ids = np.pad(input_ids, ((0, 0), (0, pad_width)), constant_values=pad_token_id)
            attention_mask = np.pad(attention_mask, ((0, 0), (0, pad_width)), constant_values=0)

        return torch.as_tensor(input_ids), torch.as_tensor(attention_mask)
