# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer helpers for MolmoAct2 preprocessing."""

from __future__ import annotations

import numpy as np
import torch
from transformers import Qwen2Tokenizer


class MolmoAct2Tokenizers:
    """Class-based tokenizer utilities used by the MolmoAct2 preprocessor."""

    def __init__(
        self,
        *,
        tokenizer_name_or_path: str | None,
    ) -> None:
        """Initialize lazy tokenizer helpers.

        Args:
            tokenizer_name_or_path: Local checkpoint path or HF id for the text tokenizer.
        """
        self.tokenizer_name_or_path = tokenizer_name_or_path

        self._tokenizer: Qwen2Tokenizer | None = None

    @property
    def tokenizer(self) -> Qwen2Tokenizer:
        """Main text tokenizer loaded lazily.

        Raises:
            ValueError: If tokenizer_name_or_path is not configured.
        """
        if self._tokenizer is not None:
            return self._tokenizer

        if not self.tokenizer_name_or_path:
            msg = "config.tokenizer_name_or_path is required for MolmoAct2 preprocessing."
            raise ValueError(msg)

        self._tokenizer = Qwen2Tokenizer.from_pretrained(
            str(self.tokenizer_name_or_path),
            local_files_only=True,
        )
        if self._tokenizer is None:
            msg = "Failed to initialize MolmoAct2 text tokenizer."
            raise ValueError(msg)
        return self._tokenizer

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

    def tokenize_prompts(self, prompt_texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize prompt text and ensure BOS insertion.

        Args:
            prompt_texts: Prompt text for each batch element.

        Returns:
            Tuple of input ids and attention mask tensors.

        Raises:
            TypeError: If required tokenizer special token ids are missing.
        """
        text_inputs = self.tokenizer(prompt_texts, padding=True)

        input_ids = np.asarray(text_inputs["input_ids"])
        attention_mask = np.asarray(text_inputs["attention_mask"])

        bos_token_id = self.tokenizer.bos_token_id
        if not isinstance(bos_token_id, int):
            eos_token_id = self.tokenizer.eos_token_id
            if not isinstance(eos_token_id, int):
                msg = "Tokenizer must define bos_token_id or eos_token_id."
                raise TypeError(msg)
            bos_token_id = eos_token_id

        pad_token_id = self.tokenizer.pad_token_id
        if not isinstance(pad_token_id, int):
            msg = "Tokenizer must define pad_token_id."
            raise TypeError(msg)

        input_ids, attention_mask = self._insert_bos(
            input_ids,
            attention_mask,
            int(bos_token_id),
            int(pad_token_id),
        )

        return torch.as_tensor(input_ids), torch.as_tensor(attention_mask)
