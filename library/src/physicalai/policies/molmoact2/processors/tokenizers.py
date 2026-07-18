# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer helpers for MolmoAct2 preprocessing.

The tokenizer is a ``Qwen2Tokenizer``. ``Qwen2Tokenizer.from_pretrained(...,
local_files_only=True)`` is called exactly once per ``MolmoAct2Tokenizers``
instance, in the ``tokenizer`` property below. The vocab file
(``tokenizer.json``) is resolved eagerly in ``__init__`` so that:

- an exported checkpoint can rebuild the tokenizer on a different host where
  the original checkpoint snapshot directory no longer exists, by pulling
  only ``tokenizer.json`` from the configured repo id;
- the tokenizer options (``tokenizer_config.json``) are carried on the policy
  config (loaded once by ``from_hf.py``) so runtime only needs the vocab file.

When the config carries no ``tokenizer_config`` options, the options are
downloaded from the Hub as a fallback (option B).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RemoteEntryNotFoundError
from transformers import Qwen2Tokenizer

from physicalai.policies.molmoact2.config import DEFAULT_MOLMOACT2_REPO_ID

_TOKENIZER_JSON_FILENAME = "tokenizer.json"
_TOKENIZER_CONFIG_FILENAME = "tokenizer_config.json"


class MolmoAct2Tokenizers:
    """Class-based tokenizer utilities used by the MolmoAct2 preprocessor."""

    def __init__(
        self,
        *,
        tokenizer_name_or_path: str | None,
        tokenizer_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize lazy tokenizer helpers and resolve the tokenizer directory.

        Args:
            tokenizer_name_or_path: Local checkpoint path or HF repo id for the
                text tokenizer. When ``None`` the canonical
                ``DEFAULT_MOLMOACT2_REPO_ID`` repo is used.
            tokenizer_config: Optional parsed ``tokenizer_config.json`` options
                (loaded by ``from_hf.py``). When provided, only
                ``tokenizer.json`` is fetched from the repo at runtime. When
                ``None``, ``tokenizer_config.json`` is also fetched from the
                Hub as a fallback.
        """
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self._tokenizer_config = tokenizer_config
        self._tokenizer: Qwen2Tokenizer | None = None
        # Holds the TemporaryDirectory alive for the lifetime of this instance
        # so the resolved tokenizer files remain on disk.
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._tokenizer_dir = self._resolve_tokenizer_dir()

    def _resolve_tokenizer_dir(self) -> str:
        """Resolve a local directory containing the tokenizer files.

        Returns:
            Path to a directory containing ``tokenizer.json`` (and the
            tokenizer config) that can be passed to
            ``Qwen2Tokenizer.from_pretrained(..., local_files_only=True)``.

        Raises:
            FileNotFoundError: If ``tokenizer.json`` cannot be resolved either
                locally or from the Hub.
        """
        repo_or_path = self.tokenizer_name_or_path or DEFAULT_MOLMOACT2_REPO_ID
        local_path = Path(repo_or_path)

        # Case 1: a local directory already has the tokenizer vocab file.
        if local_path.is_dir() and (local_path / _TOKENIZER_JSON_FILENAME).is_file():
            self._ensure_tokenizer_config_local(local_path)
            return str(local_path)

        # Case 2: treat the value as a Hugging Face repo id and download only
        # ``tokenizer.json`` (the vocab) from the Hub.
        repo_id = repo_or_path
        try:
            tokenizer_json_path = hf_hub_download(repo_id, _TOKENIZER_JSON_FILENAME)
        except RemoteEntryNotFoundError as exc:
            msg = (
                f"Could not resolve MolmoAct2 tokenizer: no local '{_TOKENIZER_JSON_FILENAME}' "
                f"and '{_TOKENIZER_JSON_FILENAME}' not found in repo '{repo_id}'."
            )
            raise FileNotFoundError(msg) from exc

        # Stage tokenizer.json plus the tokenizer config options into a temp
        # directory so ``from_pretrained`` can load both with
        # ``local_files_only=True``.
        self._temp_dir = tempfile.TemporaryDirectory(prefix="molmoact2-tokenizer-")
        temp_path = Path(self._temp_dir.name)
        shutil.copy2(tokenizer_json_path, temp_path / _TOKENIZER_JSON_FILENAME)
        self._write_tokenizer_config(temp_path, repo_id)
        return str(temp_path)

    def _write_tokenizer_config(self, temp_path: Path, repo_id: str) -> None:
        """Write ``tokenizer_config.json`` into the staging directory.

        Uses the options carried on the config when available, otherwise
        downloads them from the Hub (option B fallback).

        Args:
            temp_path: Staging directory to write into.
            repo_id: Hugging Face repo id used for the fallback download.
        """
        config_payload: dict[str, Any] | None = self._tokenizer_config
        if config_payload is None:
            config_payload = self._download_tokenizer_config(repo_id)
        if config_payload is None:
            # No config available; let ``from_pretrained`` apply its defaults.
            return
        with (temp_path / _TOKENIZER_CONFIG_FILENAME).open("w", encoding="utf-8") as f:
            json.dump(config_payload, f)

    @staticmethod
    def _download_tokenizer_config(repo_id: str) -> dict[str, Any] | None:
        """Download ``tokenizer_config.json`` from the Hub as a fallback.

        Tries the configured repo id first, then the canonical MolmoAct2 repo.

        Args:
            repo_id: Hugging Face repo id to download from.

        Returns:
            Parsed tokenizer config payload, or ``None`` when unavailable.
        """
        for candidate in (repo_id, DEFAULT_MOLMOACT2_REPO_ID):
            if not candidate:
                continue
            try:
                downloaded = hf_hub_download(candidate, _TOKENIZER_CONFIG_FILENAME)
            except RemoteEntryNotFoundError:
                continue
            with Path(downloaded).open(encoding="utf-8") as f:
                return json.load(f)
        return None

    @staticmethod
    def _ensure_tokenizer_config_local(local_path: Path) -> None:
        """No-op for local directories that already contain a tokenizer config.

        Args:
            local_path: Local checkpoint directory that already contains
                ``tokenizer.json``. ``tokenizer_config.json`` is left in
                whatever state it is in; ``from_pretrained`` will use it when
                present and fall back to its own defaults otherwise.
        """
        del local_path

    @property
    def tokenizer(self) -> Qwen2Tokenizer:
        """Main text tokenizer loaded lazily.

        ``Qwen2Tokenizer.from_pretrained(..., local_files_only=True)`` is
        called at most once per instance.

        Raises:
            ValueError: If the tokenizer failed to initialize.
        """
        if self._tokenizer is not None:
            return self._tokenizer

        self._tokenizer = Qwen2Tokenizer.from_pretrained(
            self._tokenizer_dir,
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
