# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the MolmoAct2 tokenizer wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from physicalai.policies.molmoact2.processors.tokenizers import MolmoAct2Tokenizers

_MOCK_TOKENIZER_REPO = "mock-org/mock-tokenizer"


class TestTokenizerSetup:
    def test_uses_local_directory(self, mock_hf_repo: Path) -> None:
        tokenizers = MolmoAct2Tokenizers(tokenizer_name_or_path=str(mock_hf_repo))
        assert tokenizers._tokenizer_dir == str(mock_hf_repo)

    def test_downloads_tokenizer(self, patch_hf_hub_download) -> None:
        tokenizers = MolmoAct2Tokenizers(
            tokenizer_name_or_path=_MOCK_TOKENIZER_REPO
        )
        assert Path(tokenizers._tokenizer_dir, "tokenizer.json").is_file()

    def test_uses_supplied_tokenizer_config(self, patch_hf_hub_download) -> None:
        tokenizers = MolmoAct2Tokenizers(
            tokenizer_name_or_path=_MOCK_TOKENIZER_REPO,
            tokenizer_config={
                "bos_token": "<|im_end|>",
                "eos_token": "<|im_end|>",
                "pad_token": "<|endoftext|>",
            },
        )
        assert Path(
            tokenizers._tokenizer_dir,
            "tokenizer_config.json",
        ).is_file()

    def test_downloads_tokenizer_config(self, patch_hf_hub_download) -> None:
        tokenizers = MolmoAct2Tokenizers(
            tokenizer_name_or_path=_MOCK_TOKENIZER_REPO
        )
        assert Path(
            tokenizers._tokenizer_dir,
            "tokenizer_config.json",
        ).is_file()

    def test_tokenizer_loaded_once(self, patch_hf_hub_download) -> None:
        tokenizers = MolmoAct2Tokenizers(
            tokenizer_name_or_path=_MOCK_TOKENIZER_REPO
        )
        with patch(
            "physicalai.policies.molmoact2.processors.tokenizers.Qwen2Tokenizer.from_pretrained"
        ) as spy:
            assert tokenizers.tokenizer is tokenizers.tokenizer
        spy.assert_called_once()


class TestTokenization:
    def test_placeholder_token_exists(self, patch_hf_hub_download) -> None:
        tokenizers = MolmoAct2Tokenizers(
            tokenizer_name_or_path=_MOCK_TOKENIZER_REPO
        )
        token_id = tokenizers.tokenizer.convert_tokens_to_ids("<|image|>")
        assert isinstance(token_id, int)
        assert token_id != tokenizers.tokenizer.unk_token_id


class TestInsertBos:
    def test_insert_bos(self) -> None:
        out_ids, _ = MolmoAct2Tokenizers._insert_bos(
            np.array([[5, 6, 7]]),
            np.array([[1, 1, 1]]),
            bos_token_id=99,
            pad_token_id=0,
        )
        assert out_ids.shape == (1, 4)
        assert out_ids[0, 0] == 99

    def test_insert_bos_noop_when_present(self) -> None:
        input_ids = np.array([[99, 6, 7]])
        out_ids, _ = MolmoAct2Tokenizers._insert_bos(
            input_ids,
            np.array([[1, 1, 1]]),
            bos_token_id=99,
            pad_token_id=0,
        )
        np.testing.assert_array_equal(out_ids, input_ids)

    def test_insert_bos_empty_sequence(self) -> None:
        out_ids, _ = MolmoAct2Tokenizers._insert_bos(
            np.empty((1, 0), dtype=np.int64),
            np.empty((1, 0), dtype=np.int64),
            bos_token_id=99,
            pad_token_id=0,
        )
        assert out_ids.shape == (1, 1)
        assert out_ids[0, 0] == 99
