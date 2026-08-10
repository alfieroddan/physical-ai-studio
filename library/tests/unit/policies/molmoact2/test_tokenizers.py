# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the MolmoAct2 tokenizer wrapper."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from physicalai.policies.molmoact2.processors.tokenizers import (
    MolmoAct2Tokenizers,
    _drop_output_only_added_tokens,
)


class TestTokenizerSetup:
    def test_openvino_view_keeps_prompt_tokens(self) -> None:
        class StubQwenTokenizer:
            added_tokens_decoder = {
                1: SimpleNamespace(content="<state_7>"),
                2: SimpleNamespace(content="<action_output>"),
                3: SimpleNamespace(content="<action_7>"),
                4: SimpleNamespace(content="<extra_7>"),
                5: SimpleNamespace(content="<|image|>"),
            }

        tokenizer = StubQwenTokenizer()
        trimmed = _drop_output_only_added_tokens(tokenizer)  # type: ignore[arg-type]

        assert tokenizer.added_tokens_decoder.keys() == {1, 2, 3, 4, 5}
        assert trimmed.added_tokens_decoder.keys() == {1, 2, 5}

    def test_uses_local_directory(self, mock_hf_repo: Path) -> None:
        tokenizers = MolmoAct2Tokenizers(tokenizer_name_or_path=str(mock_hf_repo))
        assert tokenizers._tokenizer_dir == str(mock_hf_repo)

    def test_rejects_missing_local_tokenizer(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="tokenizer.json"):
            MolmoAct2Tokenizers(tokenizer_name_or_path=str(tmp_path))

    def test_tokenizer_loaded_once(self, mock_hf_repo: Path) -> None:
        tokenizer_config = {
            "bos_token": "<|im_end|>",
            "extra_special_tokens": ["<im_start>", "<|image|>"],
            "model_max_length": 1010000,
        }
        tokenizers = MolmoAct2Tokenizers(
            tokenizer_name_or_path=str(mock_hf_repo),
            tokenizer_config=tokenizer_config,
        )
        with patch(
            "physicalai.policies.molmoact2.processors.tokenizers.Qwen2Tokenizer.from_pretrained"
        ) as spy:
            assert tokenizers.tokenizer is tokenizers.tokenizer
        spy.assert_called_once()
        assert spy.call_args.kwargs == {
            "local_files_only": True,
            **tokenizer_config,
        }


class TestTokenization:
    @staticmethod
    def _stub_tokenizer(tokenizers: MolmoAct2Tokenizers) -> None:
        class StubTokenizer:
            bos_token_id = 99
            eos_token_id = 98
            pad_token_id = 0

            def __call__(self, prompt_texts, **kwargs):
                width = kwargs["max_length"] if kwargs["padding"] == "max_length" else 2
                return {
                    "input_ids": [[5, 6, *([0] * (width - 2))] for _ in prompt_texts],
                    "attention_mask": [[1, 1, *([0] * (width - 2))] for _ in prompt_texts],
                }

        tokenizers._tokenizer = StubTokenizer()  # type: ignore[assignment]

    def test_placeholder_token_exists(self, mock_hf_repo: Path) -> None:
        tokenizers = MolmoAct2Tokenizers(tokenizer_name_or_path=str(mock_hf_repo))
        token_id = tokenizers.tokenizer.convert_tokens_to_ids("<|image|>")
        assert isinstance(token_id, int)
        assert token_id != tokenizers.tokenizer.unk_token_id

    def test_defaults_to_fixed_length_including_bos(self, mock_hf_repo: Path) -> None:
        tokenizers = MolmoAct2Tokenizers(tokenizer_name_or_path=str(mock_hf_repo), max_token_len=16)
        self._stub_tokenizer(tokenizers)

        input_ids, attention_mask = tokenizers.tokenize_prompts(["short prompt"])

        assert input_ids.shape == (1, 16)
        assert attention_mask.shape == (1, 16)
        assert int(input_ids[0, 0]) == tokenizers.tokenizer.bos_token_id
        assert int(attention_mask[0, -1]) == 0

    def test_longest_only_pads_to_batch_width(self, mock_hf_repo: Path) -> None:
        tokenizers = MolmoAct2Tokenizers(
            tokenizer_name_or_path=str(mock_hf_repo),
            max_token_len=16,
            padding="longest",
        )
        self._stub_tokenizer(tokenizers)

        input_ids, attention_mask = tokenizers.tokenize_prompts(["short prompt"])

        assert input_ids.shape == (1, 3)
        assert attention_mask.shape == (1, 3)
        assert int(input_ids[0, 0]) == tokenizers.tokenizer.bos_token_id


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
