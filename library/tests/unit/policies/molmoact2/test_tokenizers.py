# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for MolmoAct2 tokenizer utilities."""

from pathlib import Path
import re
from shutil import copyfile
from unittest.mock import Mock

import pytest
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError

from physicalai.policies.molmoact2.processors.tokenizers import MolmoAct2Tokenizers

_MOLMOACT2_REPOSITORY = "allenai/MolmoAct2"
_MOLMOACT2_REVISION = "e432d85f6e039edca44afb93c262f3084ab72a9c"


class StubTokenizer:
    bos_token_id = 9
    eos_token_id = 8
    pad_token_id = 0

    def __call__(self, prompts: list[str], **kwargs: object) -> dict[str, list[list[int]]]:
        width = int(kwargs["max_length"]) if kwargs["padding"] == "max_length" else 2  # type: ignore[call-overload]
        return {
            "input_ids": [[5, 6, *([0] * (width - 2))] for _ in prompts],
            "attention_mask": [[1, 1, *([0] * (width - 2))] for _ in prompts],
        }


def test_loads_local_tokenizer_once(tokenizer_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loader = Mock(return_value=StubTokenizer())
    monkeypatch.setattr(
        "physicalai.policies.molmoact2.processors.tokenizers.Qwen2Tokenizer.from_pretrained",
        loader,
    )
    tokenizers = MolmoAct2Tokenizers(tokenizer_name_or_path=str(tokenizer_dir))

    assert tokenizers._qwen_tokenizer() is tokenizers._qwen_tokenizer()
    loader.assert_called_once_with(str(tokenizer_dir), local_files_only=True)


def test_accepts_explicit_tokenizer_json_path(tokenizer_dir: Path) -> None:
    tokenizers = MolmoAct2Tokenizers(tokenizer_name_or_path=str(tokenizer_dir / "tokenizer.json"))

    assert tokenizers._tokenizer_dir == str(tokenizer_dir)


def test_downloads_pinned_default_when_local_tokenizer_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    downloaded_path = tmp_path / "download" / "tokenizer.json"
    downloaded_path.parent.mkdir()
    downloaded_path.write_text("{}", encoding="utf-8")
    download = Mock(return_value=str(downloaded_path))
    monkeypatch.setattr("physicalai.policies.molmoact2.processors.tokenizers.hf_hub_download", download)

    tokenizers = MolmoAct2Tokenizers(tokenizer_name_or_path=str(tmp_path / "missing"))

    assert tokenizers._tokenizer_dir == str(downloaded_path.parent)
    download.assert_called_once_with(
        repo_id=_MOLMOACT2_REPOSITORY,
        filename="tokenizer.json",
        revision=_MOLMOACT2_REVISION,
    )
    assert "downloading the pinned default tokenizer" in caplog.text


def test_missing_tokenizer_reports_local_override_when_download_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "physicalai.policies.molmoact2.processors.tokenizers.hf_hub_download",
        Mock(side_effect=LocalEntryNotFoundError("offline")),
    )

    with pytest.raises(FileNotFoundError, match="Supply a valid tokenizer_json_path"):
        MolmoAct2Tokenizers(tokenizer_name_or_path=str(tmp_path / "missing"))


@pytest.mark.parametrize(("padding", "width"), [("max_length", 6), ("longest", 3)])
def test_tokenization_inserts_bos(
    tokenizer_dir: Path,
    padding: str,
    width: int,
) -> None:
    tokenizers = MolmoAct2Tokenizers(
        tokenizer_name_or_path=str(tokenizer_dir),
        max_token_len=6,
        padding=padding,  # type: ignore[arg-type]
    )
    tokenizers._tokenizer = StubTokenizer()  # type: ignore[assignment]

    input_ids, attention_mask = tokenizers.tokenize_prompts(["task"])

    assert input_ids.shape == attention_mask.shape == (1, width)
    assert input_ids[0, 0].item() == 9


@pytest.fixture(scope="module")
def real_tokenizer_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    download_dir = tmp_path_factory.mktemp("molmoact2-download")
    tokenizer_path = hf_hub_download(
        repo_id=_MOLMOACT2_REPOSITORY,
        filename="tokenizer.json",
        revision=_MOLMOACT2_REVISION,
        local_dir=download_dir,
    )
    tokenizer_dir = tmp_path_factory.mktemp("molmoact2-tokenizer")
    copyfile(tokenizer_path, tokenizer_dir / "tokenizer.json")
    return tokenizer_dir


@pytest.mark.integration
@pytest.mark.requires_download
def test_real_molmoact2_tokenizer_tokenizes_prompts(real_tokenizer_dir: Path) -> None:
    tokenizers = MolmoAct2Tokenizers(
        tokenizer_name_or_path=str(real_tokenizer_dir),
        max_token_len=32,
    )

    input_ids, attention_mask = tokenizers.tokenize_prompts(["pick up the cube", "move left"])
    tokenizer = tokenizers._qwen_tokenizer()

    assert input_ids.shape == attention_mask.shape == (2, 32)
    assert input_ids.dtype == torch.int64
    assert attention_mask.dtype == torch.int64
    assert tokenizer.bos_token_id is None
    assert isinstance(tokenizer.eos_token_id, int)
    assert input_ids[:, 0].tolist() == [tokenizer.eos_token_id] * 2
    assert attention_mask[:, 0].tolist() == [1, 1]
    assert tokenizers.tokenizer is tokenizers.tokenizer


@pytest.mark.integration
@pytest.mark.requires_download
def test_real_molmoact2_openvino_tokenizer_drops_output_tokens(real_tokenizer_dir: Path) -> None:
    tokenizers = MolmoAct2Tokenizers(tokenizer_name_or_path=str(real_tokenizer_dir))
    source_tokens = tokenizers._qwen_tokenizer().added_tokens_decoder
    filtered_tokens = tokenizers.tokenizer.added_tokens_decoder

    assert any(token.content.startswith("<action_") for token in source_tokens.values())
    assert any(token.content.startswith("<extra_") for token in source_tokens.values())
    assert not any(re.match(r"^<(?:action|extra)_\d+>$", token.content) for token in filtered_tokens.values())
    assert any(token.content == "<action_start>" for token in filtered_tokens.values())
    assert any(token.content == "<action_end>" for token in filtered_tokens.values())
