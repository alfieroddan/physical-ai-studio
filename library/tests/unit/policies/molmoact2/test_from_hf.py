# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MolmoAct2 HuggingFace loading helpers.

All HuggingFace Hub interactions are mocked; no network access is required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from huggingface_hub.errors import RemoteEntryNotFoundError

from physicalai.policies.molmoact2.config import (
    DEFAULT_MOLMOACT2_REPO_ID,
    MOLMOACT2_IMAGE_PLACEHOLDER_TOKEN_ID,
    MolmoAct2Config,
)
from physicalai.policies.molmoact2.from_hf import (
    _load_tokenizer_config,
    build_config_from_hf_config,
    load_hf_pretrained_container,
)
from physicalai.utils.hf_utils import HuggingfacePolicyContainer


def _remote_not_found(msg: str) -> RemoteEntryNotFoundError:
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(404, request=request)
    return RemoteEntryNotFoundError(msg, response=response)


def _make_hf_config() -> dict[str, Any]:
    return {
        "model_type": "molmoact2",
        "vit_config": {"image_default_input_size": [28, 28]},
        "text_config": {"hidden_size": 64, "num_attention_heads": 4, "num_hidden_layers": 1},
        "action_expert_config": {"hidden_size": 64, "num_layers": 1},
        "max_action_dim": 4,
        "chunk_size": 4,
        "n_action_steps": 4,
    }


def _make_norm_stats() -> dict[str, Any]:
    return {
        "metadata_by_tag": {
            "libero": {
                "camera_keys": ["observation.images.image"],
                "state_key": "observation.state",
                "state_stats": {"mean": [0.0] * 6, "std": [1.0] * 6, "q01": [-1.0] * 6, "q99": [1.0] * 6},
                "action_key": "action",
                "action_stats": {"mean": [0.0] * 6, "std": [1.0] * 6, "q01": [-1.0] * 6, "q99": [1.0] * 6},
                "setup_type": "tabletop",
                "control_mode": "joint",
            }
        },
    }


def _build_config(**overrides: Any):
    """Build a config from the test HF config with chunk_size==n_action_steps==4."""
    defaults: dict[str, Any] = {
        "checkpoint_path": "/tmp/checkpoint",
        "n_action_steps": 4,
        "norm_tag": None,
        "norm_stats": None,
    }
    defaults.update(overrides)
    return build_config_from_hf_config(_make_hf_config(), **defaults)


class TestLoadTokenizerConfig:
    def test_prefers_local_file(self, mock_hf_repo: Path) -> None:
        result = _load_tokenizer_config(str(mock_hf_repo), repo_id="some/repo")
        assert result is not None
        assert result["tokenizer_class"] == "Qwen2Tokenizer"

    def test_falls_back_to_repo(self, tmp_path: Path, mock_hf_repo: Path, monkeypatch) -> None:

        captured: list[str] = []

        def _fake_download(repo_id: str, filename: str, **kwargs: Any) -> str:
            captured.append(repo_id)
            if filename == "tokenizer_config.json" and mock_hf_repo.joinpath(filename).is_file():
                return str(mock_hf_repo / filename)
            msg = f"{filename} not found"
            raise _remote_not_found(msg)

        from physicalai.policies.molmoact2 import from_hf as from_hf_module

        monkeypatch.setattr(from_hf_module, "hf_hub_download", _fake_download)

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = _load_tokenizer_config(str(empty_dir), repo_id="custom/repo")
        assert result is not None
        assert "custom/repo" in captured
        assert DEFAULT_MOLMOACT2_REPO_ID not in captured

    def test_falls_back_to_default_repo(self, tmp_path: Path, mock_hf_repo: Path, monkeypatch) -> None:

        attempts: list[str] = []
        first_repo_only = {"custom/repo": True}

        def _fake_download(repo_id: str, filename: str, **kwargs: Any) -> str:
            attempts.append(repo_id)
            if first_repo_only.get(repo_id) and filename == "tokenizer_config.json":
                first_repo_only[repo_id] = False
                msg = f"{filename} missing in {repo_id}"
                raise _remote_not_found(msg)
            if filename == "tokenizer_config.json" and mock_hf_repo.joinpath(filename).is_file():
                return str(mock_hf_repo / filename)
            msg = f"{filename} not found"
            raise _remote_not_found(msg)

        from physicalai.policies.molmoact2 import from_hf as from_hf_module

        monkeypatch.setattr(from_hf_module, "hf_hub_download", _fake_download)

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = _load_tokenizer_config(str(empty_dir), repo_id="custom/repo")
        assert result is not None
        assert attempts == ["custom/repo", DEFAULT_MOLMOACT2_REPO_ID]

    def test_returns_none_when_unavailable(self, tmp_path: Path, monkeypatch) -> None:

        def _fake_download(repo_id: str, filename: str, **kwargs: Any) -> str:
            msg = f"{filename} not found in {repo_id}"
            raise _remote_not_found(msg)

        from physicalai.policies.molmoact2 import from_hf as from_hf_module

        monkeypatch.setattr(from_hf_module, "hf_hub_download", _fake_download)

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = _load_tokenizer_config(str(empty_dir), repo_id="custom/repo")
        assert result is None


class TestBuildConfigFromHfConfig:
    def test_hardcodes_image_placeholder_token_id(self) -> None:
        config = _build_config()
        assert config.image_placeholder_token_id == MOLMOACT2_IMAGE_PLACEHOLDER_TOKEN_ID

    def test_checkpoint_image_token_ids_override_defaults(self) -> None:
        config = build_config_from_hf_config(
            {
                **_make_hf_config(),
                "image_start_token_id": 10,
                "image_end_token_id": 11,
                "image_patch_id": 12,
                "image_col_id": 13,
                "low_res_image_start_token_id": 14,
                "image_low_res_id": 15,
                "frame_start_token_id": 16,
                "frame_end_token_id": 17,
            },
            checkpoint_path="/tmp/checkpoint",
        )
        assert config.image_start_token_id == 10
        assert config.image_end_token_id == 11
        assert config.image_patch_id == 12
        assert config.image_col_id == 13
        assert config.low_res_image_start_token_id == 14
        assert config.image_low_res_id == 15
        assert config.frame_start_token_id == 16
        assert config.frame_end_token_id == 17

    def test_tokenizer_name_or_path_uses_repo_id(self) -> None:
        config = _build_config(repo_id="allenai/MolmoAct2-LIBERO")
        assert config.tokenizer_name_or_path == "allenai/MolmoAct2-LIBERO"

    def test_tokenizer_name_or_path_falls_back_to_default(self) -> None:
        config = _build_config(repo_id=None)
        assert config.tokenizer_name_or_path == DEFAULT_MOLMOACT2_REPO_ID

    def test_carries_tokenizer_config(self) -> None:
        tok_cfg = {"bos_token": "<|im_end|>"}
        config = _build_config(tokenizer_config=tok_cfg)
        assert config.tokenizer_config == tok_cfg

    def test_chunk_size_override_wins_over_hf_config(self) -> None:
        """Caller-provided chunk_size overrides the value in the HF config payload.

        n_action_steps must be set consistently since MolmoAct2Config validates
        that n_action_steps <= chunk_size.
        """
        hf_config = _make_hf_config()
        assert hf_config["chunk_size"] == 4
        config = build_config_from_hf_config(
            hf_config,
            checkpoint_path="/tmp/checkpoint",
            chunk_size=12,
            n_action_steps=4,
        )
        assert config.chunk_size == 12
        assert config.n_action_steps == 4

    def test_chunk_size_defaults_to_thirty_when_not_overridden(self) -> None:
        """When the HF config omits chunk_size, the override default of 30 applies."""
        hf_config = _make_hf_config()
        del hf_config["chunk_size"]
        config = build_config_from_hf_config(
            hf_config,
            checkpoint_path="/tmp/checkpoint",
        )
        assert config.chunk_size == 30

    def test_use_random_input_noise_override_wins_over_hf_config(self) -> None:
        """Caller-provided use_random_input_noise overrides the HF config value."""
        hf_config = _make_hf_config()
        hf_config["use_random_input_noise"] = True
        config = build_config_from_hf_config(
            hf_config,
            checkpoint_path="/tmp/checkpoint",
            use_random_input_noise=False,
        )
        assert config.use_random_input_noise is False

    def test_use_random_input_noise_defaults_to_false_when_not_overridden(self) -> None:
        """When the HF config omits use_random_input_noise, the default of False applies."""
        hf_config = _make_hf_config()
        config = build_config_from_hf_config(
            hf_config,
            checkpoint_path="/tmp/checkpoint",
        )
        assert config.use_random_input_noise is False

    def test_requires_checkpoint_path(self) -> None:
        with pytest.raises(ValueError, match="checkpoint_path is required"):
            build_config_from_hf_config(_make_hf_config(), checkpoint_path=None)

    def test_builds_features_from_norm_stats(
        self, molmoact2_features: tuple[list, list]
    ) -> None:
        config = _build_config(norm_stats=_make_norm_stats(), norm_tag="libero")
        assert any(f.name == "state" for f in config.input_features)
        assert any(f.name == "action" for f in config.output_features)
        assert config.setup_type == "tabletop"
        assert config.control_mode == "joint"

    def test_feature_overrides_win_by_name(
        self, molmoact2_features: tuple[list, list]
    ) -> None:
        inputs, outputs = molmoact2_features
        override_state = [
            f for f in inputs if f.ftype.value == "STATE"
        ][0]
        config = _build_config(
            norm_stats=_make_norm_stats(),
            norm_tag="libero",
            input_features=[override_state],
            output_features=outputs,
        )
        named = {f.name: f for f in config.input_features if f.name is not None}
        assert "state" in named


class TestLoadHfPretrainedContainer:
    def test_local_checkpoint_populates_fields(self, mock_hf_repo: Path) -> None:
        container = load_hf_pretrained_container(str(mock_hf_repo))
        assert isinstance(container, HuggingfacePolicyContainer)
        assert container.repo_id is None
        assert container.tokenizer_config is not None
        assert container.tokenizer_config["tokenizer_class"] == "Qwen2Tokenizer"
        assert container.hf_config["model_type"] == "molmoact2"
        assert container.norm_stats is not None
        assert container.processor_config is not None

    def test_hub_checkpoint_sets_repo_id(
        self, mock_hf_repo: Path, patch_download_policy_artifacts
    ) -> None:
        def _fake_ensure(repo_id, checkpoint_location, hub_kwargs=None) -> None:
            tgt = Path(checkpoint_location) / "processor_config.json"
            if not tgt.exists():
                src = mock_hf_repo / "processor_config.json"
                tgt.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

        with patch(
            "physicalai.policies.molmoact2.from_hf._ensure_processor_assets_downloaded",
            side_effect=_fake_ensure,
        ):
            container = load_hf_pretrained_container("allenai/MolmoAct2-LIBERO")
        assert container.repo_id == "allenai/MolmoAct2-LIBERO"
        assert container.tokenizer_config is not None

    def test_local_missing_weights_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        (empty / "config.json").write_text("{}", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="model.safetensors"):
            load_hf_pretrained_container(str(empty))


class TestEnsureProcessorAssetsDownloaded:
    def test_only_downloads_processor_config(
        self, mock_hf_repo: Path, tmp_path: Path, monkeypatch
    ) -> None:
        from physicalai.policies.molmoact2 import from_hf as from_hf_module

        captured: list[str] = []

        def _fake_download(repo_id: str, filename: str, **kwargs: Any) -> str:
            captured.append(filename)
            return str(mock_hf_repo / filename)

        monkeypatch.setattr(from_hf_module, "hf_hub_download", _fake_download)

        target = tmp_path / "snapshot"
        target.mkdir()
        from_hf_module._ensure_processor_assets_downloaded(
            "allenai/MolmoAct2", str(target), hub_kwargs={}
        )
        assert captured == ["processor_config.json"]
        assert (target / "processor_config.json").is_file()
