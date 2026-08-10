# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MolmoAct2 HuggingFace loading helpers.

All HuggingFace Hub interactions are mocked; no network access is required.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from physicalai.data.observation import Feature, FeatureType, NormalizationParameters
from physicalai.policies.molmoact2.config import MOLMOACT2_IMAGE_PLACEHOLDER_TOKEN_ID, MolmoAct2Config
from physicalai.policies.molmoact2.from_hf import (
    _load_tokenizer_config,
    _resolve_feature_overrides,
    MolmoAct2Snapshot,
    SNAPSHOT_ALLOW_PATTERNS,
    TOKENIZER_ALLOW_PATTERNS,
    build_config_from_hf_config,
    download_policy_artifacts_from_hub,
    load_hf_pretrained_container,
    resolve_tokenizer_assets,
)


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
    def test_loads_local_file(self, mock_hf_repo: Path) -> None:
        result = _load_tokenizer_config(str(mock_hf_repo))
        assert result["tokenizer_class"] == "Qwen2Tokenizer"
        assert result["extra_special_tokens"] == ["<im_start>", "<im_end>", "<|image|>"]

    def test_resolves_tokenizer_only_snapshot(self, mock_hf_repo: Path, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def _fake_snapshot_download(repo_id: str, **kwargs: object) -> str:
            captured["repo_id"] = repo_id
            captured.update(kwargs)
            return str(mock_hf_repo)

        from physicalai.policies.molmoact2 import from_hf as from_hf_module

        monkeypatch.setattr(from_hf_module, "snapshot_download", _fake_snapshot_download)

        tokenizer_dir, tokenizer_config = resolve_tokenizer_assets("allenai/MolmoAct2")

        assert captured == {
            "repo_id": "allenai/MolmoAct2",
            "allow_patterns": TOKENIZER_ALLOW_PATTERNS,
        }
        assert tokenizer_dir == str(mock_hf_repo)
        assert tokenizer_config["model_max_length"] == 1010000

    def test_rejects_missing_tokenizer_config(self, tmp_path: Path) -> None:
        (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")

        with pytest.raises(FileNotFoundError, match="tokenizer_config.json"):
            resolve_tokenizer_assets(tmp_path)


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

    def test_tokenizer_name_or_path_uses_checkpoint(self) -> None:
        config = _build_config(repo_id="allenai/MolmoAct2-LIBERO")
        assert config.tokenizer_name_or_path == "/tmp/checkpoint"

    def test_tokenizer_name_or_path_uses_local_checkpoint_without_repo(self) -> None:
        config = _build_config(repo_id=None)
        assert config.tokenizer_name_or_path == "/tmp/checkpoint"

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

    def test_action_mode_override_is_applied_before_validation(self) -> None:
        hf_config = {**_make_hf_config(), "action_mode": "discrete"}

        config = build_config_from_hf_config(
            hf_config,
            checkpoint_path="/tmp/checkpoint",
            train_action_expert_only=True,
            action_mode="continuous",
        )

        assert config.action_mode == "continuous"
        assert config.train_action_expert_only is True

    def test_chunk_size_defaults_to_thirty_when_not_overridden(self) -> None:
        """When the HF config omits chunk_size, the override default of 30 applies."""
        hf_config = _make_hf_config()
        del hf_config["chunk_size"]
        config = build_config_from_hf_config(
            hf_config,
            checkpoint_path="/tmp/checkpoint",
        )
        assert config.chunk_size == 30

    def test_checkpoint_action_horizon_sets_chunk_and_return_steps(self) -> None:
        hf_config = _make_hf_config()
        hf_config["max_action_horizon"] = 10
        hf_config.pop("chunk_size")
        hf_config.pop("n_action_steps")

        config = build_config_from_hf_config(
            hf_config,
            checkpoint_path="/tmp/checkpoint",
        )

        assert config.chunk_size == 10
        assert config.n_action_steps == 10
        assert config.action_expert_max_action_horizon == 10

    def test_explicit_action_step_overrides_beat_checkpoint_horizon(self) -> None:
        hf_config = _make_hf_config()
        hf_config["max_action_horizon"] = 10

        config = build_config_from_hf_config(
            hf_config,
            checkpoint_path="/tmp/checkpoint",
            n_action_steps=4,
        )

        assert config.chunk_size == 10
        assert config.n_action_steps == 4

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

    def test_builds_features_from_norm_stats(self, molmoact2_features: tuple[list, list]) -> None:
        config = _build_config(norm_stats=_make_norm_stats(), norm_tag="libero")
        assert any(f.name == "state" for f in config.input_features)
        assert any(f.name == "action" for f in config.output_features)
        assert config.setup_type == "tabletop"
        assert config.control_mode == "joint"

    def test_feature_overrides_win_by_name(self, molmoact2_features: tuple[list, list]) -> None:
        inputs, outputs = molmoact2_features
        override_state = [f for f in inputs if f.ftype.value == "STATE"][0]
        config = _build_config(
            norm_stats=_make_norm_stats(),
            norm_tag="libero",
            input_features=[override_state],
            output_features=outputs,
        )
        named = {f.name: f for f in config.input_features if f.name is not None}
        assert "state" in named


class TestResolveFeatureOverrides:
    def test_renamed_camera_does_not_preserve_pretrained_normalization(self, caplog) -> None:
        normalization = NormalizationParameters(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            q01=[-1.0, -1.0, -1.0],
            q99=[1.0, 1.0, 1.0],
        )
        pretrained = [
            Feature(
                name="wrist_image",
                ftype=FeatureType.VISUAL,
                shape=(3, 28, 28),
                normalization_data=normalization,
            ),
        ]
        override = [
            Feature(name="image2", ftype=FeatureType.VISUAL, shape=(3, 28, 28)),
        ]

        with caplog.at_level(logging.WARNING, logger="physicalai.policies.molmoact2.from_hf"):
            resolved = _resolve_feature_overrides(pretrained, override)

        resolved_by_name = {feature.name: feature for feature in resolved}
        assert resolved_by_name["wrist_image"].normalization_data is normalization
        assert resolved_by_name["image2"].normalization_data is None
        assert "image2" in caplog.text
        assert "normalization will not be copied" in caplog.text

    def test_reshaped_camera_logs_warning(self, caplog) -> None:
        pretrained = [
            Feature(name="wrist_image", ftype=FeatureType.VISUAL, shape=(3, 28, 28)),
        ]
        override = [
            Feature(name="wrist_image", ftype=FeatureType.VISUAL, shape=(3, 32, 32)),
        ]

        with caplog.at_level(logging.WARNING, logger="physicalai.policies.molmoact2.from_hf"):
            resolved = _resolve_feature_overrides(pretrained, override)

        assert resolved[0].shape == (3, 32, 32)
        assert "wrist_image" in caplog.text
        assert "changed shape" in caplog.text


class TestLoadHfPretrainedContainer:
    def test_local_checkpoint_populates_fields(self, mock_hf_repo: Path) -> None:
        container = load_hf_pretrained_container(str(mock_hf_repo))
        assert isinstance(container, MolmoAct2Snapshot)
        assert container.repo_id is None
        assert container.tokenizer_config is not None
        assert container.tokenizer_config["tokenizer_class"] == "Qwen2Tokenizer"
        assert container.hf_config["model_type"] == "molmoact2"
        assert container.norm_stats is not None
        assert container.processor_config is not None

    def test_hub_checkpoint_sets_repo_id(self, mock_hf_repo: Path, patch_download_policy_artifacts) -> None:
        revision = "1dbc166cf8765166998eff31ade2eb64c8a40076"
        container = load_hf_pretrained_container("allenai/MolmoAct2-LIBERO", revision=revision)
        assert container.repo_id == "allenai/MolmoAct2-LIBERO"
        assert container.tokenizer_revision == revision
        assert container.tokenizer_config is not None

    def test_local_missing_weights_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        (empty / "config.json").write_text("{}", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="model.safetensors"):
            load_hf_pretrained_container(str(empty))

    def test_local_missing_tokenizer_raises(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "model.safetensors").touch()

        with pytest.raises(FileNotFoundError, match="tokenizer.json"):
            load_hf_pretrained_container(str(checkpoint))


class TestDownloadPolicyArtifacts:
    def test_downloads_allowlisted_snapshot(self, mock_hf_repo: Path, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def _fake_snapshot_download(repo_id: str, **kwargs: object) -> str:
            captured["repo_id"] = repo_id
            captured.update(kwargs)
            return str(mock_hf_repo)

        from physicalai.policies.molmoact2 import from_hf as from_hf_module

        monkeypatch.setattr(from_hf_module, "snapshot_download", _fake_snapshot_download)

        config, weights, processor, processor_dir, norm_stats = download_policy_artifacts_from_hub(
            "allenai/MolmoAct2-LIBERO",
            hub_kwargs={"revision": "main"},
            preprocessor_filename="processor_config.json",
            norm_stats_filename="norm_stats.json",
        )

        assert captured == {
            "repo_id": "allenai/MolmoAct2-LIBERO",
            "allow_patterns": SNAPSHOT_ALLOW_PATTERNS,
            "revision": "main",
        }
        assert config == mock_hf_repo / "config.json"
        assert weights == mock_hf_repo / "model.safetensors.index.json"
        assert processor == mock_hf_repo / "processor_config.json"
        assert processor_dir == mock_hf_repo
        assert norm_stats == mock_hf_repo / "norm_stats.json"

    def test_rejects_snapshot_without_tokenizer(self, tmp_path: Path, monkeypatch) -> None:
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "model.safetensors").touch()

        from physicalai.policies.molmoact2 import from_hf as from_hf_module

        monkeypatch.setattr(from_hf_module, "snapshot_download", lambda *args, **kwargs: str(snapshot))

        with pytest.raises(FileNotFoundError, match="tokenizer.json"):
            download_policy_artifacts_from_hub("allenai/MolmoAct2-LIBERO")
