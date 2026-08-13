# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for MolmoAct2 unit tests.

All fixtures are self-contained: no HuggingFace downloads, no full 7B model
construction. The HF Hub is mocked with a local snapshot directory built under
``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from physicalai.data.observation import Feature, FeatureType, NormalizationParameters
from physicalai.policies.molmoact2.config import (
    MOLMOACT2_IMAGE_PLACEHOLDER_TOKEN_ID,
    MolmoAct2Config,
)


@pytest.fixture
def tiny_molmoact2_config(mock_hf_repo: Path) -> MolmoAct2Config:
    """A minimal config suitable for instantiating model components in tests."""
    tokenizer_config = json.loads((mock_hf_repo / "tokenizer_config.json").read_text(encoding="utf-8"))
    return MolmoAct2Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        vocab_size=100,
        additional_vocab_size=10,
        num_hidden_layers=1,
        intermediate_size=128,
        max_position_embeddings=64,
        vision_hidden_size=64,
        vision_intermediate_size=128,
        vision_num_hidden_layers=1,
        vision_num_attention_heads=4,
        vision_num_key_value_heads=4,
        vision_head_dim=16,
        image_default_input_size=(28, 28),
        image_patch_size=14,
        image_num_pos=4,
        adapter_hidden_size=64,
        adapter_num_attention_heads=4,
        adapter_num_key_value_heads=4,
        adapter_head_dim=16,
        adapter_intermediate_size=128,
        adapter_text_hidden_size=64,
        action_expert_hidden_size=64,
        action_expert_num_heads=4,
        action_expert_mlp_ratio=2.0,
        action_expert_ffn_multiple_of=16,
        action_expert_timestep_embed_dim=16,
        n_obs_steps=2,
        chunk_size=4,
        n_action_steps=2,
        max_action_dim=4,
        image_placeholder_token_id=MOLMOACT2_IMAGE_PLACEHOLDER_TOKEN_ID,
        tokenizer_name_or_path=str(mock_hf_repo),
        tokenizer_config=tokenizer_config,
        model_dtype="float32",
    )


@pytest.fixture
def molmoact2_features() -> tuple[list[Feature], list[Feature]]:
    """Small input/output feature schemas for a single-camera, 6-dim action task."""
    input_features = [
        Feature(
            name="image",
            ftype=FeatureType.VISUAL,
            shape=(3, 28, 28),
        ),
        Feature(
            name="state",
            ftype=FeatureType.STATE,
            shape=(6,),
            normalization_data=NormalizationParameters(
                mean=[0.0] * 6,
                std=[1.0] * 6,
                q01=[-1.0] * 6,
                q99=[1.0] * 6,
            ),
        ),
    ]
    output_features = [
        Feature(
            name="action",
            ftype=FeatureType.ACTION,
            shape=(6,),
            normalization_data=NormalizationParameters(
                mean=[0.0] * 6,
                std=[1.0] * 6,
                q01=[-1.0] * 6,
                q99=[1.0] * 6,
            ),
        ),
    ]
    return input_features, output_features


_TOKENIZER_CONFIG_PAYLOAD: dict[str, Any] = {
    "add_prefix_space": False,
    "backend": "tokenizers",
    "bos_token": "<|im_end|>",
    "clean_up_tokenization_spaces": False,
    "eos_token": "<|im_end|>",
    "errors": "replace",
    "extra_special_tokens": ["<im_start>", "<im_end>", "<|image|>"],
    "model_max_length": 1010000,
    "pad_token": "",
    "tokenizer_class": "Qwen2Tokenizer",
    "unk_token": None,
}

_HF_CONFIG_PAYLOAD: dict[str, Any] = {
    "model_type": "molmoact2",
    "vit_config": {"image_default_input_size": [28, 28]},
    "text_config": {"hidden_size": 64, "num_attention_heads": 4, "num_hidden_layers": 1},
    "action_expert_config": {"hidden_size": 64, "num_layers": 1},
    "image_placeholder_token_id": MOLMOACT2_IMAGE_PLACEHOLDER_TOKEN_ID,
}

_NORM_STATS_PAYLOAD: dict[str, Any] = {
    "metadata_by_tag": {
        "libero": {
            "camera_keys": ["observation.images.image"],
            "state_key": "observation.state",
            "state_stats": {"mean": [0.0] * 6, "std": [1.0] * 6, "q01": [-1.0] * 6, "q99": [1.0] * 6},
            "action_key": "action",
            "action_stats": {"mean": [0.0] * 6, "std": [1.0] * 6, "q01": [-1.0] * 6, "q99": [1.0] * 6},
            "setup_type": " tabletop ",
            "control_mode": "joint",
        }
    },
}

_PROCESSOR_CONFIG_PAYLOAD: dict[str, Any] = {
    "image_processor": {
        "size": {"height": 28, "width": 28},
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "patch_size": 14,
        "pooling_size": [2, 2],
    },
}


def _write_minimal_tokenizer_json(path: Path) -> None:
    """Write a minimal ``tokenizer.json`` with a tiny vocab and the placeholder added token."""
    vocab = {chr(i): i for i in range(32, 64)}
    payload: dict = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": MOLMOACT2_IMAGE_PLACEHOLDER_TOKEN_ID,
                "content": "<|image|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
        ],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "post_processor": None,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": True,
            "vocab": vocab,
            "merges": [],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def mock_hf_repo(tmp_path: Path) -> Path:
    """Create a local directory mimicking a HuggingFace repo snapshot.

    Contains ``config.json``, ``norm_stats.json``, ``processor_config.json``,
    ``tokenizer_config.json``, ``tokenizer.json`` (a minimal Qwen2 vocab), and
    an empty ``model.safetensors`` index so the loader does not try to fetch
    real weights.
    """
    repo_dir = tmp_path / "mock_repo"
    repo_dir.mkdir()

    (repo_dir / "config.json").write_text(json.dumps(_HF_CONFIG_PAYLOAD), encoding="utf-8")
    (repo_dir / "norm_stats.json").write_text(json.dumps(_NORM_STATS_PAYLOAD), encoding="utf-8")
    (repo_dir / "processor_config.json").write_text(json.dumps(_PROCESSOR_CONFIG_PAYLOAD), encoding="utf-8")
    (repo_dir / "tokenizer_config.json").write_text(json.dumps(_TOKENIZER_CONFIG_PAYLOAD), encoding="utf-8")
    _write_minimal_tokenizer_json(repo_dir / "tokenizer.json")

    index_payload = {"metadata": {"total_size": 0}, "weight_map": {}}
    (repo_dir / "model.safetensors.index.json").write_text(json.dumps(index_payload), encoding="utf-8")

    return repo_dir


@pytest.fixture
def patch_hf_hub_download(monkeypatch, mock_hf_repo: Path):
    """Patch ``huggingface_hub.hf_hub_download`` to return files from ``mock_hf_repo``."""
    from huggingface_hub.errors import RemoteEntryNotFoundError

    def _fake_download(repo_id: str, filename: str, **kwargs: Any) -> str:  # noqa: ANN401
        local = mock_hf_repo / filename
        if local.is_file():
            return str(local)
        msg = f"{filename} not found in mock repo {repo_id}"
        raise RemoteEntryNotFoundError(msg)

    import physicalai.policies.molmoact2.processors.tokenizers as tokenizers_module
    import physicalai.policies.molmoact2.from_hf as from_hf_module

    monkeypatch.setattr(tokenizers_module, "hf_hub_download", _fake_download)
    monkeypatch.setattr(from_hf_module, "hf_hub_download", _fake_download)


@pytest.fixture
def patch_download_policy_artifacts(monkeypatch, mock_hf_repo: Path):
    """Patch ``download_policy_artifacts_from_hub`` to return mock repo paths."""

    def _fake_download(
        repo_id: str,
        *,
        hub_kwargs: dict[str, object] | None = None,
        config_filename: str = "config.json",
        weights_filename: str = "model.safetensors",
        preprocessor_filename: str = "processor_config.json",
        norm_stats_filename: str | None = None,
        download_preprocessor_state_files: bool = True,
    ) -> tuple[Path, Path, Path | None, Path | None, Path | None]:
        config_file = mock_hf_repo / config_filename
        weights_file = mock_hf_repo / "model.safetensors.index.json"
        preprocessor_candidate = mock_hf_repo / preprocessor_filename
        preprocessor_file: Path | None = (
            preprocessor_candidate if preprocessor_candidate.is_file() else None
        )
        preprocessor_dir = mock_hf_repo if preprocessor_file is not None else None
        norm_stats_file = (mock_hf_repo / norm_stats_filename) if norm_stats_filename else None
        if norm_stats_file is not None and not norm_stats_file.is_file():
            norm_stats_file = None
        return config_file, weights_file, preprocessor_file, preprocessor_dir, norm_stats_file

    import physicalai.policies.molmoact2.from_hf as from_hf_module

    monkeypatch.setattr(from_hf_module, "download_policy_artifacts_from_hub", _fake_download)
    return _fake_download


@pytest.fixture
def small_batch_dict() -> dict[str, Any]:
    """A minimal batch dict with one image, one state, and one task string."""
    return {
        "state": torch.zeros(2, 6),
        "task": ["pick up the block", "place it down"],
        "images.image": torch.zeros(2, 3, 28, 28),
    }
