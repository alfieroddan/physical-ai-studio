# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint loading regression tests for first-party policies.

These tests validate that policies reconstructed via Lightning
``load_from_checkpoint`` preserve config values that were resolved during
construction (e.g., pretrained config overrides), preventing topology drift
between the saved ``state_dict`` and the re-instantiated model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import lightning as L  # noqa: N812
import pytest
import torch
from physicalai.config.serializable import dataclass_to_dict
from physicalai.inference import InferenceModel
from physicalai.policies import ACT, Groot, Pi0
from physicalai.policies.pi05 import Pi05, Pi05Config
from physicalai.policies.smolvla import SmolVLA, SmolVLAConfig
from physicalai.export.mixin_policy import CONFIG_KEY, POLICY_NAME_KEY


def _minimal_export_stats() -> dict[str, dict[str, Any]]:
    """Return minimal dataset statistics required by export preprocessors."""
    return {
        "observation.state": {
            "name": "observation.state",
            "shape": (8,),
            "mean": [0.0] * 8,
            "std": [1.0] * 8,
            "q01": [-1.0] * 8,
            "q99": [1.0] * 8,
            "type": "STATE",
        },
        "observation.image": {
            "name": "observation.image",
            "shape": (3, 224, 224),
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "q01": [-1.0, -1.0, -1.0],
            "q99": [1.0, 1.0, 1.0],
            "type": "VISUAL",
        },
        "action": {
            "name": "action",
            "shape": (7,),
            "mean": [0.0] * 7,
            "std": [1.0] * 7,
            "q01": [-1.0] * 7,
            "q99": [1.0] * 7,
            "type": "ACTION",
        },
    }


def _write_checkpoint(checkpoint_path: Path, policy: Any) -> None:
    """Write a minimal Lightning-compatible checkpoint for load_from_checkpoint."""
    checkpoint = {
        "state_dict": policy.state_dict(),
        "hyper_parameters": dict(policy.hparams),
        "epoch": 0,
        "global_step": 0,
        "pytorch-lightning_version": L.__version__,
        "loops": {},
        "hparams_name": "kwargs",
    }
    torch.save(checkpoint, str(checkpoint_path))


def _write_lerobot_wrapper_checkpoint(checkpoint_path: Path, policy_name: str) -> None:
    """Write a minimal checkpoint for NamedLeRobotPolicy.load_from_checkpoint."""
    pytest.importorskip("lerobot", reason="LeRobot not installed")
    from lerobot.policies.factory import make_policy_config  # noqa: PLC0415
    from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: PLC0415

    config = make_policy_config(policy_name)
    # Many LeRobot configs default to empty features, but policy constructors
    # validate that at least one visual/state input and action output exist.
    config.input_features = {
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    config_dict = dataclass_to_dict(config)

    checkpoint = {
        CONFIG_KEY: config_dict,
        POLICY_NAME_KEY: policy_name,
        "hyper_parameters": {"policy_name": policy_name},
        "epoch": 0,
        "global_step": 0,
        "pytorch-lightning_version": L.__version__,
        "loops": {},
        "hparams_name": "kwargs",
    }
    torch.save(checkpoint, str(checkpoint_path))


@pytest.mark.parametrize(
    ("policy_cls", "init_kwargs"),
    [
        (ACT, {"chunk_size": 50, "n_action_steps": 25}),
        (Groot, {"chunk_size": 50, "n_action_steps": 25}),
        (Pi0, {"chunk_size": 50, "n_action_steps": 25}),
        (Pi05, {"chunk_size": 50, "n_action_steps": 25, "dataset_stats": _minimal_export_stats()}),
        (SmolVLA, {"chunk_size": 50, "n_action_steps": 25, "dataset_stats": _minimal_export_stats()}),
    ],
)
def test_first_party_policy_load_from_checkpoint_roundtrip(
    tmp_path: Path,
    policy_cls: type,
    init_kwargs: dict[str, Any],
) -> None:
    """All first-party policies should round-trip through torch export + InferenceModel."""
    source = policy_cls(**init_kwargs)
    if policy_cls in {Pi05, SmolVLA} and getattr(source, "_dataset_stats", None) is None:
        source._dataset_stats = _minimal_export_stats()  # noqa: SLF001

    export_dir = tmp_path / f"{policy_cls.__name__.lower()}_torch"
    source.export(export_dir, backend="torch")

    loaded = InferenceModel.load(export_dir)

    assert loaded.backend == "torch"
    assert loaded.policy_name == policy_cls.__name__.lower()


@pytest.mark.parametrize(
    "wrapper_cls",
    ["ACT", "Diffusion", "Groot", "PI0", "PI05", "PI0Fast", "SmolVLA", "XVLA"],
)
def test_lerobot_named_wrapper_load_from_checkpoint_roundtrip(tmp_path: Path, wrapper_cls: str) -> None:
    """Named LeRobot wrappers should load back as the same subclass."""
    pytest.importorskip("lerobot", reason="LeRobot not installed")
    from physicalai.policies import lerobot as lerobot_wrappers  # noqa: PLC0415

    if wrapper_cls in {"PI0", "PI05", "PI0Fast"} and not os.getenv("PHYSICALAI_RUN_GATED_WRAPPER_TESTS"):
        pytest.skip("requires gated Hugging Face model access; set PHYSICALAI_RUN_GATED_WRAPPER_TESTS=1 to enable")
    if wrapper_cls == "Groot":
        pytest.importorskip("flash_attn", reason="Groot wrapper requires FlashAttention runtime support")

    wrapper_type = getattr(lerobot_wrappers, wrapper_cls)

    ckpt_path = tmp_path / f"lerobot_{wrapper_cls.lower()}.ckpt"
    _write_lerobot_wrapper_checkpoint(ckpt_path, wrapper_type.POLICY_NAME)

    try:
        loaded = wrapper_type.load_from_checkpoint(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        if wrapper_cls == "Groot" and "flashattention2" in str(exc).lower():
            pytest.skip("requires FlashAttention2 runtime support for Groot wrapper in this environment")
        if wrapper_cls == "XVLA" and not os.getenv("PHYSICALAI_RUN_XVLA_WRAPPER_TESTS"):
            pytest.skip(
                "requires XVLA Florence config/assets; set PHYSICALAI_RUN_XVLA_WRAPPER_TESTS=1 to enable"
            )
        raise exc

    assert type(loaded) is wrapper_type
    assert loaded.policy_name == wrapper_type.POLICY_NAME


def test_pi05_load_from_checkpoint_preserves_resolved_config(monkeypatch, tmp_path: Path) -> None:
    """Pi05 should reload with config values resolved in the source checkpoint."""

    resolved_config = Pi05Config(
        normalization_mode="MEAN_STD",
        empty_cameras=1,
        n_action_steps=1,
    )

    def _fake_from_hf(self: Pi05, *args: object, **kwargs: object) -> tuple[Pi05Config, None, None]:
        del self, args, kwargs
        return resolved_config, None, None

    monkeypatch.setattr(Pi05, "_from_hf", _fake_from_hf)

    source = Pi05(pretrained_name_or_path="stub-repo")
    ckpt_path = tmp_path / "pi05.ckpt"
    _write_checkpoint(ckpt_path, source)

    loaded = Pi05.load_from_checkpoint(ckpt_path, map_location="cpu", weights_only=True)

    assert loaded._n_action_steps == 1
    assert loaded.config.n_action_steps == 1
    assert loaded.config.normalization_mode == "MEAN_STD"
    assert loaded.config.empty_cameras == 1


def test_smolvla_load_from_checkpoint_preserves_resolved_config(monkeypatch, tmp_path: Path) -> None:
    """SmolVLA should reload with config values resolved in the source checkpoint."""

    resolved_config = SmolVLAConfig(
        n_action_steps=1,
        num_vlm_layers=0,
        load_vlm_weights=True,
        expert_width_multiplier=0.5,
        prefix_length=0,
        vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Instruct",
    )

    def _fake_from_hf(self: SmolVLA, *args: object, **kwargs: object) -> tuple[SmolVLAConfig, None, None]:
        del self, args, kwargs
        return resolved_config, None, None

    monkeypatch.setattr(SmolVLA, "_from_hf", _fake_from_hf)

    source = SmolVLA(pretrained_name_or_path="stub-repo")
    ckpt_path = tmp_path / "smolvla.ckpt"
    _write_checkpoint(ckpt_path, source)

    loaded = SmolVLA.load_from_checkpoint(ckpt_path, map_location="cpu", weights_only=True)

    assert loaded._n_action_steps == 1
    assert loaded.config.n_action_steps == 1
    assert loaded.config.num_vlm_layers == 0
    assert loaded.config.load_vlm_weights is True
    assert loaded.config.expert_width_multiplier == 0.5
    assert loaded.config.prefix_length == 0
    assert loaded.config.vlm_model_name == "HuggingFaceTB/SmolVLM2-500M-Instruct"
