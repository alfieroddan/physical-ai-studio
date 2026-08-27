# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MolmoAct2 policy wrapper."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import lightning
import pytest
import torch

from physicalai.data import Observation
from physicalai.export import ExportBackend
from physicalai.policies import get_policy
from physicalai.policies.molmoact2 import MolmoAct2, MolmoAct2Config


def test_registration_and_lazy_initialization() -> None:
    policy = get_policy("molmoact2")

    assert isinstance(policy, MolmoAct2)
    assert policy.model is None
    assert policy.preprocessor is None
    assert policy.postprocessor is None
    assert policy.inputs_schema is None
    assert policy.outputs_schema is None


def test_public_processor_attributes_register_modules() -> None:
    policy = MolmoAct2(pretrained_name_or_path=None)
    preprocessor = torch.nn.Identity()
    postprocessor = torch.nn.Identity()

    policy.preprocessor = preprocessor  # type: ignore[assignment]
    policy.postprocessor = postprocessor  # type: ignore[assignment]

    assert policy.preprocessor is preprocessor
    assert policy.postprocessor is postprocessor
    assert "preprocessor" in policy._modules
    assert "postprocessor" in policy._modules


@pytest.mark.parametrize("method", ["forward", "predict_action_chunk", "compute_val_loss"])
def test_model_methods_require_initialization(method: str) -> None:
    policy = MolmoAct2(pretrained_name_or_path=None)

    with pytest.raises((TypeError, RuntimeError), match="not initialized"):
        getattr(policy, method)(Observation(state=torch.zeros(1, 4)))


def test_invalid_lora_options() -> None:
    with pytest.raises(ValueError, match="requires use_lora"):
        MolmoAct2(pretrained_name_or_path=None, enable_lora_action_expert=True)
    with pytest.raises(ValueError, match="incompatible"):
        MolmoAct2(pretrained_name_or_path=None, use_lora=True, train_action_head_only=True)


def test_from_config_uses_resolved_config(monkeypatch: pytest.MonkeyPatch) -> None:
    config = MolmoAct2Config(n_action_steps=3, chunk_size=5, use_random_input_noise=True)
    initialized: list[MolmoAct2Config] = []

    def initialize(policy: MolmoAct2, policy_config: MolmoAct2Config) -> None:
        policy.config = policy_config
        initialized.append(policy_config)

    monkeypatch.setattr(MolmoAct2, "_initialize_from_config", initialize)

    policy = MolmoAct2.from_config(config, compile_model=True, optimizer_lr=2e-5)

    assert initialized == [config]
    assert policy.pretrained_name_or_path is None
    assert (policy.n_action_steps, policy.chunk_size) == (3, 5)
    assert policy.compile_model is True
    assert policy.optimizer_lr == 2e-5


def test_load_from_checkpoint_restores_config_and_weights(
    tiny_molmoact2_config: MolmoAct2Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = MolmoAct2.from_config(tiny_molmoact2_config)
    checkpoint = {
        "state_dict": policy.state_dict(),
        "pytorch-lightning_version": lightning.__version__,
        "hyper_parameters": dict(policy.hparams),
    }
    policy.on_save_checkpoint(checkpoint)
    checkpoint_path = tmp_path / "molmoact2.ckpt"
    torch.save(checkpoint, checkpoint_path)

    def fail_pretrained_resolution(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Lightning checkpoint loading must not resolve pretrained assets")

    monkeypatch.setattr(MolmoAct2, "_from_hf", fail_pretrained_resolution)

    restored = MolmoAct2.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    assert restored.config == tiny_molmoact2_config
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


@pytest.mark.parametrize("policy_config", [None, "invalid"])
def test_load_checkpoint_requires_policy_config(policy_config: object) -> None:
    policy = MolmoAct2(pretrained_name_or_path=None)

    with pytest.raises(TypeError, match="valid policy_config"):
        policy.on_load_checkpoint({"policy_config": policy_config})


def test_restore_checkpoint_rejects_different_initialized_config(
    tiny_molmoact2_config: MolmoAct2Config,
) -> None:
    policy = MolmoAct2.from_config(tiny_molmoact2_config)
    different_config = replace(tiny_molmoact2_config, n_action_steps=1)

    with pytest.raises(ValueError, match="does not match"):
        policy._restore_policy_config(different_config.to_dict())


def test_runtime_options_are_policy_owned() -> None:
    policy = MolmoAct2(
        pretrained_name_or_path=None,
        compile_model=True,
        openvino_compress_to_fp16=True,
        gradient_checkpointing=True,
        optimizer_lr=2e-5,
    )

    assert policy.compile_model is True
    assert policy.openvino_compress_to_fp16 is True
    assert policy.gradient_checkpointing is True
    assert policy.optimizer_lr == 2e-5
    assert "compile_model" not in policy.hparams


def test_openvino_compression_warns_for_non_float32_config() -> None:
    policy = MolmoAct2(pretrained_name_or_path=None, openvino_compress_to_fp16=True)

    with pytest.warns(UserWarning, match="only converts FP32 constants"):
        policy._validate_export_settings(MolmoAct2Config(model_dtype="bfloat16"))


def test_openvino_compression_is_used_by_export(
    tiny_molmoact2_config: MolmoAct2Config,
) -> None:
    policy = MolmoAct2(pretrained_name_or_path=None, openvino_compress_to_fp16=True)
    policy.config = tiny_molmoact2_config
    policy.input_features = tiny_molmoact2_config.input_features
    policy.output_features = tiny_molmoact2_config.output_features
    policy.model = Mock()
    policy.preprocessor = Mock()
    policy.preprocessor.tokenizer.bos_token_id = 1
    policy.preprocessor.tokenizer.pad_token_id = 0

    export_args = policy.extra_export_args[ExportBackend.OPENVINO]

    assert export_args.compress_to_fp16 is True


def test_model_modifications_are_applied_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = MolmoAct2(
        pretrained_name_or_path=None,
        compile_model=True,
        gradient_checkpointing=True,
        use_lora=True,
        train_action_head_only=False,
    )
    model = Mock()
    monkeypatch.setattr(policy, "_require_model", lambda: model)

    policy._apply_model_modifications()

    model.enable_gradient_checkpointing.assert_called_once_with()
    model.enable_lora.assert_called_once_with(enable_action_expert=False)
    model.enable_compile.assert_called_once_with()


def test_supported_export_backends() -> None:
    assert MolmoAct2.get_supported_export_backends() == [ExportBackend.TORCH, ExportBackend.OPENVINO]
