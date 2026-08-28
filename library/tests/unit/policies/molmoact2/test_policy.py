# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MolmoAct2 policy wrapper."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import lightning
import pytest
import torch

from physicalai.data import Feature, FeatureType, NormalizationParameters, Observation
from physicalai.data.dataset import Dataset
from physicalai.export import ExportBackend
from physicalai.policies import get_policy
from physicalai.policies.molmoact2 import MolmoAct2, MolmoAct2Config


def test_registration_and_lazy_initialization() -> None:
    policy = get_policy("molmoact2")

    assert isinstance(policy, MolmoAct2)
    assert policy.model is None
    assert policy._preprocessor is None
    assert policy._postprocessor is None
    assert policy.inputs_schema is None
    assert policy.outputs_schema is None


def test_private_processor_attributes_register_modules() -> None:
    policy = MolmoAct2(pretrained_name_or_path=None)
    preprocessor = torch.nn.Identity()
    postprocessor = torch.nn.Identity()

    policy._preprocessor = preprocessor  # type: ignore[assignment]
    policy._postprocessor = postprocessor  # type: ignore[assignment]

    assert policy._preprocessor is preprocessor
    assert policy._postprocessor is postprocessor
    assert "_preprocessor" in policy._modules
    assert "_postprocessor" in policy._modules


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


def test_explicit_features_override_norm_tag_features_without_inheriting_statistics(tmp_path: Path) -> None:
    input_features = [
        Feature(name="overview", ftype=FeatureType.VISUAL, shape=(3, 600, 800)),
        Feature(name="state", ftype=FeatureType.STATE, shape=(4,)),
    ]
    output_features = [Feature(name="action", ftype=FeatureType.ACTION, shape=(4,))]
    policy = MolmoAct2(pretrained_name_or_path=None, norm_tag="test")
    policy.input_features = input_features
    policy.output_features = output_features
    norm_stats = {
        "metadata_by_tag": {
            "test": {
                "camera_keys": [],
                "state_key": "observation.state",
                "state_stats": {"q01": [-1.0] * 4, "q99": [1.0] * 4},
                "action_key": "action",
                "action_stats": {"q01": [-1.0] * 4, "q99": [1.0] * 4},
                "action_horizon": 30,
                "normalize_gripper": True,
            },
        },
    }

    config = policy._convert_config({}, norm_stats, {}, tmp_path)

    assert config.input_features is not None
    assert config.output_features is not None
    assert config.input_features[0] == input_features[0]
    assert config.input_features[1].name == "state"
    assert config.input_features[1].shape == (4,)
    assert config.input_features[1].normalization_data is None
    assert config.output_features[0].name == "action"
    assert config.output_features[0].shape == (4,)
    assert config.output_features[0].normalization_data is None


def test_set_features_copies_only_requested_state_normalization(
    tiny_molmoact2_config: MolmoAct2Config,
) -> None:
    state_feature = tiny_molmoact2_config.input_features[-1]
    config_without_images = replace(tiny_molmoact2_config, input_features=[state_feature])
    policy = MolmoAct2.from_config(config_without_images).eval()
    model = policy.model
    preprocessor = policy._preprocessor
    postprocessor = policy._postprocessor
    replacement_state_stats = NormalizationParameters(q01=[-2.0] * 4, q99=[2.0] * 4)
    replacement_action_stats = NormalizationParameters(q01=[-3.0] * 4, q99=[3.0] * 4)
    input_features = [
        Feature(name="overview", ftype=FeatureType.VISUAL, shape=(3, 28, 28)),
        Feature(name="left_wrist", ftype=FeatureType.VISUAL, shape=(3, 28, 28)),
        Feature(name="right_wrist", ftype=FeatureType.VISUAL, shape=(3, 28, 28)),
        Feature(
            name="robot_state",
            ftype=FeatureType.STATE,
            shape=(4,),
            normalization_data=replacement_state_stats,
        ),
    ]
    output_features = [
        Feature(
            name="robot_action",
            ftype=FeatureType.ACTION,
            shape=(4,),
            normalization_data=replacement_action_stats,
        ),
    ]

    policy.set_features(
        input_features,
        output_features,
        copy_state_normalization=True,
    )

    assert policy.model is model
    assert policy._preprocessor is not preprocessor
    assert policy._postprocessor is not postprocessor
    assert policy._preprocessor is not None and not policy._preprocessor.training
    assert policy._postprocessor is not None and not policy._postprocessor.training
    assert policy.config is not None
    assert policy.config.input_features == policy.input_features
    assert policy.config.output_features == policy.output_features
    assert [feature.name for feature in policy.input_features or []] == [
        "overview",
        "left_wrist",
        "right_wrist",
        "robot_state",
    ]
    assert policy.input_features is not None
    assert policy.output_features is not None
    assert policy.input_features[-1].normalization_data == state_feature.normalization_data
    assert policy.output_features[0].normalization_data is replacement_action_stats


def test_set_features_copies_only_requested_action_normalization(
    tiny_molmoact2_config: MolmoAct2Config,
) -> None:
    policy = MolmoAct2.from_config(tiny_molmoact2_config)
    replacement_state_stats = NormalizationParameters(q01=[-2.0] * 4, q99=[2.0] * 4)
    replacement_action_stats = NormalizationParameters(q01=[-3.0] * 4, q99=[3.0] * 4)
    input_features = [
        Feature(name="image", ftype=FeatureType.VISUAL, shape=(3, 28, 28)),
        Feature(
            name="robot_state",
            ftype=FeatureType.STATE,
            shape=(4,),
            normalization_data=replacement_state_stats,
        ),
    ]
    output_features = [
        Feature(
            name="robot_action",
            ftype=FeatureType.ACTION,
            shape=(4,),
            normalization_data=replacement_action_stats,
        ),
    ]

    policy.set_features(
        input_features,
        output_features,
        copy_action_normalization=True,
    )

    assert policy.input_features is not None
    assert policy.output_features is not None
    assert policy.input_features[-1].normalization_data is replacement_state_stats
    assert policy.output_features[0].normalization_data == tiny_molmoact2_config.output_features[0].normalization_data


def test_set_features_rejects_incompatible_normalization_shape_atomically(
    tiny_molmoact2_config: MolmoAct2Config,
) -> None:
    policy = MolmoAct2.from_config(tiny_molmoact2_config)
    config = policy.config
    input_features = policy.input_features
    preprocessor = policy._preprocessor
    replacement_inputs = [
        Feature(name="image", ftype=FeatureType.VISUAL, shape=(3, 28, 28)),
        Feature(name="state", ftype=FeatureType.STATE, shape=(5,)),
    ]

    with pytest.raises(ValueError, match="Cannot copy STATE normalization"):
        policy.set_features(
            replacement_inputs,
            list(policy.output_features or []),
            copy_state_normalization=True,
        )

    assert policy.config is config
    assert policy.input_features is input_features
    assert policy._preprocessor is preprocessor


def test_set_features_requires_initialized_policy() -> None:
    policy = MolmoAct2(pretrained_name_or_path=None)

    with pytest.raises(TypeError, match="not initialized"):
        policy.set_features([], [])


def test_setup_replaces_eager_normalization_with_dataset_normalization(
    tiny_molmoact2_config: MolmoAct2Config,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    policy = MolmoAct2.from_config(tiny_molmoact2_config)
    model = policy.model
    dataset_stats = NormalizationParameters(q01=[-2.0] * 4, q99=[2.0] * 4)
    dataset_inputs = [
        replace(feature, normalization_data=dataset_stats)
        for feature in tiny_molmoact2_config.input_features
    ]
    dataset_outputs = [
        replace(feature, normalization_data=dataset_stats)
        for feature in tiny_molmoact2_config.output_features
    ]
    train_dataset = Mock(spec=Dataset)
    trainer = Mock()
    trainer.datamodule.train_dataset = train_dataset
    policy._trainer = trainer
    monkeypatch.setattr(policy, "_dataset_features", lambda _dataset: (dataset_inputs, dataset_outputs))

    with caplog.at_level("WARNING"):
        policy.setup("fit")

    assert "replacing them with the dataset features" in caplog.text
    assert policy.model is model
    assert policy.input_features == dataset_inputs
    assert policy.output_features == dataset_outputs
    assert policy.config is not None
    assert policy.config.input_features == dataset_inputs
    assert policy.config.output_features == dataset_outputs


def test_setup_replaces_eager_feature_contract_with_dataset_contract(
    tiny_molmoact2_config: MolmoAct2Config,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    policy = MolmoAct2.from_config(tiny_molmoact2_config)
    model = policy.model
    dataset_inputs = [
        replace(tiny_molmoact2_config.input_features[0], name="other_camera"),
        *tiny_molmoact2_config.input_features[1:],
    ]
    train_dataset = Mock(spec=Dataset)
    trainer = Mock()
    trainer.datamodule.train_dataset = train_dataset
    policy._trainer = trainer
    monkeypatch.setattr(
        policy,
        "_dataset_features",
        lambda _dataset: (dataset_inputs, tiny_molmoact2_config.output_features),
    )

    with caplog.at_level("WARNING"):
        policy.setup("fit")

    assert "replacing them with the dataset features" in caplog.text
    assert policy.model is model
    assert policy.input_features == dataset_inputs
    assert policy.output_features == tiny_molmoact2_config.output_features


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
    # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch  # Test-only trusted data.
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


def test_openvino_compression_is_used_by_export(
    tiny_molmoact2_config: MolmoAct2Config,
) -> None:
    policy = MolmoAct2(pretrained_name_or_path=None, openvino_compress_to_fp16=True)
    policy.config = tiny_molmoact2_config
    policy.input_features = tiny_molmoact2_config.input_features
    policy.output_features = tiny_molmoact2_config.output_features
    policy.model = Mock()
    policy._preprocessor = Mock()
    policy._preprocessor.tokenizer.bos_token_id = 1
    policy._preprocessor.tokenizer.pad_token_id = 0

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
