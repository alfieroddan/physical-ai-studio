# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the MolmoAct2 Lightning policy wrapper.

Fast, self-contained tests with no external dependencies (no HuggingFace
downloads). Lazy init is exercised via ``MolmoAct2(repo_id=None)``: the
config built on that path is the default :class:`MolmoAct2Config`. The
HuggingFace-container loading path is covered by ``test_from_hf.py``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from physicalai.data import Observation
from physicalai.data.observation import ACTION
from physicalai.export import ExportBackend
from physicalai.policies import get_policy
from physicalai.policies.molmoact2 import MolmoAct2, MolmoAct2Config
from physicalai.policies.molmoact2.config import (
    DEFAULT_MOLMOACT2_REPO_ID,
    MOLMOACT2_IMAGE_PLACEHOLDER_TOKEN_ID,
)



class TestMolmoact2Registration:
    def test_get_policy_returns_instance(self) -> None:
        policy = get_policy("Molmoact2")
        assert isinstance(policy, MolmoAct2)

    def test_get_policy_case_insensitive(self) -> None:
        policy = get_policy("molmoact2")
        assert isinstance(policy, MolmoAct2)



class TestMolmoact2Policy:
    """Tests for MolmoAct2 Lightning policy wrapper (lazy / config path)."""

    def test_lazy_initialization(self) -> None:
        policy = MolmoAct2(repo_id=None)
        assert policy.model is None
        assert policy._preprocessor is None
        assert policy._postprocessor is None

    def test_repo_id_defaults_to_none(self) -> None:
        assert MolmoAct2.__init__.__kwdefaults__["repo_id"] is None

    def test_hyperparameters_saved(self) -> None:
        policy = MolmoAct2(
            repo_id=None,
            input_features=None,
            output_features=None,
            n_obs_steps=4,
            n_action_steps=2,
            chunk_size=8,
            use_random_input_noise=True,
        )
        assert policy.hparams["n_obs_steps"] == 4
        assert policy.hparams["n_action_steps"] == 2
        assert policy.hparams["chunk_size"] == 8
        assert policy.hparams["use_random_input_noise"] is True
        assert "repo_id" in policy.hparams

    def test_save_hyperparameters_ignores_config_and_load_weights(self) -> None:
        policy = MolmoAct2(repo_id=None)
        assert "config" not in policy.hparams
        assert "load_weights" not in policy.hparams

    def test_config_attribute_reflects_args(self) -> None:
        policy = MolmoAct2(
            repo_id=None,
            n_obs_steps=3,
            n_action_steps=1,
            chunk_size=5,
            use_random_input_noise=True,
        )
        assert policy.config is not None
        assert isinstance(policy.config, MolmoAct2Config)
        assert policy.config.n_obs_steps == 3
        assert policy.config.n_action_steps == 1
        assert policy.config.chunk_size == 5
        assert policy.config.use_random_input_noise is True
        assert policy.config.tokenizer_name_or_path == DEFAULT_MOLMOACT2_REPO_ID
        assert policy.config.image_placeholder_token_id == MOLMOACT2_IMAGE_PLACEHOLDER_TOKEN_ID

    def test_resolves_tokenizer_assets_lazily(self, tmp_path: Path, monkeypatch) -> None:
        policy = MolmoAct2(repo_id=None)
        tokenizer_config = {
            "extra_special_tokens": ["<im_start>", "<|image|>"],
            "model_max_length": 1010000,
        }
        monkeypatch.setattr(
            "physicalai.policies.molmoact2.policy.resolve_tokenizer_assets",
            lambda source: (str(tmp_path), tokenizer_config),
        )

        policy._ensure_tokenizer_assets()

        assert policy.config.tokenizer_name_or_path == str(tmp_path)
        assert policy.config.tokenizer_config == tokenizer_config

    def test_action_mode_override_is_applied_to_config(self) -> None:
        policy = MolmoAct2(repo_id=None, action_mode="discrete")
        assert policy.config.action_mode == "discrete"

    def test_only_one_feature_set_rejected(self) -> None:
        from physicalai.data.observation import Feature, FeatureType

        one_feature = [Feature(name="state", ftype=FeatureType.STATE, shape=(4,))]
        with pytest.raises(ValueError, match="Need both input and output"):
            MolmoAct2(repo_id=None, input_features=one_feature, output_features=None)

    @pytest.mark.parametrize("method", ["forward", "predict_action_chunk"])
    def test_methods_raise_without_model(self, method: str) -> None:
        policy = MolmoAct2(repo_id=None)
        obs = Observation(state=torch.randn(1, 4))
        with pytest.raises(ValueError, match="not initialized"):
            getattr(policy, method)(obs)

    def test_schemas_none_when_model_uninitialized(self) -> None:
        policy = MolmoAct2(repo_id=None)
        assert policy.inputs_schema is None
        assert policy.outputs_schema is None

    def test_input_features_raise_when_uninitialized(self) -> None:
        policy = MolmoAct2(
            repo_id=None,
            input_features=None,
            output_features=None,
        )
        policy.config.input_features = None
        with pytest.raises(ValueError, match="input features"):
            _ = policy.input_features

    def test_output_features_raise_when_uninitialized(self) -> None:
        policy = MolmoAct2(
            repo_id=None,
            input_features=None,
            output_features=None,
        )
        policy.config.output_features = None
        with pytest.raises(ValueError, match="output features"):
            _ = policy.output_features


class TestMolmoact2LoRAArgs:
    """Tests for the use_lora argument wiring on the policy wrapper."""

    def test_use_lora_stored_in_config_and_hparams(self) -> None:
        policy = MolmoAct2(repo_id=None, use_lora=True, lora_rank=8)
        assert policy.config.use_lora is True
        assert policy.config.lora_rank == 8
        assert policy.hparams["use_lora"] is True
        assert policy.hparams["lora_rank"] == 8

    def test_enable_lora_action_expert_stored_in_config(self) -> None:
        policy = MolmoAct2(repo_id=None, use_lora=True, enable_lora_action_expert=True)
        assert policy.config.enable_lora_action_expert is True

    def test_enable_lora_action_expert_without_use_lora_raises(self) -> None:
        with pytest.raises(ValueError, match="enable_lora_action_expert requires use_lora"):
            MolmoAct2(repo_id=None, enable_lora_action_expert=True)


class TestMolmoact2SupportedExportBackends:
    def test_returns_torch_and_openvino(self) -> None:
        backends = MolmoAct2.get_supported_export_backends()
        assert list(backends) == [ExportBackend.TORCH, ExportBackend.OPENVINO]


class TestMolmoact2ExportArgs:
    def test_export_sample_forces_fixed_tokenizer_padding(self, molmoact2_features) -> None:
        input_features, output_features = molmoact2_features

        class TestPolicy(MolmoAct2):
            @property
            def sample_input(self):
                return {"task": "sample"}

        policy = TestPolicy(repo_id=None, tokenizer_padding="longest")
        policy.config.input_features = input_features
        policy.config.output_features = output_features

        class RecordingPreprocessor(torch.nn.Module):
            padding: str | None = None

            def forward(self, batch, *, tokenizer_padding=None):
                del batch
                self.padding = tokenizer_padding
                return {"input_ids": torch.ones((1, 4), dtype=torch.int64)}

        preprocessor = RecordingPreprocessor()
        policy._preprocessor = preprocessor  # type: ignore[assignment]

        sample = policy._get_default_export_input_sample()

        assert preprocessor.padding == "max_length"
        assert policy.config.tokenizer_padding == "longest"
        assert sample is not None
        assert sample["input_ids"].shape == (1, 4)

    def test_openvino_manifest_components(self, molmoact2_features) -> None:
        input_features, output_features = molmoact2_features
        policy = MolmoAct2(repo_id=None)
        policy.config.input_features = input_features
        policy.config.output_features = output_features
        policy.config.image_processor_size = {"height": 28, "width": 28}
        policy.config.tokenizer_config = {
            "bos_token": "<|im_end|>",
            "extra_special_tokens": ["<im_start>", "<im_end>", "<|image|>"],
            "model_max_length": 1010000,
        }
        policy.model = torch.nn.Identity()  # type: ignore[assignment]
        preprocessor = torch.nn.Identity()
        preprocessor.tokenizer = SimpleNamespace(bos_token_id=11, eos_token_id=12, pad_token_id=0)
        policy._preprocessor = preprocessor  # type: ignore[assignment]

        openvino_args = policy.extra_export_args["openvino"]

        assert [spec.type for spec in openvino_args.preprocessors_specs] == [
            "molmoact2",
            "ov_tokenizer",
            "molmoact2_inputs",
        ]
        assert [spec.type for spec in openvino_args.postprocessors_specs] == ["molmoact2_postprocess"]
        assert openvino_args.export_tokenizer is True
        assert openvino_args.via_onnx is False
        assert openvino_args.outputs == [ACTION]

        raw, tokenizer, model_inputs = openvino_args.preprocessors_specs
        assert raw.flat_params["image_keys"] == ["image"]
        assert raw.flat_params["state_stats"]["q01"] == [-1.0] * 6
        assert tokenizer.flat_params == {
            "artifact": "tokenizer.xml",
        }
        assert model_inputs.flat_params["action_dim"] == 6
        assert model_inputs.flat_params["bos_token_id"] == 11
        assert model_inputs.flat_params["pad_token_id"] == 0
        assert model_inputs.flat_params["image_size"] == (28, 28)
        assert openvino_args.postprocessors_specs[0].flat_params["action_stats"]["q99"] == [1.0] * 6

