# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the MolmoAct2 Lightning policy wrapper.

Fast, self-contained tests with no external dependencies (no HuggingFace
downloads). Lazy init is exercised via ``MolmoAct2(repo_id=None)``: the
config built on that path is the default :class:`MolmoAct2Config`. The
HuggingFace-container loading path is covered by ``test_from_hf.py``.
"""

from __future__ import annotations

import pytest
import torch

from physicalai.data import Observation
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

    def test_default_repo_id_constant(self) -> None:
        assert MolmoAct2.__init__.__defaults__ is not None
        repo_index = MolmoAct2.__init__.__code__.co_varnames.index("repo_id")
        default = MolmoAct2.__init__.__defaults__[repo_index - 1]
        assert default == DEFAULT_MOLMOACT2_REPO_ID

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

    def test_action_mode_non_continuous_rejected(self) -> None:
        with pytest.raises(ValueError, match="continous action mode"):
            MolmoAct2(repo_id=None, action_mode="discrete")

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
    def test_returns_torch_only(self) -> None:
        backends = MolmoAct2.get_supported_export_backends()
        assert list(backends) == [ExportBackend.TORCH]
