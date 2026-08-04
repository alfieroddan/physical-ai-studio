# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MolmoAct2 configuration dataclasses."""

from __future__ import annotations

import pytest

from physicalai.config import Config
from physicalai.policies.molmoact2.config import (
    DEFAULT_MOLMOACT2_REPO_ID,
    MOLMOACT2_FRAME_END_TOKEN_ID,
    MOLMOACT2_FRAME_START_TOKEN_ID,
    MOLMOACT2_IMAGE_COL_ID,
    MOLMOACT2_IMAGE_END_TOKEN_ID,
    MOLMOACT2_IMAGE_LOW_RES_ID,
    MOLMOACT2_IMAGE_PATCH_ID,
    MOLMOACT2_IMAGE_PLACEHOLDER_TOKEN_ID,
    MOLMOACT2_IMAGE_START_TOKEN_ID,
    MOLMOACT2_LOW_RES_IMAGE_START_TOKEN_ID,
    MolmoAct2Config,
)


class TestMolmoAct2Constants:
    def test_default_repo_id_value(self) -> None:
        assert DEFAULT_MOLMOACT2_REPO_ID == "allenai/MolmoAct2"

    def test_image_placeholder_token_id_value(self) -> None:
        assert MOLMOACT2_IMAGE_PLACEHOLDER_TOKEN_ID == 154629


class TestMolmoAct2Config:
    def test_default_config(self) -> None:
        config = MolmoAct2Config()
        assert config.model_type == "molmoact2"
        assert config.n_obs_steps == 30
        assert config.chunk_size == 30
        assert config.n_action_steps == 30
        assert config.max_action_dim == 32
        assert config.action_mode == "continuous"
        assert config.state_format == "discrete"
        assert config.image_num_pos == 729

    def test_default_tokenizer_values(self) -> None:
        config = MolmoAct2Config()
        assert config.tokenizer_name_or_path == DEFAULT_MOLMOACT2_REPO_ID
        assert config.image_placeholder_token_id == MOLMOACT2_IMAGE_PLACEHOLDER_TOKEN_ID
        assert config.tokenizer_config is None

    def test_default_image_token_ids(self) -> None:
        config = MolmoAct2Config()
        assert config.image_start_token_id == MOLMOACT2_IMAGE_START_TOKEN_ID
        assert config.image_end_token_id == MOLMOACT2_IMAGE_END_TOKEN_ID
        assert config.image_patch_id == MOLMOACT2_IMAGE_PATCH_ID
        assert config.image_col_id == MOLMOACT2_IMAGE_COL_ID
        assert config.low_res_image_start_token_id == MOLMOACT2_LOW_RES_IMAGE_START_TOKEN_ID
        assert config.image_low_res_id == MOLMOACT2_IMAGE_LOW_RES_ID
        assert config.frame_start_token_id == MOLMOACT2_FRAME_START_TOKEN_ID
        assert config.frame_end_token_id == MOLMOACT2_FRAME_END_TOKEN_ID

    def test_custom_config(self) -> None:
        config = MolmoAct2Config(
            chunk_size=10,
            n_action_steps=5,
            n_obs_steps=2,
            max_action_dim=16,
            action_mode="continuous",
        )
        assert config.chunk_size == 10
        assert config.n_action_steps == 5
        assert config.n_obs_steps == 2
        assert config.max_action_dim == 16

    def test_max_action_horizon_alias(self) -> None:
        config = MolmoAct2Config(chunk_size=42)
        assert config.max_action_horizon == 42

    def test_inheritance(self) -> None:
        config = MolmoAct2Config()
        assert isinstance(config, Config)

    def test_serialization_round_trip(self) -> None:
        config = MolmoAct2Config(chunk_size=12, n_action_steps=6, optimizer_lr=1e-4, max_action_dim=8)
        data = config.to_dict()
        assert data["chunk_size"] == 12
        assert data["optimizer_lr"] == 1e-4
        restored = MolmoAct2Config.from_dict(data)
        assert restored.chunk_size == 12
        assert restored.n_action_steps == 6
        assert restored.max_action_dim == 8
        assert restored.optimizer_lr == 1e-4

    def test_serializer_carries_tokenizer_config(self) -> None:
        tok_cfg = {"bos_token": "<|im_end|>", "pad_token": ""}
        config = MolmoAct2Config(tokenizer_config=tok_cfg)
        data = config.to_dict()
        assert data["tokenizer_config"] == tok_cfg
        restored = MolmoAct2Config.from_dict(data)
        assert restored.tokenizer_config == tok_cfg

    def test_n_action_steps_validation_below_one(self) -> None:
        with pytest.raises(ValueError, match="n_action_steps"):
            MolmoAct2Config(chunk_size=10, n_action_steps=0)

    def test_n_action_steps_exceeds_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="cannot be greater than chunk_size"):
            MolmoAct2Config(chunk_size=4, n_action_steps=8)

    def test_chunk_size_below_one(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            MolmoAct2Config(chunk_size=0)

    def test_n_obs_steps_below_one(self) -> None:
        with pytest.raises(ValueError, match="n_obs_steps"):
            MolmoAct2Config(n_obs_steps=0)

    def test_max_action_dim_below_one(self) -> None:
        with pytest.raises(ValueError, match="max_action_dim"):
            MolmoAct2Config(max_action_dim=0)

    def test_train_action_expert_only_requires_continuous(self) -> None:
        with pytest.raises(ValueError, match="train_action_expert_only"):
            MolmoAct2Config(action_mode="discrete", train_action_expert_only=True)

    def test_use_lora_defaults_off_and_train_action_expert_only_defaults_false(self) -> None:
        config = MolmoAct2Config()
        assert config.use_lora is False
        assert config.enable_lora_action_expert is False
        assert config.train_action_expert_only is False
        assert config.lora_rank == 64
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.05
        assert config.lora_bias == "none"

    def test_use_lora_incompatible_with_train_action_expert_only(self) -> None:
        with pytest.raises(ValueError, match="use_lora is incompatible with train_action_expert_only"):
            MolmoAct2Config(use_lora=True, train_action_expert_only=True)

    def test_enable_lora_action_expert_requires_use_lora(self) -> None:
        with pytest.raises(ValueError, match="enable_lora_action_expert requires use_lora"):
            MolmoAct2Config(enable_lora_action_expert=True)

    def test_lora_rank_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="lora_rank"):
            MolmoAct2Config(use_lora=True, lora_rank=0)

    def test_lora_dropout_range(self) -> None:
        with pytest.raises(ValueError, match="lora_dropout"):
            MolmoAct2Config(use_lora=True, lora_dropout=1.0)

    def test_lora_bias_must_be_valid(self) -> None:
        with pytest.raises(ValueError, match="lora_bias"):
            MolmoAct2Config(use_lora=True, lora_bias="bogus")  # type: ignore[arg-type]

    def test_use_lora_serialization_round_trip(self) -> None:
        config = MolmoAct2Config(use_lora=True, lora_rank=8, lora_alpha=4, lora_dropout=0.1)
        data = config.to_dict()
        assert data["use_lora"] is True
        assert data["lora_rank"] == 8
        restored = MolmoAct2Config.from_dict(data)
        assert restored.use_lora is True
        assert restored.lora_rank == 8

    def test_flow_matching_num_steps_validation(self) -> None:
        with pytest.raises(ValueError, match="flow_matching_num_steps"):
            MolmoAct2Config(flow_matching_num_steps=0)

    def test_flow_matching_cutoff_validation(self) -> None:
        with pytest.raises(ValueError, match="flow_matching_cutoff"):
            MolmoAct2Config(flow_matching_cutoff=1.5)

    def test_flow_matching_beta_alpha_validation(self) -> None:
        with pytest.raises(ValueError, match="flow_matching_beta_alpha"):
            MolmoAct2Config(flow_matching_beta_alpha=0.0)

    def test_flow_matching_beta_beta_validation(self) -> None:
        with pytest.raises(ValueError, match="flow_matching_beta_beta"):
            MolmoAct2Config(flow_matching_beta_beta=-1.0)

    def test_optimizer_lr_validation(self) -> None:
        with pytest.raises(ValueError, match="optimizer_lr"):
            MolmoAct2Config(optimizer_lr=0.0)

    def test_scheduler_warmup_validation(self) -> None:
        with pytest.raises(ValueError, match="scheduler_warmup_steps"):
            MolmoAct2Config(scheduler_warmup_steps=-1)

    def test_text_fields_are_flat(self) -> None:
        config = MolmoAct2Config(hidden_size=128, num_attention_heads=4, num_hidden_layers=2)
        assert config.hidden_size == 128
        assert config.num_attention_heads == 4
        assert config.num_hidden_layers == 2
        assert config.vocab_size == 154_624
        assert config.max_position_embeddings == 4096
        assert config.use_cache is True

    def test_image_num_patch_property(self) -> None:
        config = MolmoAct2Config(image_default_input_size=(28, 28), image_patch_size=14)
        assert config.image_num_patch == (2, 2)

    def test_add_action_expert_false_keeps_flat_settings(self) -> None:
        config = MolmoAct2Config(add_action_expert=False)
        assert config.action_expert_num_layers == config.num_hidden_layers

    def test_state_format_discrete_only(self) -> None:
        with pytest.raises(ValueError, match="state_format"):
            MolmoAct2Config(state_format="continuous")  # type: ignore[arg-type]
