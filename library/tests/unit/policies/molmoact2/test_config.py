# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MolmoAct2 configuration dataclasses."""

from __future__ import annotations

import pytest

from physicalai.config import Config
from physicalai.policies.molmoact2.config import (
    DEFAULT_MOLMOACT2_REPO_ID,
    MOLMOACT2_IMAGE_PLACEHOLDER_TOKEN_ID,
    MolmoAct2ActionExpertConfig,
    MolmoAct2AdapterConfig,
    MolmoAct2Config,
    MolmoAct2ImageProcessorConfig,
    MolmoAct2ProcessorConfig,
    MolmoAct2TextConfig,
    MolmoAct2VitConfig,
    MolmoAct2VideoProcessorConfig,
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

    def test_default_tokenizer_values(self) -> None:
        config = MolmoAct2Config()
        assert config.tokenizer_name_or_path == DEFAULT_MOLMOACT2_REPO_ID
        assert config.image_placeholder_token_id == MOLMOACT2_IMAGE_PLACEHOLDER_TOKEN_ID
        assert config.tokenizer_config is None

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

    def test_depth_mode_validation(self) -> None:
        with pytest.raises(ValueError, match="depth_mode"):
            MolmoAct2Config(depth_mode=-1)

    def test_num_depth_codes_validation(self) -> None:
        with pytest.raises(ValueError, match="num_depth_codes"):
            MolmoAct2Config(num_depth_codes=0)

    def test_num_action_tokens_validation(self) -> None:
        with pytest.raises(ValueError, match="num_action_tokens"):
            MolmoAct2Config(num_action_tokens=-1)

    def test_optimizer_lr_validation(self) -> None:
        with pytest.raises(ValueError, match="optimizer_lr"):
            MolmoAct2Config(optimizer_lr=0.0)

    def test_scheduler_warmup_validation(self) -> None:
        with pytest.raises(ValueError, match="scheduler_warmup_steps"):
            MolmoAct2Config(scheduler_warmup_steps=-1)

    def test_text_config_property_aliases(self) -> None:
        text = MolmoAct2TextConfig(hidden_size=128, num_attention_heads=4, num_hidden_layers=2)
        config = MolmoAct2Config(text_config=text)
        assert config.hidden_size == 128
        assert config.num_attention_heads == 4
        assert config.num_hidden_layers == 2
        assert config.vocab_size == text.vocab_size
        assert config.max_position_embeddings == text.max_position_embeddings
        assert config.use_cache is text.use_cache

    def test_image_num_patch_property(self) -> None:
        vit = MolmoAct2VitConfig(image_default_input_size=(28, 28), image_patch_size=14)
        config = MolmoAct2Config(vit_config=vit)
        assert config.image_num_patch == (2, 2)

    def test_add_action_expert_false_clears_config(self) -> None:
        config = MolmoAct2Config(add_action_expert=False)
        assert config.action_expert_config is None

    def test_state_format_discrete_only(self) -> None:
        with pytest.raises(ValueError, match="state_format"):
            MolmoAct2Config(state_format="continuous")  # type: ignore[arg-type]


class TestMolmoAct2TextConfig:
    def test_default_config(self) -> None:
        config = MolmoAct2TextConfig()
        assert config.hidden_size == 3584
        assert config.num_hidden_layers == 48
        assert config.num_attention_heads == 28
        assert config.head_dim == 128
        assert config.vocab_size == 152_064

    def test_default_key_value_heads_resolved(self) -> None:
        config = MolmoAct2TextConfig(num_key_value_heads=None)
        assert config.num_key_value_heads == config.num_attention_heads

    def test_invalid_num_attention_heads(self) -> None:
        with pytest.raises(ValueError, match="num_attention_heads"):
            MolmoAct2TextConfig(num_attention_heads=0)

    def test_invalid_num_key_value_heads(self) -> None:
        with pytest.raises(ValueError, match="num_key_value_heads"):
            MolmoAct2TextConfig(num_key_value_heads=0)


class TestMolmoAct2VitConfig:
    def test_image_num_patch(self) -> None:
        config = MolmoAct2VitConfig(image_default_input_size=(56, 84), image_patch_size=14)
        assert config.image_num_patch == (4, 6)


class TestMolmoAct2ActionExpertConfig:
    def test_default_config(self) -> None:
        config = MolmoAct2ActionExpertConfig()
        assert config.max_action_horizon == 32
        assert config.max_action_dim == 32
        assert config.num_layers == 32


class TestMolmoAct2ProcessorConfig:
    def test_default_config(self) -> None:
        config = MolmoAct2ProcessorConfig()
        assert config.processor_class == "MolmoAct2Processor"
        assert config.image_use_col_tokens is True
        assert isinstance(config.image_processor, MolmoAct2ImageProcessorConfig)
        assert isinstance(config.video_processor, MolmoAct2VideoProcessorConfig)

    def test_coerce_from_dict(self) -> None:
        payload = {
            "processor_class": "MolmoAct2Processor",
            "image_processor": {"size": {"height": 7, "width": 7}},
            "video_processor": {"num_frames": 3},
        }
        config = MolmoAct2ProcessorConfig.from_dict(payload)
        assert isinstance(config.image_processor, MolmoAct2ImageProcessorConfig)
        assert config.image_processor.size["height"] == 7
        assert isinstance(config.video_processor, MolmoAct2VideoProcessorConfig)
        assert config.video_processor.num_frames == 3
