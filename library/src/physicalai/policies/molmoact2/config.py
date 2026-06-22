# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration for MolmoAct2 model.

This module provides a dataclass configuration for the MolmoAct2
vision-language-action model in physicalai format.

Example (CLI):
    physicalai fit --config configs/physicalai/molmoact2.yaml

Example (API):
    >>> from physicalai.policies.molmoact2 import MolmoAct2Config
    >>> config = MolmoAct2Config()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from physicalai.config import Config

if TYPE_CHECKING:
    from physicalai.data import Feature


@dataclass
class MolmoAct2VitConfig(Config):
    """Vision transformer component configuration for MolmoAct2."""

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    image_default_input_size: tuple[int, int] = (378, 378)
    image_patch_size: int = 14
    image_num_pos: int = 577
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    initializer_range: float = 0.02
    float32_attention: bool = True
    attn_implementation: str = "eager"

    @property
    def image_num_patch(self) -> tuple[int, int]:
        """Return (height_patches, width_patches) for configured input size."""
        h, w = self.image_default_input_size
        return h // self.image_patch_size, w // self.image_patch_size


@dataclass
class MolmoAct2AdapterConfig(Config):
    """Vision adapter/pooling component configuration for MolmoAct2."""

    vit_layers: tuple[int, ...] = (-3, -9)
    pooling_attention_mask: bool = False
    hidden_size: int = 1152
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    float32_attention: bool = True
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    hidden_act: str = "silu"
    intermediate_size: int = 18_944
    text_hidden_size: int = 3584
    image_feature_dropout: float = 0.0
    initializer_range: float = 0.02
    attn_implementation: str = "eager"


@dataclass
class MolmoAct2TextConfig(Config):
    """Text transformer component configuration for MolmoAct2."""

    hidden_size: int = 3584
    num_attention_heads: int = 28
    num_key_value_heads: int | None = 4
    head_dim: int = 128
    vocab_size: int = 152_064
    additional_vocab_size: int = 128
    qkv_bias: bool = True
    num_hidden_layers: int = 48
    intermediate_size: int = 18_944
    hidden_act: str = "silu"
    embedding_dropout: float = 0.0
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    max_position_embeddings: int = 4096
    rope_theta: float = 1_000_000.0
    rope_scaling: dict[str, Any] | None = None
    rope_scaling_layers: list[int] | None = None
    use_qk_norm: bool = False
    qk_norm_type: str = "olmo"
    layer_norm_eps: float = 1e-6
    norm_after: bool = False
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = False
    attn_implementation: str = "eager"

    def __post_init__(self) -> None:
        """Validate and normalize text-attention head settings.

        Raises:
            ValueError: If any attention-head setting is invalid.
        """
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.num_attention_heads < 1:
            msg = f"num_attention_heads must be >= 1, got {self.num_attention_heads}"
            raise ValueError(msg)
        if self.num_key_value_heads < 1:
            msg = f"num_key_value_heads must be >= 1, got {self.num_key_value_heads}"
            raise ValueError(msg)


@dataclass
class MolmoAct2ActionExpertConfig(Config):
    """Action expert component configuration for MolmoAct2."""

    max_action_horizon: int = 32
    max_action_dim: int = 32
    hidden_size: int = 1024
    num_layers: int = 32
    num_heads: int = 16
    mlp_ratio: float = 8.0 / 3.0
    ffn_multiple_of: int = 256
    timestep_embed_dim: int = 256
    dropout: float = 0.0
    attn_dropout: float = 0.0
    context_layer_norm: bool = True
    qk_norm: bool = True
    qk_norm_eps: float = 1e-6
    rope: bool = True
    causal_attn: bool = False


@dataclass
class MolmoAct2Config(Config):
    """Top-level configuration for MolmoAct2 with split component sub-configs."""

    # Component sub-configs
    vit_config: MolmoAct2VitConfig = field(default_factory=MolmoAct2VitConfig)
    adapter_config: MolmoAct2AdapterConfig = field(default_factory=MolmoAct2AdapterConfig)
    text_config: MolmoAct2TextConfig = field(default_factory=MolmoAct2TextConfig)
    action_expert_config: MolmoAct2ActionExpertConfig | None = field(default_factory=MolmoAct2ActionExpertConfig)

    # Input and rollout structure
    n_obs_steps: int = 30
    chunk_size: int = 30
    n_action_steps: int = 30

    # Input / output features
    input_features: list[Feature] = field(default_factory=list)
    output_features: list[Feature] = field(default_factory=list)

    # Norm tag
    norm_tag: str | None = None

    # Action/state core settings
    max_action_dim: int = 32
    action_mode: Literal["continuous", "discrete", "both"] = "both"
    state_format: Literal["discrete"] = "discrete"

    # Flow matching
    flow_matching_num_steps: int = 10
    flow_matching_cutoff: float = 1.0
    flow_matching_time_offset: float = 0.001
    flow_matching_time_scale: float = 0.999
    flow_matching_beta_alpha: float = 1.0
    flow_matching_beta_beta: float = 1.5
    mask_action_dim_padding: bool = True

    # Depth reasoning
    enable_depth_reasoning: bool = False
    depth_mode: int = 2
    num_depth_codes: int = 100
    action_expert_depth_gate: bool = False
    action_expert_depth_gate_per_layer: bool = False
    action_expert_depth_gate_init_bias: float = -4.0

    # Vision/image special tokens
    image_start_token_id: int | None = None
    low_res_image_start_token_id: int | None = None
    image_end_token_id: int | None = None
    image_low_res_id: int | None = None
    image_patch_id: int | None = None
    image_col_id: int | None = None
    frame_start_token_id: int | None = None
    frame_end_token_id: int | None = None
    use_frame_special_tokens: bool = True

    # Action tokenization
    action_output_token_id: int | None = None
    action_start_token_id: int | None = None
    action_end_token_id: int | None = None
    action_token_start_id: int | None = None
    num_action_tokens: int = 0

    # Depth tokenization
    depth_output_token_id: int | None = None
    depth_start_token_id: int | None = None
    depth_end_token_id: int | None = None
    depth_token_start_id: int | None = None
    num_depth_tokens: int = 0

    # State tokenization
    state_start_token_id: int | None = None
    state_end_token_id: int | None = None
    state_token_start_id: int | None = None
    num_state_tokens: int = 0

    # Prompt and expert controls
    add_setup_tokens: bool = True
    add_control_tokens: bool = True
    add_action_expert: bool = True
    setup_type: str = ""
    control_mode: str = ""

    # Initialization and assets
    norm_stats_filename: str = "norm_stats.json"
    tokenizer_name_or_path: str | None = None
    processor_assets_path: str | None = None
    initializer_range: float = 0.02

    # Runtime options
    compile_model: bool = False

    @property
    def max_action_horizon(self) -> int:
        """Alias used by the upstream HF config schema."""
        return self.chunk_size

    @property
    def num_attention_heads(self) -> int:
        """Expose text attention heads for compatibility with HF-style access."""
        return self.text_config.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        """Expose text key-value heads for compatibility with HF-style access.

        Raises:
            ValueError: If ``text_config.num_key_value_heads`` is unset.
        """
        if self.text_config.num_key_value_heads is None:
            msg = "text_config.num_key_value_heads must be set before access."
            raise ValueError(msg)
        return self.text_config.num_key_value_heads

    @property
    def head_dim(self) -> int:
        """Expose text attention head dimension for compatibility."""
        return self.text_config.head_dim

    @property
    def num_hidden_layers(self) -> int:
        """Expose text depth for compatibility with HF-style access."""
        return self.text_config.num_hidden_layers

    @property
    def hidden_size(self) -> int:
        """Expose text hidden size for compatibility with HF-style access."""
        return self.text_config.hidden_size

    @property
    def vocab_size(self) -> int:
        """Expose text vocabulary size for compatibility with HF-style access."""
        return self.text_config.vocab_size

    @property
    def max_position_embeddings(self) -> int:
        """Expose text max positions for compatibility with HF-style access."""
        return self.text_config.max_position_embeddings

    @property
    def image_num_patch(self) -> tuple[int, int]:
        """Expose image patch grid via nested ViT config."""
        return self.vit_config.image_num_patch

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        self._validate_rollout_settings()
        self._validate_flow_matching_settings()
        self._validate_depth_and_token_settings()
        self._sync_action_expert_settings()

    def _validate_rollout_settings(self) -> None:
        if self.chunk_size < 1:
            msg = f"chunk_size must be >= 1, got {self.chunk_size}"
            raise ValueError(msg)
        if self.n_action_steps < 1:
            msg = f"n_action_steps must be >= 1, got {self.n_action_steps}"
            raise ValueError(msg)
        if self.n_action_steps > self.chunk_size:
            msg = f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            raise ValueError(msg)
        if self.n_obs_steps < 1:
            msg = f"n_obs_steps must be >= 1, got {self.n_obs_steps}"
            raise ValueError(msg)
        if self.max_action_dim < 1:
            msg = f"max_action_dim must be >= 1, got {self.max_action_dim}"
            raise ValueError(msg)

    def _validate_flow_matching_settings(self) -> None:
        if self.flow_matching_num_steps < 1:
            msg = f"flow_matching_num_steps must be >= 1, got {self.flow_matching_num_steps}"
            raise ValueError(msg)
        if not 0.0 <= self.flow_matching_cutoff <= 1.0:
            msg = f"flow_matching_cutoff must be in [0.0, 1.0], got {self.flow_matching_cutoff}"
            raise ValueError(msg)
        if not 0.0 <= self.flow_matching_time_offset <= 1.0:
            msg = f"flow_matching_time_offset must be in [0.0, 1.0], got {self.flow_matching_time_offset}"
            raise ValueError(msg)
        if not 0.0 <= self.flow_matching_time_scale <= 1.0:
            msg = f"flow_matching_time_scale must be in [0.0, 1.0], got {self.flow_matching_time_scale}"
            raise ValueError(msg)
        if self.flow_matching_beta_alpha <= 0.0:
            msg = f"flow_matching_beta_alpha must be > 0.0, got {self.flow_matching_beta_alpha}"
            raise ValueError(msg)
        if self.flow_matching_beta_beta <= 0.0:
            msg = f"flow_matching_beta_beta must be > 0.0, got {self.flow_matching_beta_beta}"
            raise ValueError(msg)

    def _validate_depth_and_token_settings(self) -> None:
        if self.depth_mode < 0:
            msg = f"depth_mode must be >= 0, got {self.depth_mode}"
            raise ValueError(msg)
        if self.num_depth_codes < 1:
            msg = f"num_depth_codes must be >= 1, got {self.num_depth_codes}"
            raise ValueError(msg)
        if self.num_action_tokens < 0:
            msg = f"num_action_tokens must be >= 0, got {self.num_action_tokens}"
            raise ValueError(msg)
        if self.num_depth_tokens < 0:
            msg = f"num_depth_tokens must be >= 0, got {self.num_depth_tokens}"
            raise ValueError(msg)
        if self.num_state_tokens < 0:
            msg = f"num_state_tokens must be >= 0, got {self.num_state_tokens}"
            raise ValueError(msg)

    def _sync_action_expert_settings(self) -> None:
        if not self.add_action_expert:
            self.action_expert_config = None
            return
        if self.action_expert_config is None:
            self.action_expert_config = MolmoAct2ActionExpertConfig()
        self.action_expert_config.max_action_horizon = int(self.chunk_size)
        self.action_expert_config.max_action_dim = int(self.max_action_dim)
        if self.state_format != "discrete":
            msg = "MolmoAct2 export supports only state_format='discrete'."
            raise ValueError(msg)
