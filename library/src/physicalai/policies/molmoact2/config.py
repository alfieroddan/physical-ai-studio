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

DEFAULT_MOLMOACT2_REPO_ID = "allenai/MolmoAct2"
# The image special-token layout is shared by the supported MolmoAct2 checkpoints.
MOLMOACT2_IMAGE_START_TOKEN_ID = 154624
MOLMOACT2_IMAGE_END_TOKEN_ID = 154625
MOLMOACT2_IMAGE_PATCH_ID = 154626
MOLMOACT2_IMAGE_COL_ID = 154627
MOLMOACT2_LOW_RES_IMAGE_START_TOKEN_ID = 154628
MOLMOACT2_IMAGE_PLACEHOLDER_TOKEN_ID = 154629
MOLMOACT2_IMAGE_LOW_RES_ID = 154630
MOLMOACT2_FRAME_START_TOKEN_ID = 154631
MOLMOACT2_FRAME_END_TOKEN_ID = 154632


@dataclass
class MolmoAct2Config(Config):
    """Flat canonical configuration for the MolmoAct2 policy and model.

    Fields are intentionally owned by this single dataclass. Component
    constructors receive only the arguments they declare; they do not receive
    or mutate this policy-level config.

    Args:
        Text fields: ``hidden_size`` through ``text_attn_implementation``
            define the language transformer.
        Vision fields: ``vision_hidden_size`` through
            ``vision_attn_implementation`` define the image encoder.
        Adapter fields: ``adapter_vit_layers`` through
            ``adapter_attn_implementation`` define image pooling and
            projection into text space.
        Action-expert fields: ``action_expert_max_action_horizon`` through
            ``action_expert_causal_attn`` define the flow-matching denoiser.
        Preprocessor fields: ``num_state_tokens`` through
            ``normalization_mode`` define tokenization, image processing, and
            observation/action normalization.
        Rollout and flow-matching fields: ``n_obs_steps``, ``chunk_size``,
            ``n_action_steps``, and ``flow_matching_*`` control action chunk
            generation and training targets.
        Runtime fields: ``compile_model``, LoRA fields, and checkpoint fields
            control compilation, fine-tuning, and pretrained asset loading.

    A policy builds this config from defaults or a pretrained checkpoint, then
    applies only explicit non-``None`` constructor overrides.
    """

    model_type: str = "molmoact2"

    # Text transformer
    hidden_size: int = 2560
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 8
    head_dim: int = 128
    vocab_size: int = 154_624
    additional_vocab_size: int = 128
    qkv_bias: bool = False
    num_hidden_layers: int = 36
    intermediate_size: int = 9728
    hidden_act: str = "silu"
    max_position_embeddings: int = 16_384
    rope_theta: float = 5_000_000.0
    use_qk_norm: bool = True
    qk_norm_type: str = "qwen3"
    layer_norm_eps: float = 1e-6
    norm_after: bool = False
    use_cache: bool = True
    text_attn_implementation: str = "sdpa"

    # Vision transformer
    vision_hidden_size: int = 1152
    vision_intermediate_size: int = 4304
    vision_num_hidden_layers: int = 27
    vision_num_attention_heads: int = 16
    vision_num_key_value_heads: int = 16
    vision_head_dim: int = 72
    vision_hidden_act: str = "gelu_pytorch_tanh"
    vision_layer_norm_eps: float = 1e-6
    image_default_input_size: tuple[int, int] = (378, 378)
    image_patch_size: int = 14
    image_num_pos: int = 729
    vision_attention_dropout: float = 0.0
    vision_residual_dropout: float = 0.0
    vision_attn_implementation: str = "sdpa"

    # Vision adapter
    adapter_vit_layers: tuple[int, ...] = (-3, -9)
    adapter_pooling_attention_mask: bool = True
    adapter_hidden_size: int = 1152
    adapter_num_attention_heads: int = 16
    adapter_num_key_value_heads: int = 16
    adapter_head_dim: int = 72
    adapter_attention_dropout: float = 0.0
    adapter_residual_dropout: float = 0.0
    adapter_hidden_act: str = "silu"
    adapter_intermediate_size: int = 9728
    adapter_text_hidden_size: int = 2560
    image_feature_dropout: float = 0.0
    adapter_attn_implementation: str = "sdpa"

    # Action expert
    action_expert_max_action_horizon: int = 32
    action_expert_max_action_dim: int = 32
    action_expert_hidden_size: int = 768
    action_expert_num_layers: int = 32
    action_expert_num_heads: int = 8
    action_expert_mlp_ratio: float = 4.0
    action_expert_ffn_multiple_of: int = 256
    action_expert_timestep_embed_dim: int = 256
    action_expert_context_layer_norm: bool = True
    action_expert_qk_norm: bool = True
    action_expert_qk_norm_eps: float = 1e-6
    action_expert_rope: bool = True
    action_expert_causal_attn: bool = False

    # Preprocessor and image processor
    num_state_tokens: int = 256
    setup_type: str = ""
    control_mode: str = ""
    add_setup_tokens: bool = True
    add_control_tokens: bool = True
    tokenizer_name_or_path: str = DEFAULT_MOLMOACT2_REPO_ID
    tokenizer_revision: str | None = None
    tokenizer_max_length: int = 256
    tokenizer_padding: Literal["max_length", "longest"] = "max_length"
    tokenizer_config: dict[str, Any] | None = None
    image_processor_crop_mode: str = "resize"
    image_processor_mean: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    image_processor_std: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    image_processor_patch_size: int = 14
    image_processor_pooling_size: list[int] = field(default_factory=lambda: [2, 2])
    image_processor_size: dict[str, int] = field(default_factory=lambda: {"height": 378, "width": 378})
    image_use_col_tokens: bool = True
    use_single_crop_col_tokens: bool | None = False
    use_single_crop_start_token: bool = True
    normalization_mode: str = "QUANTILES"

    # Export
    openvino_compress_to_fp16: bool = False

    # Input and rollout structure
    n_obs_steps: int = 1
    chunk_size: int = 30
    n_action_steps: int = 30

    # Input / output features
    input_features: list[Feature] | None = field(default_factory=list)
    output_features: list[Feature] | None = field(default_factory=list)

    # Norm tag
    norm_tag: str | None = None

    # Action/state core settings
    max_action_dim: int = 32
    action_mode: Literal["continuous", "discrete", "both"] = "continuous"
    state_format: Literal["discrete"] = "discrete"

    # SO-100/101 joint frame transform (pre-#777 LeRobot calibration).
    # When enabled, joint observations/actions are mapped robot->checkpoint on
    # the way in and checkpoint->robot on the way out. Defaults reproduce the
    # LeRobot backward-compatibility correction for SO-101. These are shared by
    # the preprocessor and postprocessor, so they stay top-level.
    adapt_to_so101: bool = False
    joint_signs: list[float] = field(default_factory=lambda: [1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    joint_offsets: list[float] = field(default_factory=lambda: [0.0, 90.0, 90.0, 0.0, 0.0, 0.0])

    # Flow matching
    flow_matching_num_steps: int = 10
    # Number of independent (timestep, noise) samples drawn per training example
    # and averaged in the flow-matching loss. The action expert is re-run per
    # sample but the (expensive) VLM encoder runs once per example, so this is a
    # cheap way to reduce per-step loss variance. Matches the reference MolmoAct2
    # training recipe's default of 8.
    num_flow_timesteps: int = 8
    flow_matching_cutoff: float = 1.0
    flow_matching_time_offset: float = 0.001
    flow_matching_time_scale: float = 0.999
    flow_matching_beta_alpha: float = 1.0
    flow_matching_beta_beta: float = 1.5
    mask_action_dim_padding: bool = True
    # Start flow matching from sampled Gaussian noise instead of zeros. Kept off
    # by default so the exported graph stays deterministic and RNG-free.
    use_random_input_noise: bool = False

    # Vision/image special tokens
    image_start_token_id: int | None = MOLMOACT2_IMAGE_START_TOKEN_ID
    low_res_image_start_token_id: int | None = MOLMOACT2_LOW_RES_IMAGE_START_TOKEN_ID
    image_end_token_id: int | None = MOLMOACT2_IMAGE_END_TOKEN_ID
    image_low_res_id: int | None = MOLMOACT2_IMAGE_LOW_RES_ID
    image_placeholder_token_id: int = MOLMOACT2_IMAGE_PLACEHOLDER_TOKEN_ID
    image_patch_id: int | None = MOLMOACT2_IMAGE_PATCH_ID
    image_col_id: int | None = MOLMOACT2_IMAGE_COL_ID
    frame_start_token_id: int | None = MOLMOACT2_FRAME_START_TOKEN_ID
    frame_end_token_id: int | None = MOLMOACT2_FRAME_END_TOKEN_ID

    # Prompt and expert controls
    add_action_expert: bool = True

    # Local path to the pretrained checkpoint snapshot directory carrying
    # ``model.safetensors`` (and the processor/tokenizer assets). Populated by
    # ``build_config_from_hf_config`` so the policy can locate its weights.
    # Left ``None`` for configs built by :func:`make_molmoact2_config`.
    checkpoint_path: str | None = None

    # Runtime options
    compile_model: bool = False
    model_dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"

    # Fine-tuning controls
    train_action_expert_only: bool = False
    gradient_checkpointing: bool = False

    # LoRA fine-tuning. When ``use_lora`` is ``True``, low-rank adapters are
    # applied to the VLM (text transformer + vision backbone) linears via
    # ``peft.get_peft_model``. ``enable_lora_action_expert`` additionally
    # extends the adapter targets to the action-expert linears; it requires
    # ``use_lora`` to be ``True``. LoRA is incompatible with
    # ``train_action_expert_only`` (which freezes the VLM entirely).
    use_lora: bool = False
    enable_lora_action_expert: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: Literal["all", "lora_only", "none"] = "none"

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        self._validate_rollout_settings()
        self._validate_flow_matching_params()
        self._validate_text_settings()
        self._sync_action_expert_settings()

    @property
    def image_num_patch(self) -> tuple[int, int]:
        """Configured ``(height_patches, width_patches)`` image grid."""
        height, width = self.image_default_input_size
        return height // self.image_patch_size, width // self.image_patch_size

    @property
    def max_action_horizon(self) -> int:
        """Checkpoint-compatible alias for the generated action chunk length."""
        return self.chunk_size

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
        if self.train_action_expert_only and self.action_mode != "continuous":
            msg = "MolmoAct2 train_action_expert_only requires action_mode='continuous'."
            raise ValueError(msg)
        if self.use_lora and self.train_action_expert_only:
            msg = "MolmoAct2 use_lora is incompatible with train_action_expert_only."
            raise ValueError(msg)
        if self.enable_lora_action_expert and not self.use_lora:
            msg = "MolmoAct2 enable_lora_action_expert requires use_lora=True."
            raise ValueError(msg)
        if self.model_dtype not in {"float32", "bfloat16", "float16"}:
            msg = (
                f"Unsupported model_dtype={self.model_dtype!r}. "
                "Expected 'float32', 'bfloat16', or 'float16'."
            )
            raise ValueError(msg)
        if self.lora_rank < 1:
            msg = f"MolmoAct2 lora_rank must be >= 1, got {self.lora_rank}."
            raise ValueError(msg)
        if self.lora_dropout < 0.0 or self.lora_dropout >= 1.0:
            msg = f"MolmoAct2 lora_dropout must be in [0.0, 1.0), got {self.lora_dropout}."
            raise ValueError(msg)
        if self.lora_bias not in {"none", "all", "lora_only"}:
            msg = f"MolmoAct2 lora_bias must be one of 'none', 'all', 'lora_only', got {self.lora_bias!r}."
            raise ValueError(msg)

    def _validate_flow_matching_params(self) -> None:
        if self.flow_matching_num_steps < 1:
            msg = f"flow_matching_num_steps must be >= 1, got {self.flow_matching_num_steps}"
            raise ValueError(msg)
        if self.num_flow_timesteps < 1:
            msg = f"num_flow_timesteps must be >= 1, got {self.num_flow_timesteps}"
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

    def _validate_text_settings(self) -> None:
        if self.tokenizer_padding not in {"max_length", "longest"}:
            msg = f"tokenizer_padding must be one of 'max_length' or 'longest', got {self.tokenizer_padding!r}."
            raise ValueError(msg)
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.num_attention_heads < 1:
            msg = f"num_attention_heads must be >= 1, got {self.num_attention_heads}"
            raise ValueError(msg)
        if self.num_key_value_heads < 1:
            msg = f"num_key_value_heads must be >= 1, got {self.num_key_value_heads}"
            raise ValueError(msg)
        if self.num_state_tokens < 0:
            msg = f"num_state_tokens must be >= 0, got {self.num_state_tokens}"
            raise ValueError(msg)

    def _sync_action_expert_settings(self) -> None:
        self.action_expert_num_layers = int(self.num_hidden_layers)
        self.action_expert_max_action_horizon = int(self.chunk_size)
        self.action_expert_max_action_dim = int(self.max_action_dim)
        if self.state_format != "discrete":
            msg = "MolmoAct2 export supports only state_format='discrete'."
            raise ValueError(msg)
