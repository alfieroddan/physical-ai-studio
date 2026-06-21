# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 policy implementation."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch

from physicalai.data.observation import Feature, Observation
from physicalai.policies.base import Policy

from .config import (
    MolmoAct2ActionExpertConfig,
    MolmoAct2AdapterConfig,
    MolmoAct2Config,
    MolmoAct2TextConfig,
    MolmoAct2VitConfig,
)
from .from_hf import build_config_from_hf_config, load_hf_pretrained_container
from .model import MolmoAct2Model
from .processors import make_molmoact2_preprocessors

if TYPE_CHECKING:
    from .processors import MolmoAct2Postprocessor, MolmoAct2Preprocessor


class MolmoAct2(Policy):
    """MolmoAct2 Policy."""

    def __init__(  # noqa: PLR0913
        self,
        # Input / output features
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        # location download model weights from hf or load from local dir
        hf_repo_id_or_pretrained_path: str | Path | None = None,
        # norm tag - for pretrained models to look for normalisation in json
        norm_tag: str | None = None,
        # Input and rollout structure
        n_obs_steps: int = 30,
        chunk_size: int = 30,
        n_action_steps: int = 30,
        # Action/state core settings
        max_action_dim: int = 32,
        action_mode: Literal["continuous", "discrete", "both"] = "both",
        state_format: Literal["discrete"] = "discrete",
        *,
        # Vision transformer component
        vit_hidden_size: int = 1152,
        vit_intermediate_size: int = 4304,
        vit_num_hidden_layers: int = 27,
        vit_num_attention_heads: int = 16,
        vit_num_key_value_heads: int = 16,
        vit_head_dim: int = 72,
        vit_hidden_act: str = "gelu_pytorch_tanh",
        vit_layer_norm_eps: float = 1e-6,
        vit_image_default_input_size: tuple[int, int] = (378, 378),
        vit_image_patch_size: int = 14,
        vit_image_num_pos: int = 577,
        vit_attention_dropout: float = 0.0,
        vit_residual_dropout: float = 0.0,
        vit_initializer_range: float = 0.02,
        vit_float32_attention: bool = True,
        vit_attn_implementation: str = "eager",
        # Vision adapter component
        adapter_vit_layers: tuple[int, ...] = (-3, -9),
        adapter_pooling_attention_mask: bool = False,
        adapter_hidden_size: int = 1152,
        adapter_num_attention_heads: int = 16,
        adapter_num_key_value_heads: int = 16,
        adapter_head_dim: int = 72,
        adapter_float32_attention: bool = True,
        adapter_attention_dropout: float = 0.0,
        adapter_residual_dropout: float = 0.0,
        adapter_hidden_act: str = "silu",
        adapter_intermediate_size: int = 18_944,
        adapter_text_hidden_size: int = 3584,
        adapter_image_feature_dropout: float = 0.0,
        adapter_initializer_range: float = 0.02,
        adapter_attn_implementation: str = "eager",
        # Text transformer component
        text_hidden_size: int = 3584,
        text_num_attention_heads: int = 28,
        text_num_key_value_heads: int | None = 4,
        text_head_dim: int = 128,
        text_vocab_size: int = 152_064,
        text_additional_vocab_size: int = 128,
        text_qkv_bias: bool = True,
        text_num_hidden_layers: int = 48,
        text_intermediate_size: int = 18_944,
        text_hidden_act: str = "silu",
        text_embedding_dropout: float = 0.0,
        text_attention_dropout: float = 0.0,
        text_residual_dropout: float = 0.0,
        text_max_position_embeddings: int = 4096,
        text_rope_theta: float = 1_000_000.0,
        text_rope_scaling: dict[str, Any] | None = None,
        text_rope_scaling_layers: list[int] | None = None,
        text_use_qk_norm: bool = False,
        text_qk_norm_type: str = "olmo",
        text_layer_norm_eps: float = 1e-6,
        text_norm_after: bool = False,
        text_initializer_range: float = 0.02,
        text_use_cache: bool = True,
        text_tie_word_embeddings: bool = False,
        text_attn_implementation: str = "eager",
        # Action expert component
        action_expert_max_action_horizon: int = 32,
        action_expert_max_action_dim: int = 32,
        action_expert_hidden_size: int = 1024,
        action_expert_num_layers: int = 32,
        action_expert_num_heads: int = 16,
        action_expert_mlp_ratio: float = 8.0 / 3.0,
        action_expert_ffn_multiple_of: int = 256,
        action_expert_timestep_embed_dim: int = 256,
        action_expert_dropout: float = 0.0,
        action_expert_attn_dropout: float = 0.0,
        action_expert_context_layer_norm: bool = True,
        action_expert_qk_norm: bool = True,
        action_expert_qk_norm_eps: float = 1e-6,
        action_expert_rope: bool = True,
        action_expert_causal_attn: bool = False,
        # Flow matching
        flow_matching_num_steps: int = 10,
        flow_matching_cutoff: float = 1.0,
        flow_matching_time_offset: float = 0.001,
        flow_matching_time_scale: float = 0.999,
        flow_matching_beta_alpha: float = 1.0,
        flow_matching_beta_beta: float = 1.5,
        mask_action_dim_padding: bool = True,
        # Depth reasoning
        enable_depth_reasoning: bool = False,
        depth_mode: int = 2,
        num_depth_codes: int = 100,
        action_expert_depth_gate: bool = False,
        action_expert_depth_gate_per_layer: bool = False,
        action_expert_depth_gate_init_bias: float = -4.0,
        # Vision/image special tokens
        image_start_token_id: int | None = None,
        low_res_image_start_token_id: int | None = None,
        image_end_token_id: int | None = None,
        image_low_res_id: int | None = None,
        image_patch_id: int | None = None,
        image_col_id: int | None = None,
        frame_start_token_id: int | None = None,
        frame_end_token_id: int | None = None,
        use_frame_special_tokens: bool = True,
        # Action tokenization
        action_output_token_id: int | None = None,
        action_start_token_id: int | None = None,
        action_end_token_id: int | None = None,
        action_token_start_id: int | None = None,
        num_action_tokens: int = 0,
        # Depth tokenization
        depth_output_token_id: int | None = None,
        depth_start_token_id: int | None = None,
        depth_end_token_id: int | None = None,
        depth_token_start_id: int | None = None,
        num_depth_tokens: int = 0,
        # State tokenization
        state_start_token_id: int | None = None,
        state_end_token_id: int | None = None,
        state_token_start_id: int | None = None,
        num_state_tokens: int = 0,
        # Prompt and expert controls
        add_setup_tokens: bool = True,
        add_control_tokens: bool = True,
        add_action_expert: bool = True,
        # Initialization and assets
        norm_stats_filename: str = "norm_stats.json",
        initializer_range: float = 0.02,
        # Runtime options
        compile_model: bool = False,
    ) -> None:
        """Initialize MolmoAct2 policy.

        Raises:
            ValueError: If required features are missing or pretrained norm tag is not provided.
        """
        super().__init__(n_action_steps=n_action_steps)

        # Check input / output features
        if not input_features or not output_features:
            msg_str = "Model requires input and output features."
            raise ValueError(msg_str)

        self.hf_container = None
        if hf_repo_id_or_pretrained_path is not None:
            # check norm tag exists, otherwise can't load normalisation stats
            if not norm_tag:
                msg_str = "If loading from HuggingFace, norm_tag is required to load stats from norm_stats.json."
                raise ValueError(msg_str)
            self.hf_container = load_hf_pretrained_container(
                hf_repo_id_or_pretrained_path,
                norm_stats_filename=norm_stats_filename,
            )
            self.config = build_config_from_hf_config(
                self.hf_container.hf_config,
                norm_stats=self.hf_container.norm_stats,
                input_features=input_features,
                output_features=output_features,
                norm_tag=norm_tag,
                n_obs_steps=n_obs_steps,
                chunk_size=chunk_size,
                n_action_steps=n_action_steps,
                max_action_dim=max_action_dim,
            )
        else:
            self.config = self._build_config_from_init_args_only(
                vit_hidden_size=vit_hidden_size,
                vit_intermediate_size=vit_intermediate_size,
                vit_num_hidden_layers=vit_num_hidden_layers,
                vit_num_attention_heads=vit_num_attention_heads,
                vit_num_key_value_heads=vit_num_key_value_heads,
                vit_head_dim=vit_head_dim,
                vit_hidden_act=vit_hidden_act,
                vit_layer_norm_eps=vit_layer_norm_eps,
                vit_image_default_input_size=vit_image_default_input_size,
                vit_image_patch_size=vit_image_patch_size,
                vit_image_num_pos=vit_image_num_pos,
                vit_attention_dropout=vit_attention_dropout,
                vit_residual_dropout=vit_residual_dropout,
                vit_initializer_range=vit_initializer_range,
                vit_float32_attention=vit_float32_attention,
                vit_attn_implementation=vit_attn_implementation,
                adapter_vit_layers=adapter_vit_layers,
                adapter_pooling_attention_mask=adapter_pooling_attention_mask,
                adapter_hidden_size=adapter_hidden_size,
                adapter_num_attention_heads=adapter_num_attention_heads,
                adapter_num_key_value_heads=adapter_num_key_value_heads,
                adapter_head_dim=adapter_head_dim,
                adapter_float32_attention=adapter_float32_attention,
                adapter_attention_dropout=adapter_attention_dropout,
                adapter_residual_dropout=adapter_residual_dropout,
                adapter_hidden_act=adapter_hidden_act,
                adapter_intermediate_size=adapter_intermediate_size,
                adapter_text_hidden_size=adapter_text_hidden_size,
                adapter_image_feature_dropout=adapter_image_feature_dropout,
                adapter_initializer_range=adapter_initializer_range,
                adapter_attn_implementation=adapter_attn_implementation,
                text_hidden_size=text_hidden_size,
                text_num_attention_heads=text_num_attention_heads,
                text_num_key_value_heads=text_num_key_value_heads,
                text_head_dim=text_head_dim,
                text_vocab_size=text_vocab_size,
                text_additional_vocab_size=text_additional_vocab_size,
                text_qkv_bias=text_qkv_bias,
                text_num_hidden_layers=text_num_hidden_layers,
                text_intermediate_size=text_intermediate_size,
                text_hidden_act=text_hidden_act,
                text_embedding_dropout=text_embedding_dropout,
                text_attention_dropout=text_attention_dropout,
                text_residual_dropout=text_residual_dropout,
                text_max_position_embeddings=text_max_position_embeddings,
                text_rope_theta=text_rope_theta,
                text_rope_scaling=text_rope_scaling,
                text_rope_scaling_layers=text_rope_scaling_layers,
                text_use_qk_norm=text_use_qk_norm,
                text_qk_norm_type=text_qk_norm_type,
                text_layer_norm_eps=text_layer_norm_eps,
                text_norm_after=text_norm_after,
                text_initializer_range=text_initializer_range,
                text_use_cache=text_use_cache,
                text_tie_word_embeddings=text_tie_word_embeddings,
                text_attn_implementation=text_attn_implementation,
                action_expert_max_action_horizon=action_expert_max_action_horizon,
                action_expert_max_action_dim=action_expert_max_action_dim,
                action_expert_hidden_size=action_expert_hidden_size,
                action_expert_num_layers=action_expert_num_layers,
                action_expert_num_heads=action_expert_num_heads,
                action_expert_mlp_ratio=action_expert_mlp_ratio,
                action_expert_ffn_multiple_of=action_expert_ffn_multiple_of,
                action_expert_timestep_embed_dim=action_expert_timestep_embed_dim,
                action_expert_dropout=action_expert_dropout,
                action_expert_attn_dropout=action_expert_attn_dropout,
                action_expert_context_layer_norm=action_expert_context_layer_norm,
                action_expert_qk_norm=action_expert_qk_norm,
                action_expert_qk_norm_eps=action_expert_qk_norm_eps,
                action_expert_rope=action_expert_rope,
                action_expert_causal_attn=action_expert_causal_attn,
                n_obs_steps=n_obs_steps,
                chunk_size=chunk_size,
                n_action_steps=n_action_steps,
                max_action_dim=max_action_dim,
                action_mode=action_mode,
                state_format=state_format,
                flow_matching_num_steps=flow_matching_num_steps,
                flow_matching_cutoff=flow_matching_cutoff,
                flow_matching_time_offset=flow_matching_time_offset,
                flow_matching_time_scale=flow_matching_time_scale,
                flow_matching_beta_alpha=flow_matching_beta_alpha,
                flow_matching_beta_beta=flow_matching_beta_beta,
                mask_action_dim_padding=mask_action_dim_padding,
                enable_depth_reasoning=enable_depth_reasoning,
                depth_mode=depth_mode,
                num_depth_codes=num_depth_codes,
                action_expert_depth_gate=action_expert_depth_gate,
                action_expert_depth_gate_per_layer=action_expert_depth_gate_per_layer,
                action_expert_depth_gate_init_bias=action_expert_depth_gate_init_bias,
                image_start_token_id=image_start_token_id,
                low_res_image_start_token_id=low_res_image_start_token_id,
                image_end_token_id=image_end_token_id,
                image_low_res_id=image_low_res_id,
                image_patch_id=image_patch_id,
                image_col_id=image_col_id,
                frame_start_token_id=frame_start_token_id,
                frame_end_token_id=frame_end_token_id,
                use_frame_special_tokens=use_frame_special_tokens,
                action_output_token_id=action_output_token_id,
                action_start_token_id=action_start_token_id,
                action_end_token_id=action_end_token_id,
                action_token_start_id=action_token_start_id,
                num_action_tokens=num_action_tokens,
                depth_output_token_id=depth_output_token_id,
                depth_start_token_id=depth_start_token_id,
                depth_end_token_id=depth_end_token_id,
                depth_token_start_id=depth_token_start_id,
                num_depth_tokens=num_depth_tokens,
                state_start_token_id=state_start_token_id,
                state_end_token_id=state_end_token_id,
                state_token_start_id=state_token_start_id,
                num_state_tokens=num_state_tokens,
                add_setup_tokens=add_setup_tokens,
                add_control_tokens=add_control_tokens,
                add_action_expert=add_action_expert,
                norm_stats_filename=norm_stats_filename,
                initializer_range=initializer_range,
                compile_model=compile_model,
                input_features=input_features,
                output_features=output_features,
            )

        # captures raw init args
        self.save_hyperparameters(ignore=["config", "hf_repo_id_or_pretrained_path", "compile_model"])

        # model, pre and post processor lazy init, resolved later in _initalize_model
        self._model: MolmoAct2Model | None = None
        self._preprocessor: MolmoAct2Preprocessor | None = None
        self._postprocessor: MolmoAct2Postprocessor | None = None

    @classmethod
    def _build_config_from_init_args_only(  # noqa: PLR0913
        cls,
        *,
        # Vision transformer component
        vit_hidden_size: int,
        vit_intermediate_size: int,
        vit_num_hidden_layers: int,
        vit_num_attention_heads: int,
        vit_num_key_value_heads: int,
        vit_head_dim: int,
        vit_hidden_act: str,
        vit_layer_norm_eps: float,
        vit_image_default_input_size: tuple[int, int],
        vit_image_patch_size: int,
        vit_image_num_pos: int,
        vit_attention_dropout: float,
        vit_residual_dropout: float,
        vit_initializer_range: float,
        vit_float32_attention: bool,
        vit_attn_implementation: str,
        # Vision adapter component
        adapter_vit_layers: tuple[int, ...],
        adapter_pooling_attention_mask: bool,
        adapter_hidden_size: int,
        adapter_num_attention_heads: int,
        adapter_num_key_value_heads: int,
        adapter_head_dim: int,
        adapter_float32_attention: bool,
        adapter_attention_dropout: float,
        adapter_residual_dropout: float,
        adapter_hidden_act: str,
        adapter_intermediate_size: int,
        adapter_text_hidden_size: int,
        adapter_image_feature_dropout: float,
        adapter_initializer_range: float,
        adapter_attn_implementation: str,
        # Text transformer component
        text_hidden_size: int,
        text_num_attention_heads: int,
        text_num_key_value_heads: int | None,
        text_head_dim: int,
        text_vocab_size: int,
        text_additional_vocab_size: int,
        text_qkv_bias: bool,
        text_num_hidden_layers: int,
        text_intermediate_size: int,
        text_hidden_act: str,
        text_embedding_dropout: float,
        text_attention_dropout: float,
        text_residual_dropout: float,
        text_max_position_embeddings: int,
        text_rope_theta: float,
        text_rope_scaling: dict[str, Any] | None,
        text_rope_scaling_layers: list[int] | None,
        text_use_qk_norm: bool,
        text_qk_norm_type: str,
        text_layer_norm_eps: float,
        text_norm_after: bool,
        text_initializer_range: float,
        text_use_cache: bool,
        text_tie_word_embeddings: bool,
        text_attn_implementation: str,
        # Action expert component
        action_expert_max_action_horizon: int,
        action_expert_max_action_dim: int,
        action_expert_hidden_size: int,
        action_expert_num_layers: int,
        action_expert_num_heads: int,
        action_expert_mlp_ratio: float,
        action_expert_ffn_multiple_of: int,
        action_expert_timestep_embed_dim: int,
        action_expert_dropout: float,
        action_expert_attn_dropout: float,
        action_expert_context_layer_norm: bool,
        action_expert_qk_norm: bool,
        action_expert_qk_norm_eps: float,
        action_expert_rope: bool,
        action_expert_causal_attn: bool,
        # Input and rollout structure
        n_obs_steps: int,
        chunk_size: int,
        n_action_steps: int,
        # Action/state core settings
        max_action_dim: int,
        action_mode: Literal["continuous", "discrete", "both"],
        state_format: Literal["discrete"],
        # Flow matching
        flow_matching_num_steps: int,
        flow_matching_cutoff: float,
        flow_matching_time_offset: float,
        flow_matching_time_scale: float,
        flow_matching_beta_alpha: float,
        flow_matching_beta_beta: float,
        mask_action_dim_padding: bool,
        # Depth reasoning
        enable_depth_reasoning: bool,
        depth_mode: int,
        num_depth_codes: int,
        action_expert_depth_gate: bool,
        action_expert_depth_gate_per_layer: bool,
        action_expert_depth_gate_init_bias: float,
        # Vision/image special tokens
        image_start_token_id: int | None,
        low_res_image_start_token_id: int | None,
        image_end_token_id: int | None,
        image_low_res_id: int | None,
        image_patch_id: int | None,
        image_col_id: int | None,
        frame_start_token_id: int | None,
        frame_end_token_id: int | None,
        use_frame_special_tokens: bool,
        # Action tokenization
        action_output_token_id: int | None,
        action_start_token_id: int | None,
        action_end_token_id: int | None,
        action_token_start_id: int | None,
        num_action_tokens: int,
        # Depth tokenization
        depth_output_token_id: int | None,
        depth_start_token_id: int | None,
        depth_end_token_id: int | None,
        depth_token_start_id: int | None,
        num_depth_tokens: int,
        # State tokenization
        state_start_token_id: int | None,
        state_end_token_id: int | None,
        state_token_start_id: int | None,
        num_state_tokens: int,
        # Prompt and expert controls
        add_setup_tokens: bool,
        add_control_tokens: bool,
        add_action_expert: bool,
        # Initialization and assets
        norm_stats_filename: str,
        initializer_range: float,
        # Runtime options
        compile_model: bool,
        # Input / output features
        input_features: list[Feature],
        output_features: list[Feature],
        norm_tag: str | None = None,
    ) -> MolmoAct2Config:
        """Build policy config from constructor args only.

        Args:
            vit_hidden_size: Vision transformer hidden size.
            vit_intermediate_size: Vision transformer intermediate size.
            vit_num_hidden_layers: Vision transformer number of layers.
            vit_num_attention_heads: Vision transformer number of attention heads.
            vit_num_key_value_heads: Vision transformer number of key/value heads.
            vit_head_dim: Vision transformer head dimension.
            vit_hidden_act: Vision transformer activation name.
            vit_layer_norm_eps: Vision transformer layer norm epsilon.
            vit_image_default_input_size: Vision transformer default input resolution.
            vit_image_patch_size: Vision transformer patch size.
            vit_image_num_pos: Vision transformer positional embedding count.
            vit_attention_dropout: Vision transformer attention dropout.
            vit_residual_dropout: Vision transformer residual dropout.
            vit_initializer_range: Vision transformer initializer range.
            vit_float32_attention: Whether to force float32 attention in ViT.
            vit_attn_implementation: ViT attention implementation.
            adapter_vit_layers: Adapter source ViT layers.
            adapter_pooling_attention_mask: Whether adapter uses pooling attention mask.
            adapter_hidden_size: Adapter hidden size.
            adapter_num_attention_heads: Adapter number of attention heads.
            adapter_num_key_value_heads: Adapter number of key/value heads.
            adapter_head_dim: Adapter head dimension.
            adapter_float32_attention: Whether to force float32 attention in adapter.
            adapter_attention_dropout: Adapter attention dropout.
            adapter_residual_dropout: Adapter residual dropout.
            adapter_hidden_act: Adapter activation name.
            adapter_intermediate_size: Adapter intermediate size.
            adapter_text_hidden_size: Adapter text hidden size.
            adapter_image_feature_dropout: Adapter image feature dropout.
            adapter_initializer_range: Adapter initializer range.
            adapter_attn_implementation: Adapter attention implementation.
            text_hidden_size: Text transformer hidden size.
            text_num_attention_heads: Text transformer number of attention heads.
            text_num_key_value_heads: Text transformer number of key/value heads.
            text_head_dim: Text transformer head dimension.
            text_vocab_size: Text vocabulary size.
            text_additional_vocab_size: Extra text vocabulary size.
            text_qkv_bias: Whether text qkv projections use bias.
            text_num_hidden_layers: Text transformer number of layers.
            text_intermediate_size: Text transformer intermediate size.
            text_hidden_act: Text transformer activation name.
            text_embedding_dropout: Text embedding dropout.
            text_attention_dropout: Text attention dropout.
            text_residual_dropout: Text residual dropout.
            text_max_position_embeddings: Max text position embeddings.
            text_rope_theta: RoPE theta value.
            text_rope_scaling: Optional RoPE scaling config.
            text_rope_scaling_layers: Optional layers for RoPE scaling.
            text_use_qk_norm: Whether to enable qk normalization.
            text_qk_norm_type: qk normalization variant.
            text_layer_norm_eps: Text layer norm epsilon.
            text_norm_after: Whether text applies norm after block.
            text_initializer_range: Text initializer range.
            text_use_cache: Whether text model uses KV cache.
            text_tie_word_embeddings: Whether text ties word embeddings.
            text_attn_implementation: Text attention implementation.
            action_expert_max_action_horizon: Action expert horizon.
            action_expert_max_action_dim: Action expert max action dimension.
            action_expert_hidden_size: Action expert hidden size.
            action_expert_num_layers: Action expert number of layers.
            action_expert_num_heads: Action expert number of heads.
            action_expert_mlp_ratio: Action expert MLP ratio.
            action_expert_ffn_multiple_of: Action expert FFN multiple.
            action_expert_timestep_embed_dim: Action expert timestep embedding dimension.
            action_expert_dropout: Action expert dropout.
            action_expert_attn_dropout: Action expert attention dropout.
            action_expert_context_layer_norm: Whether action expert uses context layer norm.
            action_expert_qk_norm: Whether action expert uses qk norm.
            action_expert_qk_norm_eps: Action expert qk norm epsilon.
            action_expert_rope: Whether action expert uses RoPE.
            action_expert_causal_attn: Whether action expert uses causal attention.
            n_obs_steps: Observation horizon.
            chunk_size: Action chunk size.
            n_action_steps: Number of executed action steps.
            max_action_dim: Maximum action dimension.
            action_mode: Action mode.
            state_format: State format.
            flow_matching_num_steps: Flow matching step count.
            flow_matching_cutoff: Flow matching cutoff.
            flow_matching_time_offset: Flow matching time offset.
            flow_matching_time_scale: Flow matching time scale.
            flow_matching_beta_alpha: Flow matching beta alpha.
            flow_matching_beta_beta: Flow matching beta beta.
            mask_action_dim_padding: Whether to mask action-dimension padding.
            enable_depth_reasoning: Whether depth reasoning is enabled.
            depth_mode: Depth reasoning mode.
            num_depth_codes: Number of depth codes.
            action_expert_depth_gate: Whether action expert uses depth gate.
            action_expert_depth_gate_per_layer: Whether depth gate is applied per layer.
            action_expert_depth_gate_init_bias: Initial depth gate bias.
            image_start_token_id: Image start token id.
            low_res_image_start_token_id: Low-res image start token id.
            image_end_token_id: Image end token id.
            image_low_res_id: Low-res image id.
            image_patch_id: Image patch id.
            image_col_id: Image column token id.
            frame_start_token_id: Frame start token id.
            frame_end_token_id: Frame end token id.
            use_frame_special_tokens: Whether to use frame special tokens.
            action_output_token_id: Action output token id.
            action_start_token_id: Action start token id.
            action_end_token_id: Action end token id.
            action_token_start_id: Action token start id.
            num_action_tokens: Number of action tokens.
            depth_output_token_id: Depth output token id.
            depth_start_token_id: Depth start token id.
            depth_end_token_id: Depth end token id.
            depth_token_start_id: Depth token start id.
            num_depth_tokens: Number of depth tokens.
            state_start_token_id: State start token id.
            state_end_token_id: State end token id.
            state_token_start_id: State token start id.
            num_state_tokens: Number of state tokens.
            add_setup_tokens: Whether setup tokens are added.
            add_control_tokens: Whether control tokens are added.
            add_action_expert: Whether action expert is enabled.
            norm_stats_filename: Normalization stats filename.
            initializer_range: Global initializer range.
            compile_model: Whether to compile the model.
            input_features: Input feature definitions.
            output_features: Output feature definitions.
            norm_tag: Optional normalization tag.

        Returns:
            Resolved ``MolmoAct2Config`` instance.
        """
        vit_config = MolmoAct2VitConfig(
            hidden_size=vit_hidden_size,
            intermediate_size=vit_intermediate_size,
            num_hidden_layers=vit_num_hidden_layers,
            num_attention_heads=vit_num_attention_heads,
            num_key_value_heads=vit_num_key_value_heads,
            head_dim=vit_head_dim,
            hidden_act=vit_hidden_act,
            layer_norm_eps=vit_layer_norm_eps,
            image_default_input_size=vit_image_default_input_size,
            image_patch_size=vit_image_patch_size,
            image_num_pos=vit_image_num_pos,
            attention_dropout=vit_attention_dropout,
            residual_dropout=vit_residual_dropout,
            initializer_range=vit_initializer_range,
            float32_attention=vit_float32_attention,
            attn_implementation=vit_attn_implementation,
        )

        adapter_config = MolmoAct2AdapterConfig(
            vit_layers=adapter_vit_layers,
            pooling_attention_mask=adapter_pooling_attention_mask,
            hidden_size=adapter_hidden_size,
            num_attention_heads=adapter_num_attention_heads,
            num_key_value_heads=adapter_num_key_value_heads,
            head_dim=adapter_head_dim,
            float32_attention=adapter_float32_attention,
            attention_dropout=adapter_attention_dropout,
            residual_dropout=adapter_residual_dropout,
            hidden_act=adapter_hidden_act,
            intermediate_size=adapter_intermediate_size,
            text_hidden_size=adapter_text_hidden_size,
            image_feature_dropout=adapter_image_feature_dropout,
            initializer_range=adapter_initializer_range,
            attn_implementation=adapter_attn_implementation,
        )

        text_config = MolmoAct2TextConfig(
            hidden_size=text_hidden_size,
            num_attention_heads=text_num_attention_heads,
            num_key_value_heads=text_num_key_value_heads,
            head_dim=text_head_dim,
            vocab_size=text_vocab_size,
            additional_vocab_size=text_additional_vocab_size,
            qkv_bias=text_qkv_bias,
            num_hidden_layers=text_num_hidden_layers,
            intermediate_size=text_intermediate_size,
            hidden_act=text_hidden_act,
            embedding_dropout=text_embedding_dropout,
            attention_dropout=text_attention_dropout,
            residual_dropout=text_residual_dropout,
            max_position_embeddings=text_max_position_embeddings,
            rope_theta=text_rope_theta,
            rope_scaling=text_rope_scaling,
            rope_scaling_layers=text_rope_scaling_layers,
            use_qk_norm=text_use_qk_norm,
            qk_norm_type=text_qk_norm_type,
            layer_norm_eps=text_layer_norm_eps,
            norm_after=text_norm_after,
            initializer_range=text_initializer_range,
            use_cache=text_use_cache,
            tie_word_embeddings=text_tie_word_embeddings,
            attn_implementation=text_attn_implementation,
        )

        action_expert_config: MolmoAct2ActionExpertConfig | None = None
        if add_action_expert:
            action_expert_config = MolmoAct2ActionExpertConfig(
                max_action_horizon=action_expert_max_action_horizon,
                max_action_dim=action_expert_max_action_dim,
                hidden_size=action_expert_hidden_size,
                num_layers=action_expert_num_layers,
                num_heads=action_expert_num_heads,
                mlp_ratio=action_expert_mlp_ratio,
                ffn_multiple_of=action_expert_ffn_multiple_of,
                timestep_embed_dim=action_expert_timestep_embed_dim,
                dropout=action_expert_dropout,
                attn_dropout=action_expert_attn_dropout,
                context_layer_norm=action_expert_context_layer_norm,
                qk_norm=action_expert_qk_norm,
                qk_norm_eps=action_expert_qk_norm_eps,
                rope=action_expert_rope,
                causal_attn=action_expert_causal_attn,
            )

        return MolmoAct2Config(
            vit_config=vit_config,
            adapter_config=adapter_config,
            text_config=text_config,
            action_expert_config=action_expert_config,
            n_obs_steps=n_obs_steps,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            max_action_dim=max_action_dim,
            action_mode=action_mode,
            state_format=state_format,
            flow_matching_num_steps=flow_matching_num_steps,
            flow_matching_cutoff=flow_matching_cutoff,
            flow_matching_time_offset=flow_matching_time_offset,
            flow_matching_time_scale=flow_matching_time_scale,
            flow_matching_beta_alpha=flow_matching_beta_alpha,
            flow_matching_beta_beta=flow_matching_beta_beta,
            mask_action_dim_padding=mask_action_dim_padding,
            enable_depth_reasoning=enable_depth_reasoning,
            depth_mode=depth_mode,
            num_depth_codes=num_depth_codes,
            action_expert_depth_gate=action_expert_depth_gate,
            action_expert_depth_gate_per_layer=action_expert_depth_gate_per_layer,
            action_expert_depth_gate_init_bias=action_expert_depth_gate_init_bias,
            image_start_token_id=image_start_token_id,
            low_res_image_start_token_id=low_res_image_start_token_id,
            image_end_token_id=image_end_token_id,
            image_low_res_id=image_low_res_id,
            image_patch_id=image_patch_id,
            image_col_id=image_col_id,
            frame_start_token_id=frame_start_token_id,
            frame_end_token_id=frame_end_token_id,
            use_frame_special_tokens=use_frame_special_tokens,
            action_output_token_id=action_output_token_id,
            action_start_token_id=action_start_token_id,
            action_end_token_id=action_end_token_id,
            action_token_start_id=action_token_start_id,
            num_action_tokens=num_action_tokens,
            depth_output_token_id=depth_output_token_id,
            depth_start_token_id=depth_start_token_id,
            depth_end_token_id=depth_end_token_id,
            depth_token_start_id=depth_token_start_id,
            num_depth_tokens=num_depth_tokens,
            state_start_token_id=state_start_token_id,
            state_end_token_id=state_end_token_id,
            state_token_start_id=state_token_start_id,
            num_state_tokens=num_state_tokens,
            add_setup_tokens=add_setup_tokens,
            add_control_tokens=add_control_tokens,
            add_action_expert=add_action_expert,
            norm_stats_filename=norm_stats_filename,
            initializer_range=initializer_range,
            compile_model=compile_model,
            norm_tag=norm_tag,
            input_features=input_features,
            output_features=output_features,
        )

    def _initialize_model(self) -> None:
        """Initialize the underlying model and preprocessors.

        Builds the model and preprocessing pipeline from the resolved config.
        """
        self._preprocessor, self._postprocessor = make_molmoact2_preprocessors(
            config=self.config,
            hf_container=self.hf_container,
        )

        self._model = MolmoAct2Model(self.config, self.hf_container)

    def setup(self, stage: str) -> None:
        """Set up model from datamodule (lazy or fine-tuning path)."""
        del stage

        self._initialize_model()

    @torch.no_grad()
    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Predict an action chunk from an observation batch.

        Args:
            batch: Input observation batch.

        Returns:
            Predicted action chunk after post-processing.

        Raises:
            ValueError: If the model or preprocessors are not initialized.
        """
        if self._model is None:
            msg = "Model is not initialized"
            raise ValueError(msg)
        if self._preprocessor is None or self._postprocessor is None:
            msg = "Preprocessor is not initialized"
            raise ValueError(msg)

        processed_batch = self._preprocessor(batch.to_dict(flatten=True))
        actions = self._model.predict_action_chunk(processed_batch)
        return self._postprocessor(actions)

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Forward pass through the policy.

        Args:
            batch: Input observation batch.

        Returns:
            Predicted action chunk.
        """
        return self.predict_action_chunk(batch)
