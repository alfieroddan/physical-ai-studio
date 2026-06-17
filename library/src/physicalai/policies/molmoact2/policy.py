# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 policy implementation."""

import json
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors.torch import load_file as load_safetensors_file
from torch._tensor import Tensor

from physicalai.data.observation import Observation
from physicalai.export import ExportablePolicyMixin
from physicalai.policies.base import Policy
from physicalai.policies.molmoact2.config import (
    MolmoAct2ActionExpertConfig,
    MolmoAct2AdapterConfig,
    MolmoAct2Config,
    MolmoAct2TextConfig,
    MolmoAct2VitConfig,
)
from physicalai.utils.hf_utils import HuggingfacePolicyContainer, download_policy_artifacts_from_hub

SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"


def _strict_load_safetensors_weights(model: torch.nn.Module, checkpoint_location: str) -> None:
    """Load safetensors weights with strict key validation.

    Mirrors the MolmoAct2 behavior from lerobot: supports sharded and single-file
    checkpoints, and validates key compatibility with the local model.

    Raises:
        RuntimeError: If checkpoint keys are incompatible with the local model.
        FileNotFoundError: If no safetensors weights are found at the checkpoint.
    """
    checkpoint_dir = Path(checkpoint_location)
    index_path = checkpoint_dir / SAFE_WEIGHTS_INDEX_NAME
    single_file_path = checkpoint_dir / SAFE_WEIGHTS_NAME

    if index_path.is_file():
        with index_path.open(encoding="utf-8") as f:
            index = json.load(f)

        weight_map = index["weight_map"]
        loaded_keys = set(weight_map)
        model_keys = set(model.state_dict())
        missing_keys = sorted(model_keys - loaded_keys)
        unexpected_keys = sorted(loaded_keys - model_keys)
        if missing_keys or unexpected_keys:
            message = ["MolmoAct2 safetensors do not match the local model implementation."]
            if missing_keys:
                message.append(f"Missing keys: {missing_keys[:8]}")
            if unexpected_keys:
                message.append(f"Unexpected keys: {unexpected_keys[:8]}")
            raise RuntimeError(" ".join(message))

        for shard_file in sorted(set(weight_map.values())):
            state_dict = load_safetensors_file(str(checkpoint_dir / shard_file), device="cpu")
            model.load_state_dict(state_dict, strict=False)
            del state_dict
        return

    if single_file_path.is_file():
        state_dict = load_safetensors_file(str(single_file_path), device="cpu")
        model.load_state_dict(state_dict, strict=True)
        return

    msg = (
        f"MolmoAct2 checkpoint at {checkpoint_location} must contain {SAFE_WEIGHTS_NAME} or {SAFE_WEIGHTS_INDEX_NAME}."
    )
    raise FileNotFoundError(msg)


def _resolve_local_weights_path(checkpoint_dir: Path) -> Path:
    """Resolve local weights file for single-file or sharded safetensors checkpoints.

    Returns:
        Path to ``model.safetensors`` or ``model.safetensors.index.json``.

    Raises:
        FileNotFoundError: If neither local weights format exists.
    """
    single_file_path = checkpoint_dir / SAFE_WEIGHTS_NAME
    index_path = checkpoint_dir / SAFE_WEIGHTS_INDEX_NAME

    if single_file_path.is_file():
        return single_file_path
    if index_path.is_file():
        return index_path

    msg = (
        f"MolmoAct2 local checkpoint at {checkpoint_dir} must contain {SAFE_WEIGHTS_NAME} or {SAFE_WEIGHTS_INDEX_NAME}."
    )
    raise FileNotFoundError(msg)


class MolmoAct2(ExportablePolicyMixin, Policy):
    """MolmoAct2 Policy."""

    def __init__(  # noqa: PLR0913
        self,
        # download model weights
        pretrained_name_or_path: str | Path | None = None,
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
        # Input and rollout structure
        n_obs_steps: int = 30,
        chunk_size: int = 30,
        n_action_steps: int = 30,
        # Action/state core settings
        max_action_dim: int = 32,
        action_mode: Literal["continuous", "discrete", "both"] = "both",
        state_format: Literal["discrete"] = "discrete",
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
        """Initialize MolmoAct2 policy."""
        super().__init__(n_action_steps=1)

        if pretrained_name_or_path is not None:
            hf_container = self._from_hf(pretrained_name_or_path)
            self.config = self._build_config_from_hf_config(hf_container.hf_config)
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
            )

        # captures raw init args
        self.save_hyperparameters(ignore=["config", "pretrained_name_or_path", "compile_model"])

    # physical ai

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Run training forward or inference chunk prediction.

        Returns:
            Training output or predicted action chunk output.
        """
        if self.training:
            return self.model(batch)
        return self.predict_action_chunk(batch)

    def predict_action_chunk(self, batch: Observation) -> Tensor:
        """Predict one action chunk during inference.

        Returns:
            Tensor containing one action chunk.
        """
        return super().predict_action_chunk(batch)

    @staticmethod
    def _hf_component_config(
        hf_config: dict[str, Any],
        component_name: str,
    ) -> dict[str, Any]:
        component_config = hf_config.get(component_name)
        if isinstance(component_config, dict):
            return component_config
        return {}

    @classmethod
    def _build_config_from_hf_config(cls, hf_config: dict[str, Any]) -> MolmoAct2Config:
        """Build config directly from HuggingFace config JSON.

        Returns:
            MolmoAct2Config initialized from HF JSON.
        """
        config_data: dict[str, Any] = {
            "vit_config": cls._hf_component_config(hf_config, "vit_config"),
            "adapter_config": cls._hf_component_config(hf_config, "adapter_config"),
            "text_config": cls._hf_component_config(hf_config, "text_config"),
        }

        top_level_keys = (
            "n_obs_steps",
            "chunk_size",
            "n_action_steps",
            "max_action_dim",
            "action_mode",
            "state_format",
            "flow_matching_num_steps",
            "flow_matching_cutoff",
            "flow_matching_time_offset",
            "flow_matching_time_scale",
            "flow_matching_beta_alpha",
            "flow_matching_beta_beta",
            "mask_action_dim_padding",
            "enable_depth_reasoning",
            "depth_mode",
            "num_depth_codes",
            "action_expert_depth_gate",
            "action_expert_depth_gate_per_layer",
            "action_expert_depth_gate_init_bias",
            "image_start_token_id",
            "low_res_image_start_token_id",
            "image_end_token_id",
            "image_low_res_id",
            "image_patch_id",
            "image_col_id",
            "frame_start_token_id",
            "frame_end_token_id",
            "use_frame_special_tokens",
            "action_output_token_id",
            "action_start_token_id",
            "action_end_token_id",
            "action_token_start_id",
            "num_action_tokens",
            "depth_output_token_id",
            "depth_start_token_id",
            "depth_end_token_id",
            "depth_token_start_id",
            "num_depth_tokens",
            "state_start_token_id",
            "state_end_token_id",
            "state_token_start_id",
            "num_state_tokens",
            "add_setup_tokens",
            "add_control_tokens",
            "add_action_expert",
            "norm_stats_filename",
            "initializer_range",
            "compile_model",
        )
        for key in top_level_keys:
            if key in hf_config:
                config_data[key] = hf_config[key]

        if config_data.get("add_action_expert") is False:
            config_data["action_expert_config"] = None
        elif "action_expert_config" in hf_config:
            config_data["action_expert_config"] = cls._hf_component_config(hf_config, "action_expert_config")

        return MolmoAct2Config.from_dict(config_data)

    @classmethod
    def _build_config_from_init_args_only(  # noqa: PLR0913
        cls,
        *,
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
        n_obs_steps: int,
        chunk_size: int,
        n_action_steps: int,
        max_action_dim: int,
        action_mode: Literal["continuous", "discrete", "both"],
        state_format: Literal["discrete"],
        flow_matching_num_steps: int,
        flow_matching_cutoff: float,
        flow_matching_time_offset: float,
        flow_matching_time_scale: float,
        flow_matching_beta_alpha: float,
        flow_matching_beta_beta: float,
        mask_action_dim_padding: bool,
        enable_depth_reasoning: bool,
        depth_mode: int,
        num_depth_codes: int,
        action_expert_depth_gate: bool,
        action_expert_depth_gate_per_layer: bool,
        action_expert_depth_gate_init_bias: float,
        image_start_token_id: int | None,
        low_res_image_start_token_id: int | None,
        image_end_token_id: int | None,
        image_low_res_id: int | None,
        image_patch_id: int | None,
        image_col_id: int | None,
        frame_start_token_id: int | None,
        frame_end_token_id: int | None,
        use_frame_special_tokens: bool,
        action_output_token_id: int | None,
        action_start_token_id: int | None,
        action_end_token_id: int | None,
        action_token_start_id: int | None,
        num_action_tokens: int,
        depth_output_token_id: int | None,
        depth_start_token_id: int | None,
        depth_end_token_id: int | None,
        depth_token_start_id: int | None,
        num_depth_tokens: int,
        state_start_token_id: int | None,
        state_end_token_id: int | None,
        state_token_start_id: int | None,
        num_state_tokens: int,
        add_setup_tokens: bool,
        add_control_tokens: bool,
        add_action_expert: bool,
        norm_stats_filename: str,
        initializer_range: float,
        compile_model: bool,
    ) -> MolmoAct2Config:
        """Build config from init args only (no HF config).

        Returns:
            MolmoAct2Config initialized from init arguments.
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
        )

    # hf
    def _from_hf(
        self,
        pretrained_name_or_path: str | Path,
        **kwargs: object,
    ) -> HuggingfacePolicyContainer:
        path = Path(pretrained_name_or_path)
        is_local = path.is_dir()

        if is_local:
            config_file = path / "config.json"
            weights_file = _resolve_local_weights_path(path)
            preprocessor_file = path / "policy_preprocessor.json"
            preprocessor_dir = path
        else:
            hub_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                in {
                    "cache_dir",
                    "force_download",
                    "resume_download",
                    "proxies",
                    "token",
                    "revision",
                    "local_files_only",
                }
            }
            config_file, weights_file, preprocessor_file, preprocessor_dir = download_policy_artifacts_from_hub(
                str(pretrained_name_or_path),
                hub_kwargs=hub_kwargs,
            )

        with Path(config_file).open(encoding="utf-8") as f:
            hf_config = json.load(f)

        checkpoint_location = str(Path(weights_file).parent)
        if self.model is not None:
            _strict_load_safetensors_weights(self.model, checkpoint_location)

        return HuggingfacePolicyContainer(
            config_file=Path(config_file),
            weights_file=Path(weights_file),
            preprocessor_file=Path(preprocessor_file) if preprocessor_file is not None else None,
            preprocessor_dir=Path(preprocessor_dir) if preprocessor_dir is not None else None,
            checkpoint_location=checkpoint_location,
            hf_config=hf_config,
        )

    # molmoact2
