# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 policy implementation."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch

from physicalai.data.observation import Feature, FeatureType, NormalizationParameters, Observation
from physicalai.policies.base import Policy
from physicalai.utils.hf_utils import HuggingfacePolicyContainer, download_policy_artifacts_from_hub

from .config import (
    MolmoAct2ActionExpertConfig,
    MolmoAct2AdapterConfig,
    MolmoAct2Config,
    MolmoAct2TextConfig,
    MolmoAct2VitConfig,
)
from .model import MolmoAct2Model
from .processors import make_molmoact2_preprocessors

if TYPE_CHECKING:
    from .processors import MolmoAct2Postprocessor, MolmoAct2Preprocessor

SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
IMAGE_SIZE_DIMS = 2


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
        *,
        # Input and rollout structure
        n_obs_steps: int = 30,
        chunk_size: int = 30,
        n_action_steps: int = 30,
        # Action/state core settings
        max_action_dim: int = 32,
        action_mode: Literal["continuous", "discrete", "both"] = "both",
        state_format: Literal["discrete"] = "discrete",
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
            self.hf_container = self._from_hf(
                hf_repo_id_or_pretrained_path,
                norm_stats_filename=norm_stats_filename,
            )
            self.config = self._build_config_from_hf_config(
                self.hf_container.hf_config,
                norm_stats=self.hf_container.norm_stats,
                norm_tag=norm_tag,
                input_features=input_features,
                output_features=output_features,
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
            )

        # captures raw init args
        self.save_hyperparameters(ignore=["config", "hf_repo_id_or_pretrained_path", "compile_model"])

        # model, pre and post processor lazy init, resolved later in _initalize_model
        self._model: MolmoAct2Model | None = None
        self._preprocessor: MolmoAct2Preprocessor | None = None
        self._postprocessor: MolmoAct2Postprocessor | None = None

    # physical ai

    def _initialize_model(self) -> None:
        """Initialize model and preprocessors for MolmoAct2.

        Builds the wrapped model and policy preprocessors from the resolved
        policy configuration.
        """
        # output model to lerobot for now
        self._model = MolmoAct2Model(self.config, self.hf_container)

        # # make and load pre / processors
        self._preprocessor, self._postprocessor = make_molmoact2_preprocessors(
            config=self.config,
        )

    @torch.no_grad()
    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Predict a chunk of actions from observation.

        Args:
            batch: Input observation batch.

        Returns:
            Action chunk tensor after post-processing.

        Raises:
            ValueError: If the model is not initialized.
        """
        if self._model is None:
            msg = "Model is not initialized"
            raise ValueError(msg)
        if self._preprocessor is None or self._postprocessor is None:
            msg = "Preprocessor is not initialized"
            raise ValueError(msg)

        processed_batch = self._preprocessor(batch)
        actions = self._model.predict_action_chunk(processed_batch)
        return self._postprocessor(actions)

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Forward pass through the model.

        Eval mode: returns action chunk predictions.

        Returns:
            Action tensor in eval mode.
        """
        return self.predict_action_chunk(batch)

    @staticmethod
    def _hf_component_config(
        hf_config: dict[str, Any],
        component_name: str,
    ) -> dict[str, Any]:
        """Extract a nested HF component config.

        Args:
            hf_config: Parsed Hugging Face ``config.json`` payload.
            component_name: Nested component key to extract.

        Returns:
            Component config as a dict, or an empty dict when missing.
        """
        component_config = hf_config.get(component_name)
        if isinstance(component_config, dict):
            return component_config
        return {}

    @staticmethod
    def _normalization_parameters_from_stats(stats: dict[str, Any]) -> NormalizationParameters:
        """Convert raw stats dict to ``NormalizationParameters``.

        Args:
            stats: Stats mapping containing normalization keys.

        Returns:
            Converted normalization parameters.
        """
        return NormalizationParameters(
            mean=stats.get("mean"),
            std=stats.get("std"),
            min=stats.get("min"),
            max=stats.get("max"),
            q01=stats.get("q01"),
            q99=stats.get("q99"),
        )

    @staticmethod
    def _resolve_norm_tag_metadata(
        norm_stats: dict[str, Any] | None,
        norm_tag: str | None,
    ) -> dict[str, Any]:
        """Resolve normalization metadata for a selected tag.

        Args:
            norm_stats: Parsed ``norm_stats.json`` payload.
            norm_tag: Tag identifying the dataset metadata block.

        Returns:
            Metadata dict for the requested tag.

        Raises:
            ValueError: If required tag inputs are missing.
            TypeError: If the stats payload structure is invalid.
        """
        if norm_tag is None:
            msg = "norm_tag is required to resolve pretrained MolmoAct2 feature normalization."
            raise ValueError(msg)
        if norm_stats is None:
            msg = "Pretrained MolmoAct2 checkpoint is missing norm_stats.json contents."
            raise ValueError(msg)

        metadata_by_tag = norm_stats.get("metadata_by_tag")
        if not isinstance(metadata_by_tag, dict):
            msg = "Invalid MolmoAct2 norm stats format: missing metadata_by_tag."
            raise TypeError(msg)

        tag_metadata = metadata_by_tag.get(norm_tag)
        if not isinstance(tag_metadata, dict):
            msg = f"norm_tag '{norm_tag}' was not found in pretrained MolmoAct2 norm stats."
            raise TypeError(msg)
        return tag_metadata

    @classmethod
    def _resolve_hf_image_size(cls, hf_config: dict[str, Any]) -> tuple[int, int]:
        """Resolve input image size from HF ViT config.

        Args:
            hf_config: Parsed Hugging Face ``config.json`` payload.

        Returns:
            Image size tuple as ``(height, width)``.

        Raises:
            TypeError: If the configured size is malformed.
        """
        image_size = cls._hf_component_config(hf_config, "vit_config").get("image_default_input_size", (378, 378))
        if isinstance(image_size, list):
            image_size = tuple(image_size)
        if not isinstance(image_size, tuple) or len(image_size) != IMAGE_SIZE_DIMS:
            msg = f"Invalid image_default_input_size in HF config: {image_size!r}"
            raise TypeError(msg)
        return image_size

    @staticmethod
    def _build_visual_features(camera_keys: list[Any], image_size: tuple[int, int]) -> list[Feature]:
        """Build visual ``Feature`` entries from camera keys.

        Args:
            camera_keys: List of camera feature keys from norm stats metadata.
            image_size: Resolved image size ``(height, width)``.

        Returns:
            Visual feature definitions.

        Raises:
            TypeError: If any camera key is not a string.
        """
        input_features: list[Feature] = []
        for camera_key in camera_keys:
            if not isinstance(camera_key, str):
                msg = f"Invalid camera key in pretrained MolmoAct2 norm stats: {camera_key!r}"
                raise TypeError(msg)
            input_features.append(
                Feature(
                    name=camera_key.removeprefix("observation.images."),
                    ftype=FeatureType.VISUAL,
                    shape=(3, image_size[0], image_size[1]),
                ),
            )
        return input_features

    @classmethod
    def _build_feature_from_stats(
        cls,
        *,
        feature_key: str,
        stats: dict[str, Any],
        prefix: str,
        ftype: FeatureType,
    ) -> Feature:
        """Build a single feature definition from stats metadata.

        Args:
            feature_key: Feature key from metadata.
            stats: Normalization stats for the feature.
            prefix: Prefix to strip from ``feature_key`` for output name.
            ftype: Feature type enum value.

        Returns:
            A constructed ``Feature``.
        """
        return Feature(
            name=feature_key.removeprefix(prefix),
            ftype=ftype,
            shape=(len(stats.get("mean", [])),),
            normalization_data=cls._normalization_parameters_from_stats(stats),
        )

    @classmethod
    def _build_features_from_norm_stats(
        cls,
        hf_config: dict[str, Any],
        norm_stats: dict[str, Any] | None,
        norm_tag: str | None,
    ) -> tuple[list[Feature], list[Feature]]:
        """Build input and output features from pretrained norm stats.

        Args:
            hf_config: Parsed Hugging Face ``config.json`` payload.
            norm_stats: Parsed ``norm_stats.json`` payload.
            norm_tag: Selected normalization metadata tag.

        Returns:
            Tuple of ``(input_features, output_features)``.

        Raises:
            TypeError: If normalization metadata has an invalid shape or type.
        """
        tag_metadata = cls._resolve_norm_tag_metadata(norm_stats, norm_tag)
        image_size = cls._resolve_hf_image_size(hf_config)

        camera_keys = tag_metadata.get("camera_keys")
        if not isinstance(camera_keys, list):
            msg = f"Invalid camera_keys for norm_tag '{norm_tag}'."
            raise TypeError(msg)

        input_features = cls._build_visual_features(camera_keys, image_size)

        state_key = tag_metadata.get("state_key")
        state_stats = tag_metadata.get("state_stats")
        if isinstance(state_key, str) and isinstance(state_stats, dict):
            input_features.append(
                cls._build_feature_from_stats(
                    feature_key=state_key,
                    stats=state_stats,
                    prefix="observation.",
                    ftype=FeatureType.STATE,
                ),
            )

        action_key = tag_metadata.get("action_key")
        action_stats = tag_metadata.get("action_stats")
        output_features: list[Feature] = []
        if isinstance(action_key, str) and isinstance(action_stats, dict):
            output_features.append(
                cls._build_feature_from_stats(
                    feature_key=action_key,
                    stats=action_stats,
                    prefix="",
                    ftype=FeatureType.ACTION,
                ),
            )

        return input_features, output_features

    @staticmethod
    def _merge_feature_override(base_feature: Feature, override_feature: Feature) -> Feature:
        """Merge override values into a base feature definition.

        Args:
            base_feature: Pretrained/base feature definition.
            override_feature: User-provided override feature definition.

        Returns:
            Merged feature with override values taking precedence.
        """
        return Feature(
            name=override_feature.name if override_feature.name is not None else base_feature.name,
            ftype=override_feature.ftype if override_feature.ftype is not None else base_feature.ftype,
            shape=override_feature.shape if override_feature.shape is not None else base_feature.shape,
            normalization_data=(
                override_feature.normalization_data
                if override_feature.normalization_data is not None
                else base_feature.normalization_data
            ),
        )

    @classmethod
    def _resolve_feature_overrides(
        cls,
        pretrained_features: list[Feature],
        override_features: list[Feature] | None,
    ) -> list[Feature]:
        """Resolve final feature list from pretrained and override inputs.

        Args:
            pretrained_features: Features derived from pretrained metadata.
            override_features: Optional user-provided feature overrides.

        Returns:
            Final ordered feature list with overrides applied.
        """
        if not override_features:
            return pretrained_features

        overrides_by_name = {feature.name: feature for feature in override_features}
        resolved_features = [
            cls._merge_feature_override(feature, overrides_by_name[feature.name])
            for feature in pretrained_features
            if feature.name in overrides_by_name
        ]
        resolved_names = {feature.name for feature in resolved_features}

        for feature in pretrained_features:
            if feature.name not in resolved_names:
                resolved_features.append(feature)

        for feature in override_features:
            if feature.name not in resolved_names:
                resolved_features.append(feature)
                resolved_names.add(feature.name)

        return resolved_features

    @classmethod
    def _build_config_from_hf_config(
        cls,
        hf_config: dict[str, Any],
        norm_stats: dict[str, Any] | None = None,
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        norm_tag: str | None = None,
        n_obs_steps: int = 30,
        chunk_size: int = 30,
        n_action_steps: int = 30,
        max_action_dim: int = 32,
    ) -> MolmoAct2Config:
        """Build policy config from Hugging Face config and local overrides.

        Args:
            hf_config: Parsed Hugging Face ``config.json`` payload.
            norm_stats: Parsed ``norm_stats.json`` payload.
            input_features: Optional feature overrides for model inputs.
            output_features: Optional feature overrides for model outputs.
            norm_tag: Selected normalization metadata tag.
            n_obs_steps: Observation horizon override.
            chunk_size: Action chunk size override.
            n_action_steps: Number of executed action steps override.
            max_action_dim: Maximum action dimension override.

        Returns:
            Resolved ``MolmoAct2Config`` instance.
        """
        config_data: dict[str, Any] = {
            "vit_config": cls._hf_component_config(hf_config, "vit_config"),
            "adapter_config": cls._hf_component_config(hf_config, "adapter_config"),
            "text_config": cls._hf_component_config(hf_config, "text_config"),
        }
        pretrained_input_features, pretrained_output_features = cls._build_features_from_norm_stats(
            hf_config,
            norm_stats,
            norm_tag,
        )
        config_data["input_features"] = cls._resolve_feature_overrides(pretrained_input_features, input_features)
        config_data["output_features"] = cls._resolve_feature_overrides(pretrained_output_features, output_features)

        top_level_keys = (
            # Input and rollout structure
            "n_obs_steps",
            "chunk_size",
            "n_action_steps",
            # Action/state core settings
            "max_action_dim",
            "action_mode",
            "state_format",
            # Flow matching
            "flow_matching_num_steps",
            "flow_matching_cutoff",
            "flow_matching_time_offset",
            "flow_matching_time_scale",
            "flow_matching_beta_alpha",
            "flow_matching_beta_beta",
            "mask_action_dim_padding",
            # Depth reasoning
            "enable_depth_reasoning",
            "depth_mode",
            "num_depth_codes",
            "action_expert_depth_gate",
            "action_expert_depth_gate_per_layer",
            "action_expert_depth_gate_init_bias",
            # Vision/image special tokens
            "image_start_token_id",
            "low_res_image_start_token_id",
            "image_end_token_id",
            "image_low_res_id",
            "image_patch_id",
            "image_col_id",
            "frame_start_token_id",
            "frame_end_token_id",
            "use_frame_special_tokens",
            # Action tokenization
            "action_output_token_id",
            "action_start_token_id",
            "action_end_token_id",
            "action_token_start_id",
            "num_action_tokens",
            # Depth tokenization
            "depth_output_token_id",
            "depth_start_token_id",
            "depth_end_token_id",
            "depth_token_start_id",
            "num_depth_tokens",
            # State tokenization
            "state_start_token_id",
            "state_end_token_id",
            "state_token_start_id",
            "num_state_tokens",
            # Prompt and expert controls
            "add_setup_tokens",
            "add_control_tokens",
            "add_action_expert",
            # Initialization and assets
            "norm_stats_filename",
            "initializer_range",
            # Runtime options
            "compile_model",
            # Norm tag (can be overridden by kwarg above, but respect hf_config if kwarg is None)
            "norm_tag",
        )

        for key in top_level_keys:
            # kwarg-supplied norm_tag takes precedence over hf_config value
            if key == "norm_tag" and norm_tag is not None:
                continue
            if key in hf_config:
                config_data[key] = hf_config[key]

        # Overrides
        config_data["norm_tag"] = norm_tag
        config_data["n_obs_steps"] = n_obs_steps
        config_data["chunk_size"] = chunk_size
        config_data["n_action_steps"] = n_action_steps
        config_data["max_action_dim"] = max_action_dim

        if config_data.get("add_action_expert") is False:
            config_data["action_expert_config"] = None

        return MolmoAct2Config.from_dict(config_data)

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
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
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
            input_features: Optional input feature definitions.
            output_features: Optional output feature definitions.
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
            input_features=input_features if input_features is not None else [],
            output_features=output_features if output_features is not None else [],
        )

    # hf
    def _from_hf(
        self,
        pretrained_name_or_path: str | Path,
        norm_stats_filename: str = "norm_stats.json",
        **kwargs: object,
    ) -> HuggingfacePolicyContainer:
        """Resolve policy artifacts from local path or Hugging Face Hub.

        Args:
            pretrained_name_or_path: Local checkpoint directory or HF repo id.
            norm_stats_filename: Normalization stats filename to resolve.
            **kwargs: Optional HF hub download kwargs.

        Returns:
            Container with resolved artifact paths and parsed config payloads.
        """
        path = Path(pretrained_name_or_path)
        is_local = path.is_dir()
        norm_stats: dict[str, Any] | None = None

        if is_local:
            config_file = path / "config.json"
            weights_file = _resolve_local_weights_path(path)
            preprocessor_file = path / "policy_preprocessor.json"
            preprocessor_dir = path
            norm_stats_file = path / norm_stats_filename
            if norm_stats_file.is_file():
                with norm_stats_file.open(encoding="utf-8") as f:
                    norm_stats = json.load(f)
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
            (
                config_file,
                weights_file,
                preprocessor_file,
                preprocessor_dir,
                norm_stats_file,
            ) = download_policy_artifacts_from_hub(
                str(pretrained_name_or_path),
                hub_kwargs=hub_kwargs,
                norm_stats_filename=norm_stats_filename,
            )
            if norm_stats_file is not None:
                with norm_stats_file.open(encoding="utf-8") as f:
                    norm_stats = json.load(f)

        with Path(config_file).open(encoding="utf-8") as f:
            hf_config = json.load(f)

        checkpoint_location = str(Path(weights_file).parent)

        return HuggingfacePolicyContainer(
            config_file=Path(config_file),
            weights_file=Path(weights_file),
            preprocessor_file=Path(preprocessor_file) if preprocessor_file is not None else None,
            preprocessor_dir=Path(preprocessor_dir) if preprocessor_dir is not None else None,
            checkpoint_location=checkpoint_location,
            hf_config=hf_config,
            norm_stats=norm_stats,
        )
