# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face loading helpers for MolmoAct2.

Loading pipeline:

1. Artifact snapshot: :func:`load_hf_pretrained_container` resolves a local
    directory or Hub repository into :class:`MolmoAct2Snapshot`, with the model
    config, weights, optional preprocessing state, normalization data, processor
    configuration, and tokenizer options.
2. Feature resolution: the selected ``norm_tag`` produces pretrained visual,
    state, and action :class:`Feature` definitions. Caller features replace
    those definitions only by name.
3. Config translation: :func:`build_config_from_hf_config` maps the nested HF
    payload, resolved snapshot assets, and explicit caller overrides into the
    flat :class:`MolmoAct2Config` used by the policy.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from huggingface_hub import snapshot_download

from physicalai.data.observation import Feature, FeatureType, NormalizationParameters

from .config import MOLMOACT2_IMAGE_PLACEHOLDER_TOKEN_ID, MolmoAct2Config

logger = logging.getLogger(__name__)

SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
TOKENIZER_NAME = "tokenizer.json"
TOKENIZER_CONFIG_NAME = "tokenizer_config.json"
TOKENIZER_ALLOW_PATTERNS = [TOKENIZER_NAME, TOKENIZER_CONFIG_NAME]
SNAPSHOT_ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.md"]
IMAGE_SIZE_DIMS = 2
# Text prompt placeholder that gets expanded into image patch tokens. Kept for
# reference; its token id is hardcoded as
# ``MOLMOACT2_IMAGE_PLACEHOLDER_TOKEN_ID`` so that ``from_hf.py`` no longer
# needs to instantiate the tokenizer.
IMAGE_PROMPT = "<|image|>"
_COMMIT_HASH_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")


class HFHubDownloadKwargs(TypedDict, total=False):
    """Supported keyword arguments for ``hf_hub_download``."""

    cache_dir: str | Path | None
    force_download: bool
    token: bool | str | None
    revision: str | None
    local_files_only: bool


def _select_hf_hub_download_kwargs(kwargs: dict[str, object]) -> HFHubDownloadKwargs:
    """Filter untyped caller options into the supported Hub download arguments.

    Returns:
        Validated arguments accepted by ``hf_hub_download``.
    """
    selected: HFHubDownloadKwargs = {}

    cache_dir = kwargs.get("cache_dir")
    if cache_dir is None or isinstance(cache_dir, (str, Path)):
        selected["cache_dir"] = cache_dir

    force_download = kwargs.get("force_download")
    if isinstance(force_download, bool):
        selected["force_download"] = force_download

    token = kwargs.get("token")
    if token is None or isinstance(token, (bool, str)):
        selected["token"] = token

    revision = kwargs.get("revision")
    if revision is None or isinstance(revision, str):
        selected["revision"] = revision

    local_files_only = kwargs.get("local_files_only")
    if isinstance(local_files_only, bool):
        selected["local_files_only"] = local_files_only

    return selected


@dataclass
class MolmoAct2Snapshot:
    """Artifacts and parsed metadata resolved for a MolmoAct2 checkpoint."""

    config_file: Path
    weights_file: Path
    preprocessor_file: Path | None
    preprocessor_dir: Path | None
    checkpoint_location: str
    hf_config: dict[str, Any]
    norm_stats: dict[str, Any] | None = None
    processor_config: dict[str, Any] | None = None
    repo_id: str | None = None
    tokenizer_revision: str | None = None
    tokenizer_config: dict[str, Any] | None = None


def _resolve_snapshot_revision(config_file: Path, requested_revision: object) -> str | None:
    """Resolve an immutable revision from a Hub request or cache snapshot path.

    Returns:
        The immutable revision, if one can be resolved.
    """
    if isinstance(requested_revision, str) and _COMMIT_HASH_RE.fullmatch(requested_revision):
        return requested_revision
    snapshot_revision = config_file.parent.name
    if _COMMIT_HASH_RE.fullmatch(snapshot_revision):
        return snapshot_revision
    return None


def download_policy_artifacts_from_hub(
    repo_id: str,
    *,
    hub_kwargs: HFHubDownloadKwargs | None = None,
    config_filename: str = "config.json",
    weights_filename: str = SAFE_WEIGHTS_NAME,
    preprocessor_filename: str = "policy_preprocessor.json",
    norm_stats_filename: str | None = None,
    download_preprocessor_state_files: bool = True,
) -> tuple[Path, Path, Path | None, Path | None, Path | None]:
    """Download and validate the MolmoAct2 Hugging Face repository snapshot.

    Returns:
        The config, weights, optional preprocessor file/directory, and optional
        normalization stats file resolved from the repository.

    Raises:
        FileNotFoundError: If a required config, weights, or tokenizer file is
            missing from the downloaded snapshot.
    """
    selected_hub_kwargs = hub_kwargs or HFHubDownloadKwargs()
    snapshot_dir = Path(
        snapshot_download(  # nosec B615 - revision remains caller-configurable during initial integration
            repo_id,
            allow_patterns=SNAPSHOT_ALLOW_PATTERNS,
            **selected_hub_kwargs,  # type: ignore[arg-type]
        ),
    )
    del download_preprocessor_state_files

    config_file = snapshot_dir / config_filename
    if not config_file.is_file():
        msg = f"MolmoAct2 repository '{repo_id}' is missing required file '{config_filename}'."
        raise FileNotFoundError(msg)

    weights_file = snapshot_dir / weights_filename
    if not weights_file.is_file():
        weights_file = snapshot_dir / f"{weights_filename}.index.json"
    if not weights_file.is_file():
        msg = (
            f"MolmoAct2 repository '{repo_id}' is missing required weights "
            f"'{weights_filename}' or '{weights_filename}.index.json'."
        )
        raise FileNotFoundError(msg)

    tokenizer_file = snapshot_dir / TOKENIZER_NAME
    if not tokenizer_file.is_file():
        msg = f"MolmoAct2 repository '{repo_id}' is missing required file '{TOKENIZER_NAME}'."
        raise FileNotFoundError(msg)
    tokenizer_config_file = snapshot_dir / TOKENIZER_CONFIG_NAME
    if not tokenizer_config_file.is_file():
        msg = f"MolmoAct2 repository '{repo_id}' is missing required file '{TOKENIZER_CONFIG_NAME}'."
        raise FileNotFoundError(msg)

    preprocessor_candidate = snapshot_dir / preprocessor_filename
    preprocessor_file = preprocessor_candidate if preprocessor_candidate.is_file() else None
    preprocessor_dir = snapshot_dir if preprocessor_file is not None else None
    norm_stats_candidate = snapshot_dir / norm_stats_filename if norm_stats_filename is not None else None
    norm_stats_file = (
        norm_stats_candidate if norm_stats_candidate is not None and norm_stats_candidate.is_file() else None
    )
    return config_file, weights_file, preprocessor_file, preprocessor_dir, norm_stats_file


def _load_tokenizer_config(checkpoint_location: str) -> dict[str, Any]:
    """Load tokenizer construction options from the resolved snapshot.

    Returns:
        Parsed tokenizer construction options.

    Raises:
        FileNotFoundError: If the tokenizer config is missing.
        TypeError: If the tokenizer config is not a JSON object.
    """
    config_path = Path(checkpoint_location) / TOKENIZER_CONFIG_NAME
    if not config_path.is_file():
        msg = f"MolmoAct2 checkpoint at {checkpoint_location} must contain '{TOKENIZER_CONFIG_NAME}'."
        raise FileNotFoundError(msg)
    with config_path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        msg = f"Invalid MolmoAct2 tokenizer config in {config_path}: expected a JSON object."
        raise TypeError(msg)
    return payload


def resolve_tokenizer_assets(
    tokenizer_name_or_path: str | Path,
    *,
    hub_kwargs: HFHubDownloadKwargs | None = None,
) -> tuple[str, dict[str, Any]]:
    """Resolve only the tokenizer files needed for preprocessing.

    Args:
        tokenizer_name_or_path: Local tokenizer directory or Hugging Face repo ID.
        hub_kwargs: Optional Hugging Face snapshot arguments.

    Returns:
        Local tokenizer directory and parsed tokenizer construction options.

    Raises:
        FileNotFoundError: If either required tokenizer file is missing.
    """
    tokenizer_path = Path(tokenizer_name_or_path)
    if tokenizer_path.is_dir():
        tokenizer_dir = tokenizer_path
    else:
        tokenizer_dir = Path(
            snapshot_download(  # nosec B615 - tokenizer-only fallback follows the configured repository revision
                str(tokenizer_name_or_path),
                allow_patterns=TOKENIZER_ALLOW_PATTERNS,
                **(hub_kwargs or HFHubDownloadKwargs()),  # type: ignore[arg-type]
            ),
        )

    tokenizer_file = tokenizer_dir / TOKENIZER_NAME
    if not tokenizer_file.is_file():
        msg = f"Tokenizer source '{tokenizer_name_or_path}' is missing required file '{TOKENIZER_NAME}'."
        raise FileNotFoundError(msg)
    return str(tokenizer_dir), _load_tokenizer_config(str(tokenizer_dir))


def _hf_component_config(hf_config: dict[str, Any], component_name: str) -> dict[str, Any]:
    """Extract a nested HF component config.

    Args:
        hf_config: Parsed Hugging Face `config.json` payload.
        component_name: Nested component key to extract.

    Returns:
        Component config as a dict, or an empty dict when missing.
    """
    component_config = hf_config.get(component_name)
    if isinstance(component_config, dict):
        return component_config
    return {}


def _normalization_parameters_from_stats(stats: dict[str, Any]) -> NormalizationParameters:
    """Convert raw stats dict to `NormalizationParameters`.

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
        mask=stats.get("mask"),
    )


def _resolve_norm_tag_metadata(norm_stats: dict[str, Any] | None, norm_tag: str | None) -> dict[str, Any]:
    """Resolve normalization metadata for a selected tag.

    Args:
        norm_stats: Parsed `norm_stats.json` payload.
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


def _resolve_hf_image_size(hf_config: dict[str, Any]) -> tuple[int, int]:
    """Resolve input image size from HF ViT config.

    Args:
        hf_config: Parsed Hugging Face `config.json` payload.

    Returns:
        Image size tuple as `(height, width)`.

    Raises:
        TypeError: If the configured size is malformed.
    """
    image_size = _hf_component_config(hf_config, "vit_config").get("image_default_input_size", (378, 378))
    if isinstance(image_size, list):
        image_size = tuple(image_size)
    if not isinstance(image_size, tuple) or len(image_size) != IMAGE_SIZE_DIMS:
        msg = f"Invalid image_default_input_size in HF config: {image_size!r}"
        raise TypeError(msg)
    return image_size


def _build_visual_features(camera_keys: list[Any], image_size: tuple[int, int]) -> list[Feature]:
    """Build visual `Feature` entries from camera keys.

    Args:
        camera_keys: List of camera feature keys from norm stats metadata.
        image_size: Resolved image size `(height, width)`.

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


def _build_feature_from_stats(
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
        prefix: Prefix to strip from `feature_key` for output name.
        ftype: Feature type enum value.

    Returns:
        A constructed `Feature`.
    """
    return Feature(
        name=feature_key.removeprefix(prefix),
        ftype=ftype,
        shape=(len(stats.get("mean", [])),),
        normalization_data=_normalization_parameters_from_stats(stats),
    )


def _build_features_from_norm_stats(
    hf_config: dict[str, Any],
    norm_stats: dict[str, Any] | None,
    norm_tag: str | None,
) -> tuple[list[Feature], list[Feature]]:
    """Build input and output features from pretrained norm stats.

    Args:
        hf_config: Parsed Hugging Face `config.json` payload.
        norm_stats: Parsed `norm_stats.json` payload.
        norm_tag: Selected normalization metadata tag.

    Returns:
        Tuple of `(input_features, output_features)`.

    Raises:
        TypeError: If normalization metadata has an invalid shape or type.
    """
    tag_metadata = _resolve_norm_tag_metadata(norm_stats, norm_tag)
    image_size = _resolve_hf_image_size(hf_config)

    camera_keys = tag_metadata.get("camera_keys")
    if not isinstance(camera_keys, list):
        msg = f"Invalid camera_keys for norm_tag '{norm_tag}'."
        raise TypeError(msg)

    input_features = _build_visual_features(camera_keys, image_size)

    state_key = tag_metadata.get("state_key")
    state_stats = tag_metadata.get("state_stats")
    if isinstance(state_key, str) and isinstance(state_stats, dict):
        input_features.append(
            _build_feature_from_stats(
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
            _build_feature_from_stats(
                feature_key=action_key,
                stats=action_stats,
                prefix="",
                ftype=FeatureType.ACTION,
            ),
        )

    return input_features, output_features


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


def _resolve_feature_overrides(
    pretrained_features: list[Feature],
    override_features: list[Feature] | None,
) -> list[Feature]:
    """Resolve final feature list from pretrained and override inputs.

    Only exact names match. A renamed feature remains independent of the
    pretrained schema and never inherits its normalization metadata. A warning
    is logged when an override name is new or when a name-matched feature has a
    different shape from its pretrained definition.

    Args:
        pretrained_features: Features derived from pretrained metadata.
        override_features: Optional user-provided feature overrides.

    Returns:
        Final ordered feature list with overrides applied.

    """
    if not override_features:
        return pretrained_features

    overrides_by_name = {feature.name: feature for feature in override_features}
    resolved_features: list[Feature] = []
    for feature in pretrained_features:
        override_feature = overrides_by_name.get(feature.name)
        if override_feature is None:
            continue
        if (
            feature.shape is not None
            and override_feature.shape is not None
            and tuple(feature.shape) != tuple(override_feature.shape)
        ):
            logger.warning(
                "MolmoAct2 override feature %r changed shape from %s to %s.",
                feature.name,
                tuple(feature.shape),
                tuple(override_feature.shape),
            )
        resolved_features.append(_merge_feature_override(feature, override_feature))
    resolved_names = {feature.name for feature in resolved_features if feature.name is not None}

    unmatched_pretrained = [feature for feature in pretrained_features if feature.name not in resolved_names]
    unmatched_overrides = [feature for feature in override_features if feature.name not in resolved_names]

    for override_feature in unmatched_overrides:
        logger.warning(
            "MolmoAct2 override feature %r has no matching pretrained feature name; "
            "pretrained normalization will not be copied.",
            override_feature.name,
        )

    resolved_features.extend(unmatched_pretrained)
    resolved_features.extend(unmatched_overrides)

    return resolved_features


def _translate_component_configs(config_data: dict[str, Any], hf_config: dict[str, Any]) -> None:
    """Copy nested Hugging Face component fields into the flat config."""
    component_field_maps = {
        "vit_config": {
            "hidden_size": "vision_hidden_size",
            "intermediate_size": "vision_intermediate_size",
            "num_hidden_layers": "vision_num_hidden_layers",
            "num_attention_heads": "vision_num_attention_heads",
            "num_key_value_heads": "vision_num_key_value_heads",
            "head_dim": "vision_head_dim",
            "hidden_act": "vision_hidden_act",
            "layer_norm_eps": "vision_layer_norm_eps",
            "image_default_input_size": "image_default_input_size",
            "image_patch_size": "image_patch_size",
            "image_num_pos": "image_num_pos",
            "attention_dropout": "vision_attention_dropout",
            "residual_dropout": "vision_residual_dropout",
            "attn_implementation": "vision_attn_implementation",
        },
        "adapter_config": {
            "vit_layers": "adapter_vit_layers",
            "pooling_attention_mask": "adapter_pooling_attention_mask",
            "hidden_size": "adapter_hidden_size",
            "num_attention_heads": "adapter_num_attention_heads",
            "num_key_value_heads": "adapter_num_key_value_heads",
            "head_dim": "adapter_head_dim",
            "attention_dropout": "adapter_attention_dropout",
            "residual_dropout": "adapter_residual_dropout",
            "hidden_act": "adapter_hidden_act",
            "intermediate_size": "adapter_intermediate_size",
            "text_hidden_size": "adapter_text_hidden_size",
            "image_feature_dropout": "image_feature_dropout",
            "attn_implementation": "adapter_attn_implementation",
        },
        "text_config": {
            key: key
            for key in (
                "hidden_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "vocab_size",
                "additional_vocab_size",
                "qkv_bias",
                "num_hidden_layers",
                "intermediate_size",
                "hidden_act",
                "max_position_embeddings",
                "rope_theta",
                "use_qk_norm",
                "qk_norm_type",
                "layer_norm_eps",
                "norm_after",
                "use_cache",
            )
        }
        | {"attn_implementation": "text_attn_implementation"},
        "action_expert_config": {
            "max_action_horizon": "action_expert_max_action_horizon",
            "max_action_dim": "action_expert_max_action_dim",
            "hidden_size": "action_expert_hidden_size",
            "num_layers": "action_expert_num_layers",
            "num_heads": "action_expert_num_heads",
            "mlp_ratio": "action_expert_mlp_ratio",
            "ffn_multiple_of": "action_expert_ffn_multiple_of",
            "timestep_embed_dim": "action_expert_timestep_embed_dim",
            "context_layer_norm": "action_expert_context_layer_norm",
            "qk_norm": "action_expert_qk_norm",
            "qk_norm_eps": "action_expert_qk_norm_eps",
            "rope": "action_expert_rope",
            "causal_attn": "action_expert_causal_attn",
        },
    }
    for component_name, field_map in component_field_maps.items():
        component_data = _hf_component_config(hf_config, component_name)
        for source_key, target_key in field_map.items():
            if source_key in component_data:
                config_data[target_key] = component_data[source_key]


def _resolve_config_features(
    config_data: dict[str, Any],
    hf_config: dict[str, Any],
    norm_stats: dict[str, Any] | None,
    norm_tag: str | None,
    input_features: list[Feature] | None,
    output_features: list[Feature] | None,
) -> None:
    """Resolve feature schemas from normalization metadata and caller overrides."""
    if norm_tag is None:
        config_data["input_features"] = list(input_features or [])
        config_data["output_features"] = list(output_features or [])
        return

    pretrained_input_features, pretrained_output_features = _build_features_from_norm_stats(
        hf_config,
        norm_stats,
        norm_tag,
    )
    config_data["input_features"] = _resolve_feature_overrides(pretrained_input_features, input_features)
    config_data["output_features"] = _resolve_feature_overrides(pretrained_output_features, output_features)


def _translate_top_level_config(config_data: dict[str, Any], hf_config: dict[str, Any]) -> None:
    """Copy supported top-level Hugging Face config fields into the flat config."""
    keys = (
        "action_mode",
        "action_expert_num_heads",
        "action_expert_num_layers",
        "action_expert_hidden_size",
        "action_expert_max_action_dim",
        "action_expert_max_action_horizon",
        "add_action_expert",
        "chunk_size",
        "compile_model",
        "flow_matching_beta_alpha",
        "flow_matching_beta_beta",
        "flow_matching_cutoff",
        "flow_matching_num_steps",
        "flow_matching_time_offset",
        "flow_matching_time_scale",
        "frame_end_token_id",
        "frame_start_token_id",
        "image_col_id",
        "image_end_token_id",
        "image_low_res_id",
        "image_patch_id",
        "image_start_token_id",
        "low_res_image_start_token_id",
        "mask_action_dim_padding",
        "max_action_dim",
        "n_action_steps",
        "n_obs_steps",
        "num_flow_timesteps",
        "state_format",
        "use_random_input_noise",
        "num_state_tokens",
        "add_setup_tokens",
        "add_control_tokens",
    )
    for key in keys:
        if key in hf_config:
            config_data[key] = hf_config[key]

    checkpoint_action_horizon = hf_config.get("max_action_horizon")
    if checkpoint_action_horizon is not None:
        config_data["chunk_size"] = checkpoint_action_horizon
        config_data["n_action_steps"] = checkpoint_action_horizon


def _translate_processor_config(config_data: dict[str, Any], processor_config: dict[str, Any] | None) -> None:
    """Copy supported image processor and token-layout fields into the flat config."""
    if processor_config is None:
        return

    image_processor = processor_config.get("image_processor", processor_config)
    processor_field_map = {
        "crop_mode": "image_processor_crop_mode",
        "image_mean": "image_processor_mean",
        "image_std": "image_processor_std",
        "patch_size": "image_processor_patch_size",
        "pooling_size": "image_processor_pooling_size",
        "size": "image_processor_size",
    }
    for source_key, target_key in processor_field_map.items():
        if source_key in image_processor:
            config_data[target_key] = image_processor[source_key]

    for key in (
        "image_use_col_tokens",
        "use_single_crop_col_tokens",
        "use_single_crop_start_token",
    ):
        if key in processor_config:
            config_data[key] = processor_config[key]


def build_config_from_hf_config(
    hf_config: dict[str, Any],
    *,
    norm_stats: dict[str, Any] | None = None,
    input_features: list[Feature] | None = None,
    output_features: list[Feature] | None = None,
    norm_tag: str | None = None,
    checkpoint_path: str | None = None,
    repo_id: str | None = None,
    tokenizer_revision: str | None = None,
    tokenizer_config: dict[str, Any] | None = None,
    processor_config: dict[str, Any] | None = None,
    **overrides: object,
) -> MolmoAct2Config:
    """Build a flat policy config from Hugging Face data and explicit overrides.

    Args:
        hf_config: Parsed Hugging Face `config.json` payload.
        norm_stats: Parsed `norm_stats.json` payload.
        input_features: Optional input feature definitions.
        output_features: Optional output feature definitions.
        norm_tag: Selected normalization metadata tag.
        checkpoint_path: Local checkpoint directory.
        repo_id: Original Hugging Face repo id, retained for API compatibility.
        tokenizer_revision: Immutable commit revision for tokenizer assets.
        tokenizer_config: Optional parsed ``tokenizer_config.json`` options to
            carry into the config so the tokenizer can be rebuilt by downloading
            only ``tokenizer.json`` at runtime.
        processor_config: Optional pre-loaded processor config dict.
        **overrides: Flat :class:`MolmoAct2Config` values. ``None`` means
            retain the value supplied by the checkpoint or dataclass default.

    Returns:
        Resolved `MolmoAct2Config` instance.

    Raises:
        ValueError: If ``checkpoint_path`` is ``None``.
        TypeError: If unkown MolmoAct2 overrides are supplied.
    """
    del repo_id

    # Stage 1: start from local defaults and translate architecture fields from
    # the nested Hugging Face config.
    config_data = MolmoAct2Config().to_dict()

    _translate_component_configs(config_data, hf_config)

    # Stage 2: resolve normalized features from the selected dataset tag, then
    # apply caller-provided feature substitutions.
    _resolve_config_features(config_data, hf_config, norm_stats, norm_tag, input_features, output_features)
    _translate_top_level_config(config_data, hf_config)

    # Stage 3: attach local snapshot assets and image-processor settings.
    config_data["norm_tag"] = norm_tag
    if checkpoint_path is None:
        msg = "checkpoint_path is required to resolve MolmoAct2 pretrained assets."
        raise ValueError(msg)
    # Carry the resolved checkpoint snapshot directory on the config so it can
    # still locate the pretrained weights to load.
    config_data["checkpoint_path"] = checkpoint_path
    config_data["tokenizer_name_or_path"] = checkpoint_path
    config_data["tokenizer_revision"] = tokenizer_revision
    config_data["tokenizer_config"] = tokenizer_config
    # Hardcoded across MolmoAct2 variants (see ``config.py``); avoids
    # instantiating the tokenizer here just to look up the placeholder id.
    config_data["image_placeholder_token_id"] = MOLMOACT2_IMAGE_PLACEHOLDER_TOKEN_ID
    _translate_processor_config(config_data, processor_config)

    if norm_stats is not None and norm_tag is not None:
        tag_metadata = _resolve_norm_tag_metadata(norm_stats, norm_tag)
        config_data["setup_type"] = str(tag_metadata.get("setup_type") or "")
        config_data["control_mode"] = str(tag_metadata.get("control_mode") or "")

    # Stage 4: validate and apply explicit caller overrides last.
    valid_fields = set(config_data)
    unknown = set(overrides) - valid_fields
    if unknown:
        msg = f"Unknown MolmoAct2 override(s): {sorted(unknown)}"
        raise TypeError(msg)
    config_data.update({key: value for key, value in overrides.items() if value is not None})

    return MolmoAct2Config.from_dict(config_data)


def load_hf_pretrained_container(  # noqa: PLR0914
    pretrained_name_or_path: str | Path,
    *,
    config_filename: str = "config.json",
    processor_filename: str = "processor_config.json",
    norm_stats_filename: str = "norm_stats.json",
    **kwargs: object,
) -> MolmoAct2Snapshot:
    """Resolve local or HF artifacts for MolmoAct2 pretrained loading.

    Args:
        pretrained_name_or_path: Local checkpoint directory or HF repo id.
        config_filename: Local filename for config json file.
        processor_filename: Local filename for processor config json file.
        norm_stats_filename: Normalization stats filename to resolve.
        **kwargs: Optional HF hub download kwargs.

    Returns:
        Container with resolved artifact paths and parsed config payloads.

    Raises:
        FileNotFoundError: When looking for local weight files.
    """
    path = Path(pretrained_name_or_path)
    is_local = path.is_dir()
    norm_stats: dict[str, Any] | None = None
    config_file: Path
    weights_file: Path
    preprocessor_file: Path | None
    preprocessor_dir: Path | None
    norm_stats_file: Path | None
    # ``None`` for local checkpoints (no Hub repo id); the original pretrained
    # identifier otherwise. This is later used as the tokenizer source.
    repo_id: str | None = None if is_local else str(pretrained_name_or_path)
    tokenizer_revision: str | None = None

    if is_local:
        config_file = path / config_filename
        weights_file = path / SAFE_WEIGHTS_NAME
        if not weights_file.is_file():
            weights_file = path / SAFE_WEIGHTS_INDEX_NAME
        if not weights_file.is_file():
            msg = f"MolmoAct2 local checkpoint at {path} must contain {SAFE_WEIGHTS_NAME} or {SAFE_WEIGHTS_INDEX_NAME}."
            raise FileNotFoundError(msg)
        preprocessor_file = path / processor_filename
        preprocessor_dir = path
        norm_stats_file = path / norm_stats_filename
        if norm_stats_file.is_file():
            with norm_stats_file.open(encoding="utf-8") as f:
                norm_stats = json.load(f)
    else:
        hub_kwargs = _select_hf_hub_download_kwargs(kwargs)
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
        tokenizer_revision = _resolve_snapshot_revision(config_file, hub_kwargs.get("revision"))
        if norm_stats_file is not None:
            with norm_stats_file.open(encoding="utf-8") as f:
                norm_stats = json.load(f)

    with Path(config_file).open(encoding="utf-8") as f:
        hf_config = json.load(f)

    checkpoint_location = str(Path(weights_file).parent)
    tokenizer_file = Path(checkpoint_location) / TOKENIZER_NAME
    if not tokenizer_file.is_file():
        msg = f"MolmoAct2 checkpoint at {checkpoint_location} must contain '{TOKENIZER_NAME}'."
        raise FileNotFoundError(msg)

    # Load processor config from checkpoint location
    processor_config: dict[str, Any] | None = None
    processor_config_path = Path(checkpoint_location) / "processor_config.json"
    if processor_config_path.exists():
        with processor_config_path.open(encoding="utf-8") as f:
            processor_config = json.load(f)

    tokenizer_config = _load_tokenizer_config(checkpoint_location)

    return MolmoAct2Snapshot(
        config_file=Path(config_file),
        weights_file=Path(weights_file),
        preprocessor_file=Path(preprocessor_file) if preprocessor_file is not None else None,
        preprocessor_dir=Path(preprocessor_dir) if preprocessor_dir is not None else None,
        checkpoint_location=checkpoint_location,
        hf_config=hf_config,
        norm_stats=norm_stats,
        processor_config=processor_config,
        repo_id=repo_id,
        tokenizer_revision=tokenizer_revision,
        tokenizer_config=tokenizer_config,
    )
