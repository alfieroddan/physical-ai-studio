# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face loading helpers for MolmoAct2.

This module keeps the Hugging Face-specific loading flow separate from the
policy itself. The steps are:

1. Resolve the checkpoint location from either a local directory or the
   Hugging Face Hub.
2. Download or open the main artifacts needed by MolmoAct2:
   - `config.json`
   - `model.safetensors` or `model.safetensors.index.json`
   - optional `policy_preprocessor.json`
   - optional `norm_stats.json`
3. Parse `config.json` and `norm_stats.json` into in-memory payloads.
4. Build pretrained input/output `Feature` definitions from the selected
   `norm_tag` metadata.
5. Merge any caller-provided feature overrides and top-level config overrides.
6. Return a `HuggingfacePolicyContainer` carrying the resolved artifact paths
   plus parsed JSON payloads for the policy to consume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RemoteEntryNotFoundError

from physicalai.data.observation import Feature, FeatureType, NormalizationParameters
from physicalai.utils.hf_utils import HuggingfacePolicyContainer, download_policy_artifacts_from_hub

from .config import MolmoAct2Config

SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
IMAGE_SIZE_DIMS = 2


def _ensure_processor_files_downloaded(
    repo_id: str,
    checkpoint_location: str,
    hub_kwargs: dict[str, object] | None = None,
) -> None:
    """Download custom processor Python files to the checkpoint snapshot directory.

    When loading from HF hub with trust_remote_code=True, transformers needs these files
    in the model cache directory to load the custom processor and configuration classes.

    Args:
        repo_id: HuggingFace model repository ID.
        checkpoint_location: Path to the checkpoint snapshot directory.
        hub_kwargs: Optional HuggingFace hub download kwargs.
    """
    processor_files = [
        "processing_molmoact2.py",
        "configuration_molmoact2.py",
        "modeling_molmoact2.py",
        "image_processing_molmoact2.py",
        "video_processing_molmoact2.py",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ]

    selected_hub_kwargs = {
        k: v
        for k, v in (hub_kwargs or {}).items()
        if k in {"cache_dir", "force_download", "resume_download", "proxies", "token", "revision", "local_files_only"}
    }

    for filename in processor_files:
        try:
            downloaded_path = hf_hub_download(
                repo_id,
                filename,
                **selected_hub_kwargs,  # type: ignore[arg-type]
            )
            # The file is now in cache; ensure it's also in the checkpoint_location snapshot dir
            target_path = Path(checkpoint_location) / filename
            if not target_path.exists():
                import shutil

                shutil.copy2(downloaded_path, target_path)
        except RemoteEntryNotFoundError:
            # File doesn't exist in repo, skip it
            pass


def _resolve_local_weights_path(checkpoint_dir: Path) -> Path:
    """Resolve a local safetensors weights file.

    Args:
        checkpoint_dir: Local checkpoint directory.

    Returns:
        Path to `model.safetensors` or `model.safetensors.index.json`.

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
        _merge_feature_override(feature, overrides_by_name[feature.name])
        for feature in pretrained_features
        if feature.name in overrides_by_name
    ]
    resolved_names = {feature.name for feature in resolved_features if feature.name is not None}

    unmatched_pretrained = [feature for feature in pretrained_features if feature.name not in resolved_names]
    unmatched_overrides = [feature for feature in override_features if feature.name not in resolved_names]

    # Name-based matching is preferred. For remaining overrides (for example,
    # camera alias remaps like wrist_image -> image2), match by type/shape to
    # preserve pretrained normalization metadata while allowing renamed keys.
    consumed_pretrained_ids: set[int] = set()
    consumed_override_ids: set[int] = set()
    for override_feature in unmatched_overrides:
        for pretrained_feature in unmatched_pretrained:
            if id(pretrained_feature) in consumed_pretrained_ids:
                continue
            if pretrained_feature.ftype != override_feature.ftype:
                continue

            pretrained_shape = tuple(pretrained_feature.shape) if pretrained_feature.shape is not None else None
            override_shape = tuple(override_feature.shape) if override_feature.shape is not None else None
            if pretrained_shape is not None and override_shape is not None and pretrained_shape != override_shape:
                continue

            resolved_features.append(_merge_feature_override(pretrained_feature, override_feature))
            consumed_pretrained_ids.add(id(pretrained_feature))
            consumed_override_ids.add(id(override_feature))
            break

    resolved_features.extend(
        [feature for feature in unmatched_pretrained if id(feature) not in consumed_pretrained_ids],
    )
    resolved_features.extend(
        [feature for feature in unmatched_overrides if id(feature) not in consumed_override_ids],
    )

    return resolved_features


def build_config_from_hf_config(
    hf_config: dict[str, Any],
    *,
    norm_stats: dict[str, Any] | None = None,
    input_features: list[Feature] | None = None,
    output_features: list[Feature] | None = None,
    norm_tag: str | None = None,
    checkpoint_path: str | None = None,
    processor_config: dict[str, Any] | None = None,
    n_obs_steps: int = 30,
    chunk_size: int = 30,
    n_action_steps: int = 30,
    max_action_dim: int = 32,
) -> MolmoAct2Config:
    """Build policy config from Hugging Face config and local overrides.

    Args:
        hf_config: Parsed Hugging Face `config.json` payload.
        norm_stats: Parsed `norm_stats.json` payload.
        input_features: Optional input feature definitions.
        output_features: Optional output feature definitions.
        norm_tag: Selected normalization metadata tag.
        checkpoint_path: Local checkpoint directory containing extended tokenizer vocab.
        processor_config: Optional pre-loaded processor config dict.
        n_obs_steps: Observation horizon override.
        chunk_size: Action chunk size override.
        n_action_steps: Number of executed action steps override.
        max_action_dim: Maximum action dimension override.

    Returns:
        Resolved `MolmoAct2Config` instance.
    """
    config_data: dict[str, Any] = {
        "vit_config": _hf_component_config(hf_config, "vit_config"),
        "adapter_config": _hf_component_config(hf_config, "adapter_config"),
        "text_config": _hf_component_config(hf_config, "text_config"),
        "action_expert_config": _hf_component_config(hf_config, "action_expert_config"),
    }
    pretrained_input_features, pretrained_output_features = _build_features_from_norm_stats(
        hf_config,
        norm_stats,
        norm_tag,
    )
    config_data["input_features"] = _resolve_feature_overrides(pretrained_input_features, input_features)
    config_data["output_features"] = _resolve_feature_overrides(pretrained_output_features, output_features)

    top_level_keys = (
        "action_mode",
        "action_end_token_id",
        "action_expert_depth_gate",
        "action_expert_depth_gate_init_bias",
        "action_expert_depth_gate_per_layer",
        "action_output_token_id",
        "action_start_token_id",
        "action_token_start_id",
        "add_action_expert",
        "add_control_tokens",
        "add_setup_tokens",
        "compile_model",
        "depth_end_token_id",
        "depth_mode",
        "depth_output_token_id",
        "depth_start_token_id",
        "depth_token_start_id",
        "enable_depth_reasoning",
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
        "initializer_range",
        "low_res_image_start_token_id",
        "mask_action_dim_padding",
        "n_obs_steps",
        "num_action_tokens",
        "num_depth_codes",
        "num_depth_tokens",
        "num_state_tokens",
        "state_end_token_id",
        "state_format",
        "state_start_token_id",
        "state_token_start_id",
        "use_frame_special_tokens",
        "image_start_token_id",
        "image_low_res_id",
        "image_patch_id",
        "image_col_id",
        "num_action_tokens",
        "depth_output_token_id",
        "depth_start_token_id",
        "depth_end_token_id",
        "depth_token_start_id",
        "max_action_dim",
        "chunk_size",
        "n_action_steps",
        "norm_stats_filename",
        "action_start_token_id",
        "add_setup_tokens",
        "action_expert_num_heads",
        "action_expert_num_layers",
        "action_expert_hidden_size",
        "action_expert_max_action_dim",
        "action_expert_max_action_horizon",
    )

    for key in top_level_keys:
        if key == "norm_tag" and norm_tag is not None:
            continue
        if key in hf_config:
            config_data[key] = hf_config[key]

    config_data["norm_tag"] = norm_tag
    config_data["tokenizer_name_or_path"] = checkpoint_path
    config_data["processor_assets_path"] = checkpoint_path
    config_data["processor_config"] = processor_config

    if norm_stats is not None and norm_tag is not None:
        tag_metadata = _resolve_norm_tag_metadata(norm_stats, norm_tag)
        config_data["setup_type"] = str(tag_metadata.get("setup_type") or "")
        config_data["control_mode"] = str(tag_metadata.get("control_mode") or "")
    config_data["n_obs_steps"] = n_obs_steps
    config_data["chunk_size"] = chunk_size
    config_data["n_action_steps"] = n_action_steps
    config_data["max_action_dim"] = max_action_dim

    if config_data.get("add_action_expert") is False:
        config_data["action_expert_config"] = None

    return MolmoAct2Config.from_dict(config_data)


def load_hf_pretrained_container(
    pretrained_name_or_path: str | Path,
    *,
    config_filename: str = "config.json",
    processor_filename: str = "processor_config.json",
    norm_stats_filename: str = "norm_stats.json",
    **kwargs: object,
) -> HuggingfacePolicyContainer:
    """Resolve local or HF artifacts for MolmoAct2 pretrained loading.

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
        config_file = path / config_filename
        weights_file = _resolve_local_weights_path(path)
        preprocessor_file = path / processor_filename
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

        # Download custom processor Python files to the snapshot directory
        checkpoint_location_temp = str(Path(weights_file).parent)
        _ensure_processor_files_downloaded(
            str(pretrained_name_or_path),
            checkpoint_location_temp,
            hub_kwargs=hub_kwargs,
        )

    with Path(config_file).open(encoding="utf-8") as f:
        hf_config = json.load(f)

    checkpoint_location = str(Path(weights_file).parent)

    # Load processor config from checkpoint location
    processor_config: dict[str, Any] | None = None
    processor_config_path = Path(checkpoint_location) / "processor_config.json"
    if processor_config_path.exists():
        with processor_config_path.open(encoding="utf-8") as f:
            processor_config = json.load(f)

    return HuggingfacePolicyContainer(
        config_file=Path(config_file),
        weights_file=Path(weights_file),
        preprocessor_file=Path(preprocessor_file) if preprocessor_file is not None else None,
        preprocessor_dir=Path(preprocessor_dir) if preprocessor_dir is not None else None,
        checkpoint_location=checkpoint_location,
        hf_config=hf_config,
        norm_stats=norm_stats,
        processor_config=processor_config,
    )
