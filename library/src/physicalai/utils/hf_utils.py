# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face Hub utility helpers shared across policies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RemoteEntryNotFoundError

_HF_HUB_DOWNLOAD_KEYS = {
    "cache_dir",
    "force_download",
    "resume_download",
    "proxies",
    "token",
    "revision",
    "local_files_only",
}


@dataclass
class HuggingfacePolicyContainer:
    """Container for artifacts resolved by policy ``_from_hf`` loaders.

    Attributes:
        config_file: Path to downloaded/resolved ``config.json``.
        weights_file: Path to weights file or weights index file.
        preprocessor_file: Optional path to ``policy_preprocessor.json``.
        preprocessor_dir: Optional directory containing preprocessor state files.
        checkpoint_location: Directory where checkpoint files live.
        hf_config: Parsed contents of ``config.json``.
    """

    config_file: Path
    weights_file: Path
    preprocessor_file: Path | None
    preprocessor_dir: Path | None
    checkpoint_location: str
    hf_config: dict[str, Any]


def _download_optional_preprocessor(
    repo_id: str,
    preprocessor_filename: str,
    *,
    hub_kwargs: dict[str, object],
) -> Path | None:
    try:
        return Path(hf_hub_download(repo_id, preprocessor_filename, **hub_kwargs))  # nosec B615  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        return None


def _download_referenced_state_files(
    repo_id: str,
    preprocessor_file: Path,
    *,
    hub_kwargs: dict[str, object],
) -> None:
    with preprocessor_file.open(encoding="utf-8") as f:
        preproc_data = json.load(f)
    for step in preproc_data.get("steps", []):
        state_file = step.get("state_file")
        if state_file:
            hf_hub_download(repo_id, state_file, **hub_kwargs)  # nosec B615  # type: ignore[arg-type]


def _download_weights_or_shards(
    repo_id: str,
    *,
    weights_filename: str,
    hub_kwargs: dict[str, object],
) -> Path:
    """Download either a single safetensors file or index+all shard files.

    Returns:
        Path to the downloaded single safetensors file or to the sharded
        safetensors index file.

    Raises:
        FileNotFoundError: If neither single-file nor sharded weights exist.
        TypeError: If the sharded index has an invalid or missing weight map.
    """
    try:
        return Path(hf_hub_download(repo_id, weights_filename, **hub_kwargs))  # nosec B615  # type: ignore[arg-type]
    except RemoteEntryNotFoundError as single_exc:
        index_filename = f"{weights_filename}.index.json"
        try:
            index_path = Path(hf_hub_download(repo_id, index_filename, **hub_kwargs))  # nosec B615  # type: ignore[arg-type]
        except RemoteEntryNotFoundError as index_exc:
            msg = (
                f"Could not find weights for repo '{repo_id}'. Expected either "
                f"'{weights_filename}' or '{index_filename}'."
            )
            raise FileNotFoundError(msg) from index_exc

        with index_path.open(encoding="utf-8") as f:
            index_payload = json.load(f)
        weight_map = index_payload.get("weight_map")
        if not isinstance(weight_map, dict):
            msg = f"Invalid sharded index format in '{index_filename}': missing 'weight_map'."
            raise TypeError(msg) from single_exc

        for shard_file in sorted(set(weight_map.values())):
            hf_hub_download(repo_id, shard_file, **hub_kwargs)  # nosec B615  # type: ignore[arg-type]
        return index_path


def download_policy_artifacts_from_hub(
    repo_id: str,
    *,
    hub_kwargs: dict[str, object] | None = None,
    config_filename: str = "config.json",
    weights_filename: str = "model.safetensors",
    preprocessor_filename: str = "policy_preprocessor.json",
    download_preprocessor_state_files: bool = True,
) -> tuple[Path, Path, Path | None, Path | None]:
    """Download standard policy artifacts from a Hugging Face model repo.

    Args:
        repo_id: Hugging Face model repository ID.
        config_filename: Model config file name in the repo.
        weights_filename: Model weights file name in the repo.
        preprocessor_filename: Optional preprocessor JSON file name.
        download_preprocessor_state_files: If True, also download files referenced
            by ``state_file`` entries inside the preprocessor JSON.
        hub_kwargs: Optional keyword arguments forwarded to
            ``huggingface_hub.hf_hub_download``.

    Returns:
        Tuple of ``(config_file, weights_file, preprocessor_file, preprocessor_dir)``.
        If the preprocessor file is missing or invalid, preprocessor values are ``None``.
        ``weights_file`` is either the single safetensors file path or the
        safetensors index path for sharded checkpoints.
    """
    selected_hub_kwargs = {
        k: v
        for k, v in (hub_kwargs or {}).items()
        if k in _HF_HUB_DOWNLOAD_KEYS
    }

    config_file = Path(
        hf_hub_download(repo_id, config_filename, **selected_hub_kwargs),  # nosec B615  # type: ignore[arg-type]
    )
    weights_file = _download_weights_or_shards(
        repo_id,
        weights_filename=weights_filename,
        hub_kwargs=selected_hub_kwargs,
    )

    preprocessor_file = _download_optional_preprocessor(
        repo_id,
        preprocessor_filename,
        hub_kwargs=selected_hub_kwargs,
    )
    if preprocessor_file is None:
        preprocessor_file = None
        preprocessor_dir = None
    else:
        preprocessor_dir = preprocessor_file.parent
        if download_preprocessor_state_files:
            try:
                _download_referenced_state_files(repo_id, preprocessor_file, hub_kwargs=selected_hub_kwargs)
            except Exception:  # noqa: BLE001
                preprocessor_file = None
                preprocessor_dir = None

    return config_file, weights_file, preprocessor_file, preprocessor_dir
