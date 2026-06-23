# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: ANN,COM812,D,E501,EM,FBT001,I001,PLR2004,S105,SIM108,TCH001,TRY
# pylint: disable=all

"""Processors for MolmoAct2."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch

from physicalai.data.observation import ACTION, IMAGES, STATE, TASK, Feature, FeatureType
from physicalai.policies.molmoact2.local_processor import load_molmoact2_processor_from_pretrained
from physicalai.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType
from physicalai.utils.hf_utils import HuggingfacePolicyContainer

ACTION_OUTPUT_TOKEN = "<action_output>"
SETUP_START_TOKEN = "<setup_start>"
SETUP_END_TOKEN = "<setup_end>"
CONTROL_START_TOKEN = "<control_start>"
CONTROL_END_TOKEN = "<control_end>"
STATE_START_TOKEN = "<state_start>"
STATE_END_TOKEN = "<state_end>"
STATE_TOKEN_PREFIX = "<state_"

_TRAILING_PUNCTUATION = ".,!?;:"
_PREFIX_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"^(?:task|instruction|language[_ ]instruction|goal)\s*[:\-]\s*",
        r"^(?:the\s+task\s+is\s+to|your\s+task\s+is\s+to)\s+",
    )
)


def _feature_by_type(features: list[Feature], feature_type: FeatureType) -> Feature | None:
    for feature in features:
        if feature.ftype == feature_type:
            return feature
    return None


def _as_tensor(value: torch.Tensor | np.ndarray | None, *, dtype: torch.dtype = torch.float32) -> torch.Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _normalize_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return ""
    for pattern in _PREFIX_PATTERNS:
        normalized = pattern.sub("", normalized, count=1).strip()
    normalized = normalized.rstrip(_TRAILING_PUNCTUATION).strip()
    return normalized.lower()


def _wrap_setup_text(setup_type: str, add_setup_tokens: bool) -> str:
    if not setup_type:
        return ""
    if not add_setup_tokens:
        return setup_type
    if setup_type.startswith(SETUP_START_TOKEN) and setup_type.endswith(SETUP_END_TOKEN):
        return setup_type
    return f"{SETUP_START_TOKEN}{setup_type}{SETUP_END_TOKEN}"


def _wrap_control_text(control_mode: str, add_control_tokens: bool) -> str:
    if not control_mode:
        return ""
    if not add_control_tokens:
        return control_mode
    if control_mode.startswith(CONTROL_START_TOKEN) and control_mode.endswith(CONTROL_END_TOKEN):
        return control_mode
    return f"{CONTROL_START_TOKEN}{control_mode}{CONTROL_END_TOKEN}"


def _build_discrete_state_string(state: np.ndarray, num_state_tokens: int) -> str:
    if num_state_tokens <= 0:
        raise ValueError(f"num_state_tokens must be > 0, got {num_state_tokens}.")
    arr = np.asarray(state, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    arr = np.clip(arr, -1.0, 1.0)
    scaled = (arr + 1.0) / 2.0 * float(num_state_tokens - 1)
    token_ids = np.clip(np.rint(scaled).astype(np.int64), 0, int(num_state_tokens) - 1).reshape(-1)
    return f"{STATE_START_TOKEN}{''.join(f'{STATE_TOKEN_PREFIX}{int(token_id)}>' for token_id in token_ids)}{STATE_END_TOKEN}"


def _build_robot_text(
    *,
    task: str,
    discrete_state_string: str,
    setup_type: str,
    control_mode: str,
    add_setup_tokens: bool,
    add_control_tokens: bool,
    num_images: int,
) -> str:
    setup_text = _wrap_setup_text(setup_type, add_setup_tokens=add_setup_tokens)
    control_text = _wrap_control_text(control_mode, add_control_tokens=add_control_tokens)
    state_clause = f" The current state of the robot is {discrete_state_string}." if discrete_state_string else ""
    prompt = (
        f"The task is to {task}. The setup is {setup_text}.{state_clause} "
        f"The expected control mode is {control_text}. Given these, what action should the robot take to complete the task?"
    )
    if num_images <= 0:
        image_prefix = ""
    elif num_images == 1:
        image_prefix = "<|image|>"
    else:
        image_prefix = "".join(f"Image {idx + 1}<|image|>" for idx in range(num_images))
    return f"{image_prefix}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{ACTION_OUTPUT_TOKEN}"


def _to_numpy_image(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape for MolmoAct2: {arr.shape}")

    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype in (np.float16, np.float32, np.float64):
        if arr.size > 0 and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _text_max_positions(config: Any, *, default: int = 4096) -> int:
    text_config = getattr(config, "text_config", None)
    if isinstance(text_config, dict):
        value = text_config.get("max_position_embeddings", default)
        return int(value)
    value = getattr(text_config, "max_position_embeddings", default)
    return int(value)


class MolmoAct2Preprocessor(torch.nn.Module):
    """Pack Observation objects into MolmoAct2 model-ready tensors."""

    def __init__(
        self,
        config: Any,
        *,
        hf_container: HuggingfacePolicyContainer | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hf_container = hf_container

        self.state_feature = _feature_by_type(config.input_features, FeatureType.STATE)
        self.action_feature = _feature_by_type(config.output_features, FeatureType.ACTION)

        all_features = {f.name: f for f in config.input_features + config.output_features if f.name}
        norm_map = {FeatureType.STATE: NormalizationType.QUANTILES, FeatureType.ACTION: NormalizationType.QUANTILES, FeatureType.VISUAL: NormalizationType.IDENTITY}
        self._normalizer = FeatureNormalizeTransform(all_features, norm_map, inverse=False)

        self.max_action_dim = int(config.max_action_dim)
        self.chunk_size = int(config.chunk_size)
        self.num_state_tokens = int(config.num_state_tokens) if int(config.num_state_tokens) > 0 else 256
        self.max_sequence_length = _text_max_positions(config)
        self.action_mode = str(config.action_mode)
        self.add_setup_tokens = bool(config.add_setup_tokens)
        self.add_control_tokens = bool(config.add_control_tokens)

        self.setup_type, self.control_mode = self._resolve_setup_and_control_mode()
        self.image_keys = [feature.name for feature in config.input_features if feature.ftype == FeatureType.VISUAL and feature.name]
        self.env_action_dim = int(self.action_feature.shape[0]) if self.action_feature and self.action_feature.shape else 0

        self._processor: Any = None

    def _resolve_setup_and_control_mode(self) -> tuple[str, str]:
        if self.hf_container is None or self.hf_container.norm_stats is None or not self.config.norm_tag:
            return "", ""
        metadata_by_tag = self.hf_container.norm_stats.get("metadata_by_tag")
        if not isinstance(metadata_by_tag, dict):
            return "", ""
        metadata = metadata_by_tag.get(self.config.norm_tag)
        if not isinstance(metadata, dict):
            return "", ""
        return str(metadata.get("setup_type") or ""), str(metadata.get("control_mode") or "")

    @property
    def processor(self) -> Any:
        if self._processor is not None:
            return self._processor

        processor_assets_path = self.config.processor_assets_path
        if not processor_assets_path:
            raise ValueError("MolmoAct2 processor requires processor_assets_path in config.")

        self._processor = load_molmoact2_processor_from_pretrained(processor_assets_path)
        return self._processor

    def _resolve_image_keys(self, observation: dict[str, Any]) -> list[str]:
        requested = [f"{IMAGES}.{name}" for name in self.image_keys if f"{IMAGES}.{name}" in observation]
        if requested:
            return requested
        fallback = [
            key
            for key in observation
            if str(key).startswith(f"{IMAGES}.") or str(key).startswith("observation.images.")
        ]
        if not fallback:
            raise ValueError("MolmoAct2 requires at least one image observation.")
        return sorted(str(key) for key in fallback)

    def _extract_images(self, observation: dict[str, Any], batch_size: int) -> list[list[np.ndarray]]:
        images_by_example: list[list[np.ndarray]] = [[] for _ in range(batch_size)]
        for key in self._resolve_image_keys(observation):
            value = observation[key]
            for batch_idx in range(batch_size):
                item = value
                if (torch.is_tensor(value) or isinstance(value, np.ndarray)) and getattr(value, "ndim", 0) >= 4:
                    item = value[batch_idx]
                images_by_example[batch_idx].append(_to_numpy_image(item))
        return images_by_example

    @staticmethod
    def _extract_tasks(observation: dict[str, Any], batch_size: int) -> list[str]:
        task_source = observation.get(TASK)
        if task_source is None:
            task_source = observation.get(f"observation.{TASK}")
        if task_source is None:
            task_source = observation.get("observation.language")

        if task_source is None:
            tasks = [""] * batch_size
        elif isinstance(task_source, str):
            tasks = [task_source] * batch_size
        elif torch.is_tensor(task_source):
            if task_source.ndim == 0:
                tasks = [str(task_source.item())] * batch_size
            else:
                tasks = [str(item) for item in task_source.detach().cpu().reshape(-1).tolist()]
        elif isinstance(task_source, np.ndarray):
            tasks = [str(item) for item in task_source.reshape(-1).tolist()]
        elif isinstance(task_source, (list, tuple)):
            tasks = [str(item) for item in task_source]
        else:
            tasks = [str(task_source)]

        if len(tasks) == 1 and batch_size > 1:
            tasks = tasks * batch_size
        if len(tasks) != batch_size:
            raise ValueError(f"Expected {batch_size} task strings, got {len(tasks)}.")
        return [_normalize_text(task) for task in tasks]

    def _pad_action(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad a normalized action tensor to max_action_dim."""
        if action.ndim == 2:
            action = action.unsqueeze(1)
        if action.ndim != 3:
            raise ValueError(f"MolmoAct2 expected action shape [B, T, D], got {tuple(action.shape)}.")
        if int(action.shape[-1]) > self.max_action_dim:
            raise ValueError(
                f"Action dim {action.shape[-1]} exceeds MolmoAct2 max_action_dim={self.max_action_dim}."
            )

        normalized = action.to(dtype=torch.float32).clamp(-1.0, 1.0)
        padded = torch.zeros((*normalized.shape[:-1], self.max_action_dim), device=normalized.device, dtype=torch.float32)
        padded[..., : normalized.shape[-1]] = normalized

        action_dim_is_pad = torch.ones((normalized.shape[0], self.max_action_dim), device=normalized.device, dtype=torch.bool)
        action_dim_is_pad[:, : normalized.shape[-1]] = False
        action_horizon_is_pad = torch.zeros(normalized.shape[:2], device=normalized.device, dtype=torch.bool)
        return padded, action_horizon_is_pad, action_dim_is_pad

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(batch, dict):
            msg = f"MolmoAct2Preprocessor.forward expects dict[str, Any], got {type(batch)}"
            raise TypeError(msg)

        # Normalize state/action in-place using feature normalization stats.
        device = next(
            (v.device for v in batch.values() if torch.is_tensor(v)),
            torch.device("cpu"),
        )
        batch = self._normalizer.to(device)(batch)

        # Extract normalized state.
        raw_state = batch.get(STATE)
        if raw_state is None:
            raw_state = batch.get(f"observation.{STATE}")
        if raw_state is None:
            raise ValueError("MolmoAct2 requires state for discrete state prompting.")
        state = torch.as_tensor(raw_state, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state = state.clamp(-1.0, 1.0)
        batch_size = state.shape[0]

        # Extract task strings and normalize text.
        tasks = self._extract_tasks(batch, batch_size)

        # Extract images as per-example numpy lists.
        images_by_example = self._extract_images(batch, batch_size)

        # Build per-example text prompts.
        state_np = state.detach().cpu().numpy()
        prompt_texts: list[str] = []
        flat_images: list[np.ndarray] = []
        for i in range(batch_size):
            flat_images.extend(images_by_example[i])
            discrete_state = _build_discrete_state_string(state_np[i], self.num_state_tokens)
            prompt_texts.append(_build_robot_text(
                task=tasks[i],
                discrete_state_string=discrete_state,
                setup_type=self.setup_type,
                control_mode=self.control_mode,
                add_setup_tokens=self.add_setup_tokens,
                add_control_tokens=self.add_control_tokens,
                num_images=len(images_by_example[i]),
            ))

        # Tokenize and encode images via HF processor.
        inputs = self.processor(text=prompt_texts, images=flat_images, return_tensors="pt", padding=True)
        if int(inputs["input_ids"].shape[1]) > self.max_sequence_length:
            raise ValueError(
                f"MolmoAct2 sequence length {int(inputs['input_ids'].shape[1])} exceeds max_sequence_length={self.max_sequence_length}."
            )

        # Pad normalized action if present.
        action_padded = None
        action_horizon_is_pad = None
        action_dim_is_pad = torch.ones((batch_size, self.max_action_dim), dtype=torch.bool)
        raw_action = batch.get(ACTION)
        if raw_action is None:
            raw_action = batch.get(f"action.{ACTION}")
        action = _as_tensor(raw_action)
        if action is not None:
            action_padded, action_horizon_is_pad, action_dim_is_pad = self._pad_action(action)
        elif self.env_action_dim > 0:
            action_dim_is_pad[:, : self.env_action_dim] = False

        packed: dict[str, Any] = dict(inputs)
        packed["task"] = tasks
        packed["state"] = state
        packed["action_dim_is_pad"] = action_dim_is_pad
        if action_horizon_is_pad is not None:
            packed["action_horizon_is_pad"] = action_horizon_is_pad
        if action_padded is not None:
            packed[ACTION] = action_padded
        return packed


class MolmoAct2Postprocessor(torch.nn.Module):
    """Convert normalized MolmoAct2 outputs back into action space."""

    def __init__(self, *, config: Any) -> None:
        super().__init__()
        action_feature = _feature_by_type(config.output_features, FeatureType.ACTION)
        self.env_action_dim = int(action_feature.shape[0]) if action_feature and action_feature.shape else int(config.max_action_dim)
        self.action_name = action_feature.name if action_feature else ACTION
        output_features = {f.name: f for f in config.output_features if f.name}
        self._denormalizer = FeatureNormalizeTransform(
            output_features,
            {FeatureType.ACTION: NormalizationType.QUANTILES},
            inverse=True,
        )

    def forward(self, outputs: torch.Tensor | dict[str, Any]) -> torch.Tensor:
        if isinstance(outputs, dict):
            if ACTION in outputs:
                action = torch.as_tensor(outputs[ACTION])
            elif "actions" in outputs:
                action = torch.as_tensor(outputs["actions"])
            else:
                raise ValueError("MolmoAct2 postprocessor expected an action tensor in outputs.")
        else:
            action = torch.as_tensor(outputs)

        action = action[..., : self.env_action_dim]
        action = action.clamp(-1.0, 1.0)
        return self._denormalizer.to(action.device)({self.action_name: action})[self.action_name]


def make_molmoact2_preprocessors(
    config: Any,
    *,
    hf_container: HuggingfacePolicyContainer | None = None,
) -> tuple[MolmoAct2Preprocessor, MolmoAct2Postprocessor]:
    preprocessor = MolmoAct2Preprocessor(config=config, hf_container=hf_container)
    postprocessor = MolmoAct2Postprocessor(config=config)
    return preprocessor, postprocessor
