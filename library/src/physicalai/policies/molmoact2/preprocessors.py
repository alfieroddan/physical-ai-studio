# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""First-party preprocessing components for MolmoAct2.

This module contains small, focused preprocessing units that can be composed by
the policy preprocessor.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch

from physicalai.data.observation import ACTION, IMAGES, STATE, Feature, FeatureType
from physicalai.policies.utils.features import feature_by_type
from physicalai.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType


class _TokenizerLike(Protocol):
    def __call__(
        self,
        text: list[str],
        *,
        return_tensors: str,
        padding: bool,
        truncation: bool,
        max_length: int,
    ) -> dict[str, torch.Tensor]: ...


class _MultimodalProcessorLike(Protocol):
    def __call__(
        self,
        text: list[str],
        *,
        images: list[np.ndarray] | None = None,
        videos: list[dict[str, Any]] | None = None,
        return_tensors: str,
        padding: bool,
    ) -> dict[str, Any]: ...


_TRAILING_PUNCTUATION = ".,!?;:"
_PREFIX_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"^(?:task|instruction|language[_ ]instruction|goal)\s*[:\-]\s*",
        r"^(?:the\s+task\s+is\s+to|your\s+task\s+is\s+to)\s+",
    )
)
_QUESTION_TRAILING_SENTENCE_PUNCTUATION = ".,!?;:,\u2026"
_QUESTION_TRAILING_CLOSERS = "\"'\u201d\u2019)]}"
_QUESTION_SURROUNDING_DELIMITERS = "\"'`\u201c\u201d\u2018\u2019[](){}"

ACTION_OUTPUT_TOKEN = "<action_output>"  # nosec B105
SETUP_START_TOKEN = "<setup_start>"  # nosec B105
SETUP_END_TOKEN = "<setup_end>"  # nosec B105
CONTROL_START_TOKEN = "<control_start>"  # nosec B105
CONTROL_END_TOKEN = "<control_end>"  # nosec B105
STATE_START_TOKEN = "<state_start>"  # nosec B105
STATE_END_TOKEN = "<state_end>"  # nosec B105
STATE_TOKEN_PREFIX = "<state_"  # nosec B105


def _normalize_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return ""

    previous = None
    while normalized and normalized != previous:
        previous = normalized
        normalized = normalized.strip().strip(_QUESTION_SURROUNDING_DELIMITERS).strip()
        for pattern in _PREFIX_PATTERNS:
            normalized = pattern.sub("", normalized, count=1).strip()
        normalized = normalized.rstrip(_QUESTION_TRAILING_SENTENCE_PUNCTUATION).rstrip()
        normalized = normalized.rstrip(_QUESTION_TRAILING_CLOSERS).rstrip()
        normalized = normalized.rstrip(_QUESTION_TRAILING_SENTENCE_PUNCTUATION).rstrip()

    chunks = [chunk.strip() for chunk in re.split(r"[.!?]+", normalized) if chunk.strip()]
    if len(chunks) > 1:
        normalized = "; ".join(chunks)

    return normalized.lower()


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HF_ACCESS_TOKEN")


class MolmoAct2StateNormalizer(torch.nn.Module):
    """Normalize state tensors for MolmoAct2 inputs.

    The normalizer applies quantile normalization using the state feature
    metadata and supports flattened observation keys such as ``state`` and
    ``observation.state`` via ``FeatureNormalizeTransform`` key matching.
    """

    def __init__(self, *, input_features: list[Feature]) -> None:
        """Initialize state normalization from resolved input features.

        Args:
            input_features: Resolved policy input features.

        Raises:
            ValueError: If no state feature is present in ``input_features``.
        """
        super().__init__()
        state_feature = feature_by_type(input_features, FeatureType.STATE)
        if state_feature is None:
            msg = "MolmoAct2 state normalization requires a STATE input feature."
            raise ValueError(msg)

        self.state_feature_name = state_feature.name or STATE
        self._normalizer = FeatureNormalizeTransform(
            {self.state_feature_name: state_feature},
            {FeatureType.STATE: NormalizationType.QUANTILES},
            inverse=False,
        )
        # Store per-dim normalization mask (True=normalize, False=passthrough e.g. gripper)
        norm_mask = (
            state_feature.normalization_data.mask
            if state_feature.normalization_data is not None
            else None
        )
        if norm_mask is not None:
            self.register_buffer("_state_norm_mask", torch.tensor(norm_mask, dtype=torch.bool))
        else:
            self._state_norm_mask = None

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Normalize state values in a flattened batch dictionary.

        Args:
            batch: Flattened observation dictionary.

        Returns:
            dict[str, Any]: A shallow copy of ``batch`` with normalized state.

        Raises:
            TypeError: If ``batch`` is not a dictionary.
        """
        if not isinstance(batch, dict):
            msg = f"MolmoAct2StateNormalizer.forward expects dict[str, Any], got {type(batch)}"
            raise TypeError(msg)

        normalized = dict(batch)
        device = next((value.device for value in normalized.values() if torch.is_tensor(value)), torch.device("cpu"))
        normalized = self._normalizer.to(device)(normalized)

        # Clamp normalized state to [-1, 1] (values outside quantile range map outside this range)
        if self.state_feature_name in normalized and torch.is_tensor(normalized[self.state_feature_name]):
            state = normalized[self.state_feature_name]
            clamped = state.clamp(-1.0, 1.0)
            if self._state_norm_mask is not None:
                mask = self._state_norm_mask
                for _ in range(state.ndim - mask.ndim):
                    mask = mask.unsqueeze(0)
                mask = mask.expand_as(state).to(state.device)
                # Validate that passthrough dims (gripper) are in valid range
                passthrough_mask = ~mask.expand_as(state)
                if bool(passthrough_mask.any()):
                    passthrough_values = state[passthrough_mask]
                    if bool(((passthrough_values < -1.0) | (passthrough_values > 1.0)).any()):
                        msg = (
                            f"MolmoAct2 {self.state_feature_name} gripper values are not under [-1, 1]. "
                            "Please set normalize_gripper=True or ensure gripper values are pre-normalized to [-1, 1]."
                        )
                        raise ValueError(msg)
                normalized[self.state_feature_name] = torch.where(mask, clamped, state)
            else:
                normalized[self.state_feature_name] = clamped

        return normalized


class MolmoAct2ProcessorProvider:
    """Provide lazily loaded tokenizer and multimodal processor instances."""

    def __init__(self, *, tokenizer_name_or_path: str | None, processor_assets_path: str | None) -> None:
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.processor_assets_path = processor_assets_path
        self._tokenizer: _TokenizerLike | None = None
        self._multimodal_processor: _MultimodalProcessorLike | None = None

    @property
    def tokenizer(self) -> _TokenizerLike:
        """Return lazily loaded tokenizer.

        Raises:
            ValueError: If no tokenizer path/id was configured.
            ImportError: If transformers is not installed.
        """
        if self._tokenizer is None:
            if not self.tokenizer_name_or_path:
                msg = "MolmoAct2 TASK preprocessing requires config.tokenizer_name_or_path."
                raise ValueError(msg)

            try:
                from transformers import Qwen2Tokenizer  # noqa: PLC0415
            except ImportError:
                Qwen2Tokenizer = None

            try:
                from transformers import AutoTokenizer  # noqa: PLC0415
            except ImportError as exc:
                msg = "MolmoAct2 TASK preprocessing requires transformers to be installed."
                raise ImportError(msg) from exc

            token = _hf_token()
            if Qwen2Tokenizer is not None:
                self._tokenizer = Qwen2Tokenizer.from_pretrained(
                    self.tokenizer_name_or_path,
                    token=token,
                )
            else:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_name_or_path,
                    trust_remote_code=False,
                    token=token,
                )

        return cast("_TokenizerLike", self._tokenizer)

    def _resolve_assets_dir(self) -> Path:
        if not self.processor_assets_path:
            msg = "MolmoAct2 multimodal preprocessing requires config.processor_assets_path."
            raise ValueError(msg)

        assets_dir = Path(self.processor_assets_path)
        if not assets_dir.is_dir():
            msg = f"MolmoAct2 processor assets directory does not exist: {assets_dir}."
            raise FileNotFoundError(msg)
        return assets_dir

    def _build_multimodal_processor(self) -> _MultimodalProcessorLike:
        assets_dir = self._resolve_assets_dir()
        processor_config_path = assets_dir / "processor_config.json"
        if not processor_config_path.exists():
            msg = f"MolmoAct2 checkpoint is missing {processor_config_path}."
            raise FileNotFoundError(msg)

        with processor_config_path.open(encoding="utf-8") as f:
            processor_config = json.load(f)

        try:
            from transformers import AutoTokenizer  # noqa: PLC0415
            from transformers import Qwen2Tokenizer  # noqa: PLC0415
            from lerobot.policies.molmoact2.hf_model.image_processing_molmoact2 import (  # noqa: PLC0415
                MolmoAct2ImageProcessor,
            )
            from lerobot.policies.molmoact2.hf_model.processing_molmoact2 import MolmoAct2Processor  # noqa: PLC0415
            from lerobot.policies.molmoact2.hf_model.video_processing_molmoact2 import (  # noqa: PLC0415
                MolmoAct2VideoProcessor,
            )
        except ImportError as exc:
            msg = "MolmoAct2 multimodal preprocessing requires transformers and lerobot to be installed."
            raise ImportError(msg) from exc

        image_processor_config = {
            key: value
            for key, value in dict(processor_config.get("image_processor") or {}).items()
            if key not in {"auto_map", "image_processor_type", "processor_class"}
        }
        video_processor_config = {
            key: value
            for key, value in dict(processor_config.get("video_processor") or {}).items()
            if key not in {"auto_map", "video_processor_type", "processor_class"}
        }

        image_processor = MolmoAct2ImageProcessor(**image_processor_config)
        video_processor = MolmoAct2VideoProcessor(**video_processor_config)
        tokenizer_source = self.tokenizer_name_or_path or str(assets_dir)
        token = _hf_token()
        try:
            tokenizer = Qwen2Tokenizer.from_pretrained(
                tokenizer_source,
                token=token,
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                trust_remote_code=False,
                token=token,
            )

        chat_template_path = assets_dir / "chat_template.jinja"
        chat_template = chat_template_path.read_text() if chat_template_path.exists() else None
        processor = MolmoAct2Processor(
            image_processor=image_processor,
            video_processor=video_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            image_use_col_tokens=processor_config.get("image_use_col_tokens", True),
            use_single_crop_col_tokens=processor_config.get("use_single_crop_col_tokens"),
            use_single_crop_start_token=processor_config.get("use_single_crop_start_token", True),
            video_use_col_tokens=processor_config.get("video_use_col_tokens", False),
            use_frame_special_tokens=processor_config.get("use_frame_special_tokens", True),
        )
        return cast("_MultimodalProcessorLike", processor)

    @property
    def multimodal_processor(self) -> _MultimodalProcessorLike | None:
        """Return lazily loaded multimodal processor when local assets exist."""
        if self._multimodal_processor is not None:
            return self._multimodal_processor

        try:
            self._multimodal_processor = self._build_multimodal_processor()
        except (FileNotFoundError, ImportError, ValueError):
            self._multimodal_processor = None
        return self._multimodal_processor


class MolmoAct2TextPreprocessor:
    """Build normalized prompt text for MolmoAct2."""

    def __init__(
        self,
        *,
        input_features: list[Feature],
        num_state_tokens: int,
        add_setup_tokens: bool,
        add_control_tokens: bool,
        setup_type: str,
        control_mode: str,
    ) -> None:
        self.num_state_tokens = int(num_state_tokens) if int(num_state_tokens) > 0 else 256
        self.add_setup_tokens = bool(add_setup_tokens)
        self.add_control_tokens = bool(add_control_tokens)
        self.setup_type = str(setup_type or "")
        self.control_mode = str(control_mode or "")
        self.state_feature_name = next(
            (
                feature.name
                for feature in input_features
                if feature.ftype == FeatureType.STATE and feature.name
            ),
            STATE,
        )

    @staticmethod
    def _wrap_setup_text(setup_type: str, *, add_setup_tokens: bool) -> str:
        if not setup_type:
            return ""
        if not add_setup_tokens:
            return setup_type
        if setup_type.startswith(SETUP_START_TOKEN) and setup_type.endswith(SETUP_END_TOKEN):
            return setup_type
        return f"{SETUP_START_TOKEN}{setup_type}{SETUP_END_TOKEN}"

    @staticmethod
    def _wrap_control_text(control_mode: str, *, add_control_tokens: bool) -> str:
        if not control_mode:
            return ""
        if not add_control_tokens:
            return control_mode
        if control_mode.startswith(CONTROL_START_TOKEN) and control_mode.endswith(CONTROL_END_TOKEN):
            return control_mode
        return f"{CONTROL_START_TOKEN}{control_mode}{CONTROL_END_TOKEN}"

    @staticmethod
    def _build_discrete_state_string(state: torch.Tensor, num_state_tokens: int) -> str:
        if num_state_tokens <= 0:
            msg = f"num_state_tokens must be > 0, got {num_state_tokens}."
            raise ValueError(msg)
        state = torch.nan_to_num(state.to(dtype=torch.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        state = state.clamp(-1.0, 1.0)
        scaled = (state + 1.0) / 2.0 * float(num_state_tokens - 1)
        token_ids = scaled.round().clamp(0, int(num_state_tokens) - 1).to(dtype=torch.int64).reshape(-1).tolist()
        state_tokens = "".join(f"{STATE_TOKEN_PREFIX}{int(token_id)}>" for token_id in token_ids)
        return f"{STATE_START_TOKEN}{state_tokens}{STATE_END_TOKEN}"

    @staticmethod
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
        setup_text = MolmoAct2TextPreprocessor._wrap_setup_text(setup_type, add_setup_tokens=add_setup_tokens)
        control_text = MolmoAct2TextPreprocessor._wrap_control_text(control_mode, add_control_tokens=add_control_tokens)
        state_clause = f" The current state of the robot is {discrete_state_string}." if discrete_state_string else ""
        prompt = (
            f"The task is to {task}. The setup is {setup_text}.{state_clause} "
            "The expected control mode is "
            f"{control_text}. Given these, what action should the robot take to complete the task?"
        )
        if num_images <= 0:
            image_prefix = ""
        elif num_images == 1:
            image_prefix = "<|image|>"
        else:
            image_prefix = "".join(f"Image {idx + 1}<|image|>" for idx in range(num_images))
        return f"{image_prefix}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{ACTION_OUTPUT_TOKEN}"

    def _state_from_batch(self, batch: dict[str, Any], batch_size: int, device: torch.device) -> torch.Tensor | None:
        state_value = batch.get(self.state_feature_name)
        if state_value is None:
            state_value = batch.get(STATE)
        if state_value is None:
            return None
        state = torch.as_tensor(state_value, dtype=torch.float32, device=device)
        # Handle missing batch dimension: [D] -> [1, D]
        if state.ndim == 1:
            state = state.unsqueeze(0)
        # If batch_size inferred as 1 but state has different first dim, use state's dim
        # This handles case where batch inference picked wrong tensor
        if batch_size == 1 and state.shape[0] != 1:
            batch_size = state.shape[0]
        if int(state.shape[0]) == 1 and batch_size > 1:
            state = state.expand(batch_size, -1)
        if int(state.shape[0]) != batch_size:
            msg = f"Expected state batch size {batch_size}, got {int(state.shape[0])}."
            raise ValueError(msg)
        return state

    @staticmethod
    def _as_text_list(
        value: object | None,
        batch_size: int,
    ) -> list[str]:
        if value is None:
            return [""] * batch_size
        if isinstance(value, str):
            texts = [value]
        elif torch.is_tensor(value):
            tensor_value = torch.as_tensor(value)
            if tensor_value.ndim == 0:
                texts = [str(tensor_value.item())]
            else:
                texts = [str(item) for item in tensor_value.detach().cpu().reshape(-1).tolist()]
        elif isinstance(value, Sequence):
            texts = [str(item) for item in value]
        else:
            texts = [str(value)]

        if len(texts) == batch_size:
            return texts
        if len(texts) == 1:
            return texts * batch_size
        msg = f"Expected {batch_size} task strings, got {len(texts)}."
        raise ValueError(msg)

    @staticmethod
    def _infer_batch_size(batch: dict[str, Any]) -> int:
        """Infer batch size from batch dictionary.
        
        Prioritizes state tensor for inference since state shape is [B, D]
        and less ambiguous than image shape which could be [C, H, W].
        """
        # First, try to find state tensor which has unambiguous shape [B, D]
        for key in ["state", "observation.state"]:
            if key in batch:
                value = batch[key]
                if torch.is_tensor(value) and value.ndim > 0:
                    return int(value.shape[0])
        
        # Fallback to first tensor found (for task or other keys)
        for value in batch.values():
            if torch.is_tensor(value) and value.ndim > 0:
                # Avoid picking image tensor (which is [C, H, W] or [B, C, H, W])
                # by checking if it looks like an image (4D or 3D with large dims)
                if value.ndim >= 3 and any(d > 100 for d in value.shape):
                    continue  # Skip image-like tensors
                return int(value.shape[0])
        return 1

    def build_prompts(self, batch: dict[str, Any]) -> tuple[list[str], list[str], int]:
        """Build normalized prompt strings and return batch size.

        Args:
            batch: Flattened observation dictionary.

        Returns:
            tuple[list[str], list[str], int]: Prompt strings, normalized task strings,
            and batch size.

        Raises:
            TypeError: If ``batch`` is not a dictionary.
        """
        if not isinstance(batch, dict):
            msg = f"MolmoAct2TextPreprocessor.build_prompts expects dict[str, Any], got {type(batch)}"
            raise TypeError(msg)

        batch_size = self._infer_batch_size(batch)
        tasks = [_normalize_text(task) for task in self._as_text_list(batch.get("task"), batch_size)]

        device = next((value.device for value in batch.values() if torch.is_tensor(value)), torch.device("cpu"))
        state = self._state_from_batch(batch, batch_size, device)
        num_images = len([key for key in batch if str(key).startswith(f"{IMAGES}.")])

        prompts: list[str] = []
        for index in range(batch_size):
            discrete_state = ""
            if state is not None:
                discrete_state = self._build_discrete_state_string(state[index], self.num_state_tokens)
            prompts.append(
                self._build_robot_text(
                    task=tasks[index],
                    discrete_state_string=discrete_state,
                    setup_type=self.setup_type,
                    control_mode=self.control_mode,
                    add_setup_tokens=self.add_setup_tokens,
                    add_control_tokens=self.add_control_tokens,
                    num_images=num_images,
                ),
            )

        return prompts, tasks, batch_size


class MolmoAct2ImagePreprocessor:
    """Extract and normalize image inputs for MolmoAct2."""

    def __init__(self, *, input_features: list[Feature]) -> None:
        self.image_feature_names = [
            str(feature.name)
            for feature in input_features
            if feature.ftype == FeatureType.VISUAL and feature.name
        ]

    @staticmethod
    def _to_numpy_image(value: object) -> np.ndarray:
        arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)

        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.ndim != 3:
            msg = f"Unsupported image shape for MolmoAct2: {arr.shape}."
            raise ValueError(msg)

        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype in {np.float16, np.float32, np.float64}:
            if arr.size > 0 and float(np.nanmax(arr)) <= 1.0:
                arr *= 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _resolve_image_keys(self, batch: dict[str, Any]) -> list[str]:
        if self.image_feature_names:
            keys: list[str] = []
            for name in self.image_feature_names:
                direct_key = f"{IMAGES}.{name}"
                if direct_key in batch:
                    keys.append(direct_key)
                    continue

                # LiberoGym uses image2 for the wrist camera while HF metadata
                # commonly names it wrist_image. Accept either naming in batches.
                if name == "wrist_image" and f"{IMAGES}.image2" in batch:
                    keys.append(f"{IMAGES}.image2")
                elif name == "image2" and f"{IMAGES}.wrist_image" in batch:
                    keys.append(f"{IMAGES}.wrist_image")

            # Preserve order and remove accidental duplicates.
            keys = list(dict.fromkeys(keys))
            if keys:
                return keys
        return sorted([str(key) for key in batch if str(key).startswith(f"{IMAGES}.")])

    def extract_images(self, batch: dict[str, Any], batch_size: int) -> list[list[np.ndarray]]:
        """Extract per-example images as numpy arrays."""
        images_by_example: list[list[np.ndarray]] = [[] for _ in range(batch_size)]
        for key in self._resolve_image_keys(batch):
            value = batch[key]
            for batch_idx in range(batch_size):
                item = value
                if (torch.is_tensor(value) or isinstance(value, np.ndarray)) and getattr(value, "ndim", 0) >= 4:
                    item = value[batch_idx]
                images_by_example[batch_idx].append(self._to_numpy_image(item))
        return images_by_example


class MolmoAct2VideoPreprocessor:
    """Extract video inputs for MolmoAct2.

    The current policy path does not yet emit video observations, but keeping
    the logic isolated here makes the multimodal composition surface explicit.
    """

    def extract_videos(self, batch: dict[str, Any], batch_size: int) -> list[dict[str, Any]]:
        del batch, batch_size
        return []


class MolmoAct2TaskTokenizer(torch.nn.Module):
    """Tokenize normalized text prompts into model language inputs."""

    def __init__(
        self,
        *,
        max_sequence_length: int,
        input_features: list[Feature],
        num_state_tokens: int,
        add_setup_tokens: bool,
        add_control_tokens: bool,
        setup_type: str,
        control_mode: str,
        tokenizer_name_or_path: str | None,
    ) -> None:
        super().__init__()
        self.max_sequence_length = int(max_sequence_length)
        self.text_preprocessor = MolmoAct2TextPreprocessor(
            input_features=input_features,
            num_state_tokens=num_state_tokens,
            add_setup_tokens=add_setup_tokens,
            add_control_tokens=add_control_tokens,
            setup_type=setup_type,
            control_mode=control_mode,
        )
        self.processor_provider = MolmoAct2ProcessorProvider(
            tokenizer_name_or_path=tokenizer_name_or_path,
            processor_assets_path=None,
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Tokenize text prompts and add language model tensors.

        Args:
            batch: Flattened observation dictionary.

        Returns:
            dict[str, Any]: Updated batch with tokenized language inputs.

        Raises:
            TypeError: If ``batch`` is not a dictionary.
        """
        if not isinstance(batch, dict):
            msg = f"MolmoAct2TaskTokenizer.forward expects dict[str, Any], got {type(batch)}"
            raise TypeError(msg)

        processed = dict(batch)
        prompts, tasks, _ = self.text_preprocessor.build_prompts(processed)

        device = next((value.device for value in processed.values() if torch.is_tensor(value)), torch.device("cpu"))

        processed["task"] = tasks

        tokenized = self.processor_provider.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
        )
        processed["input_ids"] = tokenized["input_ids"].to(device)
        processed["attention_mask"] = tokenized["attention_mask"].to(device)

        return processed


class MolmoAct2MultimodalTokenizer(torch.nn.Module):
    """Apply image/video multimodal expansion for MolmoAct2 inputs."""

    def __init__(
        self,
        *,
        input_features: list[Feature],
        num_state_tokens: int,
        add_setup_tokens: bool,
        add_control_tokens: bool,
        setup_type: str,
        control_mode: str,
        tokenizer_name_or_path: str | None,
        processor_assets_path: str | None,
    ) -> None:
        super().__init__()
        self.text_preprocessor = MolmoAct2TextPreprocessor(
            input_features=input_features,
            num_state_tokens=num_state_tokens,
            add_setup_tokens=add_setup_tokens,
            add_control_tokens=add_control_tokens,
            setup_type=setup_type,
            control_mode=control_mode,
        )
        self.image_preprocessor = MolmoAct2ImagePreprocessor(input_features=input_features)
        self.video_preprocessor = MolmoAct2VideoPreprocessor()
        self.processor_provider = MolmoAct2ProcessorProvider(
            tokenizer_name_or_path=tokenizer_name_or_path,
            processor_assets_path=processor_assets_path,
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Apply multimodal token expansion when processor assets are available."""
        if not isinstance(batch, dict):
            msg = f"MolmoAct2MultimodalTokenizer.forward expects dict[str, Any], got {type(batch)}"
            raise TypeError(msg)

        processed = dict(batch)
        prompts, tasks, batch_size = self.text_preprocessor.build_prompts(processed)
        images_by_example = self.image_preprocessor.extract_images(processed, batch_size)
        videos = self.video_preprocessor.extract_videos(processed, batch_size)

        flat_images: list[np.ndarray] = []
        for example_images in images_by_example:
            flat_images.extend(example_images)

        multimodal_processor = self.processor_provider.multimodal_processor
        if multimodal_processor is None or not (flat_images or videos):
            return processed

        device = next((value.device for value in processed.values() if torch.is_tensor(value)), torch.device("cpu"))
        processed["task"] = tasks
        tokenized = multimodal_processor(
            prompts,
            images=flat_images or None,
            videos=videos or None,
            return_tensors="pt",
            padding=True,
        )
        for key, value in tokenized.items():
            if torch.is_tensor(value):
                processed[str(key)] = value.to(device)
            else:
                processed[str(key)] = value
        return processed


class MolmoAct2ActionPadder(torch.nn.Module):
    """Pad action tensors and emit action-dimension padding masks."""

    def __init__(self, *, output_features: list[Feature], max_action_dim: int) -> None:
        super().__init__()
        action_feature = feature_by_type(output_features, FeatureType.ACTION)
        self.max_action_dim = int(max_action_dim)
        self.env_action_dim = (
            int(action_feature.shape[0]) if action_feature and action_feature.shape else int(max_action_dim)
        )

    @staticmethod
    def _infer_batch_size(batch: dict[str, Any]) -> int:
        """Infer batch size from batch dictionary.
        
        Prioritizes state tensor for inference since state shape is [B, D]
        and less ambiguous than image shape which could be [C, H, W].
        """
        # First, try to find state tensor which has unambiguous shape [B, D]
        for key in ["state", "observation.state"]:
            if key in batch:
                value = batch[key]
                if torch.is_tensor(value) and value.ndim > 0:
                    return int(value.shape[0])
        
        # Fallback to first tensor found (for task or other keys)
        for value in batch.values():
            if torch.is_tensor(value) and value.ndim > 0:
                # Avoid picking image tensor (which is [C, H, W] or [B, C, H, W])
                # by checking if it looks like an image (4D or 3D with large dims)
                if value.ndim >= 3 and any(d > 100 for d in value.shape):
                    continue  # Skip image-like tensors
                return int(value.shape[0])
        return 1

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(batch, dict):
            msg = f"MolmoAct2ActionPadder.forward expects dict[str, Any], got {type(batch)}"
            raise TypeError(msg)

        processed = dict(batch)
        batch_size = self._infer_batch_size(processed)
        device = next((value.device for value in processed.values() if torch.is_tensor(value)), torch.device("cpu"))

        raw_action = processed.get(ACTION)
        if raw_action is not None:
            action = torch.as_tensor(raw_action, dtype=torch.float32, device=device)
            if action.ndim == 2:
                action = action.unsqueeze(1)
            if action.ndim != 3:
                msg = f"MolmoAct2 expected action shape [B, T, D], got {tuple(action.shape)}."
                raise ValueError(msg)
            if int(action.shape[-1]) > self.max_action_dim:
                msg = f"Action dim {action.shape[-1]} exceeds max_action_dim={self.max_action_dim}."
                raise ValueError(msg)

            padded = torch.zeros((*action.shape[:-1], self.max_action_dim), device=device, dtype=torch.float32)
            padded[..., : action.shape[-1]] = action.to(dtype=torch.float32)
            processed[ACTION] = padded

            action_dim_is_pad = torch.ones((action.shape[0], self.max_action_dim), device=device, dtype=torch.bool)
            action_dim_is_pad[:, : action.shape[-1]] = False
            processed["action_dim_is_pad"] = action_dim_is_pad
            processed["action_horizon_is_pad"] = torch.zeros(action.shape[:2], device=device, dtype=torch.bool)
            return processed

        action_dim_is_pad = torch.ones((batch_size, self.max_action_dim), device=device, dtype=torch.bool)
        action_dim_is_pad[:, : self.env_action_dim] = False
        processed["action_dim_is_pad"] = action_dim_is_pad
        return processed
