# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Main MolmoAct2 preprocessor orchestrating composable preprocessing steps."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from transformers import Qwen2Tokenizer

from physicalai.data.observation import ACTION, FeatureType
from physicalai.policies.molmoact2.action_tokenizer import UniversalActionProcessor

from .common import feature_by_type
from .preprocess_steps import (
    ActionExtractor,
    ActionPadder,
    FeatureBatchNormalizer,
    RobotPromptEncoder,
    StateTaskImageExtractor,
)

IMAGE_PROMPT = "<|image|>"
_ACTION_TOKENIZER_BATCH_DIMS = 2

if TYPE_CHECKING:
    from physicalai.policies.molmoact2.config import MolmoAct2Config


def _action_start_token() -> str:
    return "<action_start>"


def _action_end_token() -> str:
    return "<action_end>"


def _action_token_prefix() -> str:
    return "<action_"


def _to_torch(value: object) -> object:
    converted = value
    if torch.is_tensor(value):
        converted = value
    elif isinstance(value, np.ndarray):
        converted = value if value.dtype.kind in {"U", "S", "O"} else torch.from_numpy(value)
    elif isinstance(value, (list, tuple)) and value:
        first = value[0]
        if not isinstance(first, (str, bytes, dict)) and not (
            hasattr(first, "__dict__") and not isinstance(first, (int, float, bool, list, tuple, np.ndarray))
        ):
            try:
                converted = torch.as_tensor(value)
            except (TypeError, ValueError):
                converted = value
    return converted


class MolmoAct2Preprocessor(torch.nn.Module):
    """Pack Observation objects into MolmoAct2 model-ready tensors."""

    def __init__(self, config: MolmoAct2Config) -> None:
        """Initialize preprocessing for MolmoAct2 prompt, image, and supervision packing."""
        super().__init__()
        self.config = config

        self.state_feature = feature_by_type(config.input_features, FeatureType.STATE)

        self.num_state_tokens = int(config.num_state_tokens) if int(config.num_state_tokens) > 0 else 256
        self.setup_type = str(config.setup_type or "")
        self.control_mode = str(config.control_mode or "")
        self.add_setup_tokens = bool(config.add_setup_tokens)
        self.add_control_tokens = bool(config.add_control_tokens)
        self.image_keys = [
            feature.name for feature in config.input_features if feature.ftype == FeatureType.VISUAL and feature.name
        ]

        self._normalizer_step = FeatureBatchNormalizer(
            input_features=config.input_features,
            output_features=config.output_features,
        )
        self._extractor_step = StateTaskImageExtractor(image_keys=self.image_keys)
        self._action_extractor = ActionExtractor()
        self._action_padder = ActionPadder(max_action_dim=int(config.max_action_dim))
        self._prompt_step = RobotPromptEncoder(
            num_state_tokens=self.num_state_tokens,
            setup_type=self.setup_type,
            control_mode=self.control_mode,
            add_setup_tokens=self.add_setup_tokens,
            add_control_tokens=self.add_control_tokens,
        )

        self._tokenizer: Qwen2Tokenizer | None = None
        self._action_tokenizer: UniversalActionProcessor | None = None
        self.image_placeholder_token = IMAGE_PROMPT
        self.action_mode = str(getattr(config, "action_mode", "both"))
        self.discrete_action_tokenizer = str(getattr(config, "discrete_action_tokenizer", "")).strip()
        self._action_start_id: int | None = None
        self._action_end_id: int | None = None
        self._eos_token: str = ""
        self._eos_token_id: int | None = None

    @property
    def tokenizer(self) -> Qwen2Tokenizer:
        """Checkpoint tokenizer used for text prompt packing.

        Raises:
            ValueError: If the config does not specify a tokenizer path.
            RuntimeError: If tokenizer initialization unexpectedly fails.
        """
        if self._tokenizer is not None:
            return self._tokenizer

        tokenizer_name_or_path = self.config.tokenizer_name_or_path
        if not tokenizer_name_or_path:
            msg = (
                "config.tokenizer_name_or_path is required. "
                "Provide it via constructor or set MolmoAct2Config.tokenizer_name_or_path."
            )
            raise ValueError(msg)

        self._tokenizer = Qwen2Tokenizer.from_pretrained(
            str(tokenizer_name_or_path),
            local_files_only=True,
        )
        tokenizer = self._tokenizer
        if tokenizer is None:
            msg = "MolmoAct2 tokenizer failed to initialize."
            raise RuntimeError(msg)
        return tokenizer

    @property
    def action_tokenizer(self) -> UniversalActionProcessor:
        """Discrete action tokenizer used to build autoregressive action labels.

        Raises:
            ValueError: If discrete tokenizer configuration is missing.
        """
        if self._action_tokenizer is not None:
            return self._action_tokenizer
        if not self.discrete_action_tokenizer:
            msg = "config.discrete_action_tokenizer is required for discrete MolmoAct2 training."
            raise ValueError(msg)
        self._action_tokenizer = UniversalActionProcessor.from_pretrained_local(self.discrete_action_tokenizer)
        return self._action_tokenizer

    def _ensure_label_token_ids(self) -> None:
        if self._action_start_id is not None and self._action_end_id is not None:
            return
        self._action_start_id = self._single_token_id(_action_start_token())
        self._action_end_id = self._single_token_id(_action_end_token())
        eos_token = self.tokenizer.eos_token
        self._eos_token = eos_token if isinstance(eos_token, str) else ""
        eos_token_id = self.tokenizer.eos_token_id
        self._eos_token_id = eos_token_id if isinstance(eos_token_id, int) else None

    def _single_token_id(self, token: str) -> int:
        token_ids = self.tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) != 1:
            msg = f"MolmoAct2 token {token!r} must encode to one token, got {token_ids}."
            raise ValueError(msg)
        return int(token_ids[0])

    @staticmethod
    def _tokenize_discrete_action(action: np.ndarray, processor: UniversalActionProcessor) -> list[int]:
        arr = np.asarray(action, dtype=np.float32)
        if arr.ndim == _ACTION_TOKENIZER_BATCH_DIMS:
            arr = arr[None, :, :]
        elif arr.ndim == 1:
            arr = arr[None, None, :]
        tokens_out = processor(arr)
        if isinstance(tokens_out, dict):
            tokens_out = tokens_out.get("input_ids", next(iter(tokens_out.values())))
        if isinstance(tokens_out, np.ndarray):
            tokens_out = tokens_out.tolist()
        if torch.is_tensor(tokens_out):
            tokens_out = tokens_out.detach().cpu().tolist()
        if not isinstance(tokens_out, list):
            msg = f"Unexpected discrete action tokenizer output type: {type(tokens_out)}"
            raise TypeError(msg)
        if tokens_out and isinstance(tokens_out[0], (list, tuple, np.ndarray)):
            tokens_out = tokens_out[0]
        token_ids: list[int] = []
        for token_id in tokens_out:
            if isinstance(token_id, (list, tuple, np.ndarray)):
                msg = f"Unexpected nested discrete token id type: {type(token_id)}"
                raise TypeError(msg)
            token_ids.append(int(token_id))
        return token_ids

    def _build_discrete_action_string(self, action: np.ndarray) -> str:
        token_ids = self._tokenize_discrete_action(action, self.action_tokenizer)
        pieces = "".join(f"{_action_token_prefix()}{int(token_id)}>" for token_id in token_ids)
        return f"{_action_start_token()}{pieces}{_action_end_token()}"

    def _build_labels(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self._ensure_label_token_ids()
        labels = torch.full_like(input_ids, -100)
        for batch_idx in range(input_ids.shape[0]):
            valid = attention_mask[batch_idx].to(dtype=torch.bool)
            row = input_ids[batch_idx]
            starts = (row == self._action_start_id).nonzero(as_tuple=False).flatten().tolist()
            ends = (row == self._action_end_id).nonzero(as_tuple=False).flatten().tolist()
            end_ptr = 0
            for start in starts:
                while end_ptr < len(ends) and ends[end_ptr] < start:
                    end_ptr += 1
                if end_ptr >= len(ends):
                    msg = "Found <action_start> without matching <action_end> in MolmoAct2 labels."
                    raise ValueError(msg)
                end = int(ends[end_ptr])
                label_end = end + 1
                if (
                    self._eos_token_id is not None
                    and label_end < int(row.shape[0])
                    and int(row[label_end]) == int(self._eos_token_id)
                ):
                    label_end += 1
                labels[batch_idx, start:label_end] = row[start:label_end]
                end_ptr += 1
            if not starts:
                msg = "No discrete action span found in MolmoAct2 training text."
                raise ValueError(msg)
            labels[batch_idx] = torch.where(valid, labels[batch_idx], torch.full_like(labels[batch_idx], -100))
        return labels

    @staticmethod
    def _insert_bos(
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        bos_token_id: int,
        pad_token_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
            attention_mask = attention_mask[None, :]
            squeeze = True
        else:
            squeeze = False

        batch_size, seq_len = input_ids.shape
        if seq_len == 0:
            out_ids = np.full((batch_size, 1), bos_token_id, dtype=input_ids.dtype)
            out_mask = np.ones((batch_size, 1), dtype=attention_mask.dtype)
            return (out_ids[0], out_mask[0]) if squeeze else (out_ids, out_mask)

        first_valid = (attention_mask == 1).argmax(axis=-1)
        if np.all(input_ids[np.arange(batch_size), first_valid] == bos_token_id):
            return (input_ids[0], attention_mask[0]) if squeeze else (input_ids, attention_mask)

        out_ids = np.full((batch_size, seq_len + 1), pad_token_id, dtype=input_ids.dtype)
        out_mask = np.zeros((batch_size, seq_len + 1), dtype=attention_mask.dtype)

        src = np.tile(np.arange(seq_len), (batch_size, 1))
        valid = src >= first_valid[:, None]
        tgt = src + 1
        batch_idx = np.tile(np.arange(batch_size)[:, None], (1, seq_len))

        out_ids[batch_idx[valid], tgt[valid]] = input_ids[valid]
        out_mask[batch_idx[valid], tgt[valid]] = 1
        out_ids[np.arange(batch_size), first_valid] = bos_token_id
        out_mask[np.arange(batch_size), first_valid] = 1

        return (out_ids[0], out_mask[0]) if squeeze else (out_ids, out_mask)

    def _tokenize_and_pack(
        self,
        *,
        texts: list[str],
        image_batch: torch.Tensor,
        build_labels: bool = False,
    ) -> dict[str, object]:
        text_inputs = self.tokenizer(texts, padding=True)

        input_ids = np.asarray(text_inputs["input_ids"])
        attention_mask = np.asarray(text_inputs["attention_mask"])
        bos_token_id = self.tokenizer.bos_token_id
        if not isinstance(bos_token_id, int):
            eos_token_id = self.tokenizer.eos_token_id
            if not isinstance(eos_token_id, int):
                msg = "MolmoAct2 tokenizer must define bos_token_id or eos_token_id."
                raise TypeError(msg)
            bos_token_id = eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        if not isinstance(pad_token_id, int):
            msg = "MolmoAct2 tokenizer must define pad_token_id."
            raise TypeError(msg)
        input_ids, attention_mask = self._insert_bos(
            input_ids,
            attention_mask,
            bos_token_id,
            pad_token_id,
        )

        image_placeholder_token_id = self.tokenizer.convert_tokens_to_ids(self.image_placeholder_token)
        if not isinstance(image_placeholder_token_id, int):
            msg = "MolmoAct2 image placeholder token must map to a single token id."
            raise TypeError(msg)

        data: dict[str, object] = {
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
            # Keep this as a Python int so export paths do not need to
            # specialize a symbolic scalar tensor into a concrete integer.
            "image_placeholder_token_id": image_placeholder_token_id,
            "images_bchw": image_batch,
        }

        packed = {
            key: (value if key == "image_placeholder_token_id" else _to_torch(value))
            for key, value in data.items()
        }
        if build_labels:
            input_ids_tensor = packed["input_ids"]
            attention_mask_tensor = packed["attention_mask"]
            if not torch.is_tensor(input_ids_tensor) or not torch.is_tensor(attention_mask_tensor):
                msg = "MolmoAct2 tokenizer outputs must convert to tensors for label construction."
                raise TypeError(msg)
            packed["labels"] = self._build_labels(
                cast("torch.Tensor", input_ids_tensor),
                cast("torch.Tensor", attention_mask_tensor),
            )
        return packed

    @staticmethod
    def _flatten_observation_batch(batch: dict[str, object]) -> dict[str, object]:
        """Normalize nested observation dictionaries into flat dot-key format."""
        flattened: dict[str, object] = {}
        for key, value in batch.items():
            if isinstance(value, dict):
                nested_keys: list[str] = []
                for nested_key, nested_value in value.items():
                    flat_key = f"{key}.{nested_key}"
                    flattened[flat_key] = nested_value
                    nested_keys.append(flat_key)
                flattened[f"_{key}_keys"] = nested_keys
            else:
                flattened[key] = value
        return flattened

    def forward(self, batch: dict[str, object]) -> dict[str, object]:
        """Convert a normalized observation batch into MolmoAct2 training or inference inputs.

        Args:
            batch: Observation dictionary, flattened or nested.

        Returns:
            Model-ready MolmoAct2 tensors and optional supervision targets.

        Raises:
            TypeError: If ``batch`` is not a dictionary.
            RuntimeError: If discrete labels are requested without action targets.
        """
        if not isinstance(batch, dict):
            msg = f"MolmoAct2Preprocessor.forward expects dict[str, object], got {type(batch)}"
            raise TypeError(msg)

        normalized_batch = self._normalizer_step(self._flatten_observation_batch(batch))
        bundle = self._extractor_step.extract(normalized_batch)
        prompt_pack = self._prompt_step.encode(bundle)

        image_batch = (
            torch.stack(prompt_pack.flat_images, dim=0)
            if prompt_pack.flat_images
            else torch.empty((0, 3, 0, 0))
        )

        action = self._action_extractor.extract(normalized_batch)
        build_action_labels = action is not None and self.action_mode in {"discrete", "both"}
        texts = prompt_pack.prompt_texts
        if build_action_labels:
            if action is None:
                msg = "MolmoAct2 discrete label construction requires action targets."
                raise RuntimeError(msg)
            full_texts: list[str] = []
            for idx, prompt in enumerate(prompt_pack.prompt_texts):
                answer = self._build_discrete_action_string(action[idx].detach().cpu().numpy())
                full_texts.append(f"{prompt}{answer}{self._eos_token or (self.tokenizer.eos_token or '')}")
            texts = full_texts

        inputs = self._tokenize_and_pack(
            texts=texts,
            image_batch=image_batch,
            build_labels=build_action_labels,
        )

        packed: dict[str, object] = dict(inputs)
        packed["task"] = bundle.tasks
        packed["state"] = bundle.state

        if action is not None:
            action_padded, action_horizon_is_pad, action_dim_is_pad = self._action_padder(action)
            packed[ACTION] = action_padded
            packed["action_horizon_is_pad"] = action_horizon_is_pad
            packed["action_dim_is_pad"] = action_dim_is_pad

        return packed
