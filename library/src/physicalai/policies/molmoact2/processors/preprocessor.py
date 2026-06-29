# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Main MolmoAct2 preprocessor orchestrating composable preprocessing steps."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from transformers import Qwen2Tokenizer

from physicalai.data.observation import FeatureType

from .common import feature_by_type
from .preprocess_steps import (
    FeatureBatchNormalizer,
    RobotPromptEncoder,
    StateTaskImageExtractor,
)

IMAGE_PROMPT = "<|image|>"


def _to_torch(value: Any) -> Any:
    if torch.is_tensor(value):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S", "O"}:
            return value
        return torch.from_numpy(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return value
        first = value[0]
        if isinstance(first, (str, bytes, dict)):
            return value
        if hasattr(first, "__dict__") and not isinstance(first, (int, float, bool, list, tuple, np.ndarray)):
            return value
        try:
            return torch.as_tensor(value)
        except (TypeError, ValueError):
            return value
    return value


class MolmoAct2Preprocessor(torch.nn.Module):
    """Pack Observation objects into MolmoAct2 model-ready tensors."""

    def __init__(self, config: Any) -> None:
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
        self._prompt_step = RobotPromptEncoder(
            num_state_tokens=self.num_state_tokens,
            setup_type=self.setup_type,
            control_mode=self.control_mode,
            add_setup_tokens=self.add_setup_tokens,
            add_control_tokens=self.add_control_tokens,
        )

        self._tokenizer: Qwen2Tokenizer | None = None
        self.image_placeholder_token = IMAGE_PROMPT

    @property
    def tokenizer(self) -> Qwen2Tokenizer:
        if self._tokenizer is not None:
            return self._tokenizer

        tokenizer_name_or_path = self.config.tokenizer_name_or_path
        if not tokenizer_name_or_path:
            raise ValueError(
                "config.tokenizer_name_or_path is required. "
                "Provide it via constructor or set MolmoAct2Config.tokenizer_name_or_path.",
            )

        self._tokenizer = Qwen2Tokenizer.from_pretrained(
            str(tokenizer_name_or_path),
            local_files_only=True,
        )
        return self._tokenizer

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
        prompt_texts: list[str],
        image_batch: torch.Tensor,
    ) -> dict[str, Any]:
        text_inputs = self.tokenizer(prompt_texts, padding=True)

        input_ids = np.asarray(text_inputs["input_ids"])
        attention_mask = np.asarray(text_inputs["attention_mask"])
        bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        input_ids, attention_mask = self._insert_bos(
            input_ids,
            attention_mask,
            int(bos),
            int(self.tokenizer.pad_token_id),
        )

        data: dict[str, Any] = {
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
            "image_placeholder_token_id": int(self.tokenizer.convert_tokens_to_ids(self.image_placeholder_token)),
            "images_bchw": image_batch,
        }

        return {key: _to_torch(value) for key, value in data.items()}

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(batch, dict):
            raise TypeError(f"MolmoAct2Preprocessor.forward expects dict[str, Any], got {type(batch)}")

        normalized_batch = self._normalizer_step(batch)
        bundle = self._extractor_step.extract(normalized_batch)
        prompt_pack = self._prompt_step.encode(bundle)

        image_batch = (
            torch.stack(prompt_pack.flat_images, dim=0)
            if prompt_pack.flat_images
            else torch.empty((0, 3, 0, 0))
        )
        inputs = self._tokenize_and_pack(prompt_texts=prompt_pack.prompt_texts, image_batch=image_batch)

        packed: dict[str, Any] = dict(inputs)
        packed["task"] = bundle.tasks
        packed["state"] = bundle.state
        return packed
