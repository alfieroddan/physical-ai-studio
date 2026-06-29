# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Main MolmoAct2 preprocessor orchestrating composable preprocessing steps."""

from __future__ import annotations

from typing import Any

import torch

from physicalai.data.observation import ACTION, FeatureType

from .common import feature_by_type, text_max_positions
from .local_processor import load_molmoact2_processor_from_pretrained
from .preprocess_steps import (
    ActionExtractor,
    ActionPadder,
    FeatureBatchNormalizer,
    RobotPromptEncoder,
    StateTaskImageExtractor,
)


class MolmoAct2Preprocessor(torch.nn.Module):
    """Pack Observation objects into MolmoAct2 model-ready tensors."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config

        self.state_feature = feature_by_type(config.input_features, FeatureType.STATE)
        self.action_feature = feature_by_type(config.output_features, FeatureType.ACTION)

        self.max_action_dim = int(config.max_action_dim)
        self.num_state_tokens = int(config.num_state_tokens) if int(config.num_state_tokens) > 0 else 256
        self.max_sequence_length = text_max_positions(config)
        self.setup_type = str(config.setup_type or "")
        self.control_mode = str(config.control_mode or "")
        self.add_setup_tokens = bool(config.add_setup_tokens)
        self.add_control_tokens = bool(config.add_control_tokens)
        self.image_keys = [
            feature.name for feature in config.input_features if feature.ftype == FeatureType.VISUAL and feature.name
        ]
        self.env_action_dim = int(self.action_feature.shape[0]) if self.action_feature and self.action_feature.shape else 0

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
        self._action_padder = ActionPadder(max_action_dim=self.max_action_dim)

        self._processor: Any = None
        self.register_buffer("_device_indicator", torch.empty(0), persistent=False)

    @property
    def processor(self) -> Any:
        if self._processor is not None:
            return self._processor

        tokenizer_name_or_path = self.config.tokenizer_name_or_path
        if not tokenizer_name_or_path:
            raise ValueError(
                "config.tokenizer_name_or_path is required. "
                "Provide it via constructor or set MolmoAct2Config.tokenizer_name_or_path.",
            )

        self._processor = load_molmoact2_processor_from_pretrained(
            tokenizer_name_or_path,
            processor_config=self.config.processor_config,
        )
        return self._processor

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(batch, dict):
            raise TypeError(f"MolmoAct2Preprocessor.forward expects dict[str, Any], got {type(batch)}")

        normalized_batch = self._normalizer_step(batch)
        bundle = self._extractor_step.extract(normalized_batch)
        prompt_pack = self._prompt_step.encode(bundle)

        inputs = self.processor(
            text=prompt_pack.prompt_texts,
            images=prompt_pack.flat_images,
            return_tensors="pt",
            padding=True,
        )

        if "pixel_values" in inputs and "image_token_pooling" in inputs:
            pixel_values = inputs["pixel_values"]
            image_token_pooling = inputs["image_token_pooling"]
            if torch.is_tensor(pixel_values) and pixel_values.ndim == 3 and torch.is_tensor(image_token_pooling):
                n_patches = int(pixel_values.shape[1])
                valid = image_token_pooling >= 0
                if torch.any(valid):
                    max_idx = int(image_token_pooling[valid].max().item())
                    if max_idx >= n_patches:
                        raise ValueError(
                            "image_token_pooling contains out-of-range indices for per-image local patch IDs: "
                            f"max_idx={max_idx}, n_patches={n_patches}."
                        )

        if "pixel_values_videos" in inputs and "video_token_pooling" in inputs:
            pixel_values_videos = inputs["pixel_values_videos"]
            video_token_pooling = inputs["video_token_pooling"]
            if (
                torch.is_tensor(pixel_values_videos)
                and pixel_values_videos.ndim == 3
                and torch.is_tensor(video_token_pooling)
            ):
                n_frame_patches_total = int(pixel_values_videos.shape[0] * pixel_values_videos.shape[1])
                valid = video_token_pooling >= 0
                if torch.any(valid):
                    max_idx = int(video_token_pooling[valid].max().item())
                    if max_idx >= n_frame_patches_total:
                        raise ValueError(
                            "video_token_pooling contains out-of-range indices for local frame patch IDs: "
                            f"max_idx={max_idx}, total_patches={n_frame_patches_total}."
                        )

        if int(inputs["input_ids"].shape[1]) > self.max_sequence_length:
            raise ValueError(
                f"MolmoAct2 sequence length {int(inputs['input_ids'].shape[1])} exceeds max_sequence_length={self.max_sequence_length}.",
            )

        batch_size = int(bundle.state.shape[0])
        action_dim_is_pad = torch.ones((batch_size, self.max_action_dim), dtype=torch.bool)
        action_horizon_is_pad = None
        action_padded = None

        action = ActionExtractor.extract(normalized_batch)
        if action is not None:
            action_padded, action_horizon_is_pad, action_dim_is_pad = self._action_padder(action)
        elif self.env_action_dim > 0:
            action_dim_is_pad[:, : self.env_action_dim] = False

        packed: dict[str, Any] = dict(inputs)
        packed["task"] = bundle.tasks
        packed["state"] = bundle.state
        packed["action_dim_is_pad"] = action_dim_is_pad
        if action_horizon_is_pad is not None:
            packed["action_horizon_is_pad"] = action_horizon_is_pad
        if action_padded is not None:
            packed[ACTION] = action_padded

        target_device = self._device_indicator.device
        for key, value in list(packed.items()):
            if torch.is_tensor(value):
                packed[key] = value.to(device=target_device)
        return packed
