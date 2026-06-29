# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Main MolmoAct2 preprocessor orchestrating composable preprocessing steps."""

from __future__ import annotations

from typing import Any

import torch

from physicalai.data.observation import FeatureType

from .common import feature_by_type
from .local_processor import load_molmoact2_processor_from_pretrained
from .preprocess_steps import (
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

        self._processor: Any = None

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

        packed: dict[str, Any] = dict(inputs)
        packed["task"] = bundle.tasks
        packed["state"] = bundle.state
        return packed
