# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 preprocessor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from physicalai.data.constants import IMAGE_MASKS, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.data.observation import ACTION, IMAGES, STATE, FeatureType

from .preprocess_steps import (
    ActionExtractor,
    ActionPadder,
    FeatureBatchNormalizer,
    ImagePacker,
    RobotPromptEncoder,
    StateTaskImageExtractor,
)
from .tokenizers import MolmoAct2Tokenizers

if TYPE_CHECKING:
    from physicalai.policies.molmoact2.config import MolmoAct2Config


class MolmoAct2Preprocessor(torch.nn.Module):
    """Convert observations into model-ready MolmoAct2 tensors.

    OVERVIEW:
        1. Validate batch structure.
        2. Normalize configured input/output features.
        3. Extract state, task text, and image tensors.
        4. Build robot prompt text:
           - Normalize task text.
           - Discretize state values into state tokens.
           - Add setup/control wrappers.
           - Add image placeholders.
        5. Pack images into [N_images, B, C, H, W] and image masks.
        6. Optionally extract and pad action targets to max_action_dim.
          7. Tokenize prompt text with checkpoint tokenizer.
          8. Insert BOS token if required.
          9. Assemble model input dictionary.
          10. Return packed model-ready outputs.

    The implementation favors readability and clear stage boundaries.
    """

    def __init__(self, config: MolmoAct2Config) -> None:
        """Initialize step objects used in preprocessing.

        Args:
            config: MolmoAct2 configuration.
        """
        super().__init__()
        self.config = config

        input_features = list(config.input_features or [])
        output_features = list(config.output_features or [])

        image_keys = [
            feature.name
            for feature in input_features
            if feature.ftype == FeatureType.VISUAL and feature.name
        ]

        self._normalizer = FeatureBatchNormalizer(
            input_features=input_features,
            output_features=output_features,
        )
        self._extractor = StateTaskImageExtractor(image_keys=image_keys)
        self._prompt_encoder = RobotPromptEncoder(
            num_state_tokens=int(config.num_state_tokens) if int(config.num_state_tokens) > 0 else 256,
            setup_type=str(config.setup_type or ""),
            control_mode=str(config.control_mode or ""),
            add_setup_tokens=bool(config.add_setup_tokens),
            add_control_tokens=bool(config.add_control_tokens),
        )
        self._image_packer = ImagePacker()
        self._action_extractor = ActionExtractor()
        self._action_padder = ActionPadder(max_action_dim=int(config.max_action_dim))

        self._tokenizers = MolmoAct2Tokenizers(
            tokenizer_name_or_path=config.tokenizer_name_or_path,
        )

    @staticmethod
    def _validate_batch(batch: dict[str, Any]) -> None:
        """Validate the input batch object.

        Args:
            batch: Input dictionary.

        Raises:
            TypeError: If batch is not a dictionary.
        """
        if not isinstance(batch, dict):
            msg = f"MolmoAct2Preprocessor.forward expects dict[str, object], got {type(batch)}"
            raise TypeError(msg)

    def _build_token_outputs(self, prompt_texts: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize prompt text.

        Args:
            prompt_texts: Final prompt text list.

        Returns:
            Dictionary containing tokenized prompt tensors.
        """
        input_ids, attention_mask = self._tokenizers.tokenize_prompts(prompt_texts)

        return {
            TOKENIZED_PROMPT: input_ids,
            TOKENIZED_PROMPT_MASK: attention_mask,
        }

    @staticmethod
    def _ensure_tensor_outputs(outputs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Ensure all output values are torch tensors.

        Tensor lists/tuples are stacked on dim 0; other array-like values are converted
        with ``torch.as_tensor``.

        Args:
            outputs: Packed output dictionary before final coercion.

        Returns:
            Dictionary with tensor-only values.

        Raises:
            TypeError: If any output value cannot be converted to a tensor.
        """
        tensor_outputs: dict[str, torch.Tensor] = {}
        for key, value in outputs.items():
            if torch.is_tensor(value):
                tensor_outputs[key] = value
                continue

            if isinstance(value, (list, tuple)) and value and all(torch.is_tensor(item) for item in value):
                tensor_outputs[key] = torch.stack(list(value), dim=0)
                continue

            try:
                tensor_outputs[key] = torch.as_tensor(value)
            except Exception as exc:
                msg = f"MolmoAct2Preprocessor output {key!r} is not tensor-convertible: {type(value)}"
                raise TypeError(msg) from exc

        return tensor_outputs

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Preprocess one training/inference batch.

        Args:
            batch: Input observation dictionary with image tensors in BCHW format.

        Returns:
            A packed dictionary matching MolmoAct2 model inputs.
        """
        self._validate_batch(batch)

        normalized_batch = self._normalizer(batch)
        bundle = self._extractor.extract(normalized_batch)

        prompt_pack = self._prompt_encoder.encode(bundle)
        images, image_masks = self._image_packer(bundle.images_by_example)

        action = self._action_extractor.extract(normalized_batch)
        token_outputs = self._build_token_outputs(prompt_pack.prompt_texts)

        packed: dict[str, Any] = {
            TOKENIZED_PROMPT: token_outputs[TOKENIZED_PROMPT],
            TOKENIZED_PROMPT_MASK: token_outputs[TOKENIZED_PROMPT_MASK],
            IMAGES: images,
            IMAGE_MASKS: image_masks,
            STATE: bundle.state,
        }

        if action is not None:
            action_padded, action_horizon_is_pad, action_dim_is_pad = self._action_padder(action)
            packed[ACTION] = action_padded
            packed["action_horizon_is_pad"] = action_horizon_is_pad
            packed["action_dim_is_pad"] = action_dim_is_pad

        return self._ensure_tensor_outputs(packed)
