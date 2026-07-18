# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 preprocessor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from physicalai.data.constants import IMAGE_MASKS, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.data.observation import ACTION, IMAGES, STATE, TASK, FeatureType

from .image import MolmoAct2ImageProcessor
from .inputs import build_model_inputs
from .joint_transform import JointFrameTransform
from .preprocess_steps import (
    ActionExtractor,
    ActionPadder,
    FeatureBatchNormalizer,
    ImagePacker,
    PreprocessBatchBundle,
    RobotPromptEncoder,
    StateTaskImageExtractor,
)
from .tokenizers import MolmoAct2Tokenizers

if TYPE_CHECKING:
    from physicalai.policies.molmoact2.config import MolmoAct2Config

_DEFAULT_IMAGE_SIZE = (378, 378)


def _image_input_size(config: MolmoAct2Config) -> tuple[int, int]:
    """Resolve the ``(height, width)`` images are resized to before the model.

    Returns:
        The target image ``(height, width)``.
    """
    processor_config = config.processor_config
    if processor_config is None:
        return _DEFAULT_IMAGE_SIZE
    size = processor_config.image_processor.size
    return int(size["height"]), int(size["width"])


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
                6. Tokenize prompt text with checkpoint tokenizer.
                7. Insert BOS token if required.
                8. Assemble model input dictionary.
                9. Return packed model-ready outputs.
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
            feature.name for feature in input_features if feature.ftype == FeatureType.VISUAL and feature.name
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
        self._image_packer = ImagePacker(image_size=_image_input_size(config))
        self._image_processor = (
            MolmoAct2ImageProcessor(config.processor_config.image_processor)
            if config.processor_config is not None
            else None
        )
        self._tokenizers = MolmoAct2Tokenizers(
            tokenizer_name_or_path=config.tokenizer_name_or_path,
            tokenizer_config=config.tokenizer_config,
        )
        self._action_extractor = ActionExtractor()
        self._action_padder = ActionPadder(max_action_dim=int(config.max_action_dim))
        self._joint_transform = JointFrameTransform(config.joint_signs, config.joint_offsets)

    @staticmethod
    def _validate_batch(batch: dict[str, Any]) -> None:
        """Validate the input batch object.

        Args:
            batch: Input dictionary.

        Raises:
            TypeError: If batch is not a dictionary.
            ValueError: If keys are not in batch which are required.
        """
        # check is dict
        if not isinstance(batch, dict):
            msg = f"MolmoAct2Preprocessor.forward expects dict[str, object], got {type(batch)}"
            raise TypeError(msg)

        has_state = STATE in batch or f"observation.{STATE}" in batch
        if not has_state:
            msg = f"{STATE} is expected in batch. Given keys: {list(batch.keys())}"
            raise ValueError(msg)

        has_task = TASK in batch or f"observation.{TASK}" in batch or "observation.language" in batch
        if not has_task:
            msg = f"{TASK} is expected in batch. Given keys: {list(batch.keys())}"
            raise ValueError(msg)

        has_images_nested = isinstance(batch.get(IMAGES), dict)
        has_images_flat = any(str(key).startswith(f"{IMAGES}.") for key in batch)
        if not (has_images_nested or has_images_flat):
            msg = f"{IMAGES} are expected in batch. Given keys: {list(batch.keys())}"
            raise ValueError(msg)

    def _build_token_outputs(
        self,
        prompt_texts: list[str],
        *,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Tokenize prompt text.

        Args:
            prompt_texts: Final prompt text list.
            device: Target device for token tensors.

        Returns:
            Dictionary containing tokenized prompt tensors.
        """
        input_ids, attention_mask = self._tokenizers.tokenize_prompts(prompt_texts)

        return {
            TOKENIZED_PROMPT: input_ids.to(device=device),
            TOKENIZED_PROMPT_MASK: attention_mask.to(device=device),
        }

    @staticmethod
    def _preprocess_state(bundle: PreprocessBatchBundle) -> dict[str, torch.Tensor]:
        """Build the state-only output mapping.

        Returns:
            Dictionary containing only the state tensor.
        """
        return {STATE: bundle.state}

    def _preprocess_images(self, bundle: PreprocessBatchBundle) -> dict[str, torch.Tensor]:
        """Pack visual inputs into model image tensors and masks.

        Returns:
            Dictionary containing packed images and image masks.
        """
        images, image_masks = self._image_packer(bundle.images_by_example)
        return {
            IMAGES: images,
            IMAGE_MASKS: image_masks,
        }

    def _preprocess_task_text(self, bundle: PreprocessBatchBundle) -> dict[str, torch.Tensor]:
        """Encode prompts and tokenize text inputs.

        Returns:
            Dictionary containing token ids and attention masks.
        """
        prompt_pack = self._prompt_encoder.encode(bundle)
        return self._build_token_outputs(
            prompt_pack.prompt_texts,
            device=bundle.state.device,
        )

    def _apply_input_joint_transform(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Map SO-101 joint state/actions into the checkpoint frame before normalizing.

        Returns:
            The batch with joint state/actions transformed when ``adapt_to_so101``
            is enabled, otherwise the batch unchanged.
        """
        if not self.config.adapt_to_so101:
            return batch
        batch = dict(batch)
        for key in (STATE, f"observation.{STATE}", ACTION, f"action.{ACTION}"):
            value = batch.get(key)
            if torch.is_tensor(value):
                batch[key] = self._joint_transform.to_checkpoint(value)
        return batch

    def _preprocess_action(self, normalized_batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Build padded action targets and masks for training (empty at inference).

        Returns:
            Dictionary with ``action``, ``action_horizon_is_pad`` and
            ``action_dim_is_pad`` when action targets are present, else empty.
        """
        action = self._action_extractor.extract(normalized_batch)
        if action is None:
            return {}
        padded, action_horizon_is_pad, action_dim_is_pad = self._action_padder(action)
        return {
            ACTION: padded,
            "action_horizon_is_pad": action_horizon_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
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
            A packed dictionary of fully-prepared, backbone-ready model inputs.
        """
        self._validate_batch(batch)
        batch = self._apply_input_joint_transform(batch)
        normalized_batch = self._normalizer(batch)
        bundle = self._extractor.extract(normalized_batch)

        state_outputs = self._preprocess_state(bundle)
        image_outputs = self._preprocess_images(bundle)
        text_outputs = self._preprocess_task_text(bundle)

        packed: dict[str, Any] = {}
        packed.update(text_outputs)
        packed.update(image_outputs)
        packed.update(state_outputs)
        packed.update(self._preprocess_action(normalized_batch))
        packed = self._ensure_tensor_outputs(packed)

        return self._build_model_inputs(packed)

    def _build_model_inputs(self, packed: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Turn packed prompt/images/state/action into backbone-ready model inputs.

        This runs the value-dependent host prep (image patchify, ``<|image|>``
        placeholder expansion, per-example image batching) that must live outside
        the exported model graph. ``state`` is intentionally dropped: it has
        already been discretized into ``input_ids``, so the exported model never
        consumes it as a separate input.

        Returns:
            A dict of fully-prepared model input tensors and, when training, the
            padded ``action`` target and its horizon mask.

        Raises:
            ValueError: If the processor config is missing.
        """
        if self._image_processor is None:
            msg = "MolmoAct2Preprocessor requires processor_config to build model inputs."
            raise ValueError(msg)
        prepared = build_model_inputs(packed, config=self.config, image_processor=self._image_processor)
        if ACTION in packed:
            prepared[ACTION] = packed[ACTION]
            prepared["action_horizon_is_pad"] = packed["action_horizon_is_pad"]
        return prepared
