# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: ANN,COM812,D,E501,EM,FBT001,I001,PLR2004,S105,SIM108,TCH001,TRY
# pylint: disable=all

"""Processors for MolmoAct2."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from physicalai.data.observation import ACTION, IMAGES, STATE, TASK, Feature, FeatureType
from physicalai.policies.utils.features import feature_by_type
from physicalai.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType

from .preprocessors import (
    MolmoAct2ActionPadder,
    MolmoAct2MultimodalTokenizer,
    MolmoAct2StateNormalizer,
    MolmoAct2TaskTokenizer,
)

if TYPE_CHECKING:
    from .config import MolmoAct2Config


class MolmoAct2Preprocessor(torch.nn.Module):
    """Pack Observation objects into MolmoAct2 model-ready tensors.

    For now this preprocessor only applies state normalization. Image, video,
    and text processing are intentionally separated and will be integrated as
    additional first-party components.
    """

    def __init__(
        self,
        config: MolmoAct2Config,
    ) -> None:
        """Initialize the staged MolmoAct2 preprocessor pipeline.

        Args:
            config: Resolved MolmoAct2 policy config.
        """
        super().__init__()
        self.config = config
        self.state_normalizer = MolmoAct2StateNormalizer(input_features=config.input_features)
        self.task_tokenizer = MolmoAct2TaskTokenizer(
            max_sequence_length=int(getattr(config.text_config, "max_position_embeddings", 4096)),
            input_features=config.input_features,
            num_state_tokens=config.num_state_tokens,
            add_setup_tokens=config.add_setup_tokens,
            add_control_tokens=config.add_control_tokens,
            setup_type=config.setup_type,
            control_mode=config.control_mode,
            tokenizer_name_or_path=config.tokenizer_name_or_path,
        )
        self.multimodal_tokenizer = MolmoAct2MultimodalTokenizer(
            input_features=config.input_features,
            num_state_tokens=config.num_state_tokens,
            add_setup_tokens=config.add_setup_tokens,
            add_control_tokens=config.add_control_tokens,
            setup_type=config.setup_type,
            control_mode=config.control_mode,
            tokenizer_name_or_path=config.tokenizer_name_or_path,
            processor_assets_path=config.processor_assets_path,
        )
        self.action_padder = MolmoAct2ActionPadder(
            output_features=config.output_features,
            max_action_dim=config.max_action_dim,
        )

    def _process_state(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Apply state preprocessing stage.

        Returns:
            dict[str, Any]: Batch with normalized state values.
        """
        return self.state_normalizer(batch)

    def _process_text(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Apply text preprocessing stage.

        Converts ``TASK`` text into tokenizer outputs consumed by the model.

        Returns:
            dict[str, Any]: Batch with ``input_ids`` and ``attention_mask``.
        """
        return self.task_tokenizer(batch)

    def _process_images(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Apply image preprocessing stage.

        Expands text tokens into multimodal tokens and emits image tensors.

        Returns:
            dict[str, Any]: Batch with multimodal language/vision tensors when available.
        """
        return self.multimodal_tokenizer(batch)

    def _process_action(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Apply action preprocessing stage.

        Pads actions when present and always emits action-dimension padding masks.

        Returns:
            dict[str, Any]: Batch with action padding metadata.
        """
        return self.action_padder(batch)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess one flattened observation batch.

        Args:
            batch: Flattened observation dictionary. Can be unbatched (no batch dim)
                   or already batched [B, ...].

        Returns:
            dict[str, Any]: Preprocessed batch with normalized state values.

        Raises:
            TypeError: If ``batch`` is not a dictionary.
        """
        if not isinstance(batch, dict):
            msg = f"MolmoAct2Preprocessor.forward expects dict[str, Any], got {type(batch)}"
            raise TypeError(msg)

        processed_batch = dict(batch)

        # Ensure batch dimension: if state exists and is 1D [D], convert to [1, D]
        state_key = None
        for key in ["state", "observation.state"]:
            if key in processed_batch:
                state_key = key
                break

        if state_key is not None and torch.is_tensor(processed_batch[state_key]):
            state_tensor = processed_batch[state_key]
            if state_tensor.ndim == 1:
                # Add batch dimension: [D] -> [1, D]
                processed_batch[state_key] = state_tensor.unsqueeze(0)
                # Also add batch dimension to images if they are 3D
                for key in list(processed_batch.keys()):
                    if str(key).startswith("images.") or str(key).startswith("observation.images."):
                        img = processed_batch[key]
                        if torch.is_tensor(img) and img.ndim == 3:
                            processed_batch[key] = img.unsqueeze(0)

        if STATE in processed_batch:
            processed_batch = self._process_state(processed_batch)

        if TASK in processed_batch:
            processed_batch = self._process_text(processed_batch)

        if IMAGES in processed_batch or any(str(key).startswith(f"{IMAGES}.") for key in processed_batch):
            processed_batch = self._process_images(processed_batch)

        return self._process_action(processed_batch)


class MolmoAct2Postprocessor(torch.nn.Module):
    """Convert normalized MolmoAct2 outputs back into action space.

    This module expects normalized action predictions in ``[-1, 1]``, truncates any
    padded action dimensions to the environment action size, clamps values to the
    normalized range, and applies inverse normalization using resolved
    ``output_features`` metadata.
    """

    def __init__(self, *, output_features: list[Feature], max_action_dim: int) -> None:
        """Initialize the MolmoAct2 action postprocessor.

        Args:
            output_features: Resolved output feature definitions that include
                normalization metadata.
            max_action_dim: Fallback maximum action dimension when no ACTION
                feature shape is available.
        """
        super().__init__()
        action_feature = feature_by_type(output_features, FeatureType.ACTION)
        self.env_action_dim = (
            int(action_feature.shape[0]) if action_feature and action_feature.shape else int(max_action_dim)
        )
        self.action_name = action_feature.name if action_feature else ACTION
        output_features_by_name = {f.name: f for f in output_features if f.name}
        self._denormalizer = FeatureNormalizeTransform(
            output_features_by_name,
            {FeatureType.ACTION: NormalizationType.QUANTILES},
            inverse=True,
        )
        # Store per-dim normalization mask (True=normalize, False=passthrough e.g. gripper)
        action_norm_mask = (
            action_feature.normalization_data.mask
            if action_feature and action_feature.normalization_data is not None
            else None
        )
        if action_norm_mask is not None:
            self.register_buffer("_action_norm_mask", torch.tensor(action_norm_mask, dtype=torch.bool))
        else:
            self._action_norm_mask = None

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Postprocess normalized action predictions.

        Args:
            batch: Model output batch containing an ``ACTION`` tensor.

        Returns:
            dict[str, Any]: Updated batch with denormalized action tensor.

        Raises:
            TypeError: If ``batch`` is not a dictionary.
            ValueError: If ``ACTION`` is not present in ``batch``.
        """
        if not isinstance(batch, dict):
            msg = f"MolmoAct2Postprocessor.forward expects dict[str, Any], got {type(batch)}"
            raise TypeError(msg)

        if ACTION not in batch:
            msg = "MolmoAct2 postprocessor expected an action tensor in outputs."
            raise ValueError(msg)

        processed = dict(batch)
        action = processed[ACTION]

        action = action[..., : self.env_action_dim]
        # Clamp all dims to [-1, 1] before inverse normalization (matches LeRobot
        # MolmoAct2ClampActionProcessorStep; the masked unnormalizer then passes
        # gripper dims through as-is).
        action = action.clamp(-1.0, 1.0)
        denormalized = self._denormalizer.to(action.device)({self.action_name: action})[self.action_name]
        processed[ACTION] = denormalized
        return processed


def make_molmoact2_preprocessors(
    config: MolmoAct2Config,
) -> tuple[MolmoAct2Preprocessor, MolmoAct2Postprocessor]:
    """Helper function to make pre and post processors for MolmoAct2.

    Returns:
        MolmoAct2Preprocessor: The preprocessor.
        MolmoACt2Postprocessor: The postprocessor.
    """
    preprocessor = MolmoAct2Preprocessor(config=config)
    postprocessor = MolmoAct2Postprocessor(
        output_features=config.output_features,
        max_action_dim=config.max_action_dim,
    )
    return preprocessor, postprocessor
