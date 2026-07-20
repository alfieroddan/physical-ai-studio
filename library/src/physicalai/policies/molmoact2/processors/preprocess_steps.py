# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Composable preprocessing steps for MolmoAct2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torchvision.transforms.functional as tv_functional

from physicalai.data.observation import ACTION, IMAGES, STATE, TASK, Feature, FeatureType
from physicalai.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType

from .utils import build_discrete_state_string, build_robot_text, normalize_text


class FeatureBatchNormalizer(torch.nn.Module):
    """Normalize batch features using configured feature statistics."""

    def __init__(self, *, input_features: list[Feature], output_features: list[Feature]) -> None:
        """Build the feature normalizer from the input/output feature schemas."""
        super().__init__()

        all_features = {feature.name: feature for feature in input_features + output_features if feature.name}

        state_feature = next((feature for feature in input_features if feature.ftype == FeatureType.STATE), None)
        action_feature = next((feature for feature in output_features if feature.ftype == FeatureType.ACTION), None)

        state_norm = (
            NormalizationType.QUANTILES
            if state_feature is not None and state_feature.normalization_data is not None
            else NormalizationType.IDENTITY
        )
        action_norm = (
            NormalizationType.QUANTILES
            if action_feature is not None and action_feature.normalization_data is not None
            else NormalizationType.IDENTITY
        )

        norm_map = {
            FeatureType.STATE: state_norm,
            FeatureType.ACTION: action_norm,
            FeatureType.VISUAL: NormalizationType.IDENTITY,
        }
        self._normalizer = FeatureNormalizeTransform(all_features, norm_map, inverse=False)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Normalize all configured features in-place-style and return a new dict.

        Returns:
            A new batch dict with configured features normalized in place.
        """
        device = next((value.device for value in batch.values() if torch.is_tensor(value)), torch.device("cpu"))
        return self._normalizer.to(device)(batch)


@dataclass
class PreprocessBatchBundle:
    """Intermediate typed values produced during preprocessing."""

    state: torch.Tensor
    tasks: list[str]
    images_by_example: list[list[torch.Tensor]]


class StateTaskImageExtractor:
    """Extract state, language task, and images from a flattened input batch."""

    def __init__(self, *, image_keys: list[str]) -> None:
        """Store the ordered image keys to extract from each batch."""
        self.image_keys = image_keys

    @staticmethod
    def _extract_state(batch: dict[str, Any]) -> torch.Tensor:
        raw_state = batch.get(STATE)
        if raw_state is None:
            raw_state = batch.get(f"observation.{STATE}")
        if raw_state is None:
            msg = "MolmoAct2 requires a state tensor in the input batch."
            raise ValueError(msg)

        state = torch.as_tensor(raw_state, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        return state.clamp(-1.0, 1.0)

    @staticmethod
    def _extract_tasks(batch: dict[str, Any], batch_size: int) -> list[str]:
        task_source = batch.get(TASK)
        if task_source is None:
            task_source = batch.get(f"observation.{TASK}")
        if task_source is None:
            task_source = batch.get("observation.language")

        if task_source is None:
            tasks = [""] * batch_size
        elif isinstance(task_source, str):
            tasks = [task_source] * batch_size
        elif torch.is_tensor(task_source):
            if task_source.ndim == 0:
                tasks = [str(task_source.item())] * batch_size
            else:
                tasks = [str(value) for value in task_source.detach().cpu().reshape(-1).tolist()]
        elif isinstance(task_source, (list, tuple)):
            tasks = [str(value) for value in task_source]
        else:
            tasks = [str(task_source)]

        if len(tasks) == 1 and batch_size > 1:
            tasks *= batch_size
        if len(tasks) != batch_size:
            msg = f"Expected {batch_size} task strings, got {len(tasks)}."
            raise ValueError(msg)

        return [normalize_text(task) for task in tasks]

    def _resolve_image_keys(self, batch: dict[str, Any]) -> list[str]:
        if self.image_keys:
            explicit_keys = [f"{IMAGES}.{name}" for name in self.image_keys]
            available_explicit = [key for key in explicit_keys if key in batch]
            if available_explicit:
                return available_explicit

        flat_image_keys = [str(key) for key in batch if str(key).startswith(f"{IMAGES}.") and "is_pad" not in str(key)]
        if flat_image_keys:
            return sorted(flat_image_keys)

        if isinstance(batch.get(IMAGES), dict):
            return [f"{IMAGES}.{name}" for name in batch[IMAGES] if "is_pad" not in str(name)]

        msg = "MolmoAct2 requires image tensors in BCHW format."
        raise ValueError(msg)

    @staticmethod
    def _get_image_value(batch: dict[str, Any], key: str) -> Any:  # noqa: ANN401
        if key in batch:
            return batch[key]

        if key.startswith(f"{IMAGES}.") and isinstance(batch.get(IMAGES), dict):
            nested = key.removeprefix(f"{IMAGES}.")
            images_dict = batch.get(IMAGES)
            if isinstance(images_dict, dict) and nested in images_dict:
                return images_dict[nested]

        msg = f"Image key {key!r} was not found in the batch."
        raise KeyError(msg)

    @staticmethod
    def _as_bchw_tensor(value: Any, *, key: str) -> torch.Tensor:  # noqa: ANN401
        tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
        if tensor.ndim != 4:  # noqa: PLR2004
            msg = f"Expected BCHW image tensor at {key}, got shape {tuple(tensor.shape)}"
            raise ValueError(msg)
        if int(tensor.shape[1]) != 3:  # noqa: PLR2004
            msg = f"Expected BCHW image tensor with 3 channels at {key}, got shape {tuple(tensor.shape)}"
            raise ValueError(msg)
        return tensor

    def _extract_images(self, batch: dict[str, Any], batch_size: int) -> list[list[torch.Tensor]]:
        images_by_example: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]

        for key in self._resolve_image_keys(batch):
            value = self._get_image_value(batch, key)
            bchw = self._as_bchw_tensor(value, key=key)
            if int(bchw.shape[0]) != batch_size:
                msg = f"Image batch size mismatch at {key}: expected {batch_size}, got {int(bchw.shape[0])}"
                raise ValueError(msg)

            for index in range(batch_size):
                images_by_example[index].append(bchw[index])

        return images_by_example

    def extract(self, batch: dict[str, Any]) -> PreprocessBatchBundle:
        """Extract and normalize raw state/task/image values.

        Returns:
            A :class:`PreprocessBatchBundle` carrying the extracted state
            tensor, task strings, and per-example image lists.
        """
        state = self._extract_state(batch)
        batch_size = int(state.shape[0])

        return PreprocessBatchBundle(
            state=state,
            tasks=self._extract_tasks(batch, batch_size),
            images_by_example=self._extract_images(batch, batch_size),
        )


@dataclass
class PromptPack:
    """Prompt text and optional flattened images."""

    prompt_texts: list[str]
    flat_images: list[torch.Tensor]


class RobotPromptEncoder:
    """Build readable MolmoAct2 prompt text from extracted values."""

    def __init__(
        self,
        *,
        num_state_tokens: int,
        setup_type: str,
        control_mode: str,
        add_setup_tokens: bool,
        add_control_tokens: bool,
    ) -> None:
        """Store the prompt-shaping flags used to encode each batch element."""
        self.num_state_tokens = num_state_tokens
        self.setup_type = setup_type
        self.control_mode = control_mode
        self.add_setup_tokens = add_setup_tokens
        self.add_control_tokens = add_control_tokens

    def encode(self, bundle: PreprocessBatchBundle) -> PromptPack:
        """Encode one prompt per batch element.

        Returns:
            A :class:`PromptPack` with one prompt text per batch element and
            the flattened list of image tensors.
        """
        prompt_texts: list[str] = []
        flat_images: list[torch.Tensor] = []
        for index in range(int(bundle.state.shape[0])):
            image_list = bundle.images_by_example[index]
            flat_images.extend(image_list)

            discrete_state = build_discrete_state_string(bundle.state[index], self.num_state_tokens)
            prompt = build_robot_text(
                task=bundle.tasks[index],
                discrete_state_string=discrete_state,
                setup_type=self.setup_type,
                control_mode=self.control_mode,
                add_setup_tokens=self.add_setup_tokens,
                add_control_tokens=self.add_control_tokens,
                num_images=len(image_list),
            )
            prompt_texts.append(prompt)

        return PromptPack(prompt_texts=prompt_texts, flat_images=flat_images)


class ImageResizeNormalizer(torch.nn.Module):
    """Resize per-example images to the model input size and normalize to [0, 1].

    Runs before :class:`ImagePacker` in the preprocessing pipeline. Each
    ``(C, H, W)`` image is resized on the uint8 grid (to match the reference
    numerics) and then converted to float32 in the ``[0, 1]`` range. The
    nested ``list[list[Tensor]]`` structure is preserved so :class:`ImagePacker`
    can pack the already-processed images into batch tensors.
    """

    def __init__(self, *, image_size: tuple[int, int]) -> None:
        """Store the target ``(height, width)`` the images are resized to."""
        super().__init__()
        self.height, self.width = image_size

    def _resize_normalize(self, image: torch.Tensor) -> torch.Tensor:
        """Resize a single ``(C, H, W)`` image to ``[0, 1]`` float32.

        Returns:
            Float tensor of shape ``(C, height, width)`` in the ``[0, 1]`` range.
        """
        if image.dtype == torch.uint8:
            pixels = image
        else:
            pixels = image.to(torch.float32)
            if float(pixels.max()) <= 1.0:
                pixels *= 255.0
            pixels = pixels.clamp(0, 255).to(torch.uint8)
        resized = tv_functional.resize(pixels, [self.height, self.width], antialias=False)
        return resized.to(torch.float32) / 255.0

    def forward(self, images_by_example: list[list[torch.Tensor]]) -> list[list[torch.Tensor]]:
        """Resize and normalize every image, preserving the nested list structure.

        Returns:
            The same nested ``list[list[Tensor]]`` layout as the input, with
            each image resized to ``(3, height, width)`` float32 in ``[0, 1]``.
        """
        return [[self._resize_normalize(image) for image in example_images] for example_images in images_by_example]


class ImagePacker(torch.nn.Module):
    """Pack already-resized, normalized images into batch tensors and masks."""

    def __init__(self, *, image_size: tuple[int, int]) -> None:
        """Store the target ``(height, width)`` used to shape empty outputs."""
        super().__init__()
        self.height, self.width = image_size

    def forward(self, images_by_example: list[list[torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack images into [N, B, C, H, W] and masks [N, B].

        Args:
            images_by_example: Per-example lists of ``(C, H, W)`` image tensors
                that have already been resized and normalized by
                :class:`ImageResizeNormalizer`.

        Returns:
            ``(images, masks)`` of shapes ``(N, B, 3, H, W)`` and ``(N, B)``.

        Raises:
            ValueError: If the per-element image count is inconsistent, or if
                the batch is empty.
        """
        batch_size = len(images_by_example)
        if batch_size == 0:
            empty_images = torch.empty((0, 0, 3, self.height, self.width), dtype=torch.float32)
            empty_masks = torch.empty((0, 0), dtype=torch.bool)
            return empty_images, empty_masks

        num_images = len(images_by_example[0])
        for example_images in images_by_example:
            if len(example_images) != num_images:
                msg = "MolmoAct2 requires a consistent number of images per batch element."
                raise ValueError(msg)

        if num_images == 0:
            empty_images = torch.empty((0, batch_size, 3, self.height, self.width), dtype=torch.float32)
            empty_masks = torch.empty((0, batch_size), dtype=torch.bool)
            return empty_images, empty_masks

        image_slots: list[torch.Tensor] = []
        mask_slots: list[torch.Tensor] = []
        for image_index in range(num_images):
            slot_tensor = torch.stack(
                [images_by_example[batch_index][image_index] for batch_index in range(batch_size)],
                dim=0,
            )
            image_slots.append(slot_tensor)
            mask_slots.append(torch.ones((batch_size,), dtype=torch.bool, device=slot_tensor.device))

        return torch.stack(image_slots, dim=0), torch.stack(mask_slots, dim=0)


class ActionExtractor:
    """Extract action tensor from normalized input batch when available."""

    @staticmethod
    def extract(batch: dict[str, Any]) -> torch.Tensor | None:
        """Return the action tensor from ``batch`` as float32, or ``None`` if absent."""
        raw_action = batch.get(ACTION)
        if raw_action is None:
            raw_action = batch.get(f"action.{ACTION}")
        if raw_action is None:
            return None

        if torch.is_tensor(raw_action):
            return raw_action.to(dtype=torch.float32)
        return torch.as_tensor(raw_action, dtype=torch.float32)


class ActionPadder(torch.nn.Module):
    """Pad action tensors to max_action_dim and emit padding masks."""

    def __init__(self, *, max_action_dim: int) -> None:
        """Store the target ``max_action_dim`` used to pad the last axis."""
        super().__init__()
        self.max_action_dim = int(max_action_dim)

    def forward(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return padded action, horizon mask, and dimension mask.

        Returns:
            ``(padded, action_horizon_is_pad, action_dim_is_pad)`` where the
            action is right-padded to ``max_action_dim``.

        Raises:
            ValueError: If the action is not 3-D ``[B, T, D]``, or if its
                last axis exceeds ``max_action_dim``.
        """
        if action.ndim == 2:  # noqa: PLR2004
            action = action.unsqueeze(1)
        if action.ndim != 3:  # noqa: PLR2004
            msg = f"MolmoAct2 expected action shape [B, T, D], got {tuple(action.shape)}."
            raise ValueError(msg)

        if int(action.shape[-1]) > self.max_action_dim:
            msg = f"Action dim {int(action.shape[-1])} exceeds max_action_dim={self.max_action_dim}."
            raise ValueError(msg)

        normalized = action.to(dtype=torch.float32).clamp(-1.0, 1.0)
        padded = torch.zeros(
            (*normalized.shape[:-1], self.max_action_dim),
            dtype=torch.float32,
            device=normalized.device,
        )
        padded[..., : int(normalized.shape[-1])] = normalized

        action_horizon_is_pad = torch.zeros(normalized.shape[:2], dtype=torch.bool, device=normalized.device)
        action_dim_is_pad = torch.ones(
            (normalized.shape[0], self.max_action_dim),
            dtype=torch.bool,
            device=normalized.device,
        )
        action_dim_is_pad[:, : int(normalized.shape[-1])] = False

        return padded, action_horizon_is_pad, action_dim_is_pad
