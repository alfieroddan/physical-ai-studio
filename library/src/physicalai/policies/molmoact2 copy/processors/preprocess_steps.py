# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Composable preprocessing steps for MolmoAct2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from physicalai.data.observation import ACTION, IMAGES, STATE, TASK, Feature, FeatureType, Observation
from physicalai.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType

from .common import build_discrete_state_string, build_robot_text, normalize_text


class FeatureBatchNormalizer(torch.nn.Module):
    """Normalize observation batch features using configured stats."""

    def __init__(self, *, input_features: list[Feature], output_features: list[Feature]) -> None:
        super().__init__()
        all_features = {f.name: f for f in input_features + output_features if f.name}

        state_feature = next((f for f in input_features if f.ftype == FeatureType.STATE), None)
        action_feature = next((f for f in output_features if f.ftype == FeatureType.ACTION), None)

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
        device = next((v.device for v in batch.values() if torch.is_tensor(v)), torch.device("cpu"))
        return self._normalizer.to(device)(batch)


@dataclass
class PreprocessBatchBundle:
    """Typed preprocessor intermediate values."""

    state: torch.Tensor
    tasks: list[str]
    images_by_example: list[list[torch.Tensor]]


class StateTaskImageExtractor:
    """Extract and normalize state/task/image values from a normalized batch."""

    def __init__(self, *, image_keys: list[str]) -> None:
        self.image_keys = image_keys

    def _resolve_image_keys(self, observation: dict[str, Any]) -> list[str]:
        candidate_keys = Observation.get_flattened_keys(observation, IMAGES)
        expanded_keys: list[str] = []
        for key in candidate_keys:
            key_str = str(key)
            if key_str == IMAGES and isinstance(observation.get(IMAGES), dict):
                expanded_keys.extend([f"{IMAGES}.{nested_key}" for nested_key in observation[IMAGES]])
            else:
                expanded_keys.append(key_str)

        requested = [f"{IMAGES}.{name}" for name in self.image_keys if f"{IMAGES}.{name}" in expanded_keys]
        if requested:
            return requested
        fallback = [
            key for key in expanded_keys if key.startswith(f"{IMAGES}.") or key.startswith("observation.images.")
        ]
        if not fallback:
            raise ValueError("MolmoAct2 requires at least one image observation.")
        return sorted(fallback)

    @staticmethod
    def _as_chw_tensor(image: Any) -> torch.Tensor:
        if not torch.is_tensor(image):
            raise TypeError(f"Expected torch image tensor in CHW, got {type(image)}")
        if image.ndim != 3:
            raise ValueError(f"Expected CHW image item, got shape {tuple(image.shape)}")
        if image.shape[0] != 3:
            raise ValueError(f"Expected CHW with 3 channels, got shape {tuple(image.shape)}")
        return image

    @staticmethod
    def _resolve_image_value(observation: dict[str, Any], key: str) -> Any:
        if key in observation:
            return observation[key]

        if key.startswith(f"{IMAGES}.") and isinstance(observation.get(IMAGES), dict):
            nested_key = key.removeprefix(f"{IMAGES}.")
            images = observation[IMAGES]
            if nested_key in images:
                return images[nested_key]

        if key.startswith("observation.images.") and isinstance(observation.get("observation.images"), dict):
            nested_key = key.removeprefix("observation.images.")
            images = observation["observation.images"]
            if nested_key in images:
                return images[nested_key]

        msg = f"MolmoAct2 image key {key!r} was not found in observation batch."
        raise KeyError(msg)

    def _extract_images(self, observation: dict[str, Any], batch_size: int) -> list[list[torch.Tensor]]:
        images_by_example: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        for key in self._resolve_image_keys(observation):
            value = self._resolve_image_value(observation, key)
            if not torch.is_tensor(value):
                raise TypeError(f"Expected batched image tensor/ndarray at {key}, got {type(value)}")
            if getattr(value, "ndim", 0) != 4:
                raise ValueError(
                    f"Expected batched images in BCHW format at {key}, got shape {getattr(value, 'shape', None)}"
                )
            if int(value.shape[1]) != 3:
                raise ValueError(f"Expected BCHW with 3 channels at {key}, got shape {tuple(value.shape)}")
            for batch_idx in range(batch_size):
                item = value
                if getattr(value, "ndim", 0) >= 4:
                    item = value[batch_idx]
                images_by_example[batch_idx].append(self._as_chw_tensor(item))
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
        return [normalize_text(task) for task in tasks]

    def extract(self, batch: dict[str, Any]) -> PreprocessBatchBundle:
        raw_state = batch.get(STATE)
        if raw_state is None:
            raw_state = batch.get(f"observation.{STATE}")
        if raw_state is None:
            raise ValueError("MolmoAct2 requires state for discrete state prompting.")

        state = torch.as_tensor(raw_state, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state = state.clamp(-1.0, 1.0)
        batch_size = int(state.shape[0])

        return PreprocessBatchBundle(
            state=state,
            tasks=self._extract_tasks(batch, batch_size),
            images_by_example=self._extract_images(batch, batch_size),
        )


@dataclass
class PromptPack:
    """Text prompts and flattened image list for processor input."""

    prompt_texts: list[str]
    flat_images: list[torch.Tensor]


class RobotPromptEncoder:
    """Build natural-language prompt strings from extracted batch values."""

    def __init__(
        self,
        *,
        num_state_tokens: int,
        setup_type: str,
        control_mode: str,
        add_setup_tokens: bool,
        add_control_tokens: bool,
    ) -> None:
        self.num_state_tokens = num_state_tokens
        self.setup_type = setup_type
        self.control_mode = control_mode
        self.add_setup_tokens = add_setup_tokens
        self.add_control_tokens = add_control_tokens

    def encode(self, bundle: PreprocessBatchBundle) -> PromptPack:
        state_np = bundle.state.detach().cpu().numpy()
        prompt_texts: list[str] = []
        flat_images: list[torch.Tensor] = []

        for i in range(bundle.state.shape[0]):
            flat_images.extend(bundle.images_by_example[i])
            discrete_state = build_discrete_state_string(state_np[i], self.num_state_tokens)
            prompt_texts.append(
                build_robot_text(
                    task=bundle.tasks[i],
                    discrete_state_string=discrete_state,
                    setup_type=self.setup_type,
                    control_mode=self.control_mode,
                    add_setup_tokens=self.add_setup_tokens,
                    add_control_tokens=self.add_control_tokens,
                    num_images=len(bundle.images_by_example[i]),
                )
            )

        return PromptPack(prompt_texts=prompt_texts, flat_images=flat_images)


class ActionPadder(torch.nn.Module):
    """Pad normalized action tensors to fixed max_action_dim."""

    def __init__(self, *, max_action_dim: int) -> None:
        super().__init__()
        self.max_action_dim = int(max_action_dim)

    def forward(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if action.ndim == 2:
            action = action.unsqueeze(1)
        if action.ndim != 3:
            raise ValueError(f"MolmoAct2 expected action shape [B, T, D], got {tuple(action.shape)}.")
        if int(action.shape[-1]) > self.max_action_dim:
            raise ValueError(f"Action dim {action.shape[-1]} exceeds MolmoAct2 max_action_dim={self.max_action_dim}.")

        normalized = action.to(dtype=torch.float32).clamp(-1.0, 1.0)
        padded = torch.zeros((*normalized.shape[:-1], self.max_action_dim), device=normalized.device, dtype=torch.float32)
        padded[..., : normalized.shape[-1]] = normalized

        action_dim_is_pad = torch.ones((normalized.shape[0], self.max_action_dim), device=normalized.device, dtype=torch.bool)
        action_dim_is_pad[:, : normalized.shape[-1]] = False
        action_horizon_is_pad = torch.zeros(normalized.shape[:2], device=normalized.device, dtype=torch.bool)
        return padded, action_horizon_is_pad, action_dim_is_pad


class ActionExtractor:
    """Extract and convert optional action tensors from input batch."""

    @staticmethod
    def extract(batch: dict[str, Any]) -> torch.Tensor | None:
        raw_action = batch.get(ACTION)
        if raw_action is None:
            raw_action = batch.get(f"action.{ACTION}")
        if raw_action is None:
            return None
        if torch.is_tensor(raw_action):
            return raw_action.to(dtype=torch.float32)
        return torch.as_tensor(raw_action, dtype=torch.float32)
