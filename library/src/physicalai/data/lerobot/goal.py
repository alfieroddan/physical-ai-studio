# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Task-goal conditioning for LeRobot datasets."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from lightning_utilities import module_available

from physicalai.data.goal import GoalConditionedDataset

from .converters import FormatConverter
from .dataset import _LeRobotDatasetAdapter

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from physicalai.data.observation import Observation, ObservationValue

if TYPE_CHECKING or module_available("lerobot"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
else:
    LeRobotDataset = None


def _map(fn: Callable[[torch.Tensor], torch.Tensor], value: ObservationValue) -> ObservationValue:
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: fn(torch.as_tensor(item)) for key, item in value.items()}
    return fn(torch.as_tensor(value))


def _copy_images(images: ObservationValue) -> ObservationValue:
    return _map(lambda image: image.detach().cpu().contiguous().clone(), images)


def _normalize_task(task: str) -> str:
    return " ".join(task.lower().split()).rstrip(".")


def _raw_dataset(dataset: LeRobotDataset | _LeRobotDatasetAdapter) -> LeRobotDataset:
    if isinstance(dataset, _LeRobotDatasetAdapter):
        return dataset._lerobot_dataset  # noqa: SLF001
    return dataset


@contextmanager
def _without_image_transforms(dataset: LeRobotDataset) -> Generator[None, None, None]:
    """Temporarily disable observation transforms while decoding canonical goals."""
    transforms = getattr(dataset, "image_transforms", None)
    if transforms is not None:
        dataset.set_image_transforms(None)
    try:
        yield
    finally:
        if transforms is not None:
            dataset.set_image_transforms(transforms)


@dataclass(frozen=True)
class _TaskGoals:
    """Private immutable index of one canonical image set per task."""

    task_indices: dict[str, int]
    images: dict[int, ObservationValue]

    @classmethod
    def from_dataset(cls, dataset: LeRobotDataset) -> _TaskGoals:
        """Decode the last frame of the first selected episode for every task.

        Returns:
            The canonical goals indexed by task.

        Raises:
            ValueError: If the dataset cannot provide a canonical image for each selected task.
        """
        if getattr(dataset, "delta_timestamps", None):
            msg = "Goal conditioning requires delta_timestamps=None to select a single goal frame."
            raise ValueError(msg)

        task_indices: dict[str, int] = {}
        for task, index in zip(dataset.meta.tasks.index, dataset.meta.tasks["task_index"], strict=True):
            normalized = _normalize_task(str(task))
            task_index = int(index)
            if normalized in task_indices and task_indices[normalized] != task_index:
                msg = f"Task descriptions normalize to the same key '{normalized}'."
                raise ValueError(msg)
            task_indices[normalized] = task_index

        episodes = cast("list[dict[str, Any]]", list(dataset.meta.episodes))
        episodes.sort(key=lambda episode: int(episode["dataset_from_index"]))
        known_ids = {int(episode["episode_index"]) for episode in episodes}
        selected_ids = set(dataset.episodes) if dataset.episodes is not None else known_ids
        missing_ids = selected_ids - known_ids
        if missing_ids:
            msg = f"Dataset metadata is missing selected episodes: {sorted(missing_ids)}"
            raise ValueError(msg)
        if not selected_ids:
            msg = "Cannot build task goals for a dataset with no selected episodes."
            raise ValueError(msg)

        images: dict[int, ObservationValue] = {}
        relative_start = 0
        with _without_image_transforms(dataset):
            for episode in episodes:
                episode_index = int(episode["episode_index"])
                if episode_index not in selected_ids:
                    continue
                frame_count = int(episode["dataset_to_index"]) - int(episode["dataset_from_index"])
                if frame_count <= 0:
                    msg = f"Episode {episode_index} has invalid frame bounds."
                    raise ValueError(msg)

                task_index = int(dataset.get_raw_item(relative_start)["task_index"])
                if task_index not in images:
                    observation = FormatConverter.to_observation(dataset[relative_start + frame_count - 1])
                    goal_images = _copy_images(observation.images)
                    if goal_images is None or (isinstance(goal_images, dict) and not goal_images):
                        msg = f"Episode {episode_index} has no goal images."
                        raise ValueError(msg)
                    images[task_index] = goal_images
                relative_start += frame_count

        return cls(task_indices, images)

    def for_task_index(self, task_index: int) -> ObservationValue:
        if task_index not in self.images:
            msg = f"No goal for task index {task_index} in the selected training episodes."
            raise KeyError(msg)
        return _copy_images(self.images[task_index])

    def for_description(self, task: str) -> ObservationValue:
        normalized = _normalize_task(task)
        if normalized not in self.task_indices:
            msg = f"Task '{task}' is not in the dataset. Dataset tasks: {sorted(self.task_indices)}"
            raise KeyError(msg)
        return self.for_task_index(self.task_indices[normalized])


class LeRobotGoalConditionedDataset(GoalConditionedDataset):
    """LeRobot dataset whose samples carry one canonical, unaugmented goal per task."""

    _task_goals: _TaskGoals

    def __init__(
        self,
        dataset: LeRobotDataset | _LeRobotDatasetAdapter,
        *,
        goals_from: LeRobotGoalConditionedDataset | None = None,
    ) -> None:
        """Build task goals, or share them from another split's conditioned dataset."""
        if not isinstance(dataset, _LeRobotDatasetAdapter):
            dataset = _LeRobotDatasetAdapter.from_lerobot(dataset)
        super().__init__(dataset)
        self._task_goals = (
            _TaskGoals.from_dataset(_raw_dataset(dataset)) if goals_from is None else goals_from._task_goals  # noqa: SLF001
        )

    def get_goal(self, observation: Observation) -> ObservationValue:
        """Return the canonical goal for the sample's task."""
        return self._task_goals.for_task_index(int(observation.task_index))  # type: ignore[arg-type]

    def get_task_goal(self, task: str) -> ObservationValue:
        """Return the canonical goal for a task description."""
        return self._task_goals.for_description(task)


__all__ = ["LeRobotGoalConditionedDataset"]
