# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Goal conditioning for LeRobot datasets.

A goal shows the state a policy should reach: the last frame of a demonstration.
`LeRobotGoalConditionedDataset` attaches its images to every sample as ``Observation.goal_images``.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from lightning_utilities import module_available

from physicalai.data.goal import GoalConditionedDataset

from .converters import FormatConverter
from .dataset import _LeRobotDatasetAdapter

if TYPE_CHECKING:
    from collections.abc import Callable

    from physicalai.data.observation import Observation, ObservationValue

if TYPE_CHECKING or module_available("lerobot"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
else:
    LeRobotDataset = None

GoalSource = Literal["episode"]
GoalIndices = list[int] | str | Path

_MAX_GOAL_CACHE_BYTES = 64 * 1024 * 1024


def _map(fn: Callable[[torch.Tensor], torch.Tensor], value: ObservationValue) -> ObservationValue:
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: fn(torch.as_tensor(v)) for k, v in value.items()}
    return fn(torch.as_tensor(value))


def _to_cache_image(image: torch.Tensor) -> torch.Tensor:
    image = image.detach().cpu().contiguous()
    if image.dtype == torch.uint8:
        return image
    if (
        image.is_floating_point()
        and image.numel() > 0
        and torch.isfinite(image).all()
        and image.min() >= 0
        and image.max() <= 1
    ):
        return (image * 255).round().to(torch.uint8)
    return image


def _from_cache_image(image: torch.Tensor) -> torch.Tensor:
    if image.dtype == torch.uint8:
        return image.float() / 255
    return image.clone()


def _normalize_task(task: str) -> str:
    # Tolerate formatting differences between gym and dataset task text.
    return " ".join(task.lower().split()).rstrip(".")


def _load_goal_indices(value: GoalIndices | None) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.suffix.lower() != ".json":
            msg = f"Goal index file must be JSON, got '{path.suffix}'."
            raise ValueError(msg)
        with path.open(encoding="utf-8") as file:
            value = json.load(file)
    if not isinstance(value, list) or any(type(index) is not int for index in value):
        msg = "Goal indices must be a JSON array or list of integers."
        raise ValueError(msg)
    return value


def _resolve_goal_offsets(
    episodes: list[dict[str, Any]],
    selected_episodes: list[dict[str, Any]],
    goal_indices: GoalIndices | None,
) -> dict[int, int]:
    indices = _load_goal_indices(goal_indices)
    if indices is None:
        return {}
    if len(indices) == len(episodes):
        return {int(episode["episode_index"]): index for episode, index in zip(episodes, indices, strict=True)}
    if len(indices) == len(selected_episodes):
        return {int(episode["episode_index"]): index for episode, index in zip(selected_episodes, indices, strict=True)}
    msg = (
        f"Expected {len(episodes)} goal indices for the full dataset or "
        f"{len(selected_episodes)} for the selected episodes; got {len(indices)}."
    )
    raise ValueError(msg)


def _nbytes(value: ObservationValue) -> int:
    if value is None:
        return 0
    tensors = value.values() if isinstance(value, dict) else (value,)
    total = 0
    for tensor in tensors:
        image = torch.as_tensor(tensor)
        total += image.numel() * image.element_size()
    return total


class LeRobotGoalProvider:
    """Provide each episode's own terminal frame as its goal.

    The provider reads only from the supplied dataset instance. It lazily decodes goals and
    keeps a bounded least-recently-used cache, so filtered datasets do not trigger a second
    Hugging Face dataset load and held-out episodes cannot become training goals.

    The last frame must show the task completed. Episodes that continue afterwards (e.g.
    cyclic recordings that return home, see the recording guide) should be cropped to end at
    task completion, or the goal is the home pose.

    Examples:
        >>> dataset = LeRobotDataset("lerobot/pusht", episodes=[3])
        >>> goals = LeRobotGoalProvider(dataset)
        >>> goals.get(episode_index=3, task_index=0)["top"].shape
        torch.Size([3, 96, 96])
        >>> rollout_goal = goals.get_task_goal(task_index=0)
    """

    def __init__(
        self,
        dataset: LeRobotDataset,
        source: GoalSource = "episode",
        goal_indices: GoalIndices | None = None,
    ) -> None:
        """Index terminal frames without loading another dataset or decoding images.

        Args:
            dataset: The LeRobot dataset instance wrapped by the goal-conditioned dataset.
                Goals are limited to its selected episodes.
            source: Only ``"episode"`` is supported.
            goal_indices: Frame offsets within episodes, as a list or path to a JSON list.
                Offsets are zero-based; negative offsets count from the end (``-1`` is the last
                frame). A full-dataset list is indexed by episode order; a selected-split list is
                indexed by selected episode order. ``None`` selects each episode's last frame.

        Raises:
            ValueError: If temporal windows are configured, ``source`` is unknown, or an index is invalid.
        """
        if getattr(dataset, "delta_timestamps", None):
            msg = "Goal conditioning requires delta_timestamps=None to select a single terminal frame."
            raise ValueError(msg)
        if source != "episode":
            msg = f"Unknown goal source '{source}', expected 'episode'."
            raise ValueError(msg)

        self.source = source
        self._dataset = dataset
        self._task_indices = {
            _normalize_task(str(task)): int(index)
            for task, index in zip(dataset.meta.tasks.index, dataset.meta.tasks["task_index"], strict=True)
        }
        episodes = cast("list[dict[str, Any]]", list(dataset.meta.episodes))
        ordered_episodes = sorted(episodes, key=lambda item: int(item["dataset_from_index"]))
        selected_ids = (
            set(dataset.episodes)
            if dataset.episodes is not None
            else {int(episode["episode_index"]) for episode in episodes}
        )
        known_ids = {int(episode["episode_index"]) for episode in episodes}
        missing_ids = selected_ids - known_ids
        if missing_ids:
            msg = f"Dataset metadata is missing selected episodes: {sorted(missing_ids)}"
            raise ValueError(msg)
        if not selected_ids:
            msg = "Cannot build goal provider for a dataset with no selected episodes."
            raise ValueError(msg)

        selected_episodes = [episode for episode in ordered_episodes if int(episode["episode_index"]) in selected_ids]
        offsets = _resolve_goal_offsets(ordered_episodes, selected_episodes, goal_indices)

        self._episode_frames: dict[int, int] = {}
        self._task_episodes: dict[int, list[int]] = {}
        relative_frame = 0
        for episode in ordered_episodes:
            episode_index = int(episode["episode_index"])
            frame_count = int(episode["dataset_to_index"]) - int(episode["dataset_from_index"])
            if frame_count <= 0:
                msg = f"Episode {episode_index} has invalid frame bounds."
                raise ValueError(msg)
            if episode_index not in selected_ids:
                continue

            frame_offset = offsets.get(episode_index, -1)
            if frame_offset < 0:
                frame_offset += frame_count
            if frame_offset >= frame_count or frame_offset < 0:
                msg = (
                    f"Goal frame offset {frame_offset} is outside episode {episode_index}, "
                    f"which contains {frame_count} frames."
                )
                raise ValueError(msg)

            goal_frame = relative_frame + frame_offset
            raw_item = dataset.get_raw_item(goal_frame)
            self._episode_frames[episode_index] = goal_frame
            self._task_episodes.setdefault(int(raw_item["task_index"]), []).append(episode_index)
            relative_frame += frame_count

        self._goal_cache: OrderedDict[int, ObservationValue] = OrderedDict()
        self._goal_cache_bytes = 0

    @classmethod
    def from_dataset(
        cls,
        dataset: LeRobotDataset,
        source: GoalSource = "episode",
        goal_indices: GoalIndices | None = None,
    ) -> LeRobotGoalProvider:
        """Build goals from the supplied dataset without reopening it.

        Returns:
            A provider that reads goals from ``dataset``.
        """
        return cls(dataset, source=source, goal_indices=goal_indices)

    def get(self, episode_index: int, task_index: int) -> ObservationValue:  # noqa: ARG002
        """Return the terminal image for the sample's episode.

        Raises:
            KeyError: If there is no goal for the episode / task.
        """
        if episode_index not in self._episode_frames:
            msg = f"No goal for episode index {episode_index} in the wrapped dataset."
            raise KeyError(msg)
        return self._goal(episode_index)

    def task_index(self, task: str) -> int:
        """Return the dataset's index for a task description, e.g. ``Observation.task`` from a gym.

        Raises:
            KeyError: If the dataset has no such task.
        """
        normalized = _normalize_task(task)
        if normalized not in self._task_indices:
            msg = f"Task '{task}' is not in the dataset. Dataset tasks: {sorted(self._task_indices)}"
            raise KeyError(msg)
        return self._task_indices[normalized]

    def get_task_goal(self, task_index: int) -> ObservationValue:
        """Return the first selected episode's terminal image for a task, e.g. for rollouts."""
        return self._goal(self._episodes_for_task(task_index)[0])

    def _episodes_for_task(self, task_index: int) -> list[int]:
        if task_index not in self._task_episodes:
            msg = f"No goal for task index {task_index} in the wrapped dataset."
            raise KeyError(msg)
        return self._task_episodes[task_index]

    def _goal(self, episode_index: int) -> ObservationValue:
        if episode_index in self._goal_cache:
            self._goal_cache.move_to_end(episode_index)
            return _map(_from_cache_image, self._goal_cache[episode_index])

        observation = FormatConverter.to_observation(self._dataset[self._episode_frames[episode_index]])
        observed_episode = observation.episode_index
        if observed_episode is None:
            msg = f"Resolved terminal frame for episode {episode_index} has no episode index."
            raise ValueError(msg)
        observed_episode_index = int(torch.as_tensor(observed_episode).item())
        if observed_episode_index != episode_index:
            msg = f"Resolved terminal frame for episode {episode_index} belongs to episode {observed_episode_index}."
            raise ValueError(msg)
        images = _map(_to_cache_image, observation.images)
        if images is None or (isinstance(images, dict) and not images):
            msg = f"Episode {episode_index} has no goal images."
            raise ValueError(msg)

        image_bytes = _nbytes(images)
        if image_bytes <= _MAX_GOAL_CACHE_BYTES:
            while self._goal_cache and self._goal_cache_bytes + image_bytes > _MAX_GOAL_CACHE_BYTES:
                _, evicted = self._goal_cache.popitem(last=False)
                self._goal_cache_bytes -= _nbytes(evicted)
            self._goal_cache[episode_index] = images
            self._goal_cache_bytes += image_bytes
        return _map(_from_cache_image, images)


class LeRobotGoalConditionedDataset(GoalConditionedDataset):
    """LeRobot dataset whose samples carry goal images (``Observation.goal_images``).

    Can be passed to `LeRobotDataModule` as ``dataset``, or created for you with its
    ``goal_source`` argument.

    Examples:
        >>> dataset = LeRobotGoalConditionedDataset(LeRobotDataset("lerobot/libero"))
        >>> datamodule = LeRobotDataModule(dataset=dataset, train_batch_size=32)
        >>> rollout_goal = dataset.goals.get_task_goal(task_index=0)
    """

    def __init__(
        self,
        dataset: LeRobotDataset | _LeRobotDatasetAdapter,
        goal_source: GoalSource = "episode",
        *,
        goal_indices: GoalIndices | None = None,
        goals: LeRobotGoalProvider | None = None,
    ) -> None:
        """Wrap a dataset and build (or reuse) its goals.

        Args:
            dataset: Any split of a LeRobot dataset.
            goal_source: The sample's own episode terminal frame (currently the only source).
            goal_indices: Per-episode frame offsets as a list or JSON file. Offsets are zero-based;
                negative offsets count from the end, so ``-1`` selects the last frame. Defaults to
                the last frame of each episode.
            goals: Existing goals to share, e.g. between a train and val split. When given,
                ``goal_source`` is ignored and nothing is decoded.

        Raises:
            ValueError: If both ``goals`` and ``goal_indices`` are provided.
        """
        if goals is not None and goal_indices is not None:
            msg = "Cannot provide both 'goals' and 'goal_indices'."
            raise ValueError(msg)
        if not isinstance(dataset, _LeRobotDatasetAdapter):
            dataset = _LeRobotDatasetAdapter.from_lerobot(dataset)
        super().__init__(dataset)
        split = dataset._lerobot_dataset  # noqa: SLF001
        self.goals = goals if goals is not None else LeRobotGoalProvider.from_dataset(split, goal_source, goal_indices)

    def get_goal(self, observation: Observation) -> ObservationValue:
        """Return the goal images for a sample, according to the goal source."""
        return self.goals.get(int(observation.episode_index), int(observation.task_index))  # type: ignore[arg-type]

    def get_task_goal(self, task: str) -> ObservationValue:
        """Return the task's fixed goal images (last frame of its first episode), whatever the goal source."""
        return self.goals.get_task_goal(self.goals.task_index(task))


__all__ = ["GoalIndices", "GoalSource", "LeRobotGoalConditionedDataset", "LeRobotGoalProvider"]
