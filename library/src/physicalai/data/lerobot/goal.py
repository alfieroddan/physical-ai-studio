# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Goal conditioning for LeRobot datasets.

A goal shows the state a policy should reach: the last frame of a demonstration.
`LeRobotGoalConditionedDataset` attaches its images to every sample as ``Observation.goal_images``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, get_args

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

GoalSource = Literal["task", "episode", "task_sample"]


def _map(fn: Callable[[torch.Tensor], torch.Tensor], value: ObservationValue) -> ObservationValue:
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: fn(torch.as_tensor(v)) for k, v in value.items()}
    return fn(torch.as_tensor(value))


def _to_uint8(image: torch.Tensor) -> torch.Tensor:
    # Frames are decoded from 8-bit sources, so this round-trip is lossless.
    return (image * 255).round().to(torch.uint8)


def _normalize_task(task: str) -> str:
    # Tolerate formatting differences between gym and dataset task text.
    return " ".join(task.lower().split()).rstrip(".")


def _stack(values: list[ObservationValue]) -> ObservationValue:
    first = values[0]
    if first is None:
        return None
    if isinstance(first, dict):
        return {k: torch.stack([torch.as_tensor(v[k]) for v in values]) for k in first}  # type: ignore[index]
    return torch.stack([torch.as_tensor(v) for v in values])


class LeRobotGoalProvider:
    """Goals taken from the last frame of LeRobot episodes.

    - ``source="task"``: one goal per task, the last frame of the task's first episode.
      Every episode of a task shares it. This matches the original Patch Policy.
    - ``source="episode"``: each episode's own last frame, so goal and actions always agree.
    - ``source="task_sample"``: the last frame of a random episode of the same task, drawn per
      sample. Training then sees the rollout situation (some demo's end state, not this one's).

    The last frame must show the task completed. Episodes that continue afterwards (e.g.
    cyclic recordings that return home, see the recording guide) should be cropped to end at
    task completion, or the goal is the home pose.

    All goal frames are decoded once at construction. Images are kept as uint8 and stacked
    into one tensor per camera, so DataLoader workers share them through torch shared memory
    instead of each holding a copy.

    Examples:
        >>> full = LeRobotDataset("lerobot/pusht")
        >>> goals = LeRobotGoalProvider(full, source="task_sample")
        >>> goals.get(episode_index=3, task_index=0).images.shape
        torch.Size([3, 96, 96])
        >>> rollout_goal = goals.get_task_goal(task_index=0)
    """

    def __init__(self, dataset: LeRobotDataset, source: GoalSource = "task") -> None:
        """Decode and cache the goal frames.

        Args:
            dataset: Unfiltered dataset opened without ``delta_timestamps`` or ``image_transforms``.
                Goals are looked up by absolute frame index, and a goal may come from an
                episode outside the train/val split, so every episode must be present.
            source: ``"task"``, ``"episode"`` or ``"task_sample"``, see class docstring.

        Raises:
            ValueError: If ``dataset`` is episode-filtered or ``source`` is unknown.
        """
        if dataset.episodes is not None:
            msg = "LeRobotGoalProvider needs an unfiltered dataset (episodes=None)."
            raise ValueError(msg)
        if source not in get_args(GoalSource):
            msg = f"Unknown goal source '{source}', expected one of {get_args(GoalSource)}."
            raise ValueError(msg)

        self.source = source
        self._task_indices = {
            _normalize_task(str(task)): int(index)
            for task, index in zip(dataset.meta.tasks.index, dataset.meta.tasks["task_index"], strict=True)
        }

        def goal_frame(episode: dict[str, Any]) -> int:
            return int(episode["dataset_to_index"]) - 1

        def task_of(episode: dict[str, Any]) -> int:
            # Read from the frame itself (no video decoding): not every dataset stores
            # tasks in its episode metadata, and samples are looked up by this value.
            return int(dataset.get_raw_item(goal_frame(episode))["task_index"])

        episodes = cast("list[dict[str, Any]]", list(dataset.meta.episodes))
        if source == "task":
            # Only each task's first episode is ever used, so skip decoding the rest.
            first_episodes: dict[int, dict[str, Any]] = {}
            for episode in episodes:
                first_episodes.setdefault(task_of(episode), episode)
            episodes = list(first_episodes.values())

        # Row in the goal cache for each episode, and for each task (in episode order).
        self._episode_rows: dict[int, int] = {}
        self._task_rows: dict[int, list[int]] = {}
        for row, episode in enumerate(episodes):
            self._episode_rows[int(episode["episode_index"])] = row
            self._task_rows.setdefault(task_of(episode), []).append(row)

        observations = [FormatConverter.to_observation(dataset[goal_frame(episode)]) for episode in episodes]
        self._images = _stack([_map(_to_uint8, obs.images) for obs in observations])

    @classmethod
    def from_dataset(cls, dataset: LeRobotDataset, source: GoalSource = "task") -> LeRobotGoalProvider:
        """Build goals for any split of a LeRobot dataset.

        Goals are read from a fresh unfiltered copy: a task's goal episode may lie outside
        the split, and filtered datasets are indexed by relative row.

        Returns:
            Provider covering every episode of the dataset.
        """
        return cls(LeRobotDataset(repo_id=dataset.repo_id, root=dataset.root), source=source)

    def get(self, episode_index: int, task_index: int) -> ObservationValue:
        """Return the goal images for a training sample, according to ``source``.

        Raises:
            KeyError: If there is no goal for the episode / task.
        """
        if self.source == "episode":
            if episode_index not in self._episode_rows:
                msg = f"No goal for episode index {episode_index}."
                raise KeyError(msg)
            return self._goal(self._episode_rows[episode_index])
        if self.source == "task_sample":
            # torch RNG so each DataLoader worker draws its own (seeded) sequence.
            rows = self._rows_for_task(task_index)
            return self._goal(rows[int(torch.randint(len(rows), ()))])
        return self.get_task_goal(task_index)

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
        """Return the fixed goal images for a task (its first episode's last frame), e.g. for rollouts."""
        return self._goal(self._rows_for_task(task_index)[0])

    def _rows_for_task(self, task_index: int) -> list[int]:
        if task_index not in self._task_rows:
            msg = f"No goal for task index {task_index}."
            raise KeyError(msg)
        return self._task_rows[task_index]

    def _goal(self, row: int) -> ObservationValue:
        return _map(lambda t: t[row].float() / 255, self._images)


class LeRobotGoalConditionedDataset(GoalConditionedDataset):
    """LeRobot dataset whose samples carry goal images (``Observation.goal_images``).

    Can be passed to `LeRobotDataModule` as ``dataset``, or created for you with its
    ``goal_source`` argument.

    Examples:
        >>> dataset = LeRobotGoalConditionedDataset(LeRobotDataset("lerobot/libero"), goal_source="task_sample")
        >>> datamodule = LeRobotDataModule(dataset=dataset, train_batch_size=32)
        >>> rollout_goal = dataset.goals.get_task_goal(task_index=0)
    """

    def __init__(
        self,
        dataset: LeRobotDataset | _LeRobotDatasetAdapter,
        goal_source: GoalSource = "task",
        *,
        goals: LeRobotGoalProvider | None = None,
    ) -> None:
        """Wrap a dataset and build (or reuse) its goals.

        Args:
            dataset: Any split of a LeRobot dataset.
            goal_source: How goals are chosen, see `LeRobotGoalProvider`.
            goals: Existing goals to share, e.g. between a train and val split. When given,
                ``goal_source`` is ignored and nothing is decoded.
        """
        if not isinstance(dataset, _LeRobotDatasetAdapter):
            dataset = _LeRobotDatasetAdapter.from_lerobot(dataset)
        super().__init__(dataset)
        split = dataset._lerobot_dataset  # noqa: SLF001
        self.goals = goals if goals is not None else LeRobotGoalProvider.from_dataset(split, goal_source)

    def get_goal(self, observation: Observation) -> ObservationValue:
        """Return the goal images for a sample, according to the goal source."""
        return self.goals.get(int(observation.episode_index), int(observation.task_index))  # type: ignore[arg-type]

    def get_task_goal(self, task: str) -> ObservationValue:
        """Return the task's fixed goal images (last frame of its first episode), whatever the goal source."""
        return self.goals.get_task_goal(self.goals.task_index(task))


__all__ = ["GoalSource", "LeRobotGoalConditionedDataset", "LeRobotGoalProvider"]
