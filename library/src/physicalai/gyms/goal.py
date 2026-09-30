# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Goal conditioning for gyms.

`GoalConditionedGym` attaches the goal images for the gym's current task to every
`Observation` returned by ``reset`` / ``step``, so a goal-conditioned policy sees the same
``goal_images`` field during rollouts as it does during training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .base import Gym

if TYPE_CHECKING:
    from physicalai.data.goal import GoalConditionedDataset
    from physicalai.data.observation import Observation, ObservationValue

    from .types import SingleOrBatch


def _device(obs: Observation) -> torch.device:
    for value in (obs.images, obs.state):
        if isinstance(value, dict):
            value = next(iter(value.values()), None)  # noqa: PLW2901
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def _batch(value: ObservationValue, batch_size: int, device: torch.device) -> ObservationValue:
    def batch(tensor: Any) -> torch.Tensor:  # noqa: ANN401
        tensor = torch.as_tensor(tensor, device=device)
        return tensor.unsqueeze(0).expand(batch_size, *tensor.shape)

    if isinstance(value, dict):
        return {k: batch(v) for k, v in value.items()}
    return None if value is None else batch(value)


def _match_cameras(goal: ObservationValue, images: ObservationValue, camera_map: dict[str, str]) -> ObservationValue:
    """Put goal images in the same layout as the gym's images (dict of cameras or single tensor).

    Returns:
        The goal images, keyed like ``images``.

    Raises:
        KeyError: If the goal cameras cannot be matched to the gym cameras.
    """
    if goal is None:
        return None
    if isinstance(goal, dict):
        goal = {camera_map.get(k, k): v for k, v in goal.items()}
    if not isinstance(images, dict):
        # The gym gives a single image, so the goal must have exactly one camera too.
        if isinstance(goal, dict) and len(goal) == 1:
            return next(iter(goal.values()))
        return goal
    if not isinstance(goal, dict):
        goal = {"": goal}
    if goal.keys() == images.keys():
        return goal
    if len(goal) == 1 and len(images) == 1:
        # One camera on each side: use the gym's name.
        return {next(iter(images)): next(iter(goal.values()))}
    msg = f"Cannot match goal cameras {sorted(goal)} to gym cameras {sorted(images)}; pass camera_map."
    raise KeyError(msg)


class GoalConditionedGym(Gym):
    """Gym whose observations carry the goal for its current task.

    On every ``reset`` the gym's task description (``Observation.task``) is looked up in the
    goal dataset, and that task's fixed goal is attached to every observation of the episode,
    batched and laid out like the gym's own images. Works for single-task gyms (PushT) and
    multi-task ones (LIBERO) alike. Everything else is forwarded to the wrapped gym.

    Examples:
        >>> dataset = LeRobotGoalConditionedDataset(LeRobotDataset("lerobot/libero"))
        >>> gym = GoalConditionedGym(LiberoGym(task_suite="libero_goal", task_id=3), dataset)
        >>> obs, _ = gym.reset(seed=0)
        >>> obs.goal_images["image"].shape
        torch.Size([1, 3, 256, 256])
    """

    def __init__(self, gym: Gym, dataset: GoalConditionedDataset, camera_map: dict[str, str] | None = None) -> None:
        """Initialize the wrapper.

        Args:
            gym: Gym to wrap. Its observations must carry the task description in ``task``.
            dataset: Goal dataset the policy is trained on; provides each task's goal.
            camera_map: Renames goal cameras to gym cameras, e.g. ``{"image": "top"}``. Only
                needed when names differ and there is more than one camera.
        """
        self.gym = gym
        self.dataset = dataset
        self.camera_map = camera_map or {}
        self._goal: ObservationValue = None

        # Fail early if the gym already knows a task the dataset does not have.
        description = getattr(gym, "task_description", None)
        if isinstance(description, str) and description:
            dataset.get_task_goal(description)

    def get_goal(self, observation: Observation) -> ObservationValue:
        """Return the goal images for the task in ``observation.task``.

        Raises:
            ValueError: If the observation carries no task description.
        """
        task = observation.task
        if isinstance(task, (list, tuple)) and task:
            task = task[0]
        if not isinstance(task, str):
            msg = "GoalConditionedGym needs the gym to put its task description in Observation.task."
            raise ValueError(msg)  # noqa: TRY004
        return self.dataset.get_task_goal(task)

    def reset(
        self,
        *,
        seed: int | None = None,
        episode_index: int = 0,
        **reset_kwargs: Any,  # noqa: ANN401
    ) -> tuple[Observation, dict[str, Any] | list[dict[str, Any]]]:
        """Reset the gym and pick the goal for the new episode.

        Returns:
            A tuple ``(observation, info)`` with the goal attached to the observation.
        """
        obs, info = self.gym.reset(seed=seed, episode_index=episode_index, **reset_kwargs)
        self._goal = self.get_goal(obs)
        return self._attach(obs), info

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        Observation,
        SingleOrBatch[float],
        SingleOrBatch[bool],
        SingleOrBatch[bool],
        SingleOrBatch[dict[str, Any]],
    ]:
        """Step the gym.

        Returns:
            A tuple ``(observation, reward, terminated, truncated, info)`` with the goal attached.
        """
        obs, reward, terminated, truncated, info = self.gym.step(action)
        return self._attach(obs), reward, terminated, truncated, info

    def _attach(self, obs: Observation) -> Observation:
        if self._goal is None:
            return obs
        batch_size, device = obs.batch_size, _device(obs)
        obs.goal_images = _batch(_match_cameras(self._goal, obs.images, self.camera_map), batch_size, device)
        return obs

    def render(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Render the wrapped gym.

        Returns:
            The render output of the wrapped gym.
        """
        return self.gym.render(*args, **kwargs)

    def get_max_episode_steps(self) -> int | None:
        """Return the wrapped gym's step limit."""
        return self.gym.get_max_episode_steps()

    def close(self) -> None:
        """Close the wrapped gym."""
        self.gym.close()

    def sample_action(self) -> torch.Tensor:
        """Sample an action from the wrapped gym.

        Returns:
            A valid action.
        """
        return self.gym.sample_action()

    def to_observation(self, raw_obs: Any) -> Observation:  # noqa: ANN401
        """Convert a raw observation with the wrapped gym and attach the current goal.

        Returns:
            The observation with the goal attached.
        """
        return self._attach(self.gym.to_observation(raw_obs))

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Forward attribute access to the wrapped gym.

        Returns:
            The attribute of the wrapped gym.

        Raises:
            AttributeError: If the wrapped gym is not set yet.
        """
        # Avoid infinite recursion before __init__ has set `gym` (e.g. while unpickling).
        if name == "gym":
            raise AttributeError(name)
        return getattr(self.gym, name)


__all__ = ["GoalConditionedGym"]
