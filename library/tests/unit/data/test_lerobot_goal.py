# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for LeRobot goal conditioning: goal provider, dataset wrapper and datamodule integration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from physicalai.data.lerobot import LeRobotDataModule, LeRobotGoalConditionedDataset, LeRobotGoalProvider
from physicalai.data.lerobot.dataset import _LeRobotDatasetAdapter
from physicalai.data import Observation
from physicalai.data.observation import GOAL_IMAGES
from physicalai.gyms import GoalConditionedGym, Gym

# 4 episodes of 3 frames. Episodes 0-1 are task "a" (0), episodes 2-3 are task "b" (1).
EPISODE_LENGTH = 3
EPISODE_TASKS = ["a", "a", "b", "b"]
# Task descriptions as stored in the dataset metadata (formatting is normalised on lookup).
TASK_DESCRIPTIONS = pd.DataFrame({"task_index": [0, 1]}, index=["Task A.", "task  b"])


def _stats(dim: int) -> dict:
    return {
        "mean": np.zeros(dim),
        "std": np.ones(dim),
        "min": np.zeros(dim),
        "max": np.ones(dim),
        "q01": np.zeros(dim),
        "q99": np.ones(dim),
    }


class FakeLeRobotDataset:
    """Mimics ``LeRobotDataset`` indexing: filtered datasets are indexed by relative row.

    Every frame's image and state are filled with its absolute index, so a goal image
    reveals which frame it was taken from.
    """

    def __init__(self, repo_id=None, root=None, episodes=None, **kwargs):
        self.repo_id = repo_id
        self.root = root
        self.episodes = episodes
        self.delta_indices = None
        selected = episodes if episodes is not None else range(len(EPISODE_TASKS))
        self._frames = [ep * EPISODE_LENGTH + i for ep in selected for i in range(EPISODE_LENGTH)]
        self.meta = SimpleNamespace(
            tasks=TASK_DESCRIPTIONS,
            episodes=[
                {"episode_index": ep, "dataset_to_index": (ep + 1) * EPISODE_LENGTH} for ep in range(len(EPISODE_TASKS))
            ],
            features={
                "observation.images.top": {"shape": (3, 2, 2), "dtype": "video", "names": ["c", "h", "w"]},
                "observation.state": {"shape": (2,), "dtype": "float32", "names": None},
                "action": {"shape": (2,), "dtype": "float32", "names": None},
            },
            stats={
                "observation.images.top": _stats(3),
                "observation.state": _stats(2),
                "action": _stats(2),
            },
        )
        self.features = self.meta.features
        self.fps = 10
        self.tolerance_s = 1e-4

    def __len__(self) -> int:
        return len(self._frames)

    def get_raw_item(self, idx: int) -> dict:
        return {k: v for k, v in self[idx].items() if not k.startswith("observation.images")}

    def __getitem__(self, idx: int) -> dict:
        frame = self._frames[idx]
        episode = frame // EPISODE_LENGTH
        return {
            "observation.images.top": torch.full((3, 2, 2), frame / 255),
            "observation.state": torch.full((2,), float(frame)),
            "action": torch.zeros(2),
            "episode_index": torch.tensor(episode),
            "frame_index": torch.tensor(frame % EPISODE_LENGTH),
            "index": torch.tensor(frame),
            "task_index": torch.tensor(0 if EPISODE_TASKS[episode] == "a" else 1),
            "timestamp": torch.tensor(0.0),
        }


def _frame(goal_images: dict[str, torch.Tensor]) -> torch.Tensor:
    """Absolute frame index a (possibly batched) goal image was taken from."""
    return (goal_images["top"][..., 0, 0, 0] * 255).round().long()


def _last_frame(episode: int) -> int:
    return (episode + 1) * EPISODE_LENGTH - 1


class TestLeRobotGoalProvider:
    def test_episode_source_uses_each_episodes_last_frame(self):
        goals = LeRobotGoalProvider(FakeLeRobotDataset(), source="episode")

        for episode in range(len(EPISODE_TASKS)):
            goal = goals.get(episode_index=episode, task_index=0)
            assert torch.equal(goal["top"], torch.full((3, 2, 2), _last_frame(episode) / 255))

    def test_task_source_uses_first_episode_of_task(self):
        goals = LeRobotGoalProvider(FakeLeRobotDataset(), source="task")

        assert _frame(goals.get(episode_index=1, task_index=0)) == _last_frame(0)
        assert _frame(goals.get(episode_index=3, task_index=1)) == _last_frame(2)

    def test_task_source_only_decodes_first_episodes(self):
        goals = LeRobotGoalProvider(FakeLeRobotDataset(), source="task")

        assert goals._images["top"].shape[0] == 2

    def test_task_sample_draws_from_every_episode_of_the_task(self):
        goals = LeRobotGoalProvider(FakeLeRobotDataset(), source="task_sample")
        torch.manual_seed(0)

        drawn = {int(_frame(goals.get(episode_index=3, task_index=1))) for _ in range(50)}

        assert drawn == {_last_frame(2), _last_frame(3)}

    @pytest.mark.parametrize("source", ["task", "episode", "task_sample"])
    def test_task_goal_is_first_episode_in_every_mode(self, source):
        goals = LeRobotGoalProvider(FakeLeRobotDataset(), source=source)

        assert _frame(goals.get_task_goal(task_index=1)) == _last_frame(2)

    def test_images_roundtrip_through_uint8(self):
        goals = LeRobotGoalProvider(FakeLeRobotDataset(), source="episode")

        assert goals._images["top"].dtype == torch.uint8
        assert goals.get(episode_index=0, task_index=0)["top"].dtype == torch.float32

    def test_filtered_dataset_raises(self):
        with pytest.raises(ValueError, match="unfiltered"):
            LeRobotGoalProvider(FakeLeRobotDataset(episodes=[0, 1]))

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Unknown goal source"):
            LeRobotGoalProvider(FakeLeRobotDataset(), source="future")  # type: ignore[arg-type]

    def test_unknown_task_raises(self):
        goals = LeRobotGoalProvider(FakeLeRobotDataset(), source="task")

        with pytest.raises(KeyError, match="No goal for task index 7"):
            goals.get(episode_index=0, task_index=7)

    def test_unknown_episode_raises(self):
        goals = LeRobotGoalProvider(FakeLeRobotDataset(), source="episode")

        with pytest.raises(KeyError, match="No goal for episode index 9"):
            goals.get(episode_index=9, task_index=0)


@pytest.fixture
def fake_lerobot(monkeypatch):
    """Route every ``LeRobotDataset`` the code opens to the fake."""
    monkeypatch.setattr("physicalai.data.lerobot.dataset.LeRobotDataset", FakeLeRobotDataset)
    monkeypatch.setattr("physicalai.data.lerobot.goal.LeRobotDataset", FakeLeRobotDataset)


class TestTaskLookup:
    def test_matches_description_ignoring_formatting(self):
        goals = LeRobotGoalProvider(FakeLeRobotDataset())

        assert goals.task_index("task a") == 0
        assert goals.task_index("  TASK B. ") == 1

    def test_unknown_description_lists_dataset_tasks(self):
        goals = LeRobotGoalProvider(FakeLeRobotDataset())

        with pytest.raises(KeyError, match="task b"):
            goals.task_index("task c")


@pytest.mark.usefixtures("fake_lerobot")
class TestLeRobotGoalConditionedDataset:
    def test_goal_comes_from_outside_the_split(self):
        # Episode 3 shares task "b" with episode 2, which is not in this split.
        dataset = LeRobotGoalConditionedDataset(FakeLeRobotDataset(episodes=[3]), goal_source="task")

        obs = dataset[0]

        assert obs.episode_index == 3
        assert _frame(obs.goal_images) == _last_frame(2)
        assert torch.equal(obs.goal_images["top"], torch.full((3, 2, 2), _last_frame(2) / 255))

    def test_wraps_adapter(self):
        adapter = _LeRobotDatasetAdapter.from_lerobot(FakeLeRobotDataset(episodes=[1]))

        dataset = LeRobotGoalConditionedDataset(adapter, goal_source="episode")

        assert dataset.dataset is adapter
        assert _frame(dataset[0].goal_images) == _last_frame(1)

    def test_behaves_like_the_wrapped_dataset(self):
        dataset = LeRobotGoalConditionedDataset(FakeLeRobotDataset(episodes=[0, 1]))
        dataset.delta_indices = {"action": [0, 1]}

        assert len(dataset) == 2 * EPISODE_LENGTH
        assert set(dataset.observation_features) == {"top", "state"}
        assert dataset.fps == 10
        assert dataset.dataset._lerobot_dataset.delta_indices == {"action": [0, 1]}

    def test_shares_existing_goals(self):
        train = LeRobotGoalConditionedDataset(FakeLeRobotDataset(episodes=[0, 2]), goal_source="task")

        val = LeRobotGoalConditionedDataset(FakeLeRobotDataset(episodes=[1]), goals=train.goals)

        assert val.goals is train.goals
        assert _frame(val[0].goal_images) == _last_frame(0)


@pytest.mark.usefixtures("fake_lerobot")
class TestLeRobotDataModuleGoals:
    def test_no_goal_source_leaves_dataset_unwrapped(self):
        dm = LeRobotDataModule(repo_id="fake", train_batch_size=2)

        assert not isinstance(dm.train_dataset, LeRobotGoalConditionedDataset)
        assert dm.train_dataset[0].goal_images is None

    def test_goal_source_wraps_dataset(self):
        dm = LeRobotDataModule(repo_id="fake", episodes=[3], train_batch_size=2, goal_source="task")

        assert isinstance(dm.train_dataset, LeRobotGoalConditionedDataset)
        assert _frame(dm.train_dataset[0].goal_images) == _last_frame(2)

    def test_accepts_wrapped_dataset(self):
        dataset = LeRobotGoalConditionedDataset(FakeLeRobotDataset(), goal_source="episode")

        dm = LeRobotDataModule(dataset=dataset, train_batch_size=2)

        assert dm.train_dataset is dataset

    def test_val_split_shares_goals(self, monkeypatch):
        monkeypatch.setattr("physicalai.data.lerobot.datamodule._read_total_episodes", lambda *_: 4)
        dm = LeRobotDataModule(
            repo_id="fake",
            train_batch_size=2,
            val_split=0.25,
            val_split_seed=0,
            goal_source="task",
        )

        assert isinstance(dm.val_eval_dataset, LeRobotGoalConditionedDataset)
        assert dm.val_eval_dataset.goals is dm.train_dataset.goals

    def test_goal_source_with_wrapped_dataset_raises(self):
        dataset = LeRobotGoalConditionedDataset(FakeLeRobotDataset())

        with pytest.raises(ValueError, match="already carries goals"):
            LeRobotDataModule(dataset=dataset, goal_source="task")

    def test_lerobot_format_raises(self):
        with pytest.raises(ValueError, match="physicalai"):
            LeRobotDataModule(repo_id="fake", data_format="lerobot", goal_source="task")


@pytest.mark.usefixtures("fake_lerobot")
class TestGoalBatching:
    @pytest.fixture
    def batch(self):
        dm = LeRobotDataModule(repo_id="fake", train_batch_size=4, num_workers=0, goal_source="episode")
        dm.setup("fit")
        return next(iter(dm.train_dataloader()))

    def test_collate_stacks_goals(self, batch):
        assert batch.goal_images["top"].shape == (4, 3, 2, 2)

    def test_goals_match_their_samples(self, batch):
        expected = [_last_frame(int(episode)) for episode in batch.episode_index]

        assert _frame(batch.goal_images).tolist() == expected

    def test_to_device(self, batch):
        batch = batch.to("cpu")

        assert batch.goal_images["top"].device.type == "cpu"

    def test_index_slices_goals(self, batch):
        sample = batch[1:2]

        assert sample.goal_images["top"].shape == (1, 3, 2, 2)
        assert torch.equal(sample.goal_images["top"], batch.goal_images["top"][1:2])

    def test_to_dict_uses_goal_field_names(self, batch):
        flat = batch.to_dict()

        assert f"{GOAL_IMAGES}.top" in flat


class FakeTaskGym(Gym):
    """Minimal gym that reports a task description, like PushT / LIBERO."""

    def __init__(self, task_description: str) -> None:
        self.task_description = task_description

    def _observation(self) -> Observation:
        return Observation(
            images={"top": torch.zeros(1, 3, 2, 2)},
            state=torch.zeros(1, 2),
            task=[self.task_description],
        )

    def reset(self, *, seed=None, episode_index=0, **reset_kwargs):
        return self._observation(), {}

    def step(self, action):
        return self._observation(), 0.0, False, False, {}

    def close(self) -> None:
        pass

    def sample_action(self) -> torch.Tensor:
        return torch.zeros(2)

    def to_observation(self, raw_obs) -> Observation:
        return self._observation()


@pytest.mark.usefixtures("fake_lerobot")
class TestGoalConditionedGymWithLeRobot:
    @pytest.mark.parametrize(("description", "task_episode"), [("task a", 0), ("Task B.", 2)])
    def test_goal_matches_gym_task(self, description, task_episode):
        dataset = LeRobotGoalConditionedDataset(FakeLeRobotDataset(), goal_source="task_sample")
        gym = GoalConditionedGym(FakeTaskGym(description), dataset)

        obs, _ = gym.reset(seed=0)

        # Rollouts always use the task's fixed goal, whatever the training goal source.
        assert _frame(obs.goal_images).tolist() == [_last_frame(task_episode)]
        assert obs.goal_images["top"].shape == (1, 3, 2, 2)

    def test_unknown_task_fails_at_construction(self):
        with pytest.raises(KeyError, match="not in the dataset"):
            GoalConditionedGym(FakeTaskGym("task c"), LeRobotGoalConditionedDataset(FakeLeRobotDataset()))

    def test_missing_task_description_raises(self):
        gym = GoalConditionedGym(FakeTaskGym("task a"), LeRobotGoalConditionedDataset(FakeLeRobotDataset()))
        gym.gym.task_description = None

        with pytest.raises(ValueError, match="Observation.task"):
            gym.reset()
