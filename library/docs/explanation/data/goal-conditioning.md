# Goal Conditioning

A goal-conditioned policy receives an image of the state it should reach with every
observation. This is required for benchmarks such as LIBERO-goal, where tasks can
start from the same scene.

## Canonical task goals

Each task has one canonical goal: the last frame of that task's first selected
training episode. The same goal is used for every training and validation sample of
the task and for gym rollouts.

Goal images are attached as `Observation.goal_images` in the same layout as
`Observation.images`. The field is `None` when goal conditioning is disabled.

```python test="skip" reason="requires physicalai install and downloads data"
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from physicalai.data.lerobot import LeRobotDataModule, LeRobotGoalConditionedDataset

# Let the DataModule add task goals.
datamodule = LeRobotDataModule(
    repo_id="lerobot/libero",
    goal_source="task",
)

# Or wrap an existing dataset directly.
dataset = LeRobotGoalConditionedDataset(LeRobotDataset("lerobot/libero"))
sample = dataset[0]
```

The DataModule builds goals from the training split and shares them with eval-loss
validation. This prevents held-out episodes from becoming validation-only goals.

Observation image augmentation does not apply to goals. Goal construction temporarily
disables the LeRobot dataset's image transforms, decodes one goal per task, and then
restores the transforms before training. Goal lookups return independent tensors so a
caller cannot mutate the canonical image.

Goal conditioning requires `data_format="physicalai"` and
`delta_timestamps=None`.

## Rollouts

`GoalConditionedGym` takes the conditioned dataset and attaches its canonical goal to
every rollout observation:

```python test="skip" reason="requires LIBERO and downloads data"
from physicalai.gyms import GoalConditionedGym, LiberoGym

gym = GoalConditionedGym(
    LiberoGym(task_suite="libero_goal", task_id=3),
    dataset,
)

obs, _ = gym.reset(seed=0)
obs.goal_images["image"].shape  # torch.Size([1, 3, 256, 256])
```

The gym's `Observation.task` text is matched to a dataset task. A vectorized rollout
may repeat one task across the batch, but mixed-task batches are not supported. Goal
images are broadcast to the observation batch and matched to its camera layout. Use
`camera_map` when multi-camera names differ between the dataset and gym.

## Caveats

- The first selected episode for each task must end in the desired target state.
- Gym task text must match a dataset task (capitalization, whitespace, and a trailing
  period are ignored).
- Policies should normalize `goal_images` with the same statistics as `images`.
- Inspect a sample of goal images before training.

## See Also

- [LeRobot Data Integration](lerobot.md)
- [Observation](observation.md)
- [GoalConditionedGym](../gyms/goal.md)
