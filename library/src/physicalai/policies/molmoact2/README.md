# MolmoAct2

## Description

MolmoAct2 is a vision-language-action (VLA) policy from the Allen Institute for AI, integrated here as a first-party policy. Given camera images, robot state, and a task string, it predicts a chunk of future actions.

Useful links:

- [Blog post](https://allenai.org/blog/molmoact2)
- [LeRobot docs](https://huggingface.co/docs/lerobot/en/molmoact2)
- [Paper](https://arxiv.org/abs/2605.02881)

## Quick Start: Forward Pass

```python
import torch

from physicalai.data import Feature, FeatureType, Observation
from physicalai.devices.utils import get_device
from physicalai.policies import MolmoAct2

DEVICE = get_device()

batch = Observation(
    images={
        "overview": torch.rand(1, 3, 256, 256),
        "wrist": torch.rand(1, 3, 256, 256),
    },
    state=torch.rand(1, 6),
    task=["example input"],
).to(DEVICE)

input_features = [
    Feature(name="overview", ftype=FeatureType.VISUAL, shape=(3, 256, 256)),
    Feature(name="state", ftype=FeatureType.STATE, shape=(6,)),
]

output_features = [
    Feature(name="action", ftype=FeatureType.ACTION, shape=(6,)),
]

if __name__ == "__main__":
    policy = MolmoAct2(
        input_features=input_features,
        output_features=output_features,
    ).to(DEVICE, dtype=torch.bfloat16)
    policy.eval()

    with torch.no_grad():
        actions = policy.predict_action_chunk(batch)

    print(f"Actions shape: {actions.shape}")
    print(f"Actions: {actions}")
```

## Benchmark Example (LIBERO)

```python
import csv
import random
from pathlib import Path

import numpy as np
import torch

from physicalai.benchmark.gyms import LiberoBenchmark
from physicalai.benchmark.gyms.results import TaskResult
from physicalai.devices.utils import get_device
from physicalai.policies import MolmoAct2

DEVICE = get_device()
print("Using device:", DEVICE)

SEED = 0
TASK_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
NUM_EPISODES = 1
OBSERVATION_HEIGHT = 378
OBSERVATION_WIDTH = 378
RECORD_MODE = "all"
VIDEO_DIR = "./videos/"
RESULTS_CSV = Path("./results/libero_multi_suite_results.csv")


def set_seed(seed: int) -> None:
    """Seed all global RNGs for reproducible benchmark + model sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_combined_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    """Write per-task benchmark rows to a single CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "suite",
        "task_id",
        "task_name",
        "n_episodes",
        "success_rate",
        "avg_reward",
        "avg_episode_length",
        "avg_fps",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def task_row(suite: str, task: TaskResult) -> dict[str, str | int | float]:
    """Convert TaskResult to a CSV row with suite metadata."""
    return {
        "suite": suite,
        "task_id": task.task_id,
        "task_name": task.task_name,
        "n_episodes": task.n_episodes,
        "success_rate": task.success_rate,
        "avg_reward": task.avg_reward,
        "avg_episode_length": task.avg_episode_length,
        "avg_fps": task.avg_fps,
    }


if __name__ == "__main__":
    set_seed(SEED)

    policy = MolmoAct2(
        repo_id="allenai/MolmoAct2-LIBERO",
        norm_tag="libero",
        n_action_steps=10,
        # Start flow matching from sampled Gaussian noise (matches the reference model).
        use_random_input_noise=True,
    )
    policy = policy.to(device=DEVICE, dtype=torch.bfloat16)

    csv_rows: list[dict[str, str | int | float]] = []

    for task_suite in TASK_SUITES:
        benchmark = LiberoBenchmark(
            task_suite=task_suite,
            num_episodes=NUM_EPISODES,
            seed=SEED,
            camera_name_mapping={"image2": "wrist_image"},
            observation_height=OBSERVATION_HEIGHT,
            observation_width=OBSERVATION_WIDTH,
            record_mode=RECORD_MODE,
            video_dir=VIDEO_DIR,
        )

        try:
            results = benchmark.evaluate(policy)
            summary = results.summary()

            block_header = f"\n{'=' * 28} {task_suite} {'=' * 28}\n"
            print(block_header, end="")
            print(summary)

            csv_rows.extend(task_row(task_suite, task) for task in results.task_results)
        finally:
            for gym in benchmark.gyms:
                gym.close()

    save_combined_csv(RESULTS_CSV, csv_rows)
    print(f"\nSaved CSV results to: {RESULTS_CSV}")
```

Results on NVIDIA A100 (FP16):

| Suite          |  Tasks | Avg. Success Rate (%) | Avg. Reward | Avg. Episode Length |  Avg. FPS |
| -------------- | -----: | --------------------: | ----------: | ------------------: | --------: |
| libero_spatial |     10 |                 100.0 |        1.00 |               106.4 |     10.56 |
| libero_object  |     10 |                 100.0 |        1.00 |               132.7 |     11.82 |
| libero_goal    |     10 |                  90.0 |        0.90 |               134.2 |     11.51 |
| libero_10      |     10 |                 100.0 |        1.00 |               241.4 |     12.61 |
| **Average**    | **40** |              **97.5** |    **0.98** |           **153.7** | **11.63** |

Results on OpenVINO GPU (FP16):

| Suite          |  Tasks | Avg. Success Rate (%) | Avg. Reward | Avg. Episode Length | Avg. FPS |
| -------------- | -----: | --------------------: | ----------: | ------------------: | -------: |
| libero_spatial |     10 |                 100.0 |        1.00 |               122.0 |     14.6 |
| libero_object  |     10 |                 100.0 |        1.00 |               145.3 |     16.2 |
| libero_goal    |     10 |                  80.0 |        0.80 |               159.5 |     16.1 |
| libero_10      |     10 |                  80.0 |        0.80 |               300.2 |     16.7 |
| **Average**    | **40** |              **90.0** |    **0.90** |           **181.8** | **15.9** |

## Zero Shot SO-101

To run zero-shot MolmoAct2 on the SO-101 embodiment. Please export the model using:

```python
from physicalai.policies import MolmoAct2
from physicalai.data import Feature, FeatureType
import torch

input_features = [
    Feature(
        name="overview",
        ftype=FeatureType.VISUAL,
        shape=(3, 480, 640),
    ),
    Feature(
        name="wrist",
        ftype=FeatureType.VISUAL,
        shape=(3, 480, 640),
    ),
    Feature(
        name="state",
        ftype=FeatureType.STATE,
        shape=(6,),
    ),
]


output_features = [
    Feature(
        name="action",
        ftype=FeatureType.ACTION,
        shape=(6,),
    ),
]

if __name__ == "__main__":
    policy = MolmoAct2(
        repo_id="allenai/MolmoAct2-SO100_101",
        norm_tag="so100_so101_molmoact2",
        adapt_to_so101=True,
    ).eval()
    policy.config.sample_noise = True
    policy = policy.to(dtype=torch.bfloat16)
    policy.export("model_dir/molmoact2-so101-torch", backend="torch")
```

You can then use the [runtime framework](https://github.com/openvinotoolkit/physicalai) to run the policy.

> [!NOTE]
> When using the robot on a S0-101, start the model when the SO-101 is in an extended position i.e about to do the task. If not the model will get stuck in a loop in rest position.

## Tips and Tricks

## Repo ID and Norm Tag

The model init has two args, `repo_id` and `norm_tag`, which describe the pretrained HuggingFace snapshot and the key for the corresponding normalization statistics, depending on your embodiment. I will explain how best to use them here; we will use the base weights as an example.

MolmoAct2 has multiple [collections](https://huggingface.co/allenai/collections) on HuggingFace.

The `repo_id` is the Hugging Face repository, for example `allenai/MolmoAct2-SO100_101`, which can be found in the [MolmoAct2-SO100_101 repository](https://huggingface.co/allenai/MolmoAct2-SO100_101).

Inside the repo, there is a Files tab. For the snapshots we support, there is a file called `norm_stats.json`, e.g. [this one](https://huggingface.co/allenai/MolmoAct2-SO100_101/blob/main/norm_stats.json).

You'll see the `metadata_by_tag` key; this is `norm_tag`. We are looking to find the `norm_tag` that describes the embodiment we are using with the corresponding pretrained weights.

```json
{
  "format": "molmoact2_norm_stats.v1",
  "norm_mode": "q01_q99",
  "metadata_by_tag": {
    "so100_so101_molmoact2": {
      "action_key": "action",
      "state_key": "observation.state",
      "camera_keys": [],
      "normalize_gripper": true,
...
```

In this example, our `norm_tag` is `so100_so101_molmoact2`.

So, to conclude, to use the finetuned `MolmoAct2` on SO101, we would use:

```python
policy = MolmoAct2(
        repo_id="allenai/MolmoAct2-LIBERO",
        norm_tag="libero",
    )
```

## Action Modes & Training Status

MolmoAct2 supports `continuous`, `discrete`, and `both` action modes in its config, but **only `continuous` action mode is currently supported** — passing anything else will raise a `ValueError`. Discrete output/loss and the `both` mode are not yet implemented.

Training support is **TBD**. A minimal training loop is shown below for reference, but end-to-end training (including on SO-101) has not yet been confirmed to work and should be considered experimental until validated.

```python
from physicalai.data import LeRobotDataModule
from physicalai.policies import MolmoAct2
from physicalai.train import Trainer

if __name__ == "__main__":
    # dataset sets input / output features
    policy = MolmoAct2(
        use_random_input_noise=True,
        train_action_expert_only=True,
    )

    # datamodule
    dm = LeRobotDataModule(
        repo_id="lerobot/pusht",
        train_batch_size=1,
        val_batch_size=1,
        val_split=0.1,
        num_workers=5,
        episodes=[0, 1],
    )

    # trainer
    trainer = Trainer(max_steps=2, limit_val_batches=1, val_check_interval=1, precision="bf16-mixed")
    trainer.fit(model=policy, datamodule=dm)
```

## Video processing

We also have removed video pre-processing in this implementation. Please raise an issue if this is something you would like added back.
