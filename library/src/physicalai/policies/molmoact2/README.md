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
from physicalai.policies import MolmoAct2

DEVICE = "cuda"

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
import random

import numpy as np
import torch

from physicalai.benchmark.gyms import LiberoBenchmark
from physicalai.policies import MolmoAct2

DEVICE = "cuda"
SEED = 0


def set_seed(seed: int) -> None:
    """Seed all global RNGs for reproducible benchmark + model sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    set_seed(SEED)

    policy = MolmoAct2(
        repo_id="allenai/MolmoAct2-LIBERO",
        norm_tag="libero",
        n_action_steps=10,
    )
    # Start flow matching from sampled Gaussian noise (matches the reference model).
    policy.config.use_random_input_noise = True
    policy = policy.to(device=DEVICE, dtype=torch.bfloat16)

    benchmark = LiberoBenchmark(
        task_suite="libero_10",
        num_episodes=1,
        seed=SEED,
        camera_name_mapping={"image2": "wrist_image"},
        observation_height=378,
        observation_width=378,
        record_mode="all",
        video_dir="./videos/",
    )

    results = benchmark.evaluate(policy)
    summary = results.summary()

    block_header = f"\n{'=' * 28} libero_10 {'=' * 28}\n"
    print(block_header, end="")
    print(summary)
```

## Tips and Tricks

- When using the robot on a S0-101, start the model when the SO-101 is in an extended position i.e about to do the task. If not the model will get stuck in a loop in rest position.

## Repo ID and Norm Tag

The model init has two args, `repo_id` and `norm_tag`, which describe the pretrained HuggingFace snapshot and the key for the corresponding normalization statistics, depending on your embodiment. I will explain how best to use them here; we will use the base weights as an example.

MolmoAct2 has multiple [collections](https://huggingface.co/allenai/collections) on HuggingFace.

The `repo_id` is the Hugging Face repository, for example `allenai/MolmoAct2-SO100_101`, which can be found [here](https://huggingface.co/allenai/MolmoAct2-SO100_101).

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
    policy = MolmoAct2()
    policy.config.use_random_input_noise = True
    policy.config.train_action_expert_only = True

    dm = LeRobotDataModule(
        repo_id="lerobot/svla_so101_pickplace",
        train_batch_size=1,
        val_batch_size=1,
        val_split=0.1,
        num_workers=5,
        episodes=[0, 1],
    )

    trainer = Trainer(max_steps=2, limit_val_batches=1, val_check_interval=1, precision="bf16-mixed")
    trainer.fit(model=policy, datamodule=dm)
```

## Video processing

We also have removed video pre-processing in this implementation. Please raise an issue if this is something you would like added back.
