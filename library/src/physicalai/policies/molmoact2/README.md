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
        repo_id="MarkRedeman/dice-cleanup-combined",
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

## Examples of how to Init the Model

```python
"""Examples of the different ways to construct / load a MolmoAct2 policy."""

from pathlib import Path

from physicalai.data.observation import Feature, FeatureType
from physicalai.policies.molmoact2.policy import MolmoAct2, make_molmoact2_config

# 1. Standard pretrained load: resolve an HF repo, download + load weights.
policy = MolmoAct2(
    repo_id="allenai/MolmoAct2",
    norm_tag="franka_droid",
    load_weights=True,  # default; shown explicitly for contrast with (2)
)


# 2. Resolve HF metadata (tokenizer/processor/config) but SKIP the pretrained
#    weight download/load. Useful when you're about to load your own
#    state dict immediately after (e.g. a manually managed checkpoint that
#    isn't a Lightning checkpoint, so load_from_checkpoint doesn't apply).
policy_no_weights = MolmoAct2(
    repo_id="allenai/MolmoAct2",
    norm_tag="franka_droid",
    load_weights=False,
)

# 3. From-scratch: no pretrained checkpoint at all, features supplied
#    directly. No HF resolution happens because repo_id=None.
input_features = [
    Feature(name="top", ftype=FeatureType.VISUAL, shape=(3, 224, 224)),
    Feature(name="state", ftype=FeatureType.STATE, shape=(14,)),
]
output_features = [
    Feature(name="action", ftype=FeatureType.STATE, shape=(14,)),
]

policy_from_scratch = MolmoAct2(
    repo_id=None,
    input_features=input_features,
    output_features=output_features,
    n_obs_steps=1,
    n_action_steps=30,
    chunk_size=30,
)


# 4. from_config: reuse an already-built config, skipping HF resolution
#    entirely.
base_config = make_molmoact2_config(
    input_features=input_features,
    output_features=output_features,
    n_obs_steps=1,
    n_action_steps=30,
    chunk_size=30,
)

# 4a. No overrides -- config used exactly as-is.
policy_from_config = MolmoAct2.from_config(base_config)

# 4b. With overrides - only the fields you pass get patched via
#     dataclasses.replace; everything else on the config is untouched.
policy_from_config_overridden = MolmoAct2.from_config(
    base_config,
    n_action_steps=50,
    use_lora=True,
    lora_rank=32,
    load_weights=False,  # pass-through, not a config field
)

# 5. load_from_checkpoint: resume training / reload a finetuned Lightning
#    checkpoint. load_weights defaults to False here automatically, since
#    the checkpoint's own state_dict is about to overwrite whatever gets
#    constructed -- eagerly loading the original pretrained weights first
#    would just be discarded work.
checkpoint_path = Path("outputs/last.ckpt")
policy_resumed = MolmoAct2.load_from_checkpoint(checkpoint_path)

# Force the (normally skipped) pretrained weight load anyway, e.g. if you
# specifically need the original pretrained weights loaded first for some
# custom reason before the checkpoint's own weights get applied on top:
policy_resumed_force_pretrained = MolmoAct2.load_from_checkpoint(
    checkpoint_path,
    load_weights=True,
)
```
