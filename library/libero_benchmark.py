import random

import numpy as np
import torch

from physicalai.benchmark.gyms import LiberoBenchmark
from physicalai.data.observation import Feature, FeatureType
from physicalai.policies import MolmoAct2

SEED = 0


def set_seed(seed: int) -> None:
    """Seed all global RNGs for reproducible benchmark + model sampling.

    MolmoAct2 draws flow-matching noise from the global torch RNG
    (``torch.randn`` with ``generator=None``), so seeding here makes the
    policy's action sampling deterministic across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

input_features = [
    Feature(
        name="image",
        ftype=FeatureType.VISUAL,
        shape=(3, 378, 378),
        normalization_data=None,
    ),
    Feature(
        name="image2",
        ftype=FeatureType.VISUAL,
        shape=(3, 378, 378),
        normalization_data=None,
    ),
]
input_features.append(
    Feature(name="state", ftype=FeatureType.STATE, shape=(8,)),
)
output_features = [
    Feature(name="action", ftype=FeatureType.ACTION, shape=(7,)),
]

policy = MolmoAct2(
    input_features=input_features,
    output_features=output_features,
    repo_id="molmo-LIBERO",
    # repo_id="allenai/MolmoAct2-LIBERO",
    norm_tag="libero",
    n_action_steps=10,
)
policy.setup(stage="predict")
policy = policy.to(device=DEVICE, dtype=torch.bfloat16)

# Debug: show what keys the policy expects
print("\n=== Policy Configuration ===")
print(f"Input features: {[(f.name, f.shape) for f in policy.config.input_features]}")
print(f"Output features: {[(f.name, f.shape) for f in policy.config.output_features]}")

benchmark = LiberoBenchmark(
    task_suite="libero_10",
    num_episodes=1,
    seed=SEED,
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
