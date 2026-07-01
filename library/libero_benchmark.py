import random

import numpy as np
import torch

from physicalai.benchmark.gyms import LiberoBenchmark
from physicalai.data.observation import FeatureType
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

policy = MolmoAct2(
    repo_id="molmo-LIBERO",
    # repo_id="allenai/MolmoAct2-LIBERO",
    norm_tag="libero",
    n_action_steps=10,
)
# policy.setup(stage="predict")
policy = policy.to(device=DEVICE, dtype=torch.bfloat16)

# Debug: show what keys the policy expects
print("\n=== Policy Configuration ===")
print(f"Input features: {[(f.name, f.shape) for f in policy.config.input_features]}")
print(f"Output features: {[(f.name, f.shape) for f in policy.config.output_features]}")

state_feature = next((f for f in policy.config.input_features if f.ftype == FeatureType.STATE), None)
action_feature = next((f for f in policy.config.output_features if f.ftype == FeatureType.ACTION), None)
if state_feature is None or state_feature.normalization_data is None:
    raise RuntimeError("norm_tag='libero' did not resolve state normalization stats.")
if action_feature is None or action_feature.normalization_data is None:
    raise RuntimeError("norm_tag='libero' did not resolve action normalization stats.")

expected_state_q01 = np.array(
    [-0.31479429659059555, -0.26691552643710226, 0.5194626050191016, 2.159994551314992,
     -1.801294177865994, -0.8949778881389838, 0.003382730811955442, -0.04008920533069468],
    dtype=np.float64,
)
expected_action_q01 = np.array(
    [-0.6792031928846481, -0.7736573115323259, -0.8728073904104404, -0.10277447185825356,
     -0.15509810617083444, -0.20289961475228455, -1.0],
    dtype=np.float64,
)

if not np.allclose(np.array(state_feature.normalization_data.q01, dtype=np.float64), expected_state_q01, atol=1e-8):
    raise RuntimeError("Resolved state q01 stats do not match LIBERO norm_tag metadata.")
if not np.allclose(np.array(action_feature.normalization_data.q01, dtype=np.float64), expected_action_q01, atol=1e-8):
    raise RuntimeError("Resolved action q01 stats do not match LIBERO norm_tag metadata.")

print("Norm-tag normalization check passed for state/action q01 stats.")

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
