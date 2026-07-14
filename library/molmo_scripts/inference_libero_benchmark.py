import random

import numpy as np
import torch

from physicalai.benchmark.gyms import LiberoBenchmark
from physicalai.data.observation import Feature, FeatureType
from physicalai.inference import InferenceModel
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
    repo_id="allenai/MolmoAct2-LIBERO",
    norm_tag="libero",
    n_action_steps=10,
)
policy.config.sample_noise = True
policy = policy.to(device=DEVICE, dtype=torch.bfloat16)
policy.export("molmo-libero-export-torch", backend='torch')
del policy

inf_model = InferenceModel("molmo-libero-export-torch", device="cuda")

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

results = benchmark.evaluate(inf_model)
summary = results.summary()

block_header = f"\n{'=' * 28} libero_10 {'=' * 28}\n"
print(block_header, end="")
print(summary)
