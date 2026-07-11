"""Smoke test: inference still works and torch.export now succeeds."""

from __future__ import annotations

import torch

from physicalai.data import Feature, FeatureType
from physicalai.policies.molmoact2.policy import MolmoAct2

IMAGE_SIZE = 378
INPUT_FEATURES = [
    Feature(name="overview", ftype=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE)),
    Feature(name="state", ftype=FeatureType.STATE, shape=(8,)),
]
OUTPUT_FEATURES = [Feature(name="action", ftype=FeatureType.ACTION, shape=(7,))]

policy = MolmoAct2(
    repo_id="molmo-LIBERO",
    input_features=INPUT_FEATURES,
    output_features=OUTPUT_FEATURES,
    torch_compile=False,
)
policy.eval()

sample = policy._get_default_export_input_sample()
print("export input sample keys/shapes:")
for k, v in sample.items():
    print(f"  {k}: {tuple(v.shape)} {v.dtype}")

with torch.no_grad():
    out = policy.model.predict_action_chunk(sample)
print("predict_action_chunk OK ->", tuple(out.shape), out.dtype)

print("attempting torch.export.export(policy.model, (sample,)) ...")
ep = torch.export.export(policy.model, (sample,), strict=False)
print("torch.export OK; graph nodes:", len(list(ep.graph.nodes)))
