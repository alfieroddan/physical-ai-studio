from __future__ import annotations

import statistics
import time
from pathlib import Path

import numpy as np
import torch

from physicalai.data import Feature, FeatureType, Observation
from physicalai.inference import InferenceModel
from physicalai.policies import MolmoAct2


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 378

INPUT_FEATURES = [
	Feature(name="overview", ftype=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE)),
	Feature(name="state", ftype=FeatureType.STATE, shape=(8,)),
]

OUTPUT_FEATURES = [
	Feature(name="action", ftype=FeatureType.ACTION, shape=(7,)),
]


def make_fake_observation() -> Observation:
	return Observation(
		images={"overview": torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)},
		state=torch.rand(1, 8),
	).to(DEVICE)


def make_fake_numpy_observation() -> dict[str, object]:
	"""Flat dict for InferenceModel (BCHW uint8 — molmoact2_pre converts to float32)."""
	return {
		"images.overview": np.random.randint(0, 256, (1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8),
		"state": np.zeros((1, 8), dtype=np.float32),
		"task": ["pick up the object"],
	}


def run_ov_latency_test(
	inf_model: InferenceModel,
	obs_np: dict[str, object],
	warmup: int = 1,
	runs: int = 3,
) -> None:
	"""Latency test for the exported OpenVINO model via InferenceModel."""
	for _ in range(warmup):
		_ = inf_model.predict_action_chunk(obs_np)

	durations_ms: list[float] = []
	for _ in range(runs):
		start = time.perf_counter()
		_ = inf_model.predict_action_chunk(obs_np)
		durations_ms.append((time.perf_counter() - start) * 1000.0)

	print("OV Latency (ms):", [round(x, 2) for x in durations_ms])
	print(f"OV Latency mean={statistics.mean(durations_ms):.2f} ms, p50={statistics.median(durations_ms):.2f} ms")


def run_latency_test(policy: MolmoAct2, batch: Observation, warmup: int = 1, runs: int = 3) -> None:
	"""Latency test for the Lightning policy (torch tensors)."""
	for _ in range(warmup):
		with torch.no_grad():
			_ = policy.predict_action_chunk(batch)

	durations_ms: list[float] = []
	for _ in range(runs):
		if DEVICE == "cuda":
			torch.cuda.synchronize()
		start = time.perf_counter()
		with torch.no_grad():
			_ = policy.predict_action_chunk(batch)
		if DEVICE == "cuda":
			torch.cuda.synchronize()
		durations_ms.append((time.perf_counter() - start) * 1000.0)

	print("Latency (ms):", [round(x, 2) for x in durations_ms])
	print(f"Latency mean={statistics.mean(durations_ms):.2f} ms, p50={statistics.median(durations_ms):.2f} ms")


if __name__ == "__main__":
	policy = MolmoAct2(
		repo_id="molmo-LIBERO",
		input_features=INPUT_FEATURES,
		output_features=OUTPUT_FEATURES,
		torch_compile=False,
	)
	policy.eval()

	print("=== Lightning Policy Test ===")
	batch = make_fake_observation()
	run_latency_test(policy, batch)

	export_dir = "test_molmo_openvino"
	print("\n=== Exporting to OpenVINO ===")
	policy.to_openvino(export_dir)
	print(f"Exported to {export_dir}")

	print("\n=== OV Inference Test ===")
	inf_model = InferenceModel(export_dir)
	obs_np = make_fake_numpy_observation()
	run_ov_latency_test(inf_model, obs_np)