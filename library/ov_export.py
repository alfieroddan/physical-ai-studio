from __future__ import annotations

import statistics
import time
from pathlib import Path
import os
import sys

import numpy as np
import torch

from physicalai.data import Feature, FeatureType, Observation
from physicalai.inference import InferenceModel
from physicalai.policies import MolmoAct2

import openvino as ov
import logging

# ============================================================================
# Enhanced logging for OpenVINO + PyTorch tracing diagnostics
# ============================================================================

# Set environment variables for verbose tracing
os.environ["OPENVINO_LOG_LEVEL"] = "DEBUG"
# os.environ["OPENVINO_LOG_LEVEL"] = "TRACE"  # Uncomment for max verbosity

# PyTorch JIT logging (expects string argument)
try:
    torch._C._jit_set_logging_option(">>jit_log_api")
except Exception as e:
    logging.warning(f"Could not enable JIT logging: {e}")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

# OpenVINO conversion logging
ov.utils.types.get_element_type  # (just confirming import works)

core = ov.Core()
# The MO/OVC frontend uses Python logging under the hood
logging.getLogger("openvino").setLevel(logging.DEBUG)
logging.getLogger("torch").setLevel(logging.DEBUG)


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


def diagnose_trace_determinism(policy: MolmoAct2, batch: Observation) -> None:
	"""Capture two independent traces and diff them to diagnose non-determinism."""
	logging.info("=== Diagnosing trace determinism ===")
	
	policy.eval()
	try:
		with torch.no_grad():
			logging.info("First trace...")
			traced1 = torch.jit.trace(policy, batch, check_trace=False)
			
			logging.info("Second trace...")
			traced2 = torch.jit.trace(policy, batch, check_trace=False)
		
		graph1_str = str(traced1.graph)
		graph2_str = str(traced2.graph)
		
		if graph1_str == graph2_str:
			logging.info("✓ Traces are identical — non-determinism is NOT in model structure")
		else:
			logging.warning("✗ Traces differ — model has data-dependent control flow or state mutations")
			logging.debug(f"\n--- First trace graph (first 500 chars) ---\n{graph1_str[:500]}")
			logging.debug(f"\n--- Second trace graph (first 500 chars) ---\n{graph2_str[:500]}")
	except RuntimeError as e:
		logging.warning(f"Could not run trace determinism diagnostic: {e}")
		logging.info("Proceeding to export anyway — detailed logs will appear during conversion...")



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
	logging.info("Loading MolmoAct2 policy...")
	policy = MolmoAct2(
		repo_id="molmo-LIBERO",
		input_features=INPUT_FEATURES,
		output_features=OUTPUT_FEATURES,
		torch_compile=False,
	)
	policy.eval()

	# Diagnose trace determinism before attempting export
	print("Tracing determinism:")
	batch = make_fake_observation()
	diagnose_trace_determinism(policy, batch)
	print("#"*50)

	# print("=== Lightning Policy Test ===")
	# batch = make_fake_observation()
	# run_latency_test(policy, batch)

	export_dir = "test_molmo_openvino"
	logging.info(f"\n=== Exporting to OpenVINO ({export_dir}) ===")
	policy.to_openvino(export_dir, verbose=True)
	logging.info(f"✓ Exported to {export_dir}")

	logging.info("\n=== OV Inference Test ===")
	inf_model = InferenceModel(export_dir)
	obs_np = make_fake_numpy_observation()
	run_ov_latency_test(inf_model, obs_np)
