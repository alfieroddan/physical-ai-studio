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
		"task": ["Example prompt string"],
	}


if __name__ == "__main__":
	logging.info("Loading MolmoAct2 policy...")
	policy = MolmoAct2(
		repo_id="molmo-LIBERO",
		input_features=INPUT_FEATURES,
		output_features=OUTPUT_FEATURES,
		torch_compile=False,
	)
	policy.eval()

	export_dir = "test_molmo_openvino"
	logging.info(f"\n=== Exporting to OpenVINO ({export_dir}) ===")
	policy.to_openvino(export_dir, verbose=True)

	logging.info("\n=== Loading exported package with InferenceModel ===")
	inference_model = InferenceModel(export_dir)
	numpy_obs = make_fake_numpy_observation()
	action = inference_model(numpy_obs)
	print(action)
	logging.info(f"InferenceModel action type={type(action)}")
	if isinstance(action, dict):
		for key, value in action.items():
			logging.info(f"  {key}: shape={getattr(value, 'shape', None)} dtype={getattr(value, 'dtype', None)}")
	else:
		logging.info(f"  action shape={getattr(action, 'shape', None)} dtype={getattr(action, 'dtype', None)}")

