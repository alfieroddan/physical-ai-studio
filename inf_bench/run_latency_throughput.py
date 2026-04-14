"""Latency and throughput benchmark for exported policies across devices.

Methodology matches Cadene et al. (2026):
  - 100 forward passes after warmup
  - Diffusion / flow-matching models use 10 denoising steps per forward pass
"""

import csv
import logging
import time
from pathlib import Path
import torch

import numpy as np

from physicalai.data import LeRobotDataModule
from physicalai.inference import InferenceModel
from physicalai.inference.callbacks import LatencyMonitor, ThroughputMonitor
from physicalai.inference.runners import SinglePass
import copy

log = logging.getLogger(__name__)

CONFIGS = [
    {"name": "ACT", "export_dir": "./exports/act_openvino", "backend": "openvino", "n_denoising_steps": None},
    {"name": "ACT", "export_dir": "./exports/act_torch", "backend": "torch", "n_denoising_steps": None},
    {"name": "SmolVLA", "export_dir": "./exports/smolvla_openvino", "backend": "openvino", "n_denoising_steps": 10},
    {"name": "SmolVLA", "export_dir": "./exports/smolvla_torch", "backend": "torch", "n_denoising_steps": 10},
    {"name": "Pi0.5", "export_dir": "./exports/pi05_openvino", "backend": "openvino", "n_denoising_steps": 10},
    {"name": "Pi0.5", "export_dir": "./exports/pi05_torch", "backend": "torch", "n_denoising_steps": 10},
]

DEVICES = ["CPU", "GPU"]


DEV_MAPPING = {
    "torch": {
        "CPU": "cpu",
        "GPU": "xpu" if torch.xpu.is_available() else "cuda",
    },
    "openvino": {
        "CPU": "CPU",
        "GPU": "GPU",
    }
}


WARMUP_STEPS = 10
BENCHMARK_STEPS = 100  # matches Cadene et al. (2026)
CSV_OUT = "results.csv"


def get_observation():
    """Return a real observation from the LiberoGym.

    Returns:
        Observation dict from env.to_observation().
    """
    l_dm = LeRobotDataModule(repo_id="HuggingFaceVLA/libero", train_batch_size=1, episodes=[0])
    obs = next(iter(l_dm.train_dataloader()))
    return obs.to_numpy().to_dict(flatten=False)


def _timing_stats(timings_ms):
    """Compute latency statistics from raw per-call timings.

    Returns:
        Dict with avg_ms, std_ms, p95_ms, min_ms, max_ms, max_hz.
    """
    arr = np.array(timings_ms)
    avg_ms = float(arr.mean())
    return {
        "avg_ms": avg_ms,
        "std_ms": float(arr.std(ddof=1)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "max_hz": 1000.0 / avg_ms if avg_ms > 0 else float("nan"),
    }


def benchmark(cfg, device, obs):
    """Run benchmark for a single (config, device) pair.

    Times each forward pass manually so std can be computed, matching the
    reporting style of Cadene et al. (2026).

    Returns:
        Result dict, or None if the model could not be loaded.
    """
    name, export_dir, backend = cfg["name"], cfg["export_dir"], cfg["backend"]
    n_denoising_steps = cfg["n_denoising_steps"]

    latency = LatencyMonitor(window_size=BENCHMARK_STEPS)
    throughput = ThroughputMonitor(window_seconds=60.0)

    try:
        model = InferenceModel.load(
            export_dir,
            backend=backend,
            device=DEV_MAPPING[backend][device],
            runner=SinglePass(),
            callbacks=[latency, throughput],
        )
    except Exception as e:
        log.warning("SKIP %s [%s] [%s]: %s", name, backend, device, e)
        return None

    try:
        for _ in range(WARMUP_STEPS):
            model.select_action(copy.deepcopy(obs))

        latency.on_reset()
        throughput.on_reset()

        timings_ms = []
        for _ in range(BENCHMARK_STEPS):
            t0 = time.perf_counter()
            model.select_action(copy.deepcopy(obs))
            timings_ms.append((time.perf_counter() - t0) * 1000.0)

    except Exception as e:
        log.warning("SKIP %s [%s] [%s] inference failed: %s", name, backend, device, e)
        return None

    return {
        "name": name,
        "backend": backend,
        "device": device,
        "n_denoising_steps": n_denoising_steps if n_denoising_steps is not None else "N/A",
        **_timing_stats(timings_ms),
        "pred_per_s": throughput.throughput,
    }


def main():
    """Run all benchmarks, print a table, and save results to CSV."""
    log.info("Benchmarking %d steps per config (after %d warmup)", BENCHMARK_STEPS, WARMUP_STEPS)
    log.info("Devices: %s", DEVICES)

    obs = get_observation()

    results = [
        r
        for cfg in CONFIGS
        for device in DEVICES
        if (r := benchmark(cfg, device, obs)) is not None
    ]

    if not results:
        log.warning("No results, check that export directories exist.")
        return

    header = (
        f"{'Policy':<12} {'Backend':<12} {'Device':<6} {'Steps':>6}"
        f" {'avg ms':>8} {'std ms':>8} {'p95 ms':>8} {'min ms':>8} {'max ms':>8}"
        f" {'pred/s':>8} {'max Hz':>8}"
    )
    sep = "-" * len(header)
    log.info(header)
    log.info(sep)
    for r in results:
        steps = str(r["n_denoising_steps"])
        log.info(
            "%s %s %s %s %8.1f %8.2f %8.1f %8.1f %8.1f %8.1f %8.1f",
            f"{r['name']:<12}", f"{r['backend']:<12}", f"{r['device']:<6}", f"{steps:>6}",
            r["avg_ms"], r["std_ms"], r["p95_ms"], r["min_ms"], r["max_ms"],
            r["pred_per_s"], r["max_hz"],
        )

    with Path(CSV_OUT).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    log.info("Saved to %s", CSV_OUT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
