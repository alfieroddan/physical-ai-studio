"""Tail-latency / jitter analysis for exported policies.

Real-time control loops miss deadlines on worst-case latency, not mean. A
policy averaging 20 ms with 150 ms tail spikes is worse than a steady 30 ms
policy. This script reports p50 / p95 / p99 per (policy, backend, device)
and plots the tail distribution.
"""

import csv
import logging
import time
import copy
from pathlib import Path
import torch

import matplotlib.pyplot as plt
import numpy as np

from physicalai.data.lerobot.datamodule import LeRobotDataModule
from physicalai.inference import InferenceModel
from physicalai.inference.runners import SinglePass

log = logging.getLogger(__name__)

CONFIGS = [
    {"name": "ACT", "export_dir": "./exports/act_openvino", "backend": "openvino"},
    {"name": "ACT", "export_dir": "./exports/act_torch", "backend": "torch"},
    {"name": "SmolVLA", "export_dir": "./exports/smolvla_openvino", "backend": "openvino"},
    {"name": "SmolVLA", "export_dir": "./exports/smolvla_torch", "backend": "torch"},
    {"name": "Pi0.5", "export_dir": "./exports/pi05_openvino", "backend": "openvino"},
    {"name": "Pi0.5", "export_dir": "./exports/pi05_torch", "backend": "torch"},
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
BENCHMARK_STEPS = 500
CSV_OUT = "jitter.csv"
PLOT_OUT = "jitter.png"


def get_observation():
    """Return a real observation from the LiberoGym.

    Returns:
        Observation dict from env.to_observation().
    """
    l_dm = LeRobotDataModule(repo_id="HuggingFaceVLA/libero", train_batch_size=1, episodes=[0])
    obs = next(iter(l_dm.train_dataloader()))
    return obs.to_numpy().to_dict(flatten=False)


def benchmark(cfg, device, obs):
    """Collect per-call timings and return percentile stats."""
    name, export_dir, backend = cfg["name"], cfg["export_dir"], cfg["backend"]

    try:
        model = InferenceModel.load(export_dir, backend=backend, device=DEV_MAPPING[backend][device], runner=SinglePass())
    except Exception as e:
        log.warning("SKIP %s [%s] [%s]: %s", name, backend, device, e)
        return None

    for _ in range(WARMUP_STEPS):
        model.select_action(copy.deepcopy(obs))

    try:


        timings_ms = []
        for _ in range(BENCHMARK_STEPS):
            t0 = time.perf_counter()
            model.select_action(copy.deepcopy(obs))
            timings_ms.append((time.perf_counter() - t0) * 1000.0)
    except Exception as e:
        log.warning("SKIP %s [%s] [%s] inference failed: %s", name, backend, device, e)
        return None

    arr = np.array(timings_ms)
    return {
        "name": name,
        "backend": backend,
        "device": device,
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(arr.max()),
        "jitter_ms": float(np.percentile(arr, 99) - np.percentile(arr, 50)),
        "timings": timings_ms,
    }


def plot(results):
    """Grouped bar chart of p50 / p95 / p99 per config."""
    labels = [f"{r['name']}\n{r['backend']}/{r['device']}" for r in results]
    p50 = [r["p50_ms"] for r in results]
    p95 = [r["p95_ms"] for r in results]
    p99 = [r["p99_ms"] for r in results]

    x = np.arange(len(labels))
    width = 0.27

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    ax.bar(x - width, p50, width, label="p50")
    ax.bar(x, p95, width, label="p95")
    ax.bar(x + width, p99, width, label="p99")

    ax.set_ylabel("Latency (ms)")
    ax.set_title("Tail latency per configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(True, axis="y")
    ax.legend()

    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=150)
    log.info("Plot saved to %s", PLOT_OUT)


def main():
    """Benchmark jitter for every config and plot."""
    obs = get_observation()
    results = [
        r
        for cfg in CONFIGS
        for device in DEVICES
        if (r := benchmark(cfg, device, obs)) is not None
    ]

    if not results:
        log.warning("No results.")
        return

    header = f"{'Policy':<10} {'Backend':<10} {'Device':<6} {'p50':>8} {'p95':>8} {'p99':>8} {'jitter':>8}"
    log.info(header)
    log.info("-" * len(header))
    for r in results:
        log.info(
            "%s %s %s %8.1f %8.1f %8.1f %8.1f",
            f"{r['name']:<10}", f"{r['backend']:<10}", f"{r['device']:<6}",
            r["p50_ms"], r["p95_ms"], r["p99_ms"], r["jitter_ms"],
        )

    csv_rows = [{k: v for k, v in r.items() if k != "timings"} for r in results]
    with Path(CSV_OUT).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    log.info("Results saved to %s", CSV_OUT)

    plot(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
