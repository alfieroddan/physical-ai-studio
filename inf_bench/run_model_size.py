"""Model footprint analysis: on-disk size and peak RSS during inference.

 This script reports:

  - export directory size (bytes on disk, what ships)
  - peak process RSS during a short inference burst (runtime working set)
  - parameter count where available from the loaded model

Results: `model_size.csv` and `model_size.png` (grouped bar chart).
"""

import csv
import gc
import logging
import os
import torch
from pathlib import Path
import copy

import matplotlib.pyplot as plt
import numpy as np
from physicalai.data.lerobot.datamodule import LeRobotDataModule
import psutil

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

DEVICE = "CPU"
WARMUP_STEPS = 5
INFERENCE_STEPS = 50
CSV_OUT = "model_size.csv"
PLOT_OUT = "model_size.png"


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


def dir_size_bytes(path):
    """Total size of all files under path (follows into subdirs)."""
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def get_observation():
    """Return a real observation from the LiberoGym.

    Returns:
        Observation dict from env.to_observation().
    """
    l_dm = LeRobotDataModule(repo_id="HuggingFaceVLA/libero", train_batch_size=1, episodes=[0])
    obs = next(iter(l_dm.train_dataloader()))
    return obs.to_numpy().to_dict(flatten=False)


def measure(cfg, obs):
    """Measure disk size and peak RSS delta during inference."""
    name, export_dir, backend = cfg["name"], cfg["export_dir"], cfg["backend"]
    path = Path(export_dir)

    if not path.exists():
        log.warning("SKIP %s [%s]: %s missing", name, backend, export_dir)
        return None

    disk_mb = dir_size_bytes(path) / (1024 * 1024)

    gc.collect()
    proc = psutil.Process()
    rss_before_mb = proc.memory_info().rss / (1024 * 1024)

    try:
        model = InferenceModel.load(export_dir, backend=backend, device=DEV_MAPPING[backend][DEVICE], runner=SinglePass())
    except Exception as e:
        log.warning("SKIP %s [%s] load failed: %s", name, backend, e)
        return None

    rss_loaded_mb = proc.memory_info().rss / (1024 * 1024)
    peak_mb = rss_loaded_mb

    try:
        for _ in range(WARMUP_STEPS):
            model.select_action(copy.deepcopy(obs))
        for _ in range(INFERENCE_STEPS):
            model.select_action(copy.deepcopy(obs))
            peak_mb = max(peak_mb, proc.memory_info().rss / (1024 * 1024))
    except Exception as e:
        log.warning("SKIP %s [%s] inference failed: %s", name, backend, e)
        return None

    param_count = None
    underlying = getattr(model, "model", None) or getattr(model, "_model", None)
    if underlying is not None and hasattr(underlying, "parameters"):
        try:
            param_count = sum(p.numel() for p in underlying.parameters())
        except Exception:
            param_count = None

    del model
    gc.collect()

    return {
        "name": name,
        "backend": backend,
        "disk_mb": disk_mb,
        "load_delta_mb": rss_loaded_mb - rss_before_mb,
        "peak_rss_mb": peak_mb,
        "params_M": (param_count / 1e6) if param_count else None,
    }


def plot(results):
    """Grouped bar chart: disk vs peak RSS per config."""
    labels = [f"{r['name']}\n{r['backend']}" for r in results]
    disk = [r["disk_mb"] for r in results]
    peak = [r["peak_rss_mb"] for r in results]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), 4.5))
    ax.bar(x - width / 2, disk, width, label="Disk (MB)")
    ax.bar(x + width / 2, peak, width, label="Peak RSS (MB)")
    ax.set_ylabel("MB")
    ax.set_title("Model footprint: on-disk size vs peak RSS")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="both")
    ax.legend()

    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=150)
    log.info("Plot saved to %s", PLOT_OUT)


def main():
    """Measure every config, save CSV + plot."""
    obs = get_observation()
    results = [r for cfg in CONFIGS if (r := measure(cfg, obs)) is not None]

    if not results:
        log.warning("No results.")
        return

    header = f"{'Policy':<10} {'Backend':<10} {'Disk MB':>10} {'Load Î”MB':>10} {'Peak MB':>10} {'Params M':>10}"
    log.info(header)
    log.info("-" * len(header))
    for r in results:
        params_str = f"{r['params_M']:.1f}" if r["params_M"] is not None else "--"
        log.info(
            "%s %s %10.1f %10.1f %10.1f %10s",
            f"{r['name']:<10}", f"{r['backend']:<10}",
            r["disk_mb"], r["load_delta_mb"], r["peak_rss_mb"], params_str,
        )

    with Path(CSV_OUT).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    log.info("Results saved to %s", CSV_OUT)

    plot(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
