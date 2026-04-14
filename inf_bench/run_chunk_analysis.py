"""Analyse the effect of chunk size on latency, throughput, and action rate.

Action rate (actions/sec) = chunk_size x throughput (forward passes/sec).

A larger chunk may slow each forward pass, but if throughput doesn't drop
proportionally the net action rate still improves — this script finds that
sweet spot.

Workflow: for each policy and chunk size, train for one step, export to Torch,
benchmark, then plot.
"""

import csv
import logging
import operator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from physicalai.data import LeRobotDataModule
from physicalai.inference import InferenceModel
from physicalai.inference.callbacks import LatencyMonitor, ThroughputMonitor
from physicalai.inference.runners import SinglePass
from physicalai.policies import ACT, Pi05, SmolVLA
from physicalai.train import Trainer

log = logging.getLogger(__name__)

CHUNK_SIZES = [10, 25, 50, 100]
POLICY_CLASSES = [ACT, SmolVLA, Pi05]

DATAMODULE_REPO = "HuggingFaceVLA/libero"
EXPORT_ROOT = Path("./exports/chunk_analysis")
BACKEND = "torch"
WARMUP_STEPS = 10
BENCHMARK_STEPS = 200
CSV_OUT = "chunk_analysis.csv"
PLOT_OUT = "chunk_analysis.png"


def make_observation_from_stats(dataset_stats):
    """Build a structured observation matching the Torch adapter's Observation format.

    The Torch adapter expects field names: 'state' (array) and 'images' (dict of
    camera_name -> array). Image shapes come from the stats 'shape' field; the mean
    field only stores per-channel values, not the full spatial shape.

    Falls back to standard Libero cameras/shapes if stats lack image entries.

    Returns:
        Dict with 'state' array and 'images' dict of camera arrays.
    """
    obs = {}
    images = {}
    for key, stats in dataset_stats.items():
        is_visual = key.startswith("observation.images.") or str(stats.get("type", "")).upper() == "VISUAL"
        if is_visual:
            camera_name = key.removeprefix("observation.images.")
            shape = tuple(stats["shape"]) if stats.get("shape") else (3, 224, 224)
            images[camera_name] = np.zeros((1, *shape), dtype=np.float32)
        elif key == "observation.state":
            shape = tuple(stats["shape"]) if stats.get("shape") else (14,)
            obs["state"] = np.zeros((1, *shape), dtype=np.float32)

    if not images:
        images = {
            "agentview_image": np.zeros((1, 3, 224, 224), dtype=np.float32),
            "robot0_eye_in_hand_image": np.zeros((1, 3, 224, 224), dtype=np.float32),
        }
    obs["images"] = images

    if "state" not in obs:
        obs["state"] = np.zeros((1, 14), dtype=np.float32)

    return obs


def export_model(policy_cls, chunk_size, datamodule):
    """Train policy for one step and export to Torch.

    Returns:
        (export_path, sample_observation) or None if export failed.
    """
    name = policy_cls.__name__
    export_path = EXPORT_ROOT / f"{name.lower()}_chunk{chunk_size}"

    log.info("%s chunk_size=%d: training...", name, chunk_size)
    policy = policy_cls(chunk_size=chunk_size, n_action_steps=chunk_size, n_obs_steps=1)
    trainer = Trainer(max_steps=1, logger=False)
    trainer.fit(policy, datamodule)

    sample_obs = make_observation_from_stats(policy.hparams["dataset_stats"])

    if export_path.exists():
        log.info("%s chunk_size=%d: export already exists, skipping.", name, chunk_size)
        return export_path, sample_obs

    try:
        policy.export(str(export_path), backend=BACKEND)
    except Exception as e:
        log.warning("%s chunk_size=%d: export failed: %s", name, chunk_size, e)
        return None
    return export_path, sample_obs


def run_benchmark(policy_name, chunk_size, export_path, sample_obs):
    """Benchmark a single exported model and return metrics.

    Returns:
        Dict of benchmark metrics, or None if loading or inference failed.
    """
    latency = LatencyMonitor(window_size=BENCHMARK_STEPS)
    throughput = ThroughputMonitor(window_seconds=60.0)

    try:
        model = InferenceModel.load(
            str(export_path),
            backend=BACKEND,
            runner=SinglePass(),
            callbacks=[latency, throughput],
        )

        for _ in range(WARMUP_STEPS):
            model.select_action(sample_obs)

        latency.on_reset()
        throughput.on_reset()

        for _ in range(BENCHMARK_STEPS):
            model.select_action(sample_obs)
    except Exception as e:
        log.warning("%s chunk_size=%d: benchmark failed: %s", policy_name, chunk_size, e)
        return None

    return {
        "policy": policy_name,
        "chunk_size": chunk_size,
        "avg_ms": latency.avg_ms,
        "p95_ms": latency.p95_ms,
        "throughput": throughput.throughput,
        "action_rate": chunk_size * throughput.throughput,
    }


def plot(results):
    """Plot chunk size vs throughput and action rate, one line per policy."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    policies = sorted({r["policy"] for r in results})
    for policy_name in policies:
        rows = sorted([r for r in results if r["policy"] == policy_name], key=operator.itemgetter("chunk_size"))
        xs = [r["chunk_size"] for r in rows]
        axes[0].plot(xs, [r["throughput"] for r in rows], marker="o", label=policy_name)
        axes[1].plot(xs, [r["action_rate"] for r in rows], marker="o", label=policy_name)
        axes[2].plot(xs, [r["avg_ms"] for r in rows], marker="o", label=policy_name)

    axes[0].set_title("Throughput vs Chunk Size")
    axes[0].set_xlabel("Chunk size")
    axes[0].set_ylabel("Forward passes / sec")

    axes[1].set_title("Action Rate vs Chunk Size")
    axes[1].set_xlabel("Chunk size")
    axes[1].set_ylabel("Actions / sec  (chunk × throughput)")

    axes[2].set_title("Latency vs Chunk Size")
    axes[2].set_xlabel("Chunk size")
    axes[2].set_ylabel("Avg latency (ms)")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=150)
    log.info("Plot saved to %s", PLOT_OUT)


def main():
    """Export, benchmark, and plot for all policies and chunk sizes."""
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    datamodule = LeRobotDataModule(repo_id=DATAMODULE_REPO, num_workers=0, episodes=[0])

    results = []
    for policy_cls in POLICY_CLASSES:
        for chunk_size in CHUNK_SIZES:
            result_or_none = export_model(policy_cls, chunk_size, datamodule)
            if result_or_none is None:
                continue
            export_path, sample_obs = result_or_none
            log.info("%s chunk_size=%d: benchmarking...", policy_cls.__name__, chunk_size)
            result = run_benchmark(policy_cls.__name__, chunk_size, export_path, sample_obs)
            if result is None:
                continue
            results.append(result)
            log.info(
                "  avg_ms=%.1f  throughput=%.1f/s  action_rate=%.1f actions/s",
                result["avg_ms"], result["throughput"], result["action_rate"],
            )

    if not results:
        log.warning("No results to save.")
        return

    with Path(CSV_OUT).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    log.info("Results saved to %s", CSV_OUT)

    plot(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
