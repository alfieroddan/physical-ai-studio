"""Reactivity analysis: action staleness vs chunk size.

Chunking amortises compute but commits the robot to an open-loop plan. Action
k of a chunk is executed `(inference_latency + k / control_hz)` seconds after
the observation was captured. This script quantifies that staleness.

Sweet spot: chunk must be long enough that the next inference finishes before
actions run out (`chunk / control_hz >= latency_s`), but short enough that the
last executed action is not too stale for contact-rich / dynamic tasks.
"""

import csv
import logging
import operator
from pathlib import Path

import matplotlib.pyplot as plt

from physicalai.data import LeRobotDataModule
from physicalai.policies import ACT, Pi05, SmolVLA

from run_chunk_analysis import EXPORT_ROOT, export_model, run_benchmark

log = logging.getLogger(__name__)

CHUNK_SIZES = [10, 25, 50, 100]
POLICY_CLASSES = [ACT, SmolVLA, Pi05]
CONTROL_HZ_TARGETS = [30, 50, 100]

DATAMODULE_REPO = "HuggingFaceVLA/libero"
CSV_OUT = "reactivity.csv"
PLOT_OUT = "reactivity.png"


def staleness_ms(latency_ms, chunk_size, control_hz):
    """Staleness of the last executed action in the chunk.

    staleness = inference_latency + (chunk_size - 1) / control_hz
    """
    return latency_ms + (chunk_size - 1) * 1000.0 / control_hz


def plot(results):
    """Two panels: staleness vs chunk, and feasibility margin vs chunk.

    Feasibility margin = (chunk / control_hz) - latency. Positive means next
    inference finishes before actions run out.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    policies = sorted({r["policy"] for r in results})
    target_hz = CONTROL_HZ_TARGETS[0]

    for policy_name in policies:
        rows = sorted(
            [r for r in results if r["policy"] == policy_name and r["control_hz"] == target_hz],
            key=operator.itemgetter("chunk_size"),
        )
        xs = [r["chunk_size"] for r in rows]
        axes[0].plot(xs, [r["staleness_ms"] for r in rows], marker="o", label=policy_name)
        axes[1].plot(xs, [r["feasibility_ms"] for r in rows], marker="o", label=policy_name)

    axes[0].set_title(f"Action staleness @ {target_hz} Hz control")
    axes[0].set_xlabel("Chunk size")
    axes[0].set_ylabel("Last-action staleness (ms)")

    axes[1].set_title(f"Feasibility margin @ {target_hz} Hz control")
    axes[1].set_xlabel("Chunk size")
    axes[1].set_ylabel("chunk/hz - latency (ms)")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1)

    for ax in axes:
        ax.grid(True)
        ax.legend()

    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=150)
    log.info("Plot saved to %s", PLOT_OUT)


def main():
    """Benchmark each (policy, chunk), compute staleness at target Hz, plot."""
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    datamodule = LeRobotDataModule(repo_id=DATAMODULE_REPO, num_workers=0, episodes=[0])

    results = []
    for policy_cls in POLICY_CLASSES:
        for chunk_size in CHUNK_SIZES:
            exported = export_model(policy_cls, chunk_size, datamodule)
            if exported is None:
                continue
            export_path, sample_obs = exported
            bench = run_benchmark(policy_cls.__name__, chunk_size, export_path, sample_obs)
            if bench is None:
                continue
            latency_ms = bench["avg_ms"]
            for hz in CONTROL_HZ_TARGETS:
                stale = staleness_ms(latency_ms, chunk_size, hz)
                feasibility = (chunk_size * 1000.0 / hz) - latency_ms
                results.append({
                    "policy": policy_cls.__name__,
                    "chunk_size": chunk_size,
                    "control_hz": hz,
                    "latency_ms": latency_ms,
                    "staleness_ms": stale,
                    "feasibility_ms": feasibility,
                })
                log.info(
                    "  %s chunk=%d hz=%d  lat=%.1fms  stale=%.1fms  feas=%+.1fms",
                    policy_cls.__name__, chunk_size, hz, latency_ms, stale, feasibility,
                )

    if not results:
        log.warning("No results.")
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
