import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from physicalai.data.observation import ACTION, STATE, Feature, FeatureType, Observation
from physicalai.inference import InferenceModel
from physicalai.policies import MolmoAct2

SEED = 0
REPEATS = 20
WARMUP = 5
MODEL_REPO = "molmo-LIBERO"
EXPORT_DIR = Path("molmo-libero-export-torch-latency")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latency benchmark for MolmoAct2 Lightning and exported torch inference.")
    parser.add_argument(
        "--mode",
        choices=("both", "lightning", "inference"),
        default="both",
        help="Which benchmark path to run.",
    )
    parser.add_argument("--repeats", type=int, default=REPEATS, help="Measured iterations per stage.")
    parser.add_argument("--warmup", type=int, default=WARMUP, help="Warmup iterations before timing.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (for example: cpu, cuda, cuda:0). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=EXPORT_DIR,
        help="Directory for torch export artifacts used by inference benchmark.",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable torch.compile optimization in MolmoAct2 policy (default: enabled).",
    )
    return parser.parse_args()


def maybe_sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(name: str, fn, repeats: int, warmup: int, device: str) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    maybe_sync(device)

    times_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        maybe_sync(device)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    arr = np.asarray(times_ms, dtype=np.float64)
    result = {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "std_ms": float(arr.std()),
    }
    print(
        f"{name:30s} mean={result['mean_ms']:.2f}ms "
        f"p50={result['p50_ms']:.2f}ms p95={result['p95_ms']:.2f}ms std={result['std_ms']:.2f}ms"
    )
    return result


def make_fake_features() -> tuple[list[Feature], list[Feature]]:
    input_features = [
        Feature(name="wrist_image", ftype=FeatureType.VISUAL, shape=(3, 378, 378)),
        Feature(name=STATE, ftype=FeatureType.STATE, shape=(8,)),
    ]
    output_features = [
        Feature(name=ACTION, ftype=FeatureType.ACTION, shape=(7,)),
    ]
    return input_features, output_features


def make_fake_observation(device: str) -> Observation:
    state = torch.zeros((1, 8), dtype=torch.float32, device=device)
    image = torch.rand((1, 3, 378, 378), dtype=torch.float32, device=device)

    return Observation(
        state=state,
        images={"wrist_image": image},
        task=np.asarray(["pick up the object"], dtype=object),
    )


def make_fake_inference_inputs() -> dict[str, object]:
    return {
        "state": np.zeros((1, 8), dtype=np.float32),
        "images": {"wrist_image": np.random.rand(1, 3, 378, 378).astype(np.float32)},
        "task": ["pick up the object"],
    }


def run_lightning_benchmark(device: str, repeats: int, warmup: int, compile_model: bool) -> MolmoAct2:
    print("\n=== Lightning MolmoAct2 ===")
    input_features, output_features = make_fake_features()

    policy = MolmoAct2(
        input_features=input_features,
        output_features=output_features,
        repo_id=MODEL_REPO,
        n_action_steps=10,
        torch_compile=compile_model,
    )
    policy = policy.to(device=device, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32)
    policy.eval()

    obs = make_fake_observation(device)

    def full_step() -> torch.Tensor:
        with torch.no_grad():
            return policy.predict_action_chunk(obs)

    benchmark("lightning/full_forward", full_step, repeats, warmup, device)

    return policy


def run_exported_torch_benchmark(
    *,
    policy: MolmoAct2,
    device: str,
    repeats: int,
    warmup: int,
    export_dir: Path,
) -> None:
    print("\n=== Exported Torch (InferenceModel Forward Only) ===")

    policy.export(str(export_dir), backend="torch")
    inf_model = InferenceModel(str(export_dir), device=device, preprocessors=[], postprocessors=[])
    fake_inputs = make_fake_inference_inputs()

    def full_step() -> dict[str, np.ndarray]:
        return inf_model(fake_inputs)

    benchmark("exported/forward_only", full_step, repeats, warmup, device)


def main() -> None:
    args = parse_args()

    set_seed(args.seed)
    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    print(f"Using device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Compile: {args.compile}")
    print(f"Repeats: {args.repeats}, Warmup: {args.warmup}")
    print(f"Export dir: {args.export_dir}")

    if args.mode == "lightning":
        run_lightning_benchmark(device=device, repeats=args.repeats, warmup=args.warmup, compile_model=args.compile)
        return

    if args.mode == "both":
        policy = run_lightning_benchmark(
            device=device,
            repeats=args.repeats,
            warmup=args.warmup,
            compile_model=args.compile,
        )
        run_exported_torch_benchmark(
            policy=policy,
            device=device,
            repeats=args.repeats,
            warmup=args.warmup,
            export_dir=args.export_dir,
        )
        return

    input_features, output_features = make_fake_features()
    policy = MolmoAct2(
        input_features=input_features,
        output_features=output_features,
        repo_id=MODEL_REPO,
        n_action_steps=10,
        torch_compile=args.compile,
    )
    policy = policy.to(device=device, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32)
    policy.eval()
    run_exported_torch_benchmark(
        policy=policy,
        device=device,
        repeats=args.repeats,
        warmup=args.warmup,
        export_dir=args.export_dir,
    )


if __name__ == "__main__":
    main()
