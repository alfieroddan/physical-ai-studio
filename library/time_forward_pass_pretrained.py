import time

import torch

from physicalai.data import Feature, FeatureType, Observation
from physicalai.policies import MolmoAct2


def synchronize_if_cuda(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    batch = Observation(
        images={
            "overview": torch.rand(1, 3, 256, 256, dtype=compute_dtype),
        },
        state=torch.rand(1, 6, dtype=compute_dtype),
    ).to(device=device)

    input_features = [
        Feature(name="overview", ftype=FeatureType.VISUAL, shape=(3, 256, 256)),
        Feature(name="state", ftype=FeatureType.STATE, shape=(6,)),
    ]
    output_features = [
        Feature(name="action", ftype=FeatureType.ACTION, shape=(6,)),
    ]

    policy = MolmoAct2(
        repo_id="allenai/MolmoAct2-SO100_101",
        norm_tag="so100_so101_molmoact2",
        input_features=input_features,
        output_features=output_features,
    )
    policy.setup("inference")
    policy = policy.to(device=device, dtype=compute_dtype)

    # Warm-up pass so one-time setup does not skew timing.
    with torch.no_grad():
        batch_dict = batch.to_dict(flatten=True)
        processed_batch = policy._preprocessor(batch_dict)
        model_outputs = policy.model.predict_action_chunk(processed_batch)
        _ = policy._postprocessor(model_outputs)

    num_runs = 10
    total_timings_ms = []
    pre_timings_ms = []
    model_timings_ms = []
    post_timings_ms = []
    last_actions = None

    with torch.no_grad():
        for idx in range(num_runs):
            batch_dict = batch.to_dict(flatten=True)

            synchronize_if_cuda(device)
            pre_start = time.perf_counter()
            processed_batch = policy._preprocessor(batch_dict)
            pre_end = time.perf_counter()

            synchronize_if_cuda(device)
            model_start = time.perf_counter()
            model_outputs = policy.model.predict_action_chunk(processed_batch)
            synchronize_if_cuda(device)
            model_end = time.perf_counter()

            synchronize_if_cuda(device)
            post_start = time.perf_counter()
            actions = policy._postprocessor(model_outputs)
            post_end = time.perf_counter()

            total_ms = (post_end - pre_start) * 1000.0
            pre_ms = (pre_end - pre_start) * 1000.0
            model_ms = (model_end - model_start) * 1000.0
            post_ms = (post_end - post_start) * 1000.0

            total_timings_ms.append(total_ms)
            pre_timings_ms.append(pre_ms)
            model_timings_ms.append(model_ms)
            post_timings_ms.append(post_ms)
            last_actions = actions
            print(
                f"Run {idx + 1:02d}: total={total_ms:.3f} ms "
                f"(pre={pre_ms:.3f}, model={model_ms:.3f}, post={post_ms:.3f})",
            )

    avg_total_ms = sum(total_timings_ms) / len(total_timings_ms)
    avg_pre_ms = sum(pre_timings_ms) / len(pre_timings_ms)
    avg_model_ms = sum(model_timings_ms) / len(model_timings_ms)
    avg_post_ms = sum(post_timings_ms) / len(post_timings_ms)
    min_total_ms = min(total_timings_ms)
    max_total_ms = max(total_timings_ms)

    print("\nSummary")
    print(f"Device: {device}")
    print(f"Dtype: {compute_dtype}")
    print(f"Forward runs: {num_runs}")
    print(f"Average total: {avg_total_ms:.3f} ms")
    print(f"Average preprocessor: {avg_pre_ms:.3f} ms")
    print(f"Average model: {avg_model_ms:.3f} ms")
    print(f"Average postprocessor: {avg_post_ms:.3f} ms")
    print(f"Min total: {min_total_ms:.3f} ms")
    print(f"Max total: {max_total_ms:.3f} ms")
    if last_actions is not None:
        print(f"Last actions shape: {tuple(last_actions.shape)}")


if __name__ == "__main__":
    main()
