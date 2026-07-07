import random
from pathlib import Path

import numpy as np
import torch

from physicalai.data.observation import Feature, FeatureType, Observation
from physicalai.inference import InferenceModel
from physicalai.policies import MolmoAct2


EXPORT_DIR = Path("molmo-libero-export-torch")
SEED = 123


def reseed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def first_dtype(module: torch.nn.Module | None) -> torch.dtype | None:
    if module is None:
        return None
    for parameter in module.parameters():
        return parameter.dtype
    return None


def compare_arrays(name: str, left: np.ndarray, right: np.ndarray, *, atol: float = 1e-6) -> None:
    if left.dtype.kind in {"i", "u", "b"}:
        equal = np.array_equal(left, right)
        max_abs = 0.0 if equal else float(np.max(np.abs(left.astype(np.int64) - right.astype(np.int64))))
    else:
        equal = np.allclose(left, right, atol=atol, rtol=0)
        max_abs = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
    print(f"{name}: equal={equal} shape_left={left.shape} shape_right={right.shape} max_abs={max_abs}")


def extract_action_array(output: object) -> np.ndarray:
    if isinstance(output, dict):
        if "action" in output:
            value = output["action"]
        elif "actions" in output:
            value = output["actions"]
        else:
            value = next(iter(output.values()))
    else:
        value = output

    if torch.is_tensor(value):
        if value.dtype == torch.bfloat16:
            value = value.to(dtype=torch.float32)
        return value.detach().cpu().numpy()
    return np.asarray(value)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_features = [
        Feature(name="image", ftype=FeatureType.VISUAL, shape=(3, 378, 378), normalization_data=None),
        Feature(name="image2", ftype=FeatureType.VISUAL, shape=(3, 378, 378), normalization_data=None),
        Feature(name="state", ftype=FeatureType.STATE, shape=(8,)),
    ]
    output_features = [Feature(name="action", ftype=FeatureType.ACTION, shape=(7,))]

    policy = MolmoAct2(
        input_features=input_features,
        output_features=output_features,
        repo_id="molmo-LIBERO",
        norm_tag="libero",
        n_action_steps=10,
    ).to(device=device, dtype=torch.bfloat16)

    if EXPORT_DIR.exists():
        for path in EXPORT_DIR.iterdir():
            if path.is_file():
                path.unlink()

    policy.export(EXPORT_DIR, backend="torch")
    inference_model = InferenceModel(EXPORT_DIR, device=device)
    loaded_policy = inference_model.adapter._policy

    print("== DTYPE ==")
    print("direct_policy", first_dtype(policy))
    print("loaded_policy", first_dtype(loaded_policy))
    print("direct_model", first_dtype(policy.model))
    print("loaded_model", first_dtype(loaded_policy.model))

    observation = Observation(
        task=["open the drawer"],
        state=torch.linspace(-0.5, 0.5, steps=8, dtype=torch.float32).reshape(1, 8),
        images={
            "image": torch.zeros((1, 3, 378, 378), dtype=torch.float32),
            "image2": torch.ones((1, 3, 378, 378), dtype=torch.float32) * 0.25,
        },
    )

    obs_dict_torch = observation.to(device).to_dict(flatten=True)
    obs_dict_benchmark = observation.to_numpy().to_dict(flatten=False)
    obs_reconstructed = Observation.from_dict(obs_dict_benchmark).to_torch(device)

    policy_pre = policy._preprocessor(obs_dict_torch)
    loaded_pre = loaded_policy._preprocessor(obs_reconstructed.to_dict(flatten=True))

    print("== PREPROCESSOR ==")
    for key in ["input_ids", "attention_mask", "image_placeholder_token_id", "images_bchw", "state"]:
        left = policy_pre[key].detach().cpu().numpy() if torch.is_tensor(policy_pre[key]) else np.asarray(policy_pre[key])
        right = loaded_pre[key].detach().cpu().numpy() if torch.is_tensor(loaded_pre[key]) else np.asarray(loaded_pre[key])
        compare_arrays(key, left, right)

    print("== MODEL ==")
    reseed()
    with torch.no_grad():
        policy_model_out = extract_action_array(policy.model.predict_action_chunk(policy_pre))
    reseed()
    with torch.no_grad():
        loaded_model_out = extract_action_array(loaded_policy.model.predict_action_chunk(loaded_pre))
    compare_arrays("model_actions", policy_model_out, loaded_model_out, atol=1e-5)

    print("== POSTPROCESSOR ==")
    policy_post = policy._postprocessor(torch.from_numpy(policy_model_out).to(device)).detach().cpu().numpy()
    loaded_post = loaded_policy._postprocessor(torch.from_numpy(loaded_model_out).to(device)).detach().cpu().numpy()
    compare_arrays("post_actions", policy_post, loaded_post, atol=1e-5)

    print("== FULL POLICY ==")
    reseed()
    with torch.no_grad():
        direct_actions = extract_action_array(policy.predict_action_chunk(observation.to(device)))
    reseed()
    inferred_actions = inference_model.predict_action_chunk(obs_dict_benchmark)
    compare_arrays("predict_action_chunk", direct_actions, inferred_actions, atol=1e-5)


if __name__ == "__main__":
    main()