import torch
from pathlib import Path

from physicalai.data import Feature, FeatureType, NormalizationParameters, Observation
from physicalai.policies import MolmoAct2

device = "cuda" if torch.cuda.is_available() else "cpu"


# SO100 pretrained stats expect state-only inputs with 6 values.
batch = Observation(
    images={
        "overview": torch.rand(1, 3, 256, 256),
        "wrist": torch.rand(1, 3, 256, 256),
    },
    state=torch.rand(1, 6),
).to(device)


input_features = [
    Feature(
        name="overview",
        ftype=FeatureType.VISUAL,
        shape=(3, 256, 256),
    ),
    Feature(
        name="state",
        ftype=FeatureType.STATE,
        shape=(6,),
    ),
]


output_features = [
    Feature(
        name="action",
        ftype=FeatureType.ACTION,
        shape=(6,),
    ),
]


def print_resolved_features(label: str, features: list[Feature]) -> None:
    print(f"{label}:")
    for feature in features:
        normalization_data = feature.normalization_data
        has_normalization = isinstance(normalization_data, NormalizationParameters)
        print(
            f"  - name={feature.name}, type={feature.ftype}, shape={feature.shape}, "
            f"has_normalization={has_normalization}",
        )
        if has_normalization:
            print(
                f"    q01={normalization_data.q01}, q99={normalization_data.q99}",
            )


if __name__ == "__main__":
    # Initialize the policy
    policy = MolmoAct2(
        repo_id="allenai/MolmoAct2-SO100_101",
        norm_tag="so100_so101_molmoact2",
        input_features=input_features,
        output_features=output_features,
    )
    policy.setup("inference")

    print_resolved_features("Resolved input features", policy.config.input_features)
    print_resolved_features("Resolved output features", policy.config.output_features)

    # # Forward pass to get predicted actions
    with torch.no_grad():
        actions = policy.predict_action_chunk(batch)

    print(f"Actions shape: {actions.shape}")
    print(f"Actions: {actions}")

    export_root = Path("exports") / "molmoact2"
    torch_export_dir = export_root / "torch"

    print("\nExporting Torch checkpoint + manifest...")
    policy.export(torch_export_dir, backend="torch")
    print(f"Torch export written under: {torch_export_dir.resolve()}")
