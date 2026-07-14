import torch

from physicalai.data import Feature, FeatureType, Observation
from physicalai.policies import MolmoAct2

device = "cuda" if torch.cuda.is_available() else "cpu"

batch = Observation(
    images={
        "overview": torch.rand(1, 3, 256, 256),
        "wrist": torch.rand(1, 3, 256, 256),
    },
    state=torch.rand(1, 6),
    task=["example, input",],
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


if __name__ == "__main__":
    # Initialize the policy
    policy = MolmoAct2(
        input_features=input_features,
        output_features=output_features,
    ).to(device, dtype=torch.bfloat16)
    policy.eval()

    # Forward pass to get predicted actions
    with torch.no_grad():
        actions = policy.predict_action_chunk(batch)
    print(f"Actions shape: {actions.shape}")
    print(f"Actions: {actions}")
