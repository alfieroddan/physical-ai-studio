import torch

from physicalai.data import Feature, FeatureType
from physicalai.policies import MolmoAct2

device = "cuda" if torch.cuda.is_available() else "cpu"


input_features = [
    Feature(
        name="overview",
        ftype=FeatureType.VISUAL,
        shape=(3, 256, 256),
    ),
    Feature(
        name="state",
        ftype=FeatureType.STATE,
        shape=(8,),
    ),
]


output_features = [
    Feature(
        name="action",
        ftype=FeatureType.ACTION,
        shape=(7,),
    ),
]


if __name__ == "__main__":
    # Initialize the policy
    policy = MolmoAct2(
        repo_id="molmo-LIBERO",
        norm_tag="libero",
        input_features=input_features,
        output_features=output_features,
    )
    policy.setup("inference")

    # policy.export("test_molmo_torch", backend="torch")
    policy.to_openvino("test_molmo_openvino", )
