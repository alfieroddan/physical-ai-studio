"""Bunch of integrations to test config generation

Models from: https://huggingface.co/collections/allenai/molmoact2-models
"""

from physicalai.data.observation import Feature, FeatureType
from physicalai.policies import MolmoAct2

EXAMPLE_REPO_IDS = [
    "allenai/MolmoAct2",
    "allenai/MolmoAct2-Think",
    "allenai/MolmoAct2-LIBERO",
]

NORM_TAGS = [
    "so100_so101_molmoact2",
    "so100_so101_molmoact2",
    "libero",
]


# allenai/MolmoAct2 and allenai/MolmoAct2-Think: so100/so101 arm, state/action dim=6
SO100_INPUT_FEATURES = [
    Feature(name="image", ftype=FeatureType.VISUAL, shape=(3, 378, 378)),
    Feature(name="state", ftype=FeatureType.STATE, shape=(6,)),
]
SO100_OUTPUT_FEATURES = [
    Feature(name="action", ftype=FeatureType.ACTION, shape=(6,)),
]

# allenai/MolmoAct2-LIBERO: two cameras, state dim=8, action dim=7
LIBERO_INPUT_FEATURES = [
    Feature(name="image", ftype=FeatureType.VISUAL, shape=(3, 378, 378)),
    Feature(name="wrist_image", ftype=FeatureType.VISUAL, shape=(3, 378, 378)),
    Feature(name="state", ftype=FeatureType.STATE, shape=(8,)),
]
LIBERO_OUTPUT_FEATURES = [
    Feature(name="action", ftype=FeatureType.ACTION, shape=(7,)),
]

INPUT_FEATURES = [SO100_INPUT_FEATURES, SO100_INPUT_FEATURES, LIBERO_INPUT_FEATURES]
OUTPUT_FEATURES = [SO100_OUTPUT_FEATURES, SO100_OUTPUT_FEATURES, LIBERO_OUTPUT_FEATURES]


if __name__ == "__main__":
    for norm_tag, repo_id, inp, out in zip(NORM_TAGS, EXAMPLE_REPO_IDS, INPUT_FEATURES, OUTPUT_FEATURES, strict=False):
        policy = MolmoAct2(
            repo_id=repo_id, input_features=inp, output_features=out, norm_tag=norm_tag
        )
        policy.setup("inference")
        print(policy.config)
