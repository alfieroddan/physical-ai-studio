from physicalai.policies import MolmoAct2
from physicalai.data import Feature, FeatureType, LeRobotDataModule
from physicalai.train import Trainer


input_features = [
    Feature(
        name="overview",
        ftype=FeatureType.VISUAL,
        shape=(3, 480, 640),
        normalization_data=None,
    ),
    Feature(
        name="gripper",
        ftype=FeatureType.VISUAL,
        shape=(3, 480, 640),
        normalization_data=None,
    ),
]
input_features.append(
    Feature(name="state", ftype=FeatureType.STATE, shape=(6,)),
)
output_features = [
    Feature(name="action", ftype=FeatureType.ACTION, shape=(6,)),
]

if __name__ == "__main__":
    # option 1 with defined features, eager init
    # policy = MolmoAct2(
    #     input_features=input_features,
    #     output_features=output_features,
    #     repo_id="allenai/MolmoAct2-SO100_101"
    # )

    # # option 2 set from pretrained, eager init
    # policy = MolmoAct2(
    #     repo_id="allenai/MolmoAct2-SO100_101",
    #     norm_tag="so100_so101_molmoact2"
    # )

    # # option 3 set from dataset - dataset sets input / output features
    policy = MolmoAct2(repo_id="allenai/MolmoAct2-SO100_101")

    # datamodule
    dm = LeRobotDataModule(repo_id="MarkRedeman/dice-cleanup-combined", train_batch_size=2, val_batch_size=2, val_split=0.1, num_workers=5, episodes=[0, 1])

    # trainer
    trainer = Trainer(max_steps=4, val_check_interval=2, precision="bf16-mixed")
    trainer.fit(model=policy, datamodule=dm)
