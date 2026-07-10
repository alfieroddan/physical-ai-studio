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
    # # option 3 set from dataset - dataset sets input / output features
    policy = MolmoAct2()
    # Memory baseline: avoid joint discrete+continuous loss, checkpoint activations,
    # and train only the action expert parameters.
    policy.config.action_mode = "continuous"
    policy.config.gradient_checkpointing = True
    policy.config.train_action_expert_only = True

    # datamodule
    dm = LeRobotDataModule(repo_id="MarkRedeman/dice-cleanup-combined", train_batch_size=1, val_batch_size=1, val_split=0.1, num_workers=5, episodes=[0, 1])

    # trainer
    trainer = Trainer(max_steps=2, val_check_interval=1, precision="bf16-mixed")
    trainer.fit(model=policy, datamodule=dm)
