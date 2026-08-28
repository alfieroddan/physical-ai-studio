# MolmoAct2

MolmoAct2 is a vision-language-action policy from the Allen Institute for AI.
It consumes camera images, robot state, and a task string, then predicts a chunk
of future actions. This package provides a first-party PhysicalAI implementation
for training, Lightning checkpoints, benchmarking, and export.

- [Blog post](https://allenai.org/blog/molmoact2)
- [Paper](https://arxiv.org/abs/2605.02881)
- [LeRobot documentation](https://huggingface.co/docs/lerobot/en/molmoact2)

## Installation

```bash
pip install "physicalai-train[molmoact2]"
```

## Loading Policies

MolmoAct2 has two distinct loading paths:

- `MolmoAct2(pretrained_name_or_path=..., norm_tag=...)` initializes from a
  released Hugging Face checkpoint.
- `MolmoAct2.load_from_checkpoint(...)` restores a policy previously trained
  and saved by Lightning.

### Released Hugging Face Checkpoint

```python
import torch

from physicalai.devices.utils import get_device
from physicalai.policies import MolmoAct2

DEVICE = get_device()

policy = MolmoAct2(
    pretrained_name_or_path="allenai/MolmoAct2-LIBERO",
    norm_tag="libero",
    n_action_steps=10,
    use_random_input_noise=True,
)
policy = policy.to(device=DEVICE, dtype=torch.bfloat16).eval()
```

The normalization tag resolves the checkpoint's camera, state, action,
normalization, action-horizon, setup, and control metadata from
`norm_stats.json`.

### Lightning Checkpoint

After fitting a policy, save it through the attached Lightning trainer:

```python
trainer.fit(policy, datamodule=datamodule)
trainer.save_checkpoint("checkpoints/molmoact2.ckpt", weights_only=True)
```

Restore it directly for inference:

```python
from physicalai.policies import MolmoAct2

policy = MolmoAct2.load_from_checkpoint(
    "checkpoints/molmoact2.ckpt",
    map_location="cpu",
    weights_only=True,
).eval()
```

The Lightning checkpoint contains the resolved `MolmoAct2Config`. Loading it
rebuilds the architecture and processors, then applies the trained state dict.
It does not download or reload the original pretrained model weights.

Tokenizer files are not embedded in the Lightning checkpoint. The local path
saved in `config.tokenizer_name_or_path` must still be available when the
checkpoint is restored.

## Training

### CLI

The checked-in configuration contains the complete model, dataset, optimizer,
and trainer setup:

```bash
physicalai fit --config configs/physicalai/molmoact2.yaml
```

See [`configs/physicalai/molmoact2.yaml`](../../../../configs/physicalai/molmoact2.yaml)
for the available overrides.

### Python API

```python
from physicalai.data import LeRobotDataModule
from physicalai.policies import MolmoAct2
from physicalai.train import Trainer

policy = MolmoAct2(
    use_random_input_noise=True,
    use_lora=True,
    enable_lora_action_expert=False,
    gradient_checkpointing=True,
)

datamodule = LeRobotDataModule(
    repo_id="lerobot/pusht",
    train_batch_size=8,
    data_format="physicalai",
)

trainer = Trainer(max_epochs=30, precision="bf16-mixed")
trainer.fit(policy, datamodule=datamodule)
```

When the policy is constructed lazily, the training dataset supplies its input
and output feature contract during `setup("fit")`.

For full fine-tuning, leave both `use_lora` and `train_action_head_only` false.
For action-head-only training, use `train_action_head_only=True`. LoRA and
action-head-only training are mutually exclusive.

## Benchmarking LIBERO

```python
import torch

from physicalai.benchmark.gyms import LiberoBenchmark
from physicalai.devices.utils import get_device
from physicalai.policies import MolmoAct2

DEVICE = get_device()

policy = MolmoAct2(
    pretrained_name_or_path="allenai/MolmoAct2-LIBERO",
    norm_tag="libero",
    n_action_steps=10,
    use_random_input_noise=True,
)
policy = policy.to(device=DEVICE, dtype=torch.bfloat16).eval()

benchmark = LiberoBenchmark(
    task_suite="libero_10",
    num_episodes=20,
    seed=0,
    camera_name_mapping={"image2": "wrist_image"},
    observation_height=378,
    observation_width=378,
)

try:
    results = benchmark.evaluate(policy)
    print(results.summary())
finally:
    for gym in benchmark.gyms:
        gym.close()
```

### Reported Results

The following measurements are retained from the original integration report.
Performance depends on hardware, precision, runtime versions, and benchmark
configuration.

NVIDIA A100, FP16:

| Suite          |  Tasks | Avg. success rate (%) | Avg. reward | Avg. episode length |  Avg. FPS |
| -------------- | -----: | --------------------: | ----------: | ------------------: | --------: |
| libero_spatial |     10 |                 100.0 |        1.00 |               106.4 |     10.56 |
| libero_object  |     10 |                 100.0 |        1.00 |               132.7 |     11.82 |
| libero_goal    |     10 |                  90.0 |        0.90 |               134.2 |     11.51 |
| libero_10      |     10 |                 100.0 |        1.00 |               241.4 |     12.61 |
| **Average**    | **40** |              **97.5** |    **0.98** |           **153.7** | **11.63** |

OpenVINO GPU, FP16:

| Suite          |  Tasks | Avg. success rate (%) | Avg. reward | Avg. episode length | Avg. FPS |
| -------------- | -----: | --------------------: | ----------: | ------------------: | -------: |
| libero_spatial |     10 |                 100.0 |        1.00 |               122.0 |     14.6 |
| libero_object  |     10 |                 100.0 |        1.00 |               145.3 |     16.2 |
| libero_goal    |     10 |                  80.0 |        0.80 |               159.5 |     16.1 |
| libero_10      |     10 |                  80.0 |        0.80 |               300.2 |     16.7 |
| **Average**    | **40** |              **90.0** |    **0.90** |           **181.8** | **15.9** |

## Export

MolmoAct2 supports Torch and OpenVINO export.

Load a trained Lightning checkpoint before exporting it:

```python
from physicalai.policies import MolmoAct2

policy = MolmoAct2.load_from_checkpoint(
    "checkpoints/molmoact2.ckpt",
    map_location="cpu",
    weights_only=True,
).eval()

policy.export("exports/molmoact2-torch", backend="torch")
policy.export("exports/molmoact2-openvino", backend="openvino")
```

OpenVINO export also exports the tokenizer and requires the tokenizer assets
referenced by the restored config.

## Zero-Shot SO-101

The released SO-100/101 checkpoint uses an older joint calibration convention.
Set `adapt_to_so101=True` to transform observations and actions between that
checkpoint frame and the current robot frame.

```python
import torch

from physicalai.policies import MolmoAct2

policy = MolmoAct2(
    pretrained_name_or_path="allenai/MolmoAct2-SO100_101",
    norm_tag="so100_so101_molmoact2",
)

policy.set_features(
  input_features=input_features,
  output_features=output_features,
  copy_state_normalization=True,
  copy_action_normalization=True,
)

policy = policy.to(dtype=torch.bfloat16).eval()
policy.export("exports/molmoact2-so101-torch", backend="torch")
```

`set_features` replaces the checkpoint feature definitions without reloading
its weights. Visual features are always taken directly from `input_features`,
so the replacement can add, remove, or rename cameras. The two normalization
flags independently copy the checkpoint's state and action normalization onto
replacement features with matching shapes. Copied normalization overwrites any
normalization already present on that replacement feature.

Start the SO-101 from an extended pose near the task workspace. Starting from a
rest pose can cause the policy to remain there.

## Repository and Normalization Tags

`pretrained_name_or_path` identifies the Hugging Face repository or local
snapshot. `norm_tag` selects one entry from `metadata_by_tag` in that
snapshot's `norm_stats.json`.

```json
{
  "format": "molmoact2_norm_stats.v1",
  "norm_mode": "q01_q99",
  "metadata_by_tag": {
    "so100_so101_molmoact2": {
      "action_key": "action",
      "state_key": "observation.state",
      "camera_keys": [],
      "normalize_gripper": true,
      "action_horizon": 30
    }
  }
}
```

The repository and normalization tag must describe the same embodiment. For
example:

```python
policy = MolmoAct2(
    pretrained_name_or_path="allenai/MolmoAct2-SO100_101",
    norm_tag="so100_so101_molmoact2",
)
```

For a custom dataset, omit `norm_tag` and construct the policy lazily. The
attached PhysicalAI dataset supplies features and normalization statistics at
training setup.

Constructor-provided features override features resolved from `norm_tag`
without inheriting their normalization. For zero-shot use, initialize from the
tag first and call `set_features` with the specific normalization maps that
should be copied.

## Notes

- `n_action_steps` must be between 1 and `chunk_size`.
- `use_random_input_noise=True` starts flow matching from sampled Gaussian
  noise; otherwise inference uses deterministic zero noise.
- Lightning checkpoints restore model weights and resolved configuration, but
  do not package tokenizer files.
- Supported export backends are Torch and OpenVINO.
- Generic video augmentation from the upstream implementation is not included.
