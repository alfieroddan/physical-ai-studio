# PushT Benchmark

The PushT benchmark is derived from the Diffusion Policy paper by Columbia Engineering: https://diffusion-policy.cs.columbia.edu/.

Each PushT gym is base seeded with a value of 100000 and a max number of steps of 300. They then test 50 gyms of different seeds. The paper repeats 3 times — a total of 150 episodes.

The imitation data can be found [here](https://diffusion-policy.cs.columbia.edu/data/training/). HuggingFace also hosts an example in `LeRobotDataset` format [here](https://huggingface.co/datasets/lerobot/pusht).

## CLI Example

```bash
physicalai benchmark --config configs/benchmark/pusht.yaml --ckpt_path model.ckpt
```

## API Example

```python
from physicalai.benchmark.pusht import PushTBenchmark
from physicalai.data import LeRobotDataModule
from physicalai.policies import ACT
from physicalai.train import Trainer

# Train policy
datamodule = LeRobotDataModule(repo_id="lerobot/pusht")
policy = ACT()
trainer = Trainer(max_epochs=100)
trainer.fit(policy, datamodule)

# Load best checkpoint
policy = ACT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
policy.eval()

# Evaluate benchmark (paper protocol: seed=100000, 50 episodes)
benchmark = PushTBenchmark()
results = benchmark.evaluate(policy)
print(results.summary())
```

## Citation

```bibtex
@inproceedings{chi2023diffusionpolicy,
  title     = {Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author    = {Chi, Cheng and Feng, Siyuan and Du, Yilun and Xu, Zhenjia and Cousineau, Eric and Burchfiel, Benjamin and Song, Shuran},
  booktitle = {Proceedings of Robotics: Science and Systems (RSS)},
  year      = {2023}
}

@article{chi2024diffusionpolicy,
  title   = {Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author  = {Chi, Cheng and Xu, Zhenjia and Feng, Siyuan and Cousineau, Eric and Du, Yilun and Burchfiel, Benjamin and Tedrake, Russ and Song, Shuran},
  journal = {The International Journal of Robotics Research},
  year    = {2024}
}
```