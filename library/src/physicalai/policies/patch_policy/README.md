# Patch Policy

Patch Policy is a modular vision-to-action policy. It encodes one or more
camera views with a `timm` visual backbone and passes the resulting tokens to a
selectable action head that predicts a chunk of future robot actions.

Install its optional visual-backbone dependency with:

```bash
uv sync --extra patch_policy
```

## Model structure

```text
Camera observations dict[str, B × T × C × H × W]
    │
    ├─ order and resize camera views
    ├─ normalize each camera with dataset statistics
    ├─ stack views internally as [B, T, V, C, H, W]
    ▼
TimmFeatureEncoder
    │
    ├─ DINOv2
    ├─ WebSSL
    └─ ResNet-18
    ▼
Visual tokens [B, T × V × P, E]
    │
    ├─ PatchDiffusionHead
    │    Iteratively denoises an action chunk while cross-attending to the
    │    visual tokens.
    │
    └─ PatchBehaviorTransformerHead
         Predicts residual-VQ action codes and code-conditioned continuous
         offsets with a patch-aware causal transformer.
    ▼
Action chunk [B, chunk_size, action_dim]
    │
    ├─ denormalize with dataset statistics
    └─ retain n_action_steps
    ▼
Robot actions [B, n_action_steps, action_dim]
```

`n_obs_steps` is the number of temporal frames used for every camera. A
single-frame input `[B, C, H, W]` is shaped as `[B, 1, C, H, W]`; callers
providing a temporal window use `[B, n_obs_steps, C, H, W]`. Training datasets
provide the complete window through the policy's observation delta indices.
During inference, the encoder processes only the current frame and an embedding
memory bank retains previous encoded observations. Missing history at the start
of an episode repeats the first encoded observation, matching the original
Patch Policy, and `policy.reset()` clears it.

The feature definitions also carry dataset normalization statistics. Images
use the encoder statistics. Diffusion actions use min/max normalization to
`[-1, 1]`, matching the original policy, while Behavior Transformer actions
remain in dataset units and use `vq_action_scale` inside the quantizer.

## Basic forward pass

The following example constructs a small deterministic diffusion policy. It
uses randomly initialized weights and is intended to demonstrate the input and
output contract rather than produce useful robot actions.

```python
import torch

from physicalai.data import Feature, FeatureType, Observation
from physicalai.policies import PatchPolicy

policy = PatchPolicy(
    input_features=[
        Feature(name="front", ftype=FeatureType.VISUAL, shape=(3, 28, 28)),
        Feature(name="wrist", ftype=FeatureType.VISUAL, shape=(3, 28, 28)),
    ],
    output_features=[
        Feature(name="action", ftype=FeatureType.ACTION, shape=(6,)),
    ],
    encoder_name="dinov2",
    encoder_output="patches",
    encoder_pretrained=False,
    action_head="diffusion",
    n_obs_steps=2,
    chunk_size=8,
    n_action_steps=4,
    num_inference_steps=2,
    eta=0.0,
    use_random_input_noise=False,
).eval()

batch_size = 1
observation = Observation(
    images={
        "front": torch.rand(batch_size, 2, 3, 28, 28),
        "wrist": torch.rand(batch_size, 2, 3, 28, 28),
    },
)

with torch.no_grad():
    actions = policy(observation)

print(actions.shape)  # torch.Size([1, 4, 6])
```

For training, include the overlapping action target shaped
`[B, n_obs_steps + chunk_size - 1, action_dim]`, put the policy in training
mode, and call the same forward method:

```python
observation.action = torch.rand(batch_size, 9, 6)
policy.train()
loss, metrics = policy(observation)
```

## Main policy options

| Option                                             | Purpose                                                            |
| -------------------------------------------------- | ------------------------------------------------------------------ |
| `input_features`                                   | Ordered camera feature definitions and normalization statistics.   |
| `output_features`                                  | Ordered action features concatenated into the model action vector. |
| `encoder_name`                                     | Visual backbone: `dinov2`, `webssl`, or `resnet18`.                |
| `encoder_output`                                   | Backbone representation: `patches`, `cls`, or `average`.           |
| `encoder_pretrained`                               | Load the backbone's pretrained weights.                            |
| `image_size`                                       | Encoder input `(height, width)`, defaulting to `(224, 224)`.       |
| `action_head`                                      | `diffusion` or `behavior_transformer`.                             |
| `n_obs_steps`                                      | Number of temporal frames per camera.                              |
| `chunk_size`                                       | Number of future actions predicted by the model.                   |
| `n_action_steps`                                   | Number of predicted actions returned for execution.                |
| `hidden_dim`, `num_layers`, `num_heads`, `dropout` | Action-head transformer architecture.                              |
| `num_train_timesteps`, `num_inference_steps`       | Diffusion schedule lengths.                                        |
| `eta`                                              | DDIM stochasticity; `0.0` gives deterministic update steps.        |
| `use_random_input_noise`                           | Start diffusion sampling from random noise instead of zeros.       |

Behavior Transformer-specific quantizer and loss settings are available on
`PatchPolicyConfig`, including `vq_latent_dim`, `vq_num_embeddings`,
`vq_num_groups`, `vq_action_scale`, and the offset, secondary-code, and
focal-loss weights. A single-frame input remains one timestep even when
`n_obs_steps` is larger.
