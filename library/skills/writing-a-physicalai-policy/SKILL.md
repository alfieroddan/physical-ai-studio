---
name: writing-a-physicalai-policy
description: 'Author a new physicalai robot/VLA Policy in src/physicalai/policies/<name>/. Use when creating, scaffolding, or extending a policy (ACT, Pi0, SmolVLA, MolmoAct2 style): defining the Config dataclass, the torch Model (compute_loss / predict_action_chunk), torch preprocessor + postprocessor, the Lightning Policy wrapper (forward, training_step, configure_optimizers, setup), pretrained/HF weight loading, and registering the policy. Keywords: policy, VLA, LightningModule, Model, preprocessor, postprocessor, compute_loss, predict_action_chunk, action chunk, flow matching, Observation, Feature, normalization.'
---

# Writing a physicalai Policy

## When to Use
- Adding a brand-new policy under `src/physicalai/policies/<name>/`.
- Extending an existing policy's model, config, or pre/post-processing.
- Porting a HuggingFace/LeRobot model into the physicalai `Policy` + `Model` shape.
- For OpenVINO/ONNX export of the finished policy, use the sibling skill `exporting-a-policy-to-openvino`.

## Mental Model

A policy is split into four concerns. Keep them in separate files.

| Concern | Base class | File | Responsibility |
|---------|-----------|------|----------------|
| Config | `physicalai.data.Config` dataclass | `config.py` | Explicit, serializable hyperparameters. No logic. |
| Model | `physicalai.policies.base.Model` (`nn.Module`) | `model/` | Pure network + losses. Weight keys must match checkpoints. |
| Pre/Post | `torch.nn.Module` | `processors/` | Observation dict → model-ready tensors, and model output → action space. |
| Policy | `physicalai.policies.base.Policy` (`LightningModule`) | `policy.py` | Orchestrates config, model, processors, training, inference. |

Data flows: `Observation` → `preprocessor(batch.to_dict())` → `model` → `postprocessor({ACTION: ...})` → action tensor `(B, T, D)`.

Study a reference before writing: `src/physicalai/policies/smolvla/` (flow matching, dict-based) and `src/physicalai/policies/molmoact2/` (VLA with tokenizer + images). ACT (`src/physicalai/policies/act/`) is the simplest.

## Procedure

### 1. Directory layout
```
src/physicalai/policies/<name>/
├── __init__.py          # export <Name>, <Name>Config, <Name>Model
├── config.py            # <Name>Config dataclass
├── policy.py            # <Name>(Policy) Lightning wrapper
├── model/               # or model.py — the nn.Module(s)
│   ├── __init__.py
│   └── ...
├── processors/          # or preprocessor.py — pre/post + factory
│   ├── __init__.py
│   ├── preprocessor.py
│   ├── postprocessor.py
│   └── factory.py       # make_<name>_preprocessors(config) -> (pre, post)
└── from_hf.py           # optional: build config + load pretrained weights
```

### 2. Config (`config.py`)
- A `@dataclass` extending the project `Config` base. Only explicit fields with defaults; no behavior beyond `__post_init__` coercion of nested dataclasses.
- Include: `input_features`, `output_features`, `n_obs_steps`, `n_action_steps`, `max_action_dim`, optimizer/scheduler fields, and any model dims.
- Features are `physicalai.data.observation.Feature(name, ftype, shape, normalization_data)` with `ftype` in `FeatureType.{VISUAL, STATE, ACTION, ...}`.

### 3. Model (`model/`)
Subclass `physicalai.policies.base.Model`. Required members (see [model base](src/physicalai/policies/base/model.py)):
- `forward(batch)`: training mode → `(loss, loss_dict)`; eval mode → predicted actions.
- `compute_loss(batch) -> (Tensor, dict)`: training loss **with grad**; dict must contain `"loss"`.
- `compute_val_loss(batch) -> (Tensor, dict)`: no-grad validation metric (override for a meaningful metric like action MSE; default delegates to `compute_loss`).
- `predict_action_chunk(batch) -> Tensor` of shape `(B, n_action_steps, action_dim)`.
- Properties `reward_delta_indices`, `action_delta_indices`, `observation_delta_indices` (return `None` if unused).
- **Weight-key discipline**: if loading a pretrained checkpoint, keep submodule attribute names identical to the checkpoint keys, and load with a strict key check. See `MolmoAct2Model.load_pretrained_weights` in [wrapper.py](src/physicalai/policies/molmoact2/model/wrapper.py).
- **Export discipline**: the model's inference path must be pure-tensor — no `.item()`, `.tolist()`, Python control flow on tensor values, or device moves that depend on data. Do that host-side work in the preprocessor. This keeps the graph traceable for OpenVINO/ONNX.

### 4. Pre/Post processors (`processors/`)
Both are `torch.nn.Module` with a `forward(batch: dict) -> dict`.
- **Preprocessor**: validate the batch; normalize features (`FeatureNormalizeTransform` / `NormalizationType`); extract state/task/images; produce the exact tensor keys the model consumes. Emit only what the model needs (drop anything unused). Run all value-dependent host prep here (tokenization, resize, patchify, placeholder expansion, padding masks).
- **Postprocessor**: take `{ACTION: tensor}`, clamp, denormalize (inverse `FeatureNormalizeTransform`), optionally map to the robot frame, return `{ACTION: ...}`.
- **Factory**: `make_<name>_preprocessors(config) -> (preprocessor, postprocessor)` in `factory.py`.

### 5. Policy (`policy.py`)
Subclass `physicalai.policies.base.Policy` (a `LightningModule`). Implement:
- `__init__(...)`: build/resolve `self.config`; set `self.model = None`, `self._preprocessor = None`, `self._postprocessor = None`; call `self.save_hyperparameters(ignore=["config"])`; eagerly init when features (or a pretrained repo) are available.
- `_initialize_model()`: build processors via the factory, construct the `Model`, load pretrained weights if a checkpoint path is set.
- `setup(stage)`: lazy path — pull `input_features`/`output_features` from `self.trainer.datamodule.train_dataset` when not provided, then `_initialize_model()`.
- `forward(batch)`: `training` → `self.model(self._preprocessor(batch.to_dict()))`; else `predict_action_chunk`.
- `predict_action_chunk(batch)` (`@torch.no_grad()`): `pre → model.predict_action_chunk → post`.
- `training_step`, `compute_val_loss`.
- `configure_optimizers()` / `configure_gradient_clipping()` using config fields (AdamW + `cosine_decay_with_warmup_scheduler` is the house style; group LRs per component if needed — see `MolmoAct2.get_optim_params`).
- Expose `input_features` / `output_features` properties.

### 6. Dual-path initialization
Support both:
- **Lazy (training)**: `Policy(...)` then `trainer.fit(...)` → model built in `setup()` from the datamodule.
- **Eager (inference/checkpoint)**: `Policy(input_features=..., output_features=...)` or `Policy.load_from_checkpoint(...)` → model built in `__init__`.

### 7. Register
Add to [src/physicalai/policies/__init__.py](src/physicalai/policies/__init__.py): import `<Name>`, `<Name>Config`, `<Name>Model` and add to `__all__`. Also update `get_policy` if it dispatches by name.

## Validation
- Import check: `python -c "from physicalai.policies import <Name>"`.
- Instantiate with explicit `input_features`/`output_features`, run a fake `Observation` through `predict_action_chunk`, assert output shape `(B, n_action_steps, action_dim)`.
- Confirm a single `training_step` produces a finite loss and backprops (grads on the intended trainable params).
- If pretrained: assert strict weight-key match (no missing/unexpected keys).
- Run `get_errors` on edited files; fix real issues (ignore warnings shared by sibling policies).

## Gotchas
- Keep `Model` attribute names aligned with checkpoint keys; a mismatch silently breaks strict loading.
- Only emit model inputs the model actually consumes — extra keys cause export/inference mismatches.
- Do host-side (data-dependent) prep in the preprocessor, never in the model, or export will fail.
- `save_hyperparameters(ignore=["config"])` so the config object isn't double-serialized.
