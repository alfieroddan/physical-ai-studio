# Required Interfaces

A minimal policy needs a config, model, policy, and processors.

## Config

The serializable config contains:

- ordered input and output features, including `Feature.normalization_data`;
- model architecture values;
- `chunk_size`;
- `n_action_steps`;
- state-dict-shaping capability settings.

The policy owns the config. The model receives flat constructor values.

Use a plain frozen dataclass for configuration. Do not add `kw_only=True` unless
the config has a concrete inheritance or field-ordering requirement that needs it.

## Model

### Required

- A flat, config-free constructor.
- `compute_loss(batch)` returns loss and metrics.
- `predict_action_chunk(batch)` returns the full native action chunk.

Do not trim to `n_action_steps` or environment width in the model.

### Usually Inherited

`forward()` may dispatch by mode:

```text
training -> compute_loss(batch)
evaluation -> predict_action_chunk(batch)
```

`compute_val_loss()` may call `compute_loss()`.

### Temporal Indices

Override only the streams the model consumes:

- `observation_delta_indices`: current or past observations;
- `action_delta_indices`: supervised action steps;
- `reward_delta_indices`: reward context.

Use empty defaults when no temporal context is needed. Extra indices change dataset
sampling and batch shape.

## Policy

### Required

- `setup(stage)`: adopt or validate dataset features and their normalization data,
  without creating modules.
- `configure_model()`: defer unresolved construction, create the model once, and
  create or refresh processors when needed.
- `forward(batch)`: dispatch training and inference.
- `predict_action_chunk(batch)`: preprocess, predict, and postprocess.
- `training_step(batch, batch_idx)`: return loss and log metrics.
- `configure_optimizers()`: create the optimizer and scheduler.

`configure_model()` must preserve an existing model and its parameters. Return without
changes only when the resolved config and processor state are already synchronized.
Refresh missing or stale processors without reloading weights or reinjecting adapters.
Reject incompatible changes before mutating an initialized policy.

The base policy owns the public `config` property, shared checkpoint plumbing, device
transfer, `select_action()`, and `reset()`. A narrow policy-specific config restoration
helper may deserialize and validate config, then delegate to `configure_model()`; it
must not create modules itself.

`compute_val_loss()` and batch preparation stay on the concrete policy because they
use its model and processor types.

## Processors

The preprocessor:

- converts `Observation` values to model tensors;
- preserves configured feature order;
- normalizes inputs and action targets.

The postprocessor:

- restores action scale;
- preserves output feature order;
- removes padded dimensions;
- trims to `n_action_steps`.

Processors read normalization from the config's features. Do not duplicate those
values in separate policy state. Statistics do not determine model architecture.

## Optional Interfaces

| Interface                          | Use when                                         |
| ---------------------------------- | ------------------------------------------------ |
| Pretrained resolver                | Loading external artifacts                       |
| `set_features()` or rename support | Metadata can change without rebuilding the model |
| Export properties                  | Supporting export                                |
| PEFT targets                       | Supporting LoRA                                  |
| RTC behavior                       | Supporting real-time chunking                    |
| Gradient-checkpointing hook        | The model supports it                            |
| Custom validation loss             | Validation differs from training                 |
| Temporal indices                   | The model consumes temporal context              |

## Checklist

- Config reconstructs the model without `dataset_stats`.
- Model constructor is flat and config-free.
- Loss and full-chunk prediction are implemented.
- Every route converges on `configure_model()`; unresolved features defer construction.
- Repeated calls preserve model identity and unchanged processor identities.
- Normalization updates and missing processors refresh without rebuilding the model.
- Checkpoint restoration rejects incompatible existing models before mutation.
- Processors use ordered features and their normalization data from the config.
- Policy flow is short and explicit.
- Temporal indices match the data the model consumes.
