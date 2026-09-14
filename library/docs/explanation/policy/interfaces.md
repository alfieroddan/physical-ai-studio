# Required Interfaces

This page separates the methods a policy author must implement from behavior inherited
from the shared base classes. The distinction keeps concrete policies small and makes
reviews focus on policy-specific decisions.

## Minimal Working Policy

A new policy needs three pieces:

1. a serializable model config;
2. a model that computes loss and predicts a full action chunk;
3. a policy that materializes the model and configures its optimizer.

Processors are also required at runtime, but their implementation is a separate
boundary. The policy creates them; it does not embed their normalization or conversion
logic.

## Config Requirements

The config must describe everything needed to reconstruct the model and interpret its
inputs and outputs:

- ordered input and output features;
- architecture dimensions and behavior;
- `chunk_size`, the model prediction horizon;
- `n_action_steps`, the execution horizon;
- any state-dict-shaping capability configuration.

The policy owns and serializes the config. The model receives only flat constructor
arguments and does not retain the config object.

## Required Model Overrides

### `compute_loss(batch)`

Computes the differentiable training loss. It returns the loss tensor and a metrics
dictionary containing at least `"loss"`.

### `predict_action_chunk(batch)`

Returns the complete native action prediction with shape
`(batch_size, chunk_size, model_action_dim)`. It must not trim to `n_action_steps` or
to the environment action width.

### Flat constructor

The model constructor declares the scalar and tensor-shape values needed to construct
the network. It does not accept the policy config or `Feature` objects.

## Inherited Model Flow

The base model can provide `forward()`:

```text
training mode  -> compute_loss(batch)
evaluation mode -> predict_action_chunk(batch)
```

A concrete model overrides `forward()` only when its dispatch or return contract is
genuinely different. Repeating the standard branch is useful in explanatory docs but
is not a required production override.

`compute_val_loss()` may default to `compute_loss()`. Override it only when validation
uses different computation or metrics.

## Temporal Delta Indices

Temporal delta indices tell the data loader which time-relative samples the model
consumes. Their values are part of the model's data contract, not an optimization.

- `observation_delta_indices` selects current or historical observations;
- `action_delta_indices` selects action targets, often the complete training chunk;
- `reward_delta_indices` selects reward context when the model consumes rewards.

The base model should return empty/default indices. Override a property only when the
model actually consumes that temporal stream. For example, a policy using two prior
observations returns negative observation offsets; a chunk predictor returns action
offsets matching its supervised horizon. Do not declare context merely because the
dataset contains it: extra indices change sampling requirements and batch shape.

## Required Policy Overrides

### `configure_model()`

The sole idempotent materialization method. It resolves or consumes the complete
config, constructs model and processors, loads optional base weights, and applies
reconstruction-sensitive modifications in the required order.

It must return without work when the model already exists.

### `configure_optimizers()`

Creates the optimizer and optional scheduler from policy-owned training settings.
Optimizer settings do not belong in the model config.

## Inherited Policy Flow

The base policy owns these methods for the standard lifecycle:

- `forward()` preprocesses training/evaluation batches and delegates to the model;
- `predict_action_chunk()` preprocesses observations, invokes the model, applies
  capability transforms, and postprocesses the full chunk;
- `training_step()` computes loss and logs metrics;
- `compute_val_loss()` dispatches validation loss;
- checkpoint hooks persist and restore the resolved config;
- `select_action()` fills and consumes the execution-horizon action queue;
- `reset()` clears runtime state between episodes.

Concrete policies should not override these methods merely to repeat the shared flow.
The [implementation guide](how-to.md) shows illustrative wrappers so authors can see
what is inherited.

## Processor Contract

The preprocessor:

- converts an `Observation` into model tensors;
- preserves configured feature order;
- applies input normalization from processor or dataset state.

The postprocessor:

- maps model outputs back to configured action features;
- reverses output normalization;
- removes padded model dimensions;
- trims the runtime chunk to `n_action_steps` at the appropriate boundary.

Processor state does not define feature identity or model architecture.

## Optional Interfaces

Implement these only when the policy supports the corresponding route or capability:

| Interface | Implement when |
| --- | --- |
| Pretrained config resolver | The policy loads an external pretrained artifact |
| `set_features()` or feature rename support | A constructed model can safely adapt names/order without changing architecture |
| Export mixin properties | The policy supports export tracing or manifest generation |
| PEFT model targets | The architecture supports LoRA/PEFT adapters |
| RTC model behavior | The model supports real-time chunking |
| Gradient-checkpointing hook | The model exposes a supported activation-checkpointing mechanism |
| Custom validation loss | Validation differs from training loss |
| Temporal delta indices | The model consumes temporal observations, actions, or rewards |

Checkpoint restoration is shared lifecycle behavior, not a method every concrete
policy reimplements. Policies with unusual reconstruction requirements extend the
cooperative hooks described in [Advanced Patterns](advanced.md).

## Review Checklist

A minimal policy is complete when:

- the config can reconstruct the architecture without dataset statistics;
- the model constructor is flat and config-free;
- loss and full-chunk prediction are implemented;
- every construction route reaches the same guarded `configure_model()`;
- processors use the config's ordered features and separate normalization state;
- optimizer construction is explicit;
- inherited lifecycle methods are not duplicated;
- temporal indices describe only context the model actually consumes.
