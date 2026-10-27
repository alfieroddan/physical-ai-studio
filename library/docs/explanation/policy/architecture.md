# Policy Architecture

A policy connects a PyTorch model to data, training, inference, checkpoints, and
export. Keep the concrete policy short enough to read in one pass.

> Every construction route converges on `configure_model()`. It defers until the
> feature contract is available, creates the model once, and synchronizes processors.

## Structure

```text
policy_name/
|-- config.py        # Features and model settings
|-- model.py         # Network, loss, and prediction
|-- policy.py        # Construction and Lightning flow
|-- preprocessor.py  # Input conversion and normalization
`-- postprocessor.py # Output conversion and denormalization
```

Small policies may combine files. Keep the same ownership boundaries.

## Ownership

| Owner               | Responsibility                                                          |
| ------------------- | ----------------------------------------------------------------------- |
| Config              | Ordered features with normalization, architecture, and action horizons  |
| Policy              | Construction, processing flow, training, and optimizer settings         |
| Base policy         | Config property, checkpoint plumbing, device transfer, and action queue |
| Dataset             | Observed features and normalization values                              |
| Model               | Network, loss, temporal indices, and full-chunk prediction              |
| Processors          | Conversion, normalization, and external action shape                    |
| Pretrained resolver | Artifact config and weight paths                                        |
| Capability mixins   | PEFT, RTC, and other shared behavior                                    |
| Export mixin        | Schemas, samples, backend settings, and manifests                       |

The config does not contain optimizer settings, artifact paths, or export destinations.
The model does not store the config or accept `Feature` objects.

Keep policy construction explicit. Store each constructor value in a clearly named
attribute and pass every field explicitly in `from_config()` and when creating the
resolved config. Do not hide constructor state in a generic dictionary such as
`_config_values`; although shorter, it makes reconstruction paths harder to read,
review, and type-check.

The policy constructor is a user-facing API, not a mirror of every model argument.
Expose common choices and training controls there. The serialized config has a
different responsibility: it must include every argument needed to reconstruct the
model, including rarely changed internal options. Policy construction may leave those
fields at config defaults, but model materialization must consume them explicitly so a
complete config is sufficient to rebuild the architecture.

## Feature State

The config owns feature names, types, shapes, order, and `Feature.normalization_data`.
Dataset statistics and encoder requirements supply normalization values; the resolved
features carry them to processors and into checkpoints. Do not duplicate those values
in separate policy state or checkpoint fields.

Normalization does not determine model architecture. Compatibility checks ignore
feature statistics but preserve the ordered feature contract and architecture fields.
Processor refresh checks compare the full config, including normalization.

The dataset validates its features against the config. It does not define model
architecture through `dataset_stats`.

## Materialization

`configure_model()` is the only method that creates the model and processors.
`setup()` adopts or validates dataset features and their normalization data without
creating modules. An early call with unresolved features leaves the policy unbuilt.

```text
configure_model
    -> defer if features are unresolved
    -> resolve or reuse complete config
    -> validate compatibility with an existing model
    -> create model, load base weights, and apply reconstruction capabilities once
    -> resolve model-dependent processor metadata
    -> create or refresh processors when needed
    -> sync runtime capabilities
```

Guard model creation, not the entire hook. An existing model can still need refreshed
processors after dataset statistics change or a checkpoint restores feature statistics.
Use the constructed model's metadata, such as encoder image size, when preparing
processors; do not instantiate another encoder to discover it.

An unchanged call preserves model, config, and processor identities. A normalization
change or missing processor refreshes processors without replacing model parameters,
optimizer references, or adapters. Reject incompatible architecture or ordered feature
contracts before changing an initialized policy. Feature adaptation requires an
explicit compatibility path; see [Advanced Patterns](advanced.md#feature-adaptation).

Use this order:

1. Resolve the config and validate it against any already materialized model.
2. Create the model only when absent.
3. Load pretrained base weights and apply state-dict-shaping capabilities only on
   first materialization, never during processor refresh or checkpoint restore.
4. Create or refresh processors from the config's features and model metadata.
5. Commit synchronized policy state and sync runtime capabilities without resetting
   unchanged runtime state.

### Lightning Constraints

`configure_model()` is a Lightning lifecycle hook, so implementations must follow
these constraints:

| Constraint                                           | Required behavior                                                                |
| ---------------------------------------------------- | -------------------------------------------------------------------------------- |
| Lightning may call the hook more than once           | Preserve the model and unchanged processors; refresh processors only when needed |
| Strategies such as FSDP control module creation      | Create large modules inside the hook, not before Lightning invokes it            |
| Checkpoint tensors load after module construction    | Rebuild the exact module and adapter structure from saved config first           |
| Checkpoint restore must be self-contained            | Do not fetch a pretrained artifact or overwrite checkpoint weights               |
| Dataset state may arrive later in `setup()`          | Keep architecture independent of feature statistics and do not rebuild the model |
| Manual inference may call the hook without a Trainer | Do not depend on trainer, dataloader, or device state during construction        |
| Config fields and defaults may change                | Version or migrate saved config so old checkpoints rebuild the same structure    |

Keep the hook deterministic and free of network access. It should materialize module
structure from resolved config; `setup()` may validate dataset features and attach
normalization data to features without changing that structure.

Eager constructor calls are a manual-inference convenience. For strategies that must
control module creation, leave the policy unmaterialized until Lightning invokes the
hook. Resolve external artifacts before entering it, including encoder architecture
metadata needed when pretrained weights are disabled.

## Construction Routes

| Route           | Config source                        | Weight source              |
| --------------- | ------------------------------------ | -------------------------- |
| Explicit config | Caller                               | None                       |
| Constructor     | Features and defaults                | None                       |
| Training        | Dataset features or validated config | None                       |
| Pretrained      | Artifact config                      | Artifact weights           |
| Checkpoint      | Saved config                         | Lightning state dictionary |

All routes call `configure_model()`. Pretrained resolvers do not create models.
Checkpoint loading does not fetch the original artifact.

## Runtime Flow

```text
Observation
    -> preprocessor
    -> model
    -> full action chunk
    -> runtime capabilities
    -> postprocessor
    -> action queue
```

The model returns `(batch_size, chunk_size, model_action_dim)`. Postprocessing trims to
`n_action_steps` and removes padded dimensions.

## Checkpoints

Save the resolved config, including input and output feature normalization. During restore:

1. clear the pretrained path;
2. load the saved config;
3. validate the saved contract against an existing model, if present;
4. call `configure_model()` to construct missing modules and capabilities once and
   synchronize processors from the saved features;
5. let Lightning load tensors;
6. restore runtime state.

See [Advanced Patterns](advanced.md) for details.

## Related Pages

- [Required Interfaces](interfaces.md)
- [Implement a Policy](how-to.md)
- [Advanced Patterns](advanced.md)
- [Export API](export.md)
- [Policy Migration](migration.md)
