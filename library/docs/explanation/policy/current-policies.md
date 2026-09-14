# Current and Proposed Policies

Current policies use several construction patterns. The new design keeps several
inputs but uses one model creation method: `configure_model()`.

- **Input route**: constructor, config, dataset, pretrained artifact, or checkpoint.
- **Materialization**: creating the model and processors.

## Summary

| Concern | Current | Proposed |
| --- | --- | --- |
| Model creation | Policy-specific helpers | One guarded `configure_model()` |
| Features | Often rebuilt from `dataset_stats` | Ordered features in the policy config |
| Normalization | Mixed with feature metadata | Dataset or processor state |
| Pretrained loading | Policy-specific handling | Resolve config and weights, then build once |
| Checkpoints | Constructor args and often `dataset_stats` | Saved config builds the model before tensor loading |
| Capabilities | Policy-specific code | Shared PEFT and RTC mixins |

## Current Design

ACT, Groot, Pi0, Pi05, SmolVLA, RLDX1, and LeRobot do not share one construction
contract.

```mermaid
flowchart TD
    A[Constructor arguments] --> I[Policy-specific initializer]
    D[Dataset stats] --> I
    H[Pretrained artifact] --> R[Policy-specific resolver]
    R --> I
    C[Checkpoint hyperparameters] --> A
    I --> F[Derive features and dimensions]
    F --> M[Create model]
    F --> P[Create processors]
```

`dataset_stats` often contains both feature metadata and normalization values. This
makes statistics part of model construction.

### Current Routes

**Lazy training**

```text
Policy(...)
    -> setup("fit") reads dataset.stats
    -> policy-specific initializer
    -> model and processors
```

This is common in ACT, Groot, Pi0, Pi05, SmolVLA, and RLDX1.

**Eager construction**

```text
Policy(dataset_stats=...)
    -> policy-specific initializer
    -> model and processors
```

ACT, Pi0, Pi05, SmolVLA, and RLDX1 support this pattern. Groot also needs an action
dimension. LeRobot can use explicit features or a LeRobot config.

**Pretrained construction**

```text
Pretrained path
    -> resolve config, stats, and weights
    -> policy-specific initializer
    -> load weights and create processors
```

Each policy returns a different artifact bundle. RLDX1 also resolves shards and camera
names.

**Checkpoint restoration**

```text
Checkpoint
    -> restore constructor arguments and dataset_stats
    -> create model
    -> load state_dict
```

LeRobot uses a custom loader.

**Explicit config**

Support varies. LeRobot accepts feature dictionaries. RLDX1 merges feature overrides
into `dataset_stats`. Other policies mainly use large constructors.

### Current Initializers

| Policy | Helper |
| --- | --- |
| ACT | `_initialize_model(dataset_stats, weights_file)` |
| Groot | `_initialize_model(env_action_dim, dataset_stats)` |
| Pi0 | `_initialize_model(dataset_stats)` |
| Pi05 | `_initialize_model(dataset_stats, weight_file)` |
| SmolVLA | `_initialize_model(dataset_stats, weights_file)` |
| RLDX1 | `_initialize_model(dataset_stats, shard_files)` |
| LeRobot | `_initialize_policy(input_features, output_features, config, dataset_stats)` |

## Current Feature Ownership

```mermaid
flowchart LR
    D[Dataset] --> S[dataset_stats]
    H[Pretrained artifact] --> S
    C[Checkpoint] --> S
    S --> F[Feature contract]
    S --> N[Normalization]
    F --> M[Model]
    F --> E[Export schema]
    N --> P[Processors]
```

| Policy | Feature source | Normalization source |
| --- | --- | --- |
| ACT | `dataset_stats` | Model and feature state |
| Groot | Dataset metadata and stats | `dataset_stats` |
| Pi0 | `dataset_stats` | `dataset_stats` |
| Pi05 | Config plus `dataset_stats` | Processor and export state |
| SmolVLA | Config plus `dataset_stats` | Processor and export state |
| RLDX1 | Features merged into `dataset_stats` | Merged stats |
| LeRobot | Explicit or LeRobot features | Separate stats passed to processors |

This causes four problems:

- feature contracts vary by input route;
- export may rediscover features from stats;
- normalization changes can appear architectural;
- checkpoints may need stats before model creation.

## Proposed Design

The policy config owns model structure and ordered features. `configure_model()` is the
only materialization method.

```mermaid
flowchart TD
    Y[YAML or constructor] --> C[Policy config]
    A[Pretrained artifact] --> C
    X[Explicit config] --> C
    K[Checkpoint config] --> C
    C --> G[configure_model]
    G -->|model exists| N[No-op]
    G --> M[Model.from_config]
    G --> P[Create processors]
    A --> W[Base weights]
    W --> M
    G --> MX[Apply reconstruction mixins]
    D[Dataset stats] --> P
    D --> V[Validate features]
    C --> V
```

```python
def configure_model(self) -> None:
    if self.model is not None:
        return

    config = self._config or self._resolve_config()
    self._config = config
    self.model = MyModel.from_config(config)
    self._preprocessor, self._postprocessor = make_policy_processors(config)
```

The guard prevents duplicate construction and repeated capability injection.

### Proposed Routes

| Route | Config source | Normalization | Weights |
| --- | --- | --- | --- |
| YAML or constructor | Explicit policy inputs | Dataset or processor artifact | Fresh or selected artifact |
| Explicit config | Caller | Dataset or processor state | Fresh |
| Pretrained | Artifact config | Artifact or dataset | Artifact weights |
| Checkpoint | Saved config | Saved processor state or dataset | Checkpoint state dictionary |
| Training | Existing config validated against dataset | Dataset | Fresh or loaded |

All routes call `configure_model()`.

- Constructors may call it when all inputs are present.
- `setup("fit")` validates or adopts dataset features.
- Pretrained resolvers return config and weight paths only.
- Checkpoints restore config before Lightning loads tensors.
- Checkpoint loading never fetches the original artifact.

## Proposed Feature Ownership

```mermaid
flowchart LR
    C[Policy config] --> F[Names types shapes order]
    F --> M[Model dimensions]
    F --> P[Processor order]
    F --> E[Export schemas]
    D[Dataset] --> N[Normalization values]
    A[Processor artifact] --> N
    N --> P
    D --> V[Contract validation]
    F --> V
```

The config defines features. Dataset or processor state defines normalization. No route
parses `dataset_stats` to rebuild feature identity.

The executable prototype still supports lazy dataset feature adoption and stores
normalization on `Feature`. These are temporary migration aids.

## Mixins

```mermaid
sequenceDiagram
    participant P as Policy
    participant C as configure_model
    participant M as Model
    participant L as PEFT
    participant R as RTC
    participant S as Lightning

    P->>C: resolved config
    C->>M: construct model
    C->>M: load base weights
    C->>L: inject LoRA
    C->>R: sync RTC state
    S->>M: load checkpoint tensors
```

- PEFT changes state-dict keys. Save its settings in the config. Inject adapters before
  checkpoint tensor loading.
- RTC is runtime state. Sync it after model creation. Apply it before chunk trimming.

## Before and After

```mermaid
flowchart LR
    subgraph Current
        B1[Arguments] --> B4[Policy-specific initializer]
        B2[dataset_stats] --> B4
        B3[Artifact or checkpoint] --> B4
        B4 --> B5[Model and processors]
    end

    subgraph Proposed
        A1[Policy config] --> A4[configure_model]
        A2[Normalization state] --> A5[Processors]
        A3[External weights] --> A4
        A4 --> A6[Model]
        A4 --> A5
    end
```

See [Policy Migration](migration.md) for migration steps.
