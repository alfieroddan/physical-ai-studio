# Implement a Policy

Use `prek` for repository validation. When a local suppression is necessary, use
the standard `# noqa: <rule>` form; do not add tool-specific `ruff:` comments to
source files.

Build the config, model, processors, and policy first. Add optional features later.

## 1. Config

The config owns ordered features, including their normalization data, model settings,
and action horizons.

```python
@dataclass(frozen=True)
class MyModelConfig(Config):
    input_features: list[Feature]
    output_features: list[Feature]
    action_dim: int
    hidden_size: int = 512
    chunk_size: int = 32
    n_action_steps: int = 32
```

Do not include optimizer settings, artifact paths, or export destinations. Keep
normalization on `Feature.normalization_data`, not in a second config field or policy
state object. Processors consume these values; model construction does not.

For compatibility checks, compare configs without feature normalization:

```python
def _without_normalization(config: MyModelConfig) -> MyModelConfig:
    return replace(
        config,
        input_features=[replace(feature, normalization_data=None) for feature in config.input_features],
        output_features=[replace(feature, normalization_data=None) for feature in config.output_features],
    )
```

Use `replace` from `dataclasses`. This comparison preserves feature identity, shape,
order, and all architecture fields while ignoring statistics alone.

## 2. Model

Use a flat constructor. Implement loss and full-chunk prediction.

```python
class MyModel(TemplateModel):
    def __init__(
        self,
        *,
        action_dim: int,
        hidden_size: int = 512,
        chunk_size: int = 32,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.backbone = build_backbone(hidden_size)
        self.action_head = nn.Linear(hidden_size, chunk_size * action_dim)

    def compute_loss(self, batch):
        prediction = self._predict(batch)
        loss = F.mse_loss(prediction, batch["action"])
        return loss, {"loss": loss}

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        return self._predict(batch)
```

Return `(batch_size, chunk_size, action_dim)`. Processors handle normalization,
padding, and `n_action_steps`.

Override temporal indices only when the model needs temporal context. See
[Required Interfaces](interfaces.md#temporal-indices).

## 3. Policy Constructor

Show all lifecycle state in the constructor.

```python
class MyPolicy(TemplatePolicy):
    def __init__(
        self,
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        pretrained_name_or_path: str | Path | None = None,
        *,
        hidden_size: int = 512,
        chunk_size: int = 32,
        n_action_steps: int = 32,
        optimizer_lr: float = 1e-4,
    ) -> None:
        super().__init__(n_action_steps=n_action_steps)
        self.model = None
        self._config: MyModelConfig | None = None
        self._preprocessor = None
        self._postprocessor = None
        self._processor_config: MyModelConfig | None = None

        self._input_features = input_features
        self._output_features = output_features
        self._pretrained_name_or_path = pretrained_name_or_path
        self._base_weights_path: Path | None = None
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps
        self.optimizer_lr = optimizer_lr

        if pretrained_name_or_path is not None:
            self._config, self._base_weights_path = _resolve_config_from_hf(pretrained_name_or_path)
            self._input_features = self._config.input_features
            self._output_features = self._config.output_features

        if self._input_features is not None and self._output_features is not None:
            self.configure_model()
```

Replace features and config values rather than mutating feature statistics in place.
`_processor_config` records the last config used to build processors, including feature
normalization. It is only a refresh marker, not a second source of normalization data.

Eager configuration is a manual-inference convenience. For strategies that control
module creation, defer the final call until Lightning invokes `configure_model()`.

Keep deprecated arguments together at the end of the signature. Translate them before
config resolution. See [Policy Migration](migration.md#legacy-arguments).

## 4. Config Construction

An explicit config uses the same materialization path.

```python
@classmethod
def from_config(cls, config: MyModelConfig, *, optimizer_lr: float = 1e-4) -> "MyPolicy":
    policy = cls(
        hidden_size=config.hidden_size,
        chunk_size=config.chunk_size,
        n_action_steps=config.n_action_steps,
        optimizer_lr=optimizer_lr,
    )
    policy._config = config
    policy._input_features = config.input_features
    policy._output_features = config.output_features
    policy.configure_model()
    return policy
```

A pretrained resolver returns data only:

```python
def _resolve_config_from_hf(path: str | Path) -> tuple[MyModelConfig, Path]:
    return read_artifact_config(path), resolve_weights(path)
```

It does not create the model. Resolve external artifacts before entering
`configure_model()`, including encoder metadata required for offline construction.
The constructor sketch treats artifact config as authoritative. Validate explicit
feature overrides before accepting them when a policy supports overrides.

## 5. Dataset Setup

`setup()` is the data-policy boundary.

```python
def setup(self, stage: str) -> None:
    if stage != "fit":
        return

    dataset = self.trainer.datamodule.train_dataset
    input_features = list(dataset.observation_features.values())
    output_features = list(dataset.action_features.values())
    self._validate_or_set_features(input_features, output_features)
```

Read features directly from the dataset. Do not rebuild them from `dataset_stats`.
The validation helper compares ordered names, types, and shapes, ignoring normalization,
then adopts or replaces the feature lists with dataset normalization attached. Include
absent statistics so stale values are not retained. The helper does not create modules
or processors; Lightning's subsequent `configure_model()` call synchronizes the config
and processors. Apply encoder-specific visual statistics to features in that hook when
the policy requires them.

## 6. Materialization

Keep `configure_model()` linear. Defer unresolved inputs and guard model creation
separately from processor synchronization.

```python
def configure_model(self) -> None:
    if self._input_features is None or self._output_features is None:
        return

    config = self._config
    if config is None:
        config = MyModelConfig(
            input_features=list(self._input_features),
            output_features=list(self._output_features),
            action_dim=resolve_action_dim(self._output_features),
            hidden_size=self.hidden_size,
            chunk_size=self.chunk_size,
            n_action_steps=self.n_action_steps,
        )
    else:
        updated = replace(
            config,
            input_features=list(self._input_features),
            output_features=list(self._output_features),
        )
        if updated != config:
            config = updated

    if self.model is not None and _without_normalization(config) != _without_normalization(self._config):
        raise ValueError("Config does not match the initialized model.")

    if self.model is None:
        self.model = MyModel.from_config(config)
        if self._base_weights_path is not None:
            self.model.load_weights(self._base_weights_path)

    if (
        config != self._processor_config
        or self._preprocessor is None
        or self._postprocessor is None
    ):
        self._preprocessor, self._postprocessor = make_policy_processors(config)
        self._processor_config = config

    self._config = config
    self.hidden_size = config.hidden_size
    self.chunk_size = config.chunk_size
    self.n_action_steps = config.n_action_steps
```

When processors require model-derived metadata such as encoder image size, pass it
from the existing model and include it in refresh checks if it can change. Do not build another
model to discover metadata. Apply structure-changing capabilities only in the first
model-creation branch, after base weights. Synchronize runtime capabilities afterward
without resetting unchanged state, including any action-horizon caches owned by the base.
The compatibility check permits statistics-only changes. Feature adaptation needs
an explicit, narrower check for changes to names, shapes, or order.

## 7. Policy Flow

Keep preprocessing, model calls, and postprocessing visible.

```python
def forward(self, batch: Observation):
    if self.training:
        return self.model.compute_loss(self._prepare_batch(batch, require_actions=True))
    return self.predict_action_chunk(batch)

def predict_action_chunk(self, batch: Observation) -> Tensor:
    prepared = self._prepare_batch(batch, require_actions=False)
    chunk = self.model.predict_action_chunk(prepared)
    return self._postprocessor(chunk)

def training_step(self, batch: Observation, batch_idx: int) -> Tensor:
    loss, metrics = self.forward(batch)
    self.log_dict(metrics)
    return loss

def configure_optimizers(self):
    return torch.optim.AdamW(self.model.parameters(), lr=self.optimizer_lr)
```

## 8. Validate

Test:

1. eager features, lazy dataset setup, and explicit config construction;
2. unresolved calls leave model and processors absent;
3. unchanged calls preserve model, config, and processor identities;
4. changed normalization refreshes processors while preserving model parameters;
5. missing processors are repaired without rebuilding the model or reinjecting adapters;
6. incompatible checkpoint configs fail before policy mutation;
7. checkpoint round trips restore tensors and normalization without external lookup;
8. one loss, one prediction, output shape, and feature order;
9. optional capabilities and export only when supported.

See [Advanced Patterns](advanced.md) for optional behavior.
