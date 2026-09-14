# Implement a Policy

This guide shows the smallest useful policy shape. It keeps optional capabilities and
export customizations out of the main flow so a new author can see model construction,
training, and inference at a glance.

## 1. Define the Config

The policy-owned config contains ordered feature contracts, horizons, and architecture
values. It is serializable; the model never stores it.

```python
@dataclass(frozen=True, kw_only=True)
class MyModelConfig(Config):
    input_features: list[Feature]
    output_features: list[Feature]
    action_dim: int
    hidden_size: int = 512
    chunk_size: int = 32
    n_action_steps: int = 32
```

Keep optimizer settings, artifact paths, normalization values, and export destinations
outside this config.

## 2. Implement the Model

Use a flat constructor and implement loss plus full-chunk prediction:

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
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.backbone = build_backbone(hidden_size)
        self.action_head = nn.Linear(hidden_size, chunk_size * action_dim)

    def compute_loss(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor | float]]:
        prediction = self._predict(batch)
        loss = F.mse_loss(prediction, batch["action"])
        return loss, {"loss": loss}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        return self._predict(batch)
```

The output shape is `(batch_size, chunk_size, action_dim)`. Leave execution-horizon
trimming and environment-dimension adaptation to policy-owned processing.

The inherited `forward()` is equivalent to this illustrative wrapper:

```python
# Illustrative only: the base model provides this standard dispatch.
def forward(self, batch):
    if self.training:
        return self.compute_loss(batch)
    return self.predict_action_chunk(batch)
```

Override temporal delta-index properties only when the model consumes temporal
context. See [Required Interfaces](interfaces.md#temporal-delta-indices).

## 3. Implement the Policy

The concrete policy stores unresolved inputs and training settings, then delegates all
model creation to `configure_model()`:

```python
class MyPolicy(TemplatePolicy):
    def __init__(
        self,
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        pretrained_name_or_path: str | Path | None = None,
        *,
        n_action_steps: int = 32,
        chunk_size: int = 32,
        optimizer_lr: float = 1e-4,
    ) -> None:
        super().__init__(n_action_steps=n_action_steps)
        self._input_features = input_features
        self._output_features = output_features
        self._pretrained_name_or_path = pretrained_name_or_path
        self._n_action_steps = n_action_steps
        self._chunk_size = chunk_size
        self.optimizer_lr = optimizer_lr
        self._config: MyModelConfig | None = None
        self._preprocessor = None
        self._postprocessor = None

        if input_features is not None and output_features is not None:
            self.configure_model()
```

A direct config route sets the complete config and enters the same materialization
method:

```python
@classmethod
def from_config(cls, config: MyModelConfig, **policy_options) -> "MyPolicy":
    policy = cls(
        n_action_steps=config.n_action_steps,
        **policy_options,
    )
    policy._config = config
    policy.configure_model()
    return policy
```

## 4. Materialize Once

Keep `configure_model()` linear and easy to scan:

```python
def configure_model(self) -> None:
    if self.model is not None:
        return

    config, weights_path = self._resolve_config_and_weights()
    self._config = config
    self.model = MyModel.from_config(config)
    self._preprocessor, self._postprocessor = make_policy_processors(config)

    if weights_path is not None:
        self.model.load_weights(weights_path)
```

`_resolve_config_and_weights()` returns the existing explicit config, resolves a
pretrained artifact, or builds a fresh config from complete constructor inputs. It
does not construct the model.

The processor factory is visible at the boundary but its implementation belongs in
processor modules. It consumes the feature contract and separately supplied
normalization state.

## 5. Configure the Optimizer

Before optimizer construction, implement `setup(stage)` as the explicit data-policy
boundary. For `"fit"`, obtain ordered observation and action features from the
training dataset and validate them against the resolved config, or retain them for
lazy `configure_model()` materialization. Do not infer feature identity from
`dataset_stats`.

```python
def configure_optimizers(self):
    assert self.model is not None
    return torch.optim.AdamW(self.model.parameters(), lr=self.optimizer_lr)
```

Training settings remain policy-owned because they do not reconstruct the network.

## 6. Implement the Policy Flow

Keep the runtime and training methods explicit and linear in the concrete policy:

```python
def forward(self, batch: Observation):
    prepared = self._preprocessor(batch)
    if self.training:
        return self.model.compute_loss(prepared)
    return self.predict_action_chunk(batch)

def predict_action_chunk(self, batch: Observation) -> Tensor:
    prepared = self._preprocessor(batch)
    chunk = self.model.predict_action_chunk(prepared)
    return self._postprocessor(chunk)

def training_step(self, batch: Observation, batch_idx: int) -> Tensor:
    loss, metrics = self.forward(batch)
    self.log_dict(metrics)
    return loss
```

These methods make the policy-specific processor and model flow visible without
pulling their implementation details into the policy. The base class still owns
checkpoint plumbing and action-queue behavior.

## 7. Add a Pretrained Route When Needed

A pretrained resolver translates artifact metadata into the same config type and
returns weights separately:

```python
def _resolve_config_from_hf(
    pretrained_name_or_path: str | Path,
) -> tuple[MyModelConfig, Path]:
    config = read_artifact_config(pretrained_name_or_path)
    weights_path = resolve_weights(pretrained_name_or_path)
    return config, weights_path
```

The resolver does not initialize the model. `configure_model()` remains the only
materialization path and loads base weights only after the final architecture exists.

## 8. Validate the Policy

Test the smallest behavioral contract first:

1. construct from an explicit config;
2. call `configure_model()` twice and verify object identity is unchanged;
3. run one training loss and one full-chunk prediction;
4. verify output shape and configured feature order;
5. save and restore a checkpoint without resolving the pretrained artifact;
6. add focused capability or export tests only when those features are supported.

For optional behavior, continue with [Advanced Patterns](advanced.md). Export-capable
policies should use a dedicated export mixin described in [Export API](export.md).
