# Advanced Patterns

Add these after the minimal policy works. Keep each feature in its owning mixin.

## PEFT and LoRA

Use all three PEFT layers:

```python
class MyModelConfig(PeftConfigMixin, Config):
    ...

class MyModel(PeftModelMixin, TemplateModel):
    @classmethod
    def get_default_peft_targets(cls) -> tuple[str, ...]:
        return ("action_head",)

class MyPolicy(PeftPolicyMixin, TemplatePolicy):
    ...
```

Order:

1. Create the base model.
2. Load pretrained base weights.
3. Inject adapters.
4. Load checkpoint tensors.

PEFT settings belong in the saved config. Default target modules belong on the model.

## Gradient Checkpointing

Keep enablement on the model. The policy only invokes it.

```python
if self.gradient_checkpointing:
    self.model.gradient_checkpointing_enable()
```

Apply it after model creation and base-weight loading.

## Real-Time Chunking

Use `RTCPolicyMixin` and `RTCModelMixin`. Sync state after model creation.

RTC must receive the full action chunk. Trim to `n_action_steps` afterward.

## Checkpoint Restoration

The shared policy base owns the normal hooks:

```python
def on_save_checkpoint(self, checkpoint):
    checkpoint["model_config"] = self.config.to_dict()

@classmethod
def load_from_checkpoint(cls, path, **kwargs):
    kwargs["pretrained_name_or_path"] = None
    return super().load_from_checkpoint(path, **kwargs)

def on_load_checkpoint(self, checkpoint):
    config = self.config_type.from_dict(checkpoint["model_config"])
    if self.model is not None and _without_normalization(config) != _without_normalization(self._config):
        raise ValueError("Checkpoint config does not match the initialized model.")
    self._config = config
    self._input_features = config.input_features
    self._output_features = config.output_features
    self.configure_model()
```

This sketch assumes each policy declares `config_type` and uses the
[`_without_normalization()` comparison](how-to.md#1-config). Input and output features
already carry normalization data, so serializing the config also saves those values.
There is no separate normalization checkpoint field or policy state object.
Validate compatibility before assigning saved state, then call `configure_model()`
directly. The hook does not create modules or need a separate restoration helper.
The check permits statistics-only changes and rejects incompatible architecture or
feature contracts. Policies with explicit feature adaptation can use their narrower
compatibility check instead.

Clear the pretrained path and disable external weight initialization before
construction. Build the final module structure before Lightning loads tensors. Encoder
config lookup must also work offline; disabling pretrained weights alone does not
guarantee this. Save the architecture metadata required to reconstruct the encoder.

If the model already exists and is compatible, preserve it and refresh only stale or
missing processors. Do not reload base weights, reinject adapters, or reset unchanged
runtime state. Reject incompatible architecture and ordered feature contracts before
mutation. Extend shared hooks only for policy-specific state.

## Feature Adaptation

Allow feature changes only when model architecture stays valid.

1. Validate action width and all other model-dependent feature constraints, including
   camera routing and feature order.
2. Replace the config feature lists through an explicit adaptation path.
3. Call `configure_model()` to synchronize processors without rebuilding the model.
4. Reset runtime queues when the meaning of queued actions changes.

Normalization-only updates do not change feature identity or model architecture.
Synchronize changed normalization through the same processor-refresh path. Preserve
processor identity when the full config, including feature normalization, is unchanged.

Renaming must preserve metadata and order.

## Export Exceptions

Keep export code in a dedicated policy export mixin.

Use schemas, samples, backend parameters, and supported backends first. Override
`_get_default_export_input_sample()` only when those hooks cannot prepare tracing
inputs. Override a backend method only as a last resort.

```python
class MyPolicyExportMixin(ExportablePolicyMixin):
    def _get_default_export_input_sample(self):
        sample = super()._get_default_export_input_sample()
        return None if sample is None else adapt_trace_inputs(sample)
```

Private-hook overrides need focused export tests.

See [Policy Migration](migration.md) for migration order.
