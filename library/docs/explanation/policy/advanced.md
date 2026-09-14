# Advanced Patterns

Add these patterns only after the minimal config, model, policy, and processor flow is
working. Each capability has its own owner and ordering requirement; none should make
the core policy difficult to skim.

## PEFT and LoRA

PEFT crosses config, model, and policy boundaries. Use the existing cooperative mixins
rather than adding policy-specific adapter logic.

```python
@dataclass(frozen=True, kw_only=True)
class MyModelConfig(PeftConfigMixin, Config):
    ...

class MyModel(PeftModelMixin, TemplateModel):
    @classmethod
    def get_default_peft_targets(cls) -> tuple[str, ...]:
        return ("action_head",)

class MyPolicy(PeftPolicyMixin, TemplatePolicy):
    ...
```

Order matters:

1. construct the base model;
2. load pretrained base-model weights, when present;
3. inject adapters when `config.use_lora` is enabled;
4. let Lightning restore checkpoint tensors.

Adapter injection changes state-dict keys, so PEFT configuration is reconstruction
state and belongs in the serialized model config. The model supplies
architecture-specific default target modules; the policy mixin owns lifecycle wiring.

## Gradient Checkpointing

Gradient checkpointing is a training-time model modification. Keep its setting on the
policy unless it changes serialized architecture behavior.

```python
def _apply_model_modifications(self) -> None:
    assert self.model is not None
    if self.gradient_checkpointing:
        self.model.gradient_checkpointing_enable()
```

Apply it after model construction and base-weight loading. Keep the model-specific
enablement mechanism in the model; the policy only decides whether to invoke it.

## Real-Time Chunking

RTC is runtime state rather than base architecture. Use `RTCPolicyMixin` for the
policy lifecycle and checkpointed enabled flag, and `RTCModelMixin` for model-side
behavior.

```python
class MyModel(RTCModelMixin, TemplateModel):
    ...

class MyPolicy(RTCPolicyMixin, TemplatePolicy):
    ...
```

Synchronize RTC after the model exists. The RTC transform consumes the model's full
native action chunk before postprocessing trims it to `n_action_steps`. A model that
returns only the execution horizon removes context RTC may need.

## Checkpoint Restoration

Checkpoint reconstruction follows the architecture invariant: build the final module
structure before Lightning loads tensors.

```python
def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
    assert self._config is not None
    checkpoint["model_config"] = self._config.to_dict()

@classmethod
def load_from_checkpoint(cls, checkpoint_path, **kwargs):
    kwargs["pretrained_name_or_path"] = None
    return super().load_from_checkpoint(checkpoint_path, **kwargs)

def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
    config_data = checkpoint.get("model_config")
    if not isinstance(config_data, Mapping):
        return

    restored = MyModelConfig.from_dict(config_data)
    if self._config is not None and self._config != restored:
        raise ValueError("Checkpoint config does not match the initialized policy")

    self._config = restored
    self.configure_model()
```

In production these cooperative hooks belong in the shared base policy. A concrete
policy extends them only for policy-specific reconstruction state.

The pretrained path must be cleared before Lightning invokes the constructor. Clearing
it in `on_load_checkpoint()` is too late and can trigger an unnecessary artifact
resolution. Restoration loads weights from the checkpoint state dictionary, never
from the original artifact.

## Feature Adaptation

Support post-construction feature adaptation only when it does not change model
architecture. Validate action width, replace the policy-owned feature contract,
rebuild processors, and reset runtime queues. Never silently rebuild the model.

Renaming preserves feature metadata and order. Normalization changes remain processor
state changes; they do not redefine feature identity.

## Exceptional Export Customization

A concrete policy should not contain `to_onnx()`, `to_openvino()`, tracing, or manifest
logic. Put export behavior in a dedicated mixin such as `MyPolicyExportMixin` so the
main policy remains readable.

Start with the public export properties described in [Export API](export.md): schemas,
a raw sample, backend parameters, and supported backends. They cover ordinary policy
customization.

Some models need backend-specific preprocessing that the standard
`ExportablePolicyMixin` flow cannot express. In that exceptional case, the dedicated
export mixin may override `_get_default_export_input_sample()` or, as a last resort, a
backend method. Keep the override narrow and delegate to `super()` after adapting the
sample.

```python
class MyPolicyExportMixin(ExportablePolicyMixin):
    def _get_default_export_input_sample(self):
        sample = super()._get_default_export_input_sample()
        if sample is None:
            return None
        return adapt_trace_inputs(sample)
```

`_get_default_export_input_sample()` is private implementation plumbing, not a routine
policy interface. An override accepts maintenance coupling to the base export flow and
must have a focused export test. Do not add this override to the core policy class.

See [Export API Migration](export-api-migration.md) for moving existing policy-local
export code into this ownership model.
