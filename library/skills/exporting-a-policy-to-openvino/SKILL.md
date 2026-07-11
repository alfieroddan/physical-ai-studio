---
name: exporting-a-policy-to-openvino
description: 'Export a physicalai Policy to OpenVINO (and ONNX/Torch) so it loads and runs with the ar-pai InferenceModel. Use when adding export support to a policy, wiring extra_export_args / ComponentSpec preprocessors_specs & postprocessors_specs, writing numpy-side inference preprocessors/postprocessors in ar-pai, registering components in the component_registry, building the manifest, fixing OpenVINO input/output tensor names, static vs dynamic shapes, or debugging InferenceModel KeyError / shape mismatch after export. Keywords: OpenVINO, ONNX, export, ExportablePolicyMixin, to_openvino, convert_model, ComponentSpec, manifest, InferenceModel, component_registry, inference preprocessor, any_name, tracing.'
---

# Exporting a Policy to OpenVINO

## When to Use
- Making an existing `physicalai` Policy exportable to OpenVINO/ONNX/Torch.
- Writing the numpy inference-side pre/post processors in `ar-pai` that reproduce the torch preprocessing outside the model graph.
- Wiring `extra_export_args` with `ComponentSpec` pipelines and registering components.
- Debugging: `InferenceModel` raises `KeyError: Expected input '<num>'`, or a shape mismatch after export.
- Prereq: the policy already exists (see the sibling skill `writing-a-physicalai-policy`).

## Mental Model

Export produces a **package directory**: the traced model (`.xml`/`.onnx`/`.pt`), a `manifest.json`, and any artifacts (e.g. tokenizer). The manifest lists a **runner**, **preprocessors**, and **postprocessors** as `ComponentSpec`s. `InferenceModel(export_dir)` reconstructs the pipeline:

```
numpy obs dict → [preprocessors] → model (OV/ONNX) → [postprocessors] → action
```

Two rules dominate everything:
1. **The traced model must be pure-tensor.** All value-dependent host work (tokenization, resize, patchify, placeholder expansion, per-example batching, padding) happens in preprocessors, never inside the model. If tracing const-folds an input away (data-dependent `.item()`/`.tolist()`/Python branching), export fails or drops inputs.
2. **The numpy inference preprocessor must emit exactly the tensor keys the traced model consumes**, with matching names, dtypes, and shapes. Anything the model doesn't consume must not be emitted.

Reference implementation to copy from: `src/physicalai/policies/molmoact2/policy.py` (torch side) and `ar-pai/src/physicalai/inference/preprocessors/molmoact2*.py` (numpy side). Also `src/physicalai/policies/smolvla/policy.py`.

## Procedure

### 1. Make the Policy exportable
Mix in `ExportablePolicyMixin` (from `physicalai.export`) **before** `Policy`:
```python
class Foo(ExportablePolicyMixin, Policy): ...
```
Implement these on the policy (see [mixin_policy.py](src/physicalai/export/mixin_policy.py)):
- `get_supported_export_backends()` → e.g. `[ExportBackend.TORCH, ExportBackend.OPENVINO]`.
- `inputs_schema` → `list[InferenceFeature]` describing model inputs (VISUAL/STATE/LANGUAGE). Drives the default `sample_input` used for tracing.
- `outputs_schema` → the `ACTION` `InferenceFeature` (shape `(n_action_steps, *action_dim)`).
- `extra_export_args` → per-backend `ExportParameters` with the `ComponentSpec` pipelines (next step).
- The base `sample_input` builds a fake observation from `inputs_schema` (string features become a fixed prompt). Override it only if you need a specific sample.

### 2. Wire `extra_export_args` (the pipeline)
Return `OpenVINOExportParameters` / `TorchExportParameters` / `ONNXExportParameters` carrying `preprocessors_specs` and `postprocessors_specs` as ordered `ComponentSpec`s (from `physicalai.inference.manifest`):
```python
from physicalai.inference.manifest import ComponentSpec

preproc = [
    ComponentSpec(type="normalize", mode="quantiles", stats={<feature>: {...}}),
    ComponentSpec(type="foo_pre", tokenizer_name_or_path=..., <flat kwargs>),
]
postproc = [
    ComponentSpec(type="foo_post"),
    ComponentSpec(type="denormalize", mode="quantiles", stats={ACTION: {...}}),
]
return {
    "openvino": OpenVINOExportParameters(outputs=output_names,
        preprocessors_specs=preproc, postprocessors_specs=postproc),
    "torch": TorchExportParameters(input_names=[...], output_names=output_names,
        preprocessors_specs=preproc, postprocessors_specs=postproc),
}
```
- `ComponentSpec` uses pydantic `extra="allow"`: extra kwargs pass through as flat constructor params. Pyright will warn "No parameter named ..." — this is **expected** and matches every sibling policy (smolvla/pi05/act). Don't fight it.
- `type=` is a short name resolved by the registry (step 4). Order matters: preprocessors run top-to-bottom, postprocessors likewise.
- Built-in component types include: `normalize`, `denormalize`, `resize`, `smolvla_resize`, `new_line`, `hf_tokenizer`, `ov_tokenizer`, `action_chunk_trimmer`.
- Pass any config the numpy component needs as JSON-serializable flat params (token ids, image size dict, pooling, `max_action_dim`, `env_action_dim`, stats). Avoid param keys named `type`/`class_path`.

### 3. Write the numpy inference pre/post (`ar-pai`)
Create numpy equivalents under `ar-pai/src/physicalai/inference/preprocessors/` and `postprocessors/` that mirror the torch pre/post **bit-for-bit in behavior**:
- Preprocessor subclasses `Preprocessor`, `__call__(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]`, emitting the exact model-input keys. Reproduce every host step (resize via cv2 to match torch resize, patchify, placeholder expansion, token type ids, per-example batching, padding masks). Split large logic into a helper module (e.g. `molmoact2_inputs.py`) like the torch `processors/inputs.py`.
- Postprocessor subclasses `Postprocessor`, `__call__(outputs) -> {ACTION: ...}`: clamp + denormalize (+ optional frame transform). Do **not** re-trim action dims the model already trimmed.
- Add to the package `__init__.py` `__all__`.

### 4. Register components in the registry
Add short-name → class-path entries in [component_factory.py](ar-pai/src/physicalai/inference/component_factory.py):
```python
component_registry.register("foo_pre", "physicalai.inference.preprocessors.FooPreprocessor")
component_registry.register("foo_post", "physicalai.inference.postprocessors.FooPostprocessor")
```
The `type` in each `ComponentSpec` must match a registered name.

### 5. Export and load
```python
policy = Foo(repo_id=..., input_features=..., output_features=...); policy.eval()
policy.to_openvino("export_dir")             # writes .xml, manifest.json, artifacts
from physicalai.inference import InferenceModel
action = InferenceModel("export_dir")(numpy_obs)   # dict of numpy arrays in
```
`InferenceModel` matches preprocessor output keys against the OV model's input names (`input_node.any_name`), runs the model, then the postprocessors.

## Common Failures & Fixes

**`KeyError: Expected input '4913'` in `InferenceModel._prepare_inputs`.**
The OV adapter reads input names via `any_name`. If an input tensor is passed through / reused / aliased in the graph (e.g. straight into a submodule or reused across denoising steps), OpenVINO may give its `any_name` a numeric alias even though the friendly name is in `get_names()`. Fix by re-pinning input names **after** export — do it policy-scoped by overriding `to_openvino` (don't edit the shared mixin):
```python
@torch.no_grad()
def to_openvino(self, output_path, input_sample=None, **kw):
    super().to_openvino(output_path, input_sample=input_sample, **kw)
    self._canonicalize_openvino_input_names(output_path, input_sample)
```
`_canonicalize_openvino_input_names` reads the saved `.xml`, and for each input whose `get_names()` contains an expected key (from `input_sample.keys()`) but whose `get_any_name()` differs, calls `model_input.tensor.set_names({name})`, then re-saves. See the working version in [molmoact2/policy.py](src/physicalai/policies/molmoact2/policy.py).

**Shape mismatch, e.g. `model input [1,259]` vs `tensor (1,260)`.**
The traced OV graph has **static** shapes fixed by the tracing `sample_input`. If your sequence length depends on prompt text, a different prompt changes the length and breaks inference. Options: pad the tokenized prompt to a fixed max length (SmolVLA sets `pad_language_to="max_length"`), or export with a dynamic sequence dim. Until fixed, tracing prompt length and inference prompt length must agree.

**An input gets const-folded out of the graph / `torch.export` raises `GuardOnDataDependentSymNode`.**
The model is doing data-dependent host work. Move that work into the preprocessor so the model only sees fixed-shape tensors.

**Only outputs are named, inputs aren't.** The shared mixin's `_postprocess_openvino_model` names outputs; inputs are named by the converter from `example_input` dict keys. Reused/aliased inputs need the policy-scoped re-pin above.

## Validation
- `policy.to_openvino("export_dir")` completes; `export_dir/manifest.json` lists your pre/post `ComponentSpec`s.
- Inspect input names: `for i in ov.Core().read_model("export_dir/<name>.xml").inputs: print(i.any_name)` — all should be friendly names.
- `InferenceModel("export_dir")(numpy_obs)` returns an action of shape `(B, n_action_steps, action_dim)`.
- Numpy vs torch parity: run the same observation through the torch preprocessor and the numpy preprocessor; compare emitted keys, shapes, and values.
- To iterate on inference without a ~200s re-export, load the already-exported `export_dir` with `InferenceModel` directly.
