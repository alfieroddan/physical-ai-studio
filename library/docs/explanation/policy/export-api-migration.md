# Export API Migration

This guide moves existing policy export implementations toward the new ownership
model. It does not rename the current properties or change them into methods. The
migration is structural: export logic leaves the core policy and becomes a focused,
testable export mixin driven by the policy-owned config.

## Target Shape

```text
policy.py
    -> construction, training settings, configure_model, optimizer

export.py
    -> MyPolicyExportMixin
    -> inputs_schema, outputs_schema, sample_input
    -> extra_export_args, supported backends
    -> exceptional tracing/backend overrides
```

The concrete policy composes the mixin but does not contain export implementation
details.

## Ownership Mapping

| Existing implementation | Target owner |
| --- | --- |
| Feature names/shapes rebuilt from `dataset_stats` | Policy config; export schemas derive from ordered config features |
| `inputs_schema` in the core policy | Dedicated policy export mixin |
| `outputs_schema` in the core policy | Dedicated policy export mixin |
| Policy-specific `sample_input` | Dedicated policy export mixin; derive from schema when possible |
| `extra_export_args` in the core policy | Dedicated policy export mixin |
| Output-name lookup in backend overrides | `outputs_schema` and backend `ExportParameters` |
| Chunk trimming inside exported model output | Manifest postprocessor from `extra_export_args` |
| Custom raw-to-trace input adaptation | `sample_input`; exceptionally `_get_default_export_input_sample()` in export mixin |
| Policy-local `to_onnx()` or `to_openvino()` | Inherited backend method; retain a narrow export-mixin override only when hooks cannot express the flow |
| Supported backend assumptions | Explicit `get_supported_export_backends()` declaration |

## Migration Steps

1. Ensure the policy owns a resolved, serializable config with ordered input and output
   features.
2. Create a policy-specific export module and mixin.
3. Move `inputs_schema`, `outputs_schema`, `sample_input`, `extra_export_args`, and
   `get_supported_export_backends()` into that mixin.
4. Replace `dataset_stats` feature discovery with config-derived names, shapes, types,
   and order.
5. Keep normalization values in processor or export-component state rather than using
   them to reconstruct the feature contract.
6. Make schema properties return `None` while config is unavailable, but raise a
   targeted error when a resolved config violates the export contract.
7. Express output names, dynamic axes, preprocessors, postprocessors, and stable
   backend settings through `extra_export_args`.
8. Remove concrete-policy backend overrides that only inject parameters or reshape
   metadata.
9. If custom trace-input adaptation remains necessary, move the smallest possible
   override into the export mixin and delegate to the base implementation.
10. Verify that model construction and training do not access export properties.

## Validation Checklist

For each supported backend, verify:

- `inputs_schema` order matches raw runtime input order;
- `outputs_schema` order matches model output and manifest order;
- absent config returns `None` rather than an unrelated attribute error;
- invalid resolved features raise a message naming the violated export contract;
- the default `sample_input` has correct names, shapes, dtypes, and device;
- preprocessing produces the exact tensor dictionary accepted by tracing;
- `extra_export_args` selects the expected output names and manifest components;
- `chunk_size != n_action_steps` adds the required action-chunk trimmer;
- every advertised backend exports successfully;
- unsupported backends are not advertised;
- checkpoint-loaded policies export without consulting the original pretrained
  artifact;
- the core policy remains understandable without opening the export module.

## Compatibility

The property names and access style remain unchanged:

```python
policy.inputs_schema
policy.outputs_schema
policy.sample_input
policy.extra_export_args
```

This lets migration happen policy by policy without changing shared call sites. The
meaning becomes stricter: these are export-lifecycle properties derived from resolved
policy state, not generic training schemas or passive fields.

## Exceptional Overrides

Some existing policies customize `to_onnx()`, `to_openvino()`, or
`_get_default_export_input_sample()` because their tracing inputs differ from ordinary
runtime inputs. Do not mechanically delete those overrides. First determine whether
the behavior can move into:

1. `inputs_schema` for raw names, shapes, and dtypes;
2. `sample_input` for representative raw values;
3. the policy preprocessor for standard raw-to-model conversion;
4. `extra_export_args` for backend and manifest parameters.

Only behavior that cannot fit those hooks should remain as a narrow override in the
policy-specific export mixin, with a focused test explaining the exception.

See [Export API](export.md) for the target hook lifecycle.
