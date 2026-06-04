#!/usr/bin/env bash
set -euo pipefail

# Export-if-missing + serve script for SmolVLA LIBERO.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBRARY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PORT="${PORT:-8000}"
BACKEND="${BACKEND:-torch}"
FORCE_EXPORT="${FORCE_EXPORT:-false}"

REPO_ID="lerobot/smolvla_libero"
POLICY_CLASS="SmolVLA"
EXPORT_DIR="tmp-benchmark/exports/smolvla_libero_${BACKEND}"

if [[ ! -x "$LIBRARY_ROOT/.venv/bin/python" ]]; then
  echo "ERROR: Python venv not found at $LIBRARY_ROOT/.venv"
  exit 1
fi

run_export() {
  echo "[export] ${POLICY_CLASS} from ${REPO_ID} -> ${EXPORT_DIR}"
  (
    cd "$LIBRARY_ROOT"
    uv run python - <<PY
from physicalai.policies import ${POLICY_CLASS}

repo = "${REPO_ID}"
export_dir = "${EXPORT_DIR}"

policy = ${POLICY_CLASS}(pretrained_name_or_path=repo)
policy.eval()
policy.export(export_dir, backend="${BACKEND}")
print(f"Exported: {export_dir}")
PY
  )
}

ensure_export() {
  local manifest_path="$LIBRARY_ROOT/$EXPORT_DIR/manifest.json"
  if [[ "$FORCE_EXPORT" == "true" || ! -f "$manifest_path" ]]; then
    run_export
    return
  fi
  echo "[export] found existing export at ${EXPORT_DIR} (set FORCE_EXPORT=true to rebuild)"
}

derive_server_args_from_manifest() {
  "$LIBRARY_ROOT/.venv/bin/python" - <<PY
import json
from pathlib import Path

manifest = json.loads(Path("${EXPORT_DIR}", "manifest.json").read_text(encoding="utf-8"))
input_features = manifest.get("model", {}).get("input_features", [])
output_features = manifest.get("model", {}).get("output_features", [])

image_input_name = "images"
language_input_name = "task"
state_input_name = "state"
include_state = False
expected_action_dim = 7

for feat in input_features:
    init = feat.get("init_args", {})
    ftype = str(init.get("ftype", "")).upper()
    name = init.get("name")
    if not isinstance(name, str):
        continue
    if ftype == "VISUAL":
        image_input_name = name.split(".")[0]
    elif ftype == "LANGUAGE":
        language_input_name = name
    elif ftype == "STATE":
        state_input_name = name
        include_state = True

for feat in output_features:
    init = feat.get("init_args", {})
    ftype = str(init.get("ftype", "")).upper()
    shape = init.get("shape", [])
    if ftype == "ACTION" and isinstance(shape, list) and shape:
        expected_action_dim = int(shape[-1])

print(f"image_input_name={image_input_name}")
print(f"language_input_name={language_input_name}")
print(f"state_input_name={state_input_name}")
print(f"include_state={'true' if include_state else 'false'}")
print(f"expected_action_dim={expected_action_dim}")
PY
}

run_server() {
  local args
  args="$(derive_server_args_from_manifest)"
  eval "$args"

  echo "[serve] export=${EXPORT_DIR} backend=${BACKEND} port=${PORT}"
  echo "[serve] image_input_name=${image_input_name} language_input_name=${language_input_name} state_input_name=${state_input_name} include_state=${include_state} expected_action_dim=${expected_action_dim}"

  (
    cd "$LIBRARY_ROOT"
    uv run tmp-benchmark/inference_model_server.py \
      --args.export_dir "$EXPORT_DIR" \
      --args.backend "$BACKEND" \
      --args.image_input_name "$image_input_name" \
      --args.language_input_name "$language_input_name" \
      --args.state_input_name "$state_input_name" \
      --args.include_state="$include_state" \
      --args.include_all_images=false \
      --args.expected_action_dim="$expected_action_dim" \
      --port "$PORT"
  )
}

ensure_export
run_server
