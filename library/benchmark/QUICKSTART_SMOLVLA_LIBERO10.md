# Quick Example: SmolVLA + Torch + vla-eval LIBERO-10

This is a minimal end-to-end flow:

1. Export SmolVLA LIBERO weights with PhysicalAI.
2. Start the model server from this folder.
3. Run the LIBERO-10 benchmark from vla-eval.

## 0) Prerequisites

- You already have both repos checked out:
  - /home/alfie/Develop/ar-pai
  - /home/alfie/Develop/vla-evaluation-harness
- Python/uv environments are set up in both repos.
- GPU + Docker available for LIBERO benchmark runs.
- If needed for model download, set HF_TOKEN.

## 1) Export lerobot/smolvla_libero with PhysicalAI (Torch)

From ar-pai library root:

```bash
cd /home/alfie/Develop/ar-pai/library
source .venv/bin/activate

mkdir -p tmp-benchmark/exports

uv run python - <<'PY'
from physicalai.policies import SmolVLA

repo = "lerobot/smolvla_libero"
export_dir = "tmp-benchmark/exports/smolvla_libero_torch"

policy = SmolVLA(pretrained_name_or_path=repo)
policy.eval()
policy.export(export_dir, backend="torch")
print(f"Exported: {export_dir}")
PY
```

Notes:

- This uses SmolVLA LIBERO weights from lerobot/smolvla_libero.
- Torch export is the backend used in this flow.

## 2) Start the inference model server (Terminal A)

Still in ar-pai library root:

```bash
cd /home/alfie/Develop/ar-pai/library
source .venv/bin/activate

uv run tmp-benchmark/inference_model_server.py \
  --args.export_dir tmp-benchmark/exports/smolvla_libero_torch \
  --args.backend torch \
  --args.state_input_name state \
  --args.include_state=true \
  --args.include_all_images=false \
  --args.expected_action_dim=7 \
  --port 8000 \
  --args.image_key image \
  --args.image_input_name images \
  --args.language_input_name task
```

If your exported model expects different input names, adjust:

- --args.image_input_name
- --args.language_input_name
- --args.state_input_name

Tip: inspect tmp-benchmark/exports/smolvla_libero_torch/manifest.json and match
the server names to the input feature names in that file.

## 3) Run LIBERO-10 benchmark (Terminal B)

From vla-eval root:

```bash
cd /home/alfie/Develop/vla-evaluation-harness

# optional: activate your vla-eval env
# source .venv/bin/activate

uv run vla-eval run --config configs/benchmarks/libero/10.yaml
```

This benchmark config already points to ws://localhost:8000, so it should connect to the server from Terminal A.

## 4) Optional: quick connection check

Before full run, you can check server health:

```bash
curl -sS http://localhost:8000/health
```

Expected response contains status ok.

## 5) Troubleshooting

- Connection refused:
  - Confirm Terminal A is running on port 8000.
- Action dimension mismatch:
  - Tune --args.expected_action_dim (7 is common for LIBERO).
- Torch export fails:
  - Ensure the model and torch dependencies are available in the active environment.
- Backend mismatch:
  - Confirm you started the server with --args.backend torch.
- Input-key mismatch:
  - For SmolVLA LIBERO exports, task/state/image names are usually task/state/images.
- Slow first steps:
  - First calls may include model warm-up and backend initialization.
