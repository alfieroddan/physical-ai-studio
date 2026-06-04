"""PhysicalAI InferenceModel adapter for vla-eval model-server protocol.

Run directly:

	uv run inference_model_server.py \
	  --export_dir /path/to/export \
	  --port 8000

Or via vla-eval config:

	script: "/absolute/path/to/inference_model_server.py"
	args:
	  export_dir: "/path/to/export"
	  port: 8000
	  expected_action_dim: 7
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import (
    GRIPPER_CLOSE_POS,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    ROTATION_AA,
    STATE_EEF_POS_AA_GRIP,
    DimSpec,
)
from vla_eval.types import Action, Observation

from physicalai.inference import InferenceModel

logger = logging.getLogger(__name__)


def _normalize_args_prefix(argv: list[str]) -> list[str]:
    """Accept both direct and wrapper-style CLI args.

    - Direct script usage expects flags like ``--export_dir``.
    - ``vla-eval serve`` forwards constructor kwargs as ``--args.export_dir``.

    Also normalize explicit boolean assignments (``--flag=true/false``) into
    jsonargparse-compatible yes/no flags (``--flag`` / ``--no-flag``).
    """
    normalized: list[str] = []
    for token in argv:
        if token.startswith("--args."):
            token = "--" + token[len("--args.") :]

        if token.startswith("--") and "=" in token:
            key, value = token.split("=", 1)
            lower = value.lower()
            if lower == "true":
                normalized.append(key)
                continue
            if lower == "false":
                name = key[2:]
                normalized.append(f"--no-{name}")
                continue

        normalized.append(token)
    return normalized


class PhysicalAIInferenceModelServer(PredictModelServer):
    """vla-eval model server backed by ``physicalai.inference.InferenceModel``.

    This adapter accepts canonical vla-eval observations (``images``,
    ``task_description``, ``state``/``states``), converts them to the input
    dictionary expected by exported PhysicalAI policies, and returns action
    vectors in the vla-eval wire format (``{"actions": np.ndarray}``).
    """

    def __init__(
        self,
        export_dir: str,
        policy_name: str | None = None,
        backend: str = "auto",
        device: str = "auto",
        *,
        image_key: str | None = None,
        image_input_name: str = "image",
        language_input_name: str = "language",
        state_input_name: str = "state",
        include_state: bool = True,
        include_all_images: bool = False,
        include_raw_observation: bool = False,
        expected_action_dim: int | None = None,
        pad_or_truncate_action: bool = True,
        observation_params: dict[str, Any] | None = None,
        declare_observation_spec: bool = False,
        observation_image_keys: list[str] | None = None,
        observation_language_key: str | None = None,
        observation_state_key: str | None = None,
        chunk_size: int = 1,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)

        self.image_key = image_key
        self.image_input_name = image_input_name
        self.language_input_name = language_input_name
        self.state_input_name = state_input_name
        self.include_state = include_state
        self.include_all_images = include_all_images
        self.include_raw_observation = include_raw_observation
        self.expected_action_dim = expected_action_dim
        self.pad_or_truncate_action = pad_or_truncate_action
        self.declare_observation_spec = declare_observation_spec
        self.observation_image_keys = observation_image_keys
        self.observation_language_key = observation_language_key
        self.observation_state_key = observation_state_key
        derived_obs_params: dict[str, Any] = {"send_state": include_state}
        if include_all_images:
            # LIBERO controls whether wrist images are emitted via this parameter.
            derived_obs_params["send_wrist_image"] = True
        if observation_params:
            derived_obs_params.update(observation_params)
        self.observation_params = derived_obs_params
        self._state_dim_hint: int | None = None
        self._visual_paths_hint: list[list[str]] = []
        self._load_manifest_hints(export_dir)

        logger.info(
            "Loading InferenceModel(export_dir=%s, policy_name=%s, backend=%s, device=%s)",
            export_dir,
            policy_name,
            backend,
            device,
        )
        self._model = InferenceModel.load(
            export_dir,
            policy_name=policy_name,
            backend=backend,
            device=device,
        )
        logger.info("Loaded PhysicalAI policy: %s", self._model)

    def _load_manifest_hints(self, export_dir: str) -> None:
        manifest_path = Path(export_dir) / "manifest.json"
        if not manifest_path.exists():
            return

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to parse manifest at %s", manifest_path)
            return

        input_features = manifest.get("model", {}).get("input_features", [])
        for feature in input_features:
            init_args = feature.get("init_args", {})
            name = init_args.get("name")
            ftype = init_args.get("ftype")
            shape = init_args.get("shape", [])

            if ftype == "STATE" and name == self.state_input_name and shape:
                try:
                    self._state_dim_hint = int(shape[0])
                except Exception:
                    self._state_dim_hint = None

            if ftype == "VISUAL" and isinstance(name, str):
                prefix = f"{self.image_input_name}."
                if name.startswith(prefix):
                    self._visual_paths_hint.append(name[len(prefix) :].split("."))

        if self._state_dim_hint is not None:
            logger.info("Manifest hint: expected %s dim=%d", self.state_input_name, self._state_dim_hint)
        if self._visual_paths_hint:
            logger.info("Manifest hint: visual paths under %s -> %s", self.image_input_name, self._visual_paths_hint)

    def get_observation_params(self) -> dict[str, Any]:
        return dict(self.observation_params)

    def get_action_spec(self) -> dict[str, DimSpec]:
        # Advertise canonical 7-DoF semantics used by LIBERO-like benchmarks.
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_AA,
            "gripper": GRIPPER_CLOSE_POS,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        if not self.declare_observation_spec:
            return {}

        spec: dict[str, DimSpec] = {}
        for key in self.observation_image_keys or []:
            spec[key] = IMAGE_RGB

        if self.observation_language_key:
            spec[self.observation_language_key] = LANGUAGE

        if self.include_state and self.observation_state_key:
            spec[self.observation_state_key] = STATE_EEF_POS_AA_GRIP

        return spec

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        await super().on_episode_start(config, ctx)
        self._model.reset()

    def _choose_image(self, obs: Observation) -> np.ndarray:
        direct = obs.get("image")
        if direct is not None:
            return np.asarray(direct, dtype=np.uint8)

        images = obs.get("images", {})
        if isinstance(images, dict) and images:
            if self.image_key and self.image_key in images:
                chosen = images[self.image_key]
            elif "agentview" in images:
                chosen = images["agentview"]
            else:
                chosen = next(iter(images.values()))
            return np.asarray(chosen, dtype=np.uint8)
        return np.zeros((256, 256, 3), dtype=np.uint8)

    @staticmethod
    def _to_bchw_float(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)

        if arr.ndim == 3:
            if arr.shape[-1] in (1, 3, 4):
                if arr.shape[-1] == 4:
                    arr = arr[..., :3]
                arr = np.transpose(arr, (2, 0, 1))
            elif arr.shape[0] in (1, 3, 4):
                if arr.shape[0] == 4:
                    arr = arr[:3, ...]
            else:
                msg = f"Unsupported 3D image shape: {arr.shape}"
                raise ValueError(msg)
            arr = arr[None, ...]
        elif arr.ndim == 4:
            if arr.shape[-1] in (1, 3, 4):
                if arr.shape[-1] == 4:
                    arr = arr[..., :3]
                arr = np.transpose(arr, (0, 3, 1, 2))
            elif arr.shape[1] in (1, 3, 4):
                if arr.shape[1] == 4:
                    arr = arr[:, :3, ...]
            else:
                msg = f"Unsupported 4D image shape: {arr.shape}"
                raise ValueError(msg)
        else:
            msg = f"Unsupported image ndim={arr.ndim}, shape={arr.shape}"
            raise ValueError(msg)

        arr = arr.astype(np.float32, copy=False)
        if arr.size and arr.max() > 1.0:
            arr = arr / 255.0
        return arr

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
        target = self._state_dim_hint
        if target is not None:
            if arr.shape[0] == target:
                pass
            elif arr.shape[0] > target:
                arr = arr[:target]
            else:
                pad = np.zeros(target - arr.shape[0], dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=0)

        # SmolVLA expects batched state at inference time (B, D).
        return arr[None, :]

    def _build_visual_payload(self, obs: Observation) -> Any:
        primary = self._to_bchw_float(self._choose_image(obs))

        # SmolVLA LIBERO exports typically expect flattened keys after
        # Observation.to_dict(), e.g. images.images.camera1/camera2/camera3.
        if self.image_input_name == "images":
            paths = self._visual_paths_hint or [
                ["images", "camera1"],
                ["images", "camera2"],
                ["images", "camera3"],
            ]
            # Keep a single-level dict so Observation.to_dict() doesn't leave nested
            # dict values under flattened image keys.
            root: dict[str, Any] = {}
            for path in paths:
                root[".".join(path)] = primary
            return root

        return primary

    def _build_model_observation(self, obs: Observation) -> dict[str, Any]:
        model_obs: dict[str, Any] = {
            self.image_input_name: self._build_visual_payload(obs),
            self.language_input_name: str(obs.get("task_description", obs.get("language", ""))),
        }

        if self.include_state:
            state = obs.get("state", obs.get("states"))
            if state is not None:
                model_obs[self.state_input_name] = self._normalize_state(np.asarray(state, dtype=np.float32))

        if self.include_all_images:
            images = obs.get("images")
            if isinstance(images, dict) and self.image_input_name != "images":
                model_obs["images"] = {k: np.asarray(v, dtype=np.uint8) for k, v in images.items()}

        if self.include_raw_observation:
            model_obs["observation"] = obs

        return model_obs

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.expected_action_dim is None:
            return arr

        if arr.shape[0] == self.expected_action_dim:
            return arr

        if not self.pad_or_truncate_action:
            msg = f"Action dimension mismatch: got {arr.shape[0]}, expected {self.expected_action_dim}"
            raise ValueError(msg)

        if arr.shape[0] > self.expected_action_dim:
            return arr[: self.expected_action_dim]

        pad = np.zeros(self.expected_action_dim - arr.shape[0], dtype=np.float32)
        return np.concatenate([arr, pad], axis=0)

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        model_obs = self._build_model_observation(obs)
        action = self._model.select_action(model_obs)
        return {"actions": self._normalize_action(action)}


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    sys.argv = [sys.argv[0], *_normalize_args_prefix(sys.argv[1:])]

    run_server(PhysicalAIInferenceModelServer)
