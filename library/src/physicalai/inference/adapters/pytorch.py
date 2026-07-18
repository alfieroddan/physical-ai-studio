# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# This module extends ``physicalai.inference.adapters`` module and corresponding
# namespace according to PEP 420. ``__init__.py`` is missing intentionally.
# ruff: noqa: INP001

"""Torch runtime adapter for inference."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from physicalai.inference.adapters.base import RuntimeAdapter
from physicalai.inference.adapters.registry import adapter_registry
from physicalai.inference.manifest import Manifest

from physicalai.config import import_class
from physicalai.data.observation import Observation
from physicalai.export.backends import TorchExportParameters

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from physicalai.policies.base import Policy


@adapter_registry.register("torch", extensions=(".ckpt", ".pt"))
class TorchAdapter(RuntimeAdapter):
    """Runtime adapter for Torch models.

    This adapter loads and runs models exported via `to_torch()`
    using PyTorch's API.

    Example:
        >>> adapter = TorchAdapter()
        >>> adapter.load("model.pt")
        >>> outputs = adapter.predict({"state": state_array, "images": images_dict})
    """

    def __init__(self, device: torch.device | str = "cpu") -> None:
        """Initialize the Torch adapter.

        Args:
            device: Device for inference ('cpu', 'cuda', 'xpu', etc.)
        """
        self.device = str(device)
        self._policy: Policy | None = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []

    def load(self, model_path: Path | str) -> None:
        """Load Torch model from file.

        Args:
            model_path: Path to the .pt file created by torch.save()

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
            KeyError: If manifest is missing required entries
            TypeError: If imported policy class is missing callable load_from_checkpoint()
        """
        model_path = Path(model_path)
        if not model_path.exists():
            msg = f"Model file not found: {model_path}"
            raise FileNotFoundError(msg)
        manifest = Manifest.load(model_path.parent)
        policy_class_path = manifest.policy.source.class_path
        if not policy_class_path:
            msg = "Manifest missing 'policy.source.class_path' entry."
            raise KeyError(msg)
        try:
            checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
            checkpoint_dtype = self._infer_checkpoint_dtype(checkpoint)
            policy_class = import_class(policy_class_path)
            load_from_checkpoint = getattr(policy_class, "load_from_checkpoint", None)
            if not callable(load_from_checkpoint):
                msg = f"Imported class '{policy_class_path}' does not define callable load_from_checkpoint()."
                raise TypeError(msg)  # noqa: TRY301
            load_from_checkpoint = cast(
                "Callable[..., Policy]",
                load_from_checkpoint,
            )
            del checkpoint  # only needed it for dtype inference above; Lightning reloads the file itself

            # load_weights=False skips the pretrained HF checkpoint pull inside
            # _initialize_model(), avoiding the transient memory spike of holding
            # both pretrained weights and this checkpoint's weights at once.
            # The checkpoint's own state_dict (loaded normally by Lightning
            # right after __init__) supplies every real parameter value anyway.
            self._policy = load_from_checkpoint(
                model_path,
                map_location="cpu",
                weights_only=False,
                load_weights=False,
            )
            if checkpoint_dtype is not None:
                self._policy = self._policy.to(device=self.device, dtype=checkpoint_dtype).eval()
            else:
                self._policy = self._policy.to(self.device).eval()
            extra_export_args = cast(
                "dict[str, Any]",
                getattr(self._policy, "extra_export_args", {}),
            )
            if "torch" in extra_export_args:
                torch_export_args = cast("TorchExportParameters", extra_export_args["torch"])
            else:
                torch_export_args = TorchExportParameters()
            self._output_names = list(torch_export_args.output_names)
            self._input_names = []
        except Exception as e:
            msg = f"Failed to load Torch model from {model_path}: {e}"
            raise RuntimeError(msg) from e

    @staticmethod
    def _infer_checkpoint_dtype(checkpoint: dict[str, Any]) -> torch.dtype | None:
        """Infer a representative parameter dtype from a saved checkpoint state_dict.

        Returns:
            The dtype of the first floating-point parameter in the checkpoint
            state_dict, or ``None`` if no floating-point parameter is found.
        """
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            return None
        for value in state_dict.values():
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                return value.dtype
        return None

    def predict(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Run inference using Torch.

        Args:
            inputs: Dictionary mapping input names to numpy arrays
                (or dicts of numpy arrays for nested fields like images).

        Returns:
            Dictionary mapping output names to numpy arrays

        Raises:
            RuntimeError: If model is not loaded or inference fails
        """
        if self._policy is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        try:
            # Build Observation from numpy dict and convert to torch tensors on device
            observation = Observation.from_dict(inputs).to_torch(self.device)

            torch_outputs = self._policy(observation)
            return self._convert_outputs_to_numpy(torch_outputs)

        except Exception as e:
            msg = f"Inference failed: {e}"
            raise RuntimeError(msg) from e

    def _convert_outputs_to_numpy(self, torch_outputs: torch.Tensor | dict | list | tuple) -> dict[str, np.ndarray]:
        """Convert model outputs to numpy format.

        Args:
            torch_outputs: Model outputs (tensor, dict, list, or tuple)

        Returns:
            Dictionary mapping output names to numpy arrays

        Raises:
            TypeError: If output type is unexpected
        """

        def to_numpy(value: torch.Tensor) -> np.ndarray:
            if value.dtype == torch.bfloat16:
                value = value.to(dtype=torch.float32)
            return value.detach().cpu().numpy()

        if isinstance(torch_outputs, torch.Tensor):
            # Single output
            return {self._output_names[0]: to_numpy(torch_outputs)}
        if isinstance(torch_outputs, dict):
            # Dict output
            return {k: to_numpy(v) if isinstance(v, torch.Tensor) else v for k, v in torch_outputs.items()}
        if isinstance(torch_outputs, (list, tuple)):
            # Multiple outputs as list/tuple
            outputs_iter = zip(self._output_names, torch_outputs, strict=True)
            return {name: to_numpy(out) for name, out in outputs_iter}

        # Unexpected output type
        msg = f"Unexpected output type: {type(torch_outputs)}"
        raise TypeError(msg)

    @property
    def input_names(self) -> list[str]:
        """Get model input names.

        Returns:
            List of input tensor names

        Raises:
            RuntimeError: If model is not loaded
        """
        if self._policy is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)
        return self._input_names

    @property
    def output_names(self) -> list[str]:
        """Get model output names.

        Returns:
            List of output tensor names

        Raises:
            RuntimeError: If model is not loaded
        """
        if self._policy is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)
        return self._output_names
