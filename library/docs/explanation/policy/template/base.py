# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Local subclasses of the physicalai base Policy and Model.

These wrap the physicalai base classes so new, template-specific behavior can
be layered on without modifying physicalai source. NewPolicy and NewPolicyModel
inherit from these instead of the physicalai base classes directly.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable, Mapping
from dataclasses import replace
from os import PathLike
from pathlib import Path
from typing import IO, Any, Self, cast

from jsonargparse import FromConfigMixin
import torch
from torch import Tensor

from physicalai.data import Feature, Observation
from physicalai.policies.base import Model as BaseModel
from physicalai.policies.base import Policy as BasePolicy

from .config import NewPolicyModelConfig, resolve_action_dim
from .processor import NewPolicyPostprocessor, NewPolicyPreprocessor, make_policy_processors


class TemplateModel(BaseModel, FromConfigMixin):
    """Model base with non-strict jsonargparse config construction."""

    @classmethod
    def from_config(cls, config: object) -> Self:
        if dataclasses.is_dataclass(config) and not isinstance(config, type):
            values = {field.name: getattr(config, field.name) for field in dataclasses.fields(config)}
        elif isinstance(config, Mapping):
            values = dict(config)
        else:
            return super().from_config(cast("str | PathLike[str]", config))

        parameters = inspect.signature(cls.__init__).parameters
        return super().from_config({name: value for name, value in values.items() if name in parameters})

class TemplatePolicy(BasePolicy):
    """Extension point for template-specific Policy behavior."""

    _config: NewPolicyModelConfig | None
    _preprocessor: NewPolicyPreprocessor | None
    _postprocessor: NewPolicyPostprocessor | None

    @property
    def _config_available(self) -> bool:
        """Whether the policy has resolved a model config."""
        return self._config is not None

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | IO[bytes],
        map_location: torch.device | str | int | Callable | dict | None = None,
        hparams_file: str | Path | None = None,
        strict: bool | None = None,  # noqa: FBT001
        weights_only: bool | None = None,  # noqa: FBT001
        **kwargs: Any,  # noqa: ANN401
    ) -> Self:
        """Load a Lightning checkpoint without resolving pretrained artifacts."""
        kwargs["pretrained_name_or_path"] = None
        return super().load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=strict,
            weights_only=weights_only,
            **kwargs,
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self._config is None:
            raise RuntimeError("Policy config is not initialized")
        checkpoint["model_config"] = self._config.to_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        config_data = checkpoint.get("model_config")
        if not isinstance(config_data, Mapping):
            return

        resolved_config = NewPolicyModelConfig.from_dict(config_data)
        if self._config is not None:
            if self._config != resolved_config:
                raise ValueError("Checkpoint feature contract does not match the initialized policy")
            return

        self._config = resolved_config
        self.configure_model()

    def set_features(
        self,
        input_features: list[Feature],
        output_features: list[Feature],
    ) -> None:
        """Replace the feature contract and rebuild processors without rebuilding the model."""
        if self.model is None or self._config is None:
            raise RuntimeError("Policy model is not initialized")

        action_dim = resolve_action_dim(output_features)
        if action_dim != self._config.action_dim:
            raise ValueError(
                f"Output width {action_dim} does not match model action width {self._config.action_dim}"
            )

        config = replace(
            self._config,
            input_features=list(input_features),
            output_features=list(output_features),
            action_dim=action_dim,
        )
        self._config = config
        self._input_features = config.input_features
        self._output_features = config.output_features
        self._preprocessor, self._postprocessor = make_policy_processors(config)
        self.reset()

    def rename_features(self, mapping: Mapping[str, str]) -> None:
        """Rename resolved input features without changing their metadata or order."""
        if self._config is None:
            raise RuntimeError("Policy config is not initialized")
        if not mapping:
            return
        if any(not isinstance(name, str) or not name for name in mapping):
            raise ValueError("Source feature names must be non-empty strings")
        if any(not isinstance(name, str) or not name for name in mapping.values()):
            raise ValueError("Replacement feature names must be non-empty strings")

        current_names = {feature.name for feature in self._config.input_features}
        unknown_names = sorted(set(mapping) - current_names)
        if unknown_names:
            raise ValueError(f"Cannot rename unknown input features: {unknown_names}")

        input_features = [
            replace(feature, name=mapping[feature.name])
            if feature.name is not None and feature.name in mapping
            else feature
            for feature in self._config.input_features
        ]
        names = [feature.name for feature in input_features]
        if len(names) != len(set(names)):
            raise ValueError(f"Feature renaming creates duplicate input names: {names}")

        self.set_features(input_features, self._config.output_features)

    def _prepare_batch(self, batch: Observation, *, require_actions: bool) -> dict[str, Tensor]:
        if self._preprocessor is None:
            raise RuntimeError("Policy is not initialized")
        processed = self._preprocessor(batch.to_dict())
        if require_actions:
            if not isinstance(batch.action, Tensor):
                raise TypeError("Expected Observation.action to contain action targets")
            processed["action"] = self._preprocessor.normalize_actions(batch.action)
        return processed

    def forward(self, batch: Observation) -> Tensor | tuple[Tensor, dict[str, Tensor | float]]:
        if not isinstance(self.model, TemplateModel):
            raise RuntimeError("Policy model is not initialized")
        if self.training:
            return self.model(self._prepare_batch(batch, require_actions=True))
        return self.predict_action_chunk(batch)

    def compute_val_loss(self, batch: Observation) -> tuple[Tensor, dict[str, Tensor | float]]:
        if not isinstance(self.model, TemplateModel):
            raise RuntimeError("Policy model is not initialized")
        return self.model.compute_val_loss(self._prepare_batch(batch, require_actions=True))

    def predict_action_chunk(self, batch: Observation) -> Tensor:
        if not isinstance(self.model, TemplateModel) or self._postprocessor is None:
            raise RuntimeError("Policy is not initialized")
        actions = cast("Any", self.model).predict_action_chunk(
            self._prepare_batch(batch, require_actions=False)
        )
        return self._postprocessor(actions)

    def training_step(self, batch: Observation, batch_idx: int) -> Tensor:
        del batch_idx
        result = self(batch)
        if not isinstance(result, tuple):
            raise RuntimeError("Training forward must return loss and metrics")
        loss, metrics = result
        self.log("train/loss", metrics["loss"], prog_bar=True)
        return loss
