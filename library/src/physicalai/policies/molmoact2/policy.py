# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 policy implementation."""

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from physicalai.data.observation import Feature, Observation
from physicalai.policies.base import Policy

from .config import MolmoAct2Config
from .from_hf import build_config_from_hf_config, load_hf_pretrained_container
from .model import MolmoAct2Model
from .processors import make_molmoact2_preprocessors

if TYPE_CHECKING:
    from .processors import MolmoAct2Postprocessor, MolmoAct2Preprocessor


def make_molmoact2_config(
    *,
    input_features: list[Feature],
    output_features: list[Feature],
    n_obs_steps: int,
    chunk_size: int,
    n_action_steps: int,
    max_action_dim: int,
    tokenizer_name_or_path: str | None = None,
    processor_assets_path: str | None = None,
    processor_config: dict[str, object] | None = None,
    setup_type: str = "",
    control_mode: str = "",
) -> MolmoAct2Config:
    """Create the explicit model config for MolmoAct2.

    This function is the non-policy home for model-definition defaults.
    """
    return MolmoAct2Config(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=n_obs_steps,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        max_action_dim=max_action_dim,
        tokenizer_name_or_path=tokenizer_name_or_path,
        processor_assets_path=processor_assets_path,
        processor_config=processor_config,
        setup_type=setup_type,
        control_mode=control_mode,
    )


class MolmoAct2(Policy):
    """MolmoAct2 Policy."""

    def __init__(
        self,
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        hf_repo_id_or_pretrained_path: str | Path | None = None,
        norm_tag: str | None = None,
        n_obs_steps: int = 30,
        chunk_size: int = 30,
        n_action_steps: int = 30,
        max_action_dim: int = 32,
        tokenizer_name_or_path: str | None = None,
        processor_config: dict[str, object] | None = None,
        setup_type: str = "",
        control_mode: str = "",
        *,
        config_filename: str = "config.json",
        norm_stats_filename: str = "norm_stats.json",
        processor_filename: str = "processor_config.json",
    ) -> None:
        """Initialize MolmoAct2 policy.

        Raises:
            ValueError: If required features are missing or pretrained norm tag is not provided.
        """
        if not input_features or not output_features:
            msg_str = "Model requires input and output features."
            raise ValueError(msg_str)

        super().__init__(n_action_steps=n_action_steps)

        self.hf_container = None

        if hf_repo_id_or_pretrained_path is not None:
            if not norm_tag:
                msg_str = "If loading from HuggingFace, norm_tag is required to load stats from norm_stats.json."
                raise ValueError(msg_str)

            self.hf_container = load_hf_pretrained_container(
                hf_repo_id_or_pretrained_path,
                norm_stats_filename=norm_stats_filename,
                config_filename=config_filename,
                processor_filename=processor_filename,
            )
            self.config = build_config_from_hf_config(
                self.hf_container.hf_config,
                norm_stats=self.hf_container.norm_stats,
                input_features=input_features,
                output_features=output_features,
                n_obs_steps=n_obs_steps,
                chunk_size=chunk_size,
                norm_tag=norm_tag,
                n_action_steps=n_action_steps,
                max_action_dim=max_action_dim,
                checkpoint_path=self.hf_container.checkpoint_location,
                processor_config=self.hf_container.processor_config,
            )
        else:
            self.config = make_molmoact2_config(
                input_features=input_features,
                output_features=output_features,
                n_obs_steps=n_obs_steps,
                chunk_size=chunk_size,
                n_action_steps=n_action_steps,
                max_action_dim=max_action_dim,
                tokenizer_name_or_path=tokenizer_name_or_path,
                processor_config=processor_config,
                setup_type=setup_type,
                control_mode=control_mode,
            )

        self.save_hyperparameters(ignore=["config", "hf_repo_id_or_pretrained_path"])

        self._model: MolmoAct2Model | None = None
        self._preprocessor: MolmoAct2Preprocessor | None = None
        self._postprocessor: MolmoAct2Postprocessor | None = None

        self.missing_keys: list[str] = []
        self.unexpected_keys: list[str] = []

    @property
    def input_features(self) -> list[Feature]:
        """Return the explicit input feature contract."""
        return self.config.input_features

    @property
    def output_features(self) -> list[Feature]:
        """Return the explicit output feature contract."""
        return self.config.output_features

    def _initialize_model(self) -> None:
        """Initialize the underlying model and preprocessors."""
        self._preprocessor, self._postprocessor = make_molmoact2_preprocessors(
            config=self.config,
        )

        self._model = MolmoAct2Model(self.config)

    def setup(self, stage: str) -> None:
        """Set up model from datamodule (lazy or fine-tuning path)."""
        del stage
        self._initialize_model()

    @torch.no_grad()
    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Predict an action chunk from an observation batch."""
        if self._model is None:
            msg = "Model is not initialized"
            raise ValueError(msg)
        if self._preprocessor is None or self._postprocessor is None:
            msg = "Processors are not initialized"
            raise ValueError(msg)

        processed_batch = self._preprocessor(batch.to_dict(flatten=True))
        actions = self._model.predict_action_chunk(processed_batch)
        return self._postprocessor(actions)

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Forward pass through the policy."""
        return self.predict_action_chunk(batch)
