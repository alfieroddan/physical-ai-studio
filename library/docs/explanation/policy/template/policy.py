# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning policy definition for the native policy design."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from physicalai.data import Feature, FeatureType
from physicalai.data.dataset import Dataset
from physicalai.data.observation import STATE
from physicalai.policies.mixins import PeftPolicyMixin, RTCPolicyMixin

from .base import TemplatePolicy
from .config import NewPolicyModelConfig, resolve_action_dim
from .export import NewPolicyExportMixin
from .model import NewPolicyModel
from .processor import NewPolicyPostprocessor, NewPolicyPreprocessor, make_policy_processors


class NewPolicy(PeftPolicyMixin, RTCPolicyMixin, NewPolicyExportMixin, TemplatePolicy):  # type: ignore[misc]
    """Policy design, fake transormer-based model, and training loop for demonstration purposes."""
    def __init__(
        self,
        # input and output features are eager init
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        # pretrained checkpoint path or name, if any
        pretrained_name_or_path: str | Path | None = None,
        *,
        # model args
        n_action_steps: int = 32,
        chunk_size: int = 32,
        # weights args
        gradient_checkpointing: bool = False,
        lora_enabled: bool = False,
        rtc_enabled: bool = False,
        # training args
        optimizer_lr: float = 1e-4,
        optimizer_weight_decay: float = 0.01,
    ) -> None:

        # model params
        self._input_features = input_features
        self._output_features = output_features
        self._pretrained_name_or_path = pretrained_name_or_path
        self._n_action_steps = n_action_steps
        self._chunk_size = chunk_size

        # initialize Policy with n_action_steps for action queue
        super().__init__(n_action_steps=n_action_steps)

        # Checkpoints restore the resolved config, never mutable features or artifact locations.
        self.save_hyperparameters(ignore=["input_features", "output_features", "pretrained_name_or_path"])

        # training params
        self.gradient_checkpointing = gradient_checkpointing
        self.lora_enabled = lora_enabled
        self.rtc_enabled = rtc_enabled
        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay

        # model is initialized to None by the base Policy.__init__ above
        self._config = None

        # processors
        self._preprocessor: NewPolicyPreprocessor | None = None
        self._postprocessor: NewPolicyPostprocessor | None = None

        if pretrained_name_or_path is not None or (
            input_features is not None and output_features is not None
        ):
            self.configure_model()

    @classmethod
    def from_config(
        cls,
        config: NewPolicyModelConfig,
        *,
        gradient_checkpointing: bool = False,
        optimizer_lr: float = 1e-4,
        optimizer_weight_decay: float = 0.01,
    ) -> "NewPolicy":
        policy = cls(
            pretrained_name_or_path=None,
            n_action_steps=config.n_action_steps,
            gradient_checkpointing=gradient_checkpointing,
            optimizer_lr=optimizer_lr,
            optimizer_weight_decay=optimizer_weight_decay,
        )

        policy._config = config
        policy.configure_model()
        return policy

    def _apply_model_modifications(self) -> None:
        assert isinstance(self.model, NewPolicyModel)

        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        assert self._config is not None
        if self._config.use_lora:
            self._inject_lora()

        self._sync_rtc_to_model()

    @property
    def config(self) -> NewPolicyModelConfig:
        if self._config is None:
            raise RuntimeError("Policy config is not initialized")
        return self._config

    @staticmethod
    def _resolve_config_from_hf(
        pretrained_name_or_path: str | Path,
    ) -> tuple[NewPolicyModelConfig, Path]:
        """Fake resolver standing in for downloading and parsing Hugging Face checkpoint artifacts."""
        fake_input_features = [
            Feature(name=STATE, shape=(4,), ftype=FeatureType.STATE),
            Feature(name="front", shape=(3, 16, 16), ftype=FeatureType.VISUAL),
        ]
        fake_output_features = [Feature(name="action", shape=(2,), ftype=FeatureType.ACTION)]
        config = NewPolicyModelConfig(
            input_features=fake_input_features,
            output_features=fake_output_features,
            action_dim=resolve_action_dim(fake_output_features),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            vocab_size=32,
            chunk_size=3,
            image_size=(16, 16),
            tokenizer_max_length=6,
        )
        weights_path = Path(str(pretrained_name_or_path)) / "model.safetensors"
        return config, weights_path

    def configure_model(self) -> None:
        """Create the model once in Lightning's strategy and precision aware context."""
        if self.model is not None:
            return

        # when loading a pretrained checkpoint, keep the checkpoint config but replace only the
        # feature contract and action horizon that are known at policy construction time.
        if self._config is not None:
            config = self._config
            weights_path = None
        elif self._pretrained_name_or_path is not None:
            pretrained_config, weights_path = self._resolve_config_from_hf(self._pretrained_name_or_path)
            resolved_output_features = (
                self._output_features
                if self._output_features is not None
                else pretrained_config.output_features
            )
            config = replace(
                pretrained_config,
                input_features=(
                    self._input_features
                    if self._input_features is not None
                    else pretrained_config.input_features
                ),
                output_features=resolved_output_features,
                action_dim=resolve_action_dim(resolved_output_features),
                n_action_steps=self._n_action_steps,
                lora_enabled=self.lora_enabled,
            )
        else:
            if self._input_features is None or self._output_features is None:
                return

            weights_path = None
            # build the model config from the values already available on the policy; leave the rest
            # of the model defaults to the config/dataclass defaults.
            config = NewPolicyModelConfig(
                input_features=self._input_features,
                output_features=self._output_features,
                action_dim=resolve_action_dim(self._output_features),
                chunk_size=self._chunk_size,
                n_action_steps=self._n_action_steps,
                lora_enabled=self.lora_enabled,
            )

        self._config = config
        self._input_features = config.input_features
        self._output_features = config.output_features
        self._n_action_steps = config.n_action_steps
        self._chunk_size = config.chunk_size
        self.model = NewPolicyModel.from_config(config)
        self._preprocessor, self._postprocessor = make_policy_processors(config)  # type: ignore[assignment]

        if weights_path is not None:
            self.model.load_weights(weights_path)

        self._apply_model_modifications()

    def setup(self, stage: str) -> None:
        """Set up the model from the training dataset."""
        if stage != "fit":
            return

        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
        train_dataset = datamodule.train_dataset
        if not isinstance(train_dataset, Dataset):
            raise TypeError(f"Expected physicalai Dataset, got {type(train_dataset)}")

        dataset_input_features = list(train_dataset.observation_features.values())
        dataset_output_features = list(train_dataset.action_features.values())

        if self._config is not None:
            if (
                self._config.input_features != dataset_input_features
                or self._config.output_features != dataset_output_features
            ):
                self.set_features(dataset_input_features, dataset_output_features)
            return

        self._input_features = dataset_input_features
        self._output_features = dataset_output_features

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )
