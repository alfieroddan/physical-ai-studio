# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 policy implementation."""

from pathlib import Path
from typing import Any, Literal, cast

import torch
from physicalai.inference.data import InferenceFeature, InferenceFeatureDtype, InferenceFeatureType
from torch import Tensor

from physicalai.data.constants import IMAGE_MASKS, STATE, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.data.dataset import Dataset
from physicalai.data.observation import ACTION, IMAGES, TASK, Feature, FeatureType, NormalizationParameters, Observation
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.export.backends import ExportParameters, TorchExportParameters
from physicalai.policies.base import Policy

from .config import MolmoAct2Config
from .from_hf import build_config_from_hf_config, load_hf_pretrained_container
from .model import MolmoAct2Model
from .processors import make_molmoact2_preprocessors


def _coerce_dataset_feature(feature: Feature) -> Feature:
    normalization_data = feature.normalization_data
    copied_normalization: NormalizationParameters | None = None
    if normalization_data is not None:
        copied_normalization = NormalizationParameters(
            mean=normalization_data.mean,
            std=normalization_data.std,
            q01=normalization_data.q01,
            q99=normalization_data.q99,
            mask=normalization_data.mask,
        )

    shape = tuple(int(dim) for dim in feature.shape) if feature.shape is not None else ()
    return Feature(
        name=str(feature.name),
        ftype=FeatureType(feature.ftype),
        shape=shape,
        normalization_data=copied_normalization,
    )


def make_molmoact2_config(
    *,
    input_features: list[Feature] | None,
    output_features: list[Feature] | None,
    n_obs_steps: int,
    n_action_steps: int,
    action_mode: Literal["continuous", "discrete", "both"] = "continuous",
    torch_compile: bool = False,
) -> MolmoAct2Config:
    """Create the explicit model config for MolmoAct2.

    This function is the non-policy home for model-definition defaults.

    Args:
        input_features: List of input features the model consumes.
        output_features: List of output features the model produces.
        n_obs_steps: Number of observation steps.
        n_action_steps: Number of action steps.
        action_mode: Action supervision mode.
        torch_compile: Whether to mark the config for optimized inference.

    Returns:
        A fully populated :class:`MolmoAct2Config`.
    """
    return MolmoAct2Config(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        action_mode=action_mode,
        compile_model=torch_compile,
    )


def _as_float_list(value: object) -> list[float]:
    if torch.is_tensor(value):
        return [float(x) for x in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    if isinstance(value, (int, float)):
        return [float(value)]
    msg = f"Unsupported normalization value type: {type(value)}"
    raise TypeError(msg)


def _feature_normalization_mode(feature: Feature) -> str:
    if feature.ftype == FeatureType.VISUAL:
        return "identity"
    norm = feature.normalization_data
    if norm is None:
        return "identity"
    if norm.q01 is not None and norm.q99 is not None:
        return "quantiles"
    if norm.min is not None and norm.max is not None:
        return "min_max"
    if norm.mean is not None and norm.std is not None:
        return "mean_std"
    return "identity"


def _feature_normalization_stats(feature: Feature, mode: str) -> dict[str, list[float]]:
    norm = feature.normalization_data
    if norm is None or mode == "identity":
        return {}
    if mode == "quantiles":
        stats = {"q01": _as_float_list(norm.q01), "q99": _as_float_list(norm.q99)}
    elif mode == "min_max":
        stats = {"min": _as_float_list(norm.min), "max": _as_float_list(norm.max)}
    else:
        stats = {"mean": _as_float_list(norm.mean), "std": _as_float_list(norm.std)}
    if norm.mask is not None:
        stats["mask"] = _as_float_list(norm.mask)
    return stats


class MolmoAct2(ExportablePolicyMixin, Policy):  # pyright: ignore[reportIncompatibleMethodOverride,reportIncompatibleVariableOverride]
    """MolmoAct2 Policy."""

    def __init__(
        self,
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        repo_id: str | Path | None = "allenai/MolmoAct2",
        norm_tag: str | None = None,
        n_obs_steps: int = 30,
        n_action_steps: int = 30,
        action_mode: Literal["continuous", "discrete", "both"] = "continuous",
        *,
        torch_compile: bool = False,
    ) -> None:
        """Initialize a MolmoAct2 policy wrapper.

        Args:
            input_features: Optional observation feature schema.
            output_features: Optional action feature schema.
            repo_id: Optional pretrained checkpoint identifier or path.
            norm_tag: Optional normalization tag for pretrained checkpoints.
            n_obs_steps: Number of observation steps.
            n_action_steps: Number of predicted action steps.
            action_mode: Training/inference action mode.
            torch_compile: Whether to enable compile-oriented config flags.

        Raises:
            ValueError: If only one of input_features/output_features is provided.
            RuntimeError: If pretrained checkpoint metadata cannot be resolved.
        """
        super().__init__(n_action_steps=n_action_steps)

        # TODO(alfieroddan): enable more than just continous action mode
        if action_mode != "continous":
            msg = "Only continous action mode is currently supported."
            raise ValueError(msg)

        # check both exist, raise error if not
        if bool(input_features) != bool(output_features):
            msg = f"Need both input and output features: input: {input_features} - output: {output_features}"
            raise ValueError(msg)

        # if pretrained find hf container
        self.hf_container = None
        if repo_id is not None:
            self.hf_container = load_hf_pretrained_container(repo_id)
            if self.hf_container is None:
                msg = "Failed to resolve pretrained MolmoAct2 checkpoint metadata."
                raise RuntimeError(msg)

        # if self.hf_container exists - we should resolve the config
        if self.hf_container:
            self.config = build_config_from_hf_config(
                self.hf_container.hf_config,
                norm_stats=self.hf_container.norm_stats,
                input_features=input_features,
                output_features=output_features,
                checkpoint_path=self.hf_container.checkpoint_location,
                processor_config=self.hf_container.processor_config,
                n_obs_steps=n_obs_steps,
                norm_tag=norm_tag,
                n_action_steps=n_action_steps,
                action_mode=action_mode,
                torch_compile=torch_compile,
            )
        else:
            self.config = make_molmoact2_config(
                input_features=input_features,
                output_features=output_features,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                action_mode=action_mode,
                torch_compile=torch_compile,
            )

        self._checkpoint_location: str | None = (
            self.hf_container.checkpoint_location if self.hf_container is not None else None
        )

        # Keep repo_id in checkpoint hparams so load_from_checkpoint reconstructs
        # the same pretrained source during inference adapter reload.
        self.save_hyperparameters(ignore=["config"])

        self.model = cast("Any", None)
        self._preprocessor = cast("Any", None)
        self._postprocessor = cast("Any", None)

        # eagerly load
        if (input_features and output_features) or (repo_id and norm_tag):
            self._initialize_model()

    @staticmethod
    def _dataset_features(train_dataset: Dataset) -> tuple[list[Feature], list[Feature]]:
        input_features = [
            _coerce_dataset_feature(feature)
            for feature in train_dataset.observation_features.values()
        ]
        output_features = [
            _coerce_dataset_feature(feature)
            for feature in train_dataset.action_features.values()
        ]
        return input_features, output_features

    def _initialize_model(self) -> None:
        """Initialize the model architecture, preprocessors, and pretrained weights.

        Model construction and weight loading are kept as explicit sequential
        steps so each concern is visible and testable independently:

        1. Build preprocessor/postprocessor from config.
        2. Construct the :class:`MolmoAct2Model` (architecture only, no weights).
        3. Load pretrained weights if a checkpoint path is present in the config.
        """
        # make pre, post and model from config
        self._preprocessor, self._postprocessor = make_molmoact2_preprocessors(config=self.config)
        self.model = MolmoAct2Model(self.config)

    def setup(self, stage: str) -> None:
        """Set up model from datamodule (lazy or fine-tuning path).

        Args:
            stage: Lightning stage identifier (unused; required by the interface).

        Raises:
            TypeError: If the attached train dataset is not a physicalai Dataset.
        """
        del stage
        if self.model is not None:
            return

        if not self.config.input_features or not self.config.output_features:
            datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
            train_dataset = datamodule.train_dataset
            if not isinstance(train_dataset, Dataset):
                msg = f"Expected physicalai Dataset, got {type(train_dataset)}"
                raise TypeError(msg)

            dataset_input_features, dataset_output_features = self._dataset_features(train_dataset)
            if not self.config.input_features:
                self.config.input_features = dataset_input_features
            if not self.config.output_features:
                self.config.output_features = dataset_output_features

        if self.model is None:
            self._initialize_model()

    @torch.no_grad()
    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Predict an action chunk from an observation batch.

        Args:
            batch: Observation batch to run inference on.

        Returns:
            Predicted action tensor of shape
            ``(batch_size, action_horizon, action_dim)``.

        Raises:
            ValueError: If the model or processors ave not been initialized.
        """
        if self.model is None:
            msg = "Model is not initialized. Call setup() first."
            raise ValueError(msg)
        if self._preprocessor is None or self._postprocessor is None:
            msg = "Processors are not initialized. Call setup() first."
            raise ValueError(msg)

        preprocessor = cast("Any", self._preprocessor)
        model = cast("Any", self.model)
        postprocessor = cast("Any", self._postprocessor)

        processed_batch = preprocessor(batch.to_dict())
        actions = model.predict_action_chunk(processed_batch)
        return postprocessor({ACTION: actions})[ACTION]

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Run training or inference forward pass.

        Args:
            batch: Input observation batch.

        Returns:
            Training: tuple of loss tensor and metrics dict.
            Inference: predicted action chunk tensor.

        Raises:
            ValueError: If model or preprocessors are not initialized in training mode.
        """
        if self.training:
            if self.model is None or self._preprocessor is None:
                msg = "Model is not initialized"
                raise ValueError(msg)
            preprocessor = cast("Any", self._preprocessor)
            model = cast("Any", self.model)
            processed_batch = preprocessor(batch.to_dict())
            return model(processed_batch)
        return self.predict_action_chunk(batch)

    def training_step(self, batch: Observation, batch_idx: int) -> torch.Tensor:
        """Lightning training step.

        Returns:
            Training loss tensor.
        """
        del batch_idx
        loss, loss_dict = self(batch)
        self.log("train/loss", loss_dict["loss"], prog_bar=True)
        return loss

    def compute_val_loss(self, batch: Observation) -> tuple[Tensor, dict[str, float]]:
        """Compute validation loss and metrics.

        Args:
            batch: Input observation batch.

        Returns:
            Validation loss tensor and metrics dictionary.

        Raises:
            ValueError: If model or preprocessors are not initialized.
        """
        if self.model is None or self._preprocessor is None:
            msg = "Model is not initialized"
            raise ValueError(msg)

        preprocessor = cast("Any", self._preprocessor)
        model = cast("Any", self.model)
        processed_batch = preprocessor(batch.to_dict())
        return model.compute_val_loss(processed_batch)

    @property
    def input_features(self) -> list[Feature]:
        """Explicit input feature contract.

        Raises:
            ValueError: If the model has not been initialized with input features.
        """
        if self.config.input_features is None:
            msg = "Model has not been initialized, no input features exist yet."
            raise ValueError(msg)
        return self.config.input_features

    @property
    def output_features(self) -> list[Feature]:
        """Explicit output feature contract.

        Raises:
            ValueError: If the model has not been initialized with output features.
        """
        if self.config.output_features is None:
            msg = "Model has not been initialized, no output features exist yet."
            raise ValueError(msg)
        return self.config.output_features

    @property
    def inputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the policy's expected model inputs for export tracing.

        Derived directly from :attr:`config.input_features`. Returns ``None``
        if the model has not yet been initialized.

        Returns:
            A list of :class:`InferenceFeature` descriptors, or ``None``.
        """
        if self.model is None or self.input_features is None:
            return None

        schema: list[InferenceFeature] = []
        for feature in self.input_features:
            if feature.ftype == FeatureType.VISUAL:
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.VISUAL,
                        shape=cast("tuple", feature.shape),
                        name=f"{IMAGES}.{feature.name}",
                        dtype=InferenceFeatureDtype.FLOAT32,
                    ),
                )
            elif feature.ftype == FeatureType.STATE:
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.STATE,
                        shape=cast("tuple", feature.shape),
                        name=str(feature.name),
                        dtype=InferenceFeatureDtype.FLOAT32,
                    ),
                )
            schema.append(
                InferenceFeature(
                    ftype=InferenceFeatureType.LANGUAGE,
                    shape=(),
                    name=TASK,
                    dtype=InferenceFeatureDtype.STRING,
                ),
            )
        return schema

    @property
    def outputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the policy's model output for export.

        Derived directly from :attr:`config.output_features`. Returns ``None``
        if the model has not yet been initialized.

        Returns:
            A list of :class:`InferenceFeature` descriptors, or ``None``.
        """
        if self.model is None or self.output_features is None:
            return None

        return [
            InferenceFeature(
                ftype=InferenceFeatureType.ACTION,
                shape=(self.config.n_action_steps, *cast("tuple", feature.shape)),
                name=ACTION,
                dtype=InferenceFeatureDtype.FLOAT32,
            )
            for feature in self.output_features
        ]

    @property
    def extra_export_args(self) -> dict[str, ExportParameters]:
        """Extra backend export args for inference-time pre/post graph components."""
        normalize_by_mode: dict[str, dict[str, dict[str, list[float]]]] = {}
        for feature in self.config.input_features or []:
            if not feature.name:
                continue
            mode = _feature_normalization_mode(feature)
            normalize_by_mode.setdefault(mode, {})[feature.name] = _feature_normalization_stats(feature, mode)

        action_feature = next((f for f in (self.config.output_features or []) if f.ftype == FeatureType.ACTION), None)
        denorm_by_mode: dict[str, dict[str, dict[str, list[float]]]] = {}
        if action_feature is not None:
            action_mode = _feature_normalization_mode(action_feature)
            action_stats = _feature_normalization_stats(action_feature, action_mode)
            denorm_by_mode.setdefault(action_mode, {})[ACTION] = action_stats

        output_names = [feature.name for feature in (self.outputs_schema or [])]

        return {
            "torch": TorchExportParameters(
                input_names=[TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK, IMAGES, IMAGE_MASKS, STATE],
                output_names=output_names,
            ),
        }

    @staticmethod
    def get_supported_export_backends() -> list[str | ExportBackend]:
        """Get a list of export backends supported by policy.

        This method returns a list of supported export backends as strings.

        Returns:
            list[str | ExportBackend]: A list of supported export backends.
        """
        return [ExportBackend.TORCH]
