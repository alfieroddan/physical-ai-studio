# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 policy implementation."""

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from physicalai.inference.data import InferenceFeature, InferenceFeatureDtype, InferenceFeatureType
from physicalai.inference.manifest import ComponentSpec

from physicalai.data.observation import ACTION, IMAGES, TASK, Feature, FeatureType, Observation
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.export.backends import ONNXExportParameters, OpenVINOExportParameters, TorchExportParameters
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
    n_action_steps: int,
) -> MolmoAct2Config:
    """Create the explicit model config for MolmoAct2.

    This function is the non-policy home for model-definition defaults.

    Args:
        input_features: List of input features the model consumes.
        output_features: List of output features the model produces.
        n_obs_steps: Number of observation steps.
        n_action_steps: Number of action steps.

    Returns:
        A fully populated :class:`MolmoAct2Config`.
    """
    return MolmoAct2Config(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
    )


class MolmoAct2(ExportablePolicyMixin, Policy):
    """MolmoAct2 Policy."""

    def __init__(
        self,
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        repo_id: str | Path | None = "allenai/MolmoAct2",
        norm_tag: str | None = "so100_so101_molmoact2",
        n_obs_steps: int = 30,
        n_action_steps: int = 30,
    ) -> None:
        """Initialize MolmoAct2 policy.

        When ``repo_id`` is provided the config is built
        from the HuggingFace checkpoint and ``norm_tag`` is required to resolve
        normalisation statistics. Otherwise a fresh config is built from the
        provided features and step counts.

        Args:
            input_features: Input features the policy consumes. Required.
            output_features: Output features the policy produces. Required.
            repo_id: HuggingFace repo ID or local path to
                a pretrained checkpoint. When given, weights are loaded during
                :meth:`_initialize_model`.
            norm_tag: Tag used to select normalisation statistics from
                ``norm_stats.json`` from repo_id.
            n_obs_steps: Number of observation steps.
            n_action_steps: Number of action steps.

        Raises:
            ValueError: If ``input_features`` or ``output_features`` are not
                provided, or if ``repo_id`` is given
                without a ``norm_tag``.
        """
        if not input_features or not output_features:
            msg = "Model requires input and output features."
            raise ValueError(msg)

        super().__init__(n_action_steps=n_action_steps)

        self.hf_container = None

        if repo_id is not None:
            if not norm_tag:
                msg = "norm_tag is required when loading from HuggingFace to resolve statistics from norm_stats.json."
                raise ValueError(msg)
            self.hf_container = load_hf_pretrained_container(repo_id)
            self.config = build_config_from_hf_config(
                self.hf_container.hf_config,
                norm_stats=self.hf_container.norm_stats,
                input_features=input_features,
                output_features=output_features,
                n_obs_steps=n_obs_steps,
                norm_tag=norm_tag,
                n_action_steps=n_action_steps,
                checkpoint_path=self.hf_container.checkpoint_location,
                processor_config=self.hf_container.processor_config,
            )
        else:
            self.config = make_molmoact2_config(
                input_features=input_features,
                output_features=output_features,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
            )

        self._checkpoint_location: str | None = (
            self.hf_container.checkpoint_location if self.hf_container is not None else None
        )

        # Keep repo_id in checkpoint hparams so load_from_checkpoint reconstructs
        # the same pretrained source during inference adapter reload.
        self.save_hyperparameters(ignore=["config"])

        self.model: MolmoAct2Model | None = None
        self._preprocessor: MolmoAct2Preprocessor | None = None
        self._postprocessor: MolmoAct2Postprocessor | None = None

        # Build module tree eagerly so Lightning load_from_checkpoint can restore
        # state_dict without requiring an explicit setup() call.
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the model architecture, preprocessors, and pretrained weights.

        Model construction and weight loading are kept as explicit sequential
        steps so each concern is visible and testable independently:

        1. Build preprocessor/postprocessor from config.
        2. Construct the :class:`MolmoAct2Model` (architecture only, no weights).
        3. Load pretrained weights if a checkpoint path is present in the config.
        """
        self._preprocessor, self._postprocessor = make_molmoact2_preprocessors(
            config=self.config,
        )

        self.model = MolmoAct2Model(self.config)

        if self._checkpoint_location is not None:
            self.model.load_pretrained_weights(self._checkpoint_location)

    def setup(self, stage: str) -> None:
        """Set up model from datamodule (lazy or fine-tuning path).

        Args:
            stage: Lightning stage identifier (unused; required by the interface).
        """
        del stage
        if self.model is None or self._preprocessor is None or self._postprocessor is None:
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
            ValueError: If the model or processors have not been initialized.
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Call setup() first.")
        if self._preprocessor is None or self._postprocessor is None:
            raise ValueError("Processors are not initialized. Call setup() first.")

        processed_batch = self._preprocessor(batch.to_dict(flatten=True))
        actions = self.model.predict_action_chunk(processed_batch)
        return self._postprocessor(actions)

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Forward pass through the policy.

        Args:
            batch: Observation batch.

        Returns:
            Predicted action tensor, or a tuple of (loss, metrics) during training.
        """
        return self.predict_action_chunk(batch)

    @property
    def input_features(self) -> list[Feature]:
        """Return the explicit input feature contract."""
        return self.config.input_features

    @property
    def output_features(self) -> list[Feature]:
        """Return the explicit output feature contract."""
        return self.config.output_features

    @property
    def inputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the policy's expected model inputs for export tracing.

        Derived directly from :attr:`config.input_features`. Returns ``None``
        if the model has not yet been initialized.

        Returns:
            A list of :class:`InferenceFeature` descriptors, or ``None``.
        """
        if self.model is None:
            return None

        schema: list[InferenceFeature] = []
        for feature in self.config.input_features:
            if feature.ftype == FeatureType.VISUAL:
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.VISUAL,
                        shape=feature.shape,
                        name=f"{IMAGES}.{feature.name}",
                        dtype=InferenceFeatureDtype.FLOAT32,
                    ),
                )
            elif feature.ftype == FeatureType.STATE:
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.STATE,
                        shape=feature.shape,
                        name=feature.name,
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
        if self.model is None:
            return None

        return [
            InferenceFeature(
                ftype=InferenceFeatureType.ACTION,
                shape=(self.config.n_action_steps, *feature.shape),
                name=feature.name,
                dtype=InferenceFeatureDtype.FLOAT32,
            )
            for feature in self.config.output_features
        ]

    @property
    def extra_export_args(self) -> dict[str, object]:
        """Extra backend export args for inference-time pre/post graph components."""

        def _as_float_list(value: object) -> list[float]:
            if torch.is_tensor(value):
                return [float(x) for x in value.detach().cpu().reshape(-1).tolist()]
            if isinstance(value, (list, tuple)):
                return [float(x) for x in value]
            if isinstance(value, (int, float)):
                return [float(value)]
            msg = f"Unsupported normalization value type: {type(value)}"
            raise TypeError(msg)

        state_feature = next((f for f in self.config.input_features if f.ftype == FeatureType.STATE), None)
        action_feature = next((f for f in self.config.output_features if f.ftype == FeatureType.ACTION), None)
        visual_features = [f for f in self.config.input_features if f.ftype == FeatureType.VISUAL and f.name]

        state_stats: dict[str, list[float]] | None = None
        if state_feature is not None and state_feature.normalization_data is not None:
            state_stats = {
                "q01": _as_float_list(state_feature.normalization_data.q01),
                "q99": _as_float_list(state_feature.normalization_data.q99),
            }
            if state_feature.normalization_data.mask is not None:
                state_stats["mask"] = _as_float_list(state_feature.normalization_data.mask)

        action_stats: dict[str, list[float]] | None = None
        if action_feature is not None and action_feature.normalization_data is not None:
            action_stats = {
                "q01": _as_float_list(action_feature.normalization_data.q01),
                "q99": _as_float_list(action_feature.normalization_data.q99),
            }
            if action_feature.normalization_data.mask is not None:
                action_stats["mask"] = _as_float_list(action_feature.normalization_data.mask)

        image_keys = [feature.name for feature in visual_features if feature.name]
        env_action_dim = int(action_feature.shape[0]) if action_feature is not None and action_feature.shape else int(
            self.config.max_action_dim,
        )

        molmoact2_pre = ComponentSpec.model_validate(
            {
                "type": "molmoact2_pre",
                "tokenizer_name_or_path": str(self.config.tokenizer_name_or_path),
                "num_state_tokens": int(self.config.num_state_tokens),
                "setup_type": str(self.config.setup_type or ""),
                "control_mode": str(self.config.control_mode or ""),
                "add_setup_tokens": bool(self.config.add_setup_tokens),
                "add_control_tokens": bool(self.config.add_control_tokens),
                "state_stats": state_stats,
                "image_keys": image_keys,
            },
        )
        molmoact2_post = ComponentSpec.model_validate(
            {
                "type": "molmoact2_post",
                "action_key": ACTION,
                "env_action_dim": env_action_dim,
                "action_stats": action_stats,
            },
        )

        output_names = [feature.name for feature in (self.outputs_schema or [])]

        return {
            "onnx": ONNXExportParameters(
                exporter_kwargs={
                    "output_names": output_names,
                },
                preprocessors_specs=[molmoact2_pre],
                postprocessors_specs=[molmoact2_post],
                export_tokenizer=False,
            ),
            "openvino": OpenVINOExportParameters(
                outputs=output_names,
                preprocessors_specs=[molmoact2_pre],
                postprocessors_specs=[molmoact2_post],
                export_tokenizer=False,
            ),
            "torch": TorchExportParameters(),
        }

    @staticmethod
    def get_supported_export_backends() -> list[str | ExportBackend]:
        """Get a list of export backends supported by policy.

        This method returns a list of supported export backends as strings.

        Returns:
            list[str | ExportBackend]: A list of supported export backends.
        """
        return [ExportBackend.TORCH, ExportBackend.OPENVINO]
