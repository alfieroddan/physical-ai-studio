# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 policy implementation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huggingface_hub import snapshot_download

from physicalai.data.observation import Feature, FeatureType, NormalizationParameters
from physicalai.export import ExportablePolicyMixin
from physicalai.policies.base import Policy

from .config import MolmoAct2Config
from .model import MolmoAct2Model
from .pretrained_utils import (
    ACTION_EXPERT_CONFIG_MAP,
    ADAPTER_CONFIG_MAP,
    TEXT_CONFIG_MAP,
    TOP_LEVEL_CONFIG_MAP,
    VISION_CONFIG_MAP,
    copy_component,
)
from .processors import make_policy_processors

if TYPE_CHECKING:
    from .processors import MolmoAct2PostProcessor, MolmoAct2PreProcessor


class MolmoAct2(ExportablePolicyMixin, Policy):
    """MolmoAct2 policy wrapper for loading pretrained checkpoints and configs.

    Args:
        input_features: Input feature definitions used when initializing a local model.
        output_features: Output feature definitions used when initializing a local model.
        pretrained_name_or_path: Local path or Hugging Face repository ID for the pretrained
            checkpoint.
        norm_tag: Normalization tag identifying the dataset-specific normalization metadata.
        n_action_steps: Number of action steps predicted by the policy.
        chunk_size: Number of actions included in each action chunk.
        n_obs_steps: Number of observation steps included in the input history.
        setup_type: Optional setup identifier used by the model configuration.
        control_mode: Optional control mode used by the model configuration.
        adapt_to_so101: Whether to enable SO101-specific adaptation behavior.

    Returns:
        None: The policy is created and, when possible, initializes its model lazily.

    Raises:
        RuntimeError: If local initialization is requested without both input and output
            features.
    """

    def __init__(
        self,
        # Input and output features
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        # Pretrained model and normalization tag
        pretrained_name_or_path: str | Path | None = "allenai/MolmoAct2",
        norm_tag: str | None = None,
        *,
        # Action and observation parameters
        n_action_steps: int = 30,
        chunk_size: int = 30,
        n_obs_steps: int = 1,
        setup_type: str | None = None,
        control_mode: str | None = None,
        adapt_to_so101: bool = False,
        gradient_checkpointing: bool = False,
        use_lora: bool = False,
        train_action_head_only: bool = False,
    ) -> None:
        """Initialize a MolmoAct2 policy instance.

        Args:
            input_features: Input feature definitions used when initializing a local model.
            output_features: Output feature definitions used when initializing a local model.
            pretrained_name_or_path: Local path or Hugging Face repo ID for the pretrained
                checkpoint.
            norm_tag: Normalization tag identifying the dataset-specific normalization metadata.
            n_action_steps: Number of action steps predicted by the policy.
            chunk_size: Number of actions included in each action chunk.
            n_obs_steps: Number of observation steps included in the input history.
            setup_type: Optional setup identifier used by the model configuration.
            control_mode: Optional control mode used by the model configuration.
            adapt_to_so101: Whether to enable SO101-specific adaptation behavior.
            gradient_checkpointing: Whether to enable gradient checkpointing on the model.
            use_lora: Whether to enable LoRA adapters on the model.
            train_action_head_only: Whether to freeze the VLM and train only the action head.
        """
        # args
        self._input_features = input_features
        self._output_features = output_features
        self._pretrained_name_or_path = pretrained_name_or_path
        self._norm_tag = norm_tag
        self._n_action_steps = n_action_steps
        self._chunk_size = chunk_size
        self._n_obs_steps = n_obs_steps
        self._setup_type = setup_type
        self._control_mode = control_mode
        self._adapt_to_so101 = adapt_to_so101 or norm_tag == "so100_so101_molmoact2"
        self.gradient_checkpointing = gradient_checkpointing
        self.use_lora = use_lora
        self.train_action_head_only = train_action_head_only

        # initialize super
        super().__init__(n_action_steps=self._n_action_steps)

        # ignore input and output features, subject to change
        self.save_hyperparameters(ignore=["input_features", "output_features"])

        # pre and post processors
        self._preprocessor: MolmoAct2PreProcessor | None = None
        self._postprocessor: MolmoAct2PostProcessor | None = None

        # underlying model
        self._model: MolmoAct2Model | None = None

        # only init if features are resolved, lazy otherwise
        user_eager = input_features is not None and output_features is not None
        pretrained_eager = pretrained_name_or_path is not None and norm_tag is not None
        if user_eager or pretrained_eager:
            self.initialize_model()

    def _require_model(self) -> MolmoAct2Model:
        if not isinstance(self._model, MolmoAct2Model):
            msg = "Policy model is not initialized"
            raise TypeError(msg)
        return self._model

    def initialize_model(self) -> None:
        """Initialize the policy model and configuration from pretrained assets or local inputs.

        Args:
            None: This method reads the instance state and does not accept parameters.

        Raises:
            RuntimeError: If the instance is configured for local initialization without required
                input or output feature definitions.
        """
        # initialize model from pretrained if available
        if self._pretrained_name_or_path:
            # gather configs and weights from path (hf hub)
            hf_config, norm_stats_config, tokenizer_config, weights_path = self._from_hf(
                self._pretrained_name_or_path,
            )
            config = self._convert_config(
                hf_config,
                norm_stats_config,
                tokenizer_config,
                weights_path.parent,
            )
        else:
            if self._input_features is None or self._output_features is None:
                msg = "Input and output features are required to initialize MolmoAct2 without pretrained data."
                raise RuntimeError(msg)
            weights_path = None
            config = MolmoAct2Config(
                input_features=self._input_features,
                output_features=self._output_features,
                n_obs_steps=self._n_obs_steps,
                chunk_size=self._chunk_size,
                n_action_steps=self._n_action_steps,
                setup_type=self._setup_type or "",
                control_mode=self._control_mode or "",
                adapt_to_so101=self._adapt_to_so101,
            )

        # resulting config and weights path
        self.config = config
        self._weights_path = weights_path

        # init model
        self._initialize_from_config(config, weights_path=weights_path)

    def _initialize_from_config(
        self,
        config: MolmoAct2Config,
        *,
        weights_path: Path | None = None,
    ) -> None:
        if self._model is not None:
            msg = "Policy model is already initialized"
            raise RuntimeError(msg)

        # update instance attributes from config
        self._input_features = config.input_features
        self._output_features = config.output_features
        self._n_action_steps = config.n_action_steps
        self._chunk_size = config.chunk_size
        self._n_obs_steps = config.n_obs_steps
        self._setup_type = config.setup_type
        self._control_mode = config.control_mode
        self._adapt_to_so101 = config.adapt_to_so101

        self._model = MolmoAct2Model.from_config(config)
        self._preprocessor, self._postprocessor = make_policy_processors(config)  # type: ignore[assignment]

        if weights_path is not None:
            self._model.load_weights(weights_path)

        self._apply_model_modifications()

    def _apply_model_modifications(self) -> None:
        model = self._require_model()

        if self.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        if self.use_lora:
            model.enable_lora()

        if self.train_action_head_only:
            model.freeze_vlm()

    @staticmethod
    def _normalization_parameters(stats: dict[str, Any]) -> NormalizationParameters:
        """Build normalization metadata from saved statistics for a feature.

        Args:
            stats: Dictionary of saved normalization statistics for a single feature.

        Returns:
            NormalizationParameters: The normalized feature metadata populated from the input
                statistics.
        """
        return NormalizationParameters(
            mean=stats.get("mean"),
            std=stats.get("std"),
            min=stats.get("min"),
            max=stats.get("max"),
            q01=stats.get("q01"),
            q99=stats.get("q99"),
            mask=stats.get("mask"),
        )

    @staticmethod
    def _feature_size(stats: dict[str, Any], feature_key: str) -> int:
        """Infer the vector size for a feature from its saved normalization statistics.

        Args:
            stats: Dictionary of normalization statistics associated with a feature.
            feature_key: Name of the feature being inspected.

        Returns:
            int: The length of the feature's vector-valued normalization statistics.

        Raises:
            ValueError: If the statistics do not contain a vector-valued array for the feature.
        """
        for stat_name in ("mean", "std", "min", "max", "q01", "q99"):
            value = stats.get(stat_name)
            if isinstance(value, list):
                return len(value)
        msg = f"MolmoAct2 normalization stats for {feature_key!r} contain no vector values."
        raise ValueError(msg)

    def _resolve_norm_tag(self, norm_stats_config: dict[str, Any]) -> dict[str, Any]:
        """Return the metadata for the selected normalization tag.

        Args:
            norm_stats_config: Dictionary containing the pretrained normalization statistics.

        Returns:
            dict[str, Any]: The metadata payload associated with the configured normalization tag.

        Raises:
            ValueError: If no normalization tag has been configured for the policy.
            TypeError: If the normalization metadata is missing or malformed.
        """
        if self._norm_tag is None:
            msg = "Normalization tag is required when loading pretrained MolmoAct2 data."
            raise ValueError(msg)
        metadata_by_tag = norm_stats_config.get("metadata_by_tag")
        if not isinstance(metadata_by_tag, dict):
            msg = "MolmoAct2 norm stats are missing metadata_by_tag."
            raise TypeError(msg)
        tag_metadata = metadata_by_tag.get(self._norm_tag)
        if tag_metadata is None:
            msg = f"Normalization tag {self._norm_tag!r} was not found in MolmoAct2 norm stats."
            raise ValueError(msg)
        if not isinstance(tag_metadata, dict):
            msg = f"Normalization metadata for tag {self._norm_tag!r} is not a JSON object."
            raise TypeError(msg)
        return tag_metadata

    def _create_features_from_norm_stats(
        self,
        tag_metadata: dict[str, Any],
        image_size: tuple[int, int],
    ) -> tuple[list[Feature], list[Feature]]:
        """Create input and output feature definitions from normalization metadata.

        Args:
            tag_metadata: Metadata describing the selected normalization tag.
            image_size: Spatial dimensions used to construct visual feature shapes.

        Returns:
            tuple[list[Feature], list[Feature]]: Input and output feature definitions derived from
                the normalization metadata.

        Raises:
            TypeError: If camera, state, or action metadata is missing or malformed.
        """
        camera_keys = tag_metadata.get("camera_keys")
        if not isinstance(camera_keys, list) or not all(isinstance(key, str) for key in camera_keys):
            msg = f"Invalid camera_keys for normalization tag {self._norm_tag!r}."
            raise TypeError(msg)

        input_features = [
            Feature(
                name=camera_key.removeprefix("observation.images."),
                ftype=FeatureType.VISUAL,
                shape=(3, *image_size),
            )
            for camera_key in camera_keys
        ]

        state_key = tag_metadata.get("state_key")
        state_stats = tag_metadata.get("state_stats")
        if not isinstance(state_key, str) or not isinstance(state_stats, dict):
            msg = f"Invalid state metadata for normalization tag {self._norm_tag!r}."
            raise TypeError(msg)
        input_features.append(
            Feature(
                name=state_key.removeprefix("observation."),
                ftype=FeatureType.STATE,
                shape=(self._feature_size(state_stats, state_key),),
                normalization_data=self._normalization_parameters(state_stats),
            ),
        )

        action_key = tag_metadata.get("action_key")
        action_stats = tag_metadata.get("action_stats")
        if not isinstance(action_key, str) or not isinstance(action_stats, dict):
            msg = f"Invalid action metadata for normalization tag {self._norm_tag!r}."
            raise TypeError(msg)
        output_features = [
            Feature(
                name=action_key,
                ftype=FeatureType.ACTION,
                shape=(self._feature_size(action_stats, action_key),),
                normalization_data=self._normalization_parameters(action_stats),
            ),
        ]
        return input_features, output_features

    def _convert_config(
        self,
        hf_config: dict[str, Any],
        norm_stats_config: dict[str, Any],
        tokenizer_config: dict[str, Any],
        snapshot_dir: Path,
    ) -> MolmoAct2Config:
        """Convert Hugging Face metadata into the library's MolmoAct2 config object.

        Args:
            hf_config: Raw Hugging Face configuration dictionary.
            norm_stats_config: Normalization statistics metadata loaded from the checkpoint.
            tokenizer_config: Tokenizer configuration loaded from the checkpoint.
            snapshot_dir: Directory containing the checkpoint snapshot.

        Returns:
            MolmoAct2Config: The converted configuration object used by the policy.

        Raises:
            TypeError: If the normalization metadata is malformed for the selected tag or action
                horizon.
        """
        flat_config: dict[str, Any] = {}
        copy_component(hf_config, flat_config, "text_config", TEXT_CONFIG_MAP)
        copy_component(hf_config, flat_config, "vit_config", VISION_CONFIG_MAP)
        copy_component(hf_config, flat_config, "adapter_config", ADAPTER_CONFIG_MAP)
        copy_component(hf_config, flat_config, "action_expert_config", ACTION_EXPERT_CONFIG_MAP)
        copy_component(hf_config, flat_config, None, TOP_LEVEL_CONFIG_MAP)

        # convert lists to tuples
        for tuple_field in ("image_default_input_size", "adapter_vit_layers"):
            value = flat_config.get(tuple_field)
            if isinstance(value, list):
                flat_config[tuple_field] = tuple(value)

        # create config from flattened configuration
        config = MolmoAct2Config(**flat_config)

        # determine normalization mode based on norm_stats_config
        normalization_modes = {
            "q01_q99": "QUANTILES",
            "mean_std": "MEAN_STD",
        }
        norm_mode = norm_stats_config.get("norm_mode")
        normalization_mode = normalization_modes.get(str(norm_mode), config.normalization_mode)

        input_features = self._input_features
        output_features = self._output_features
        chunk_size = self._chunk_size
        normalize_gripper = config.normalize_gripper
        setup_type = self._setup_type or config.setup_type
        control_mode = self._control_mode or config.control_mode

        if self._norm_tag is not None:
            tag_metadata = self._resolve_norm_tag(norm_stats_config)
            input_features, output_features = self._create_features_from_norm_stats(
                tag_metadata,
                config.image_default_input_size,
            )
            action_horizon = tag_metadata.get("action_horizon")
            if not isinstance(action_horizon, int):
                msg = f"Invalid action_horizon for normalization tag {self._norm_tag!r}."
                raise TypeError(msg)
            chunk_size = action_horizon
            normalize_gripper = bool(tag_metadata.get("normalize_gripper", False))
            if self._setup_type is None:
                setup_type = str(tag_metadata.get("setup_type") or "")
            if self._control_mode is None:
                control_mode = str(tag_metadata.get("control_mode") or "")

        return replace(
            config,
            input_features=input_features,
            output_features=output_features,
            norm_tag=self._norm_tag,
            normalize_gripper=normalize_gripper,
            chunk_size=chunk_size,
            n_action_steps=self._n_action_steps,
            n_obs_steps=self._n_obs_steps,
            setup_type=setup_type,
            control_mode=control_mode,
            adapt_to_so101=self._adapt_to_so101,
            normalization_mode=normalization_mode,
            tokenizer_config=tokenizer_config,
            tokenizer_name_or_path=str(snapshot_dir),
        )

    @staticmethod
    def _from_hf(
        pretrained_name_or_path: str | Path,
    ) -> tuple[dict, dict, dict, Path]:
        """Load and validate a MolmoAct2 checkpoint from a local path or Hugging Face repo.

        Args:
            pretrained_name_or_path: Local path or Hugging Face repository ID of the checkpoint.

        Returns:
            tuple[dict, dict, dict, Path]: The Hugging Face config, normalization stats, tokenizer
                config, and checkpoint weight file path.

        Raises:
            FileNotFoundError: If required checkpoint files are missing.
            TypeError: If a required JSON payload is malformed.
        """
        path = Path(pretrained_name_or_path)

        if not path.is_dir():
            path = Path(
                snapshot_download(
                    repo_id=str(pretrained_name_or_path),
                    allow_patterns=[
                        "config.json",
                        "norm_stats.json",
                        "processor_config.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "*.safetensors",
                        "model.safetensors.index.json",
                    ],
                ),
            )

        config_file = path / "config.json"
        norm_stats_file = path / "norm_stats.json"
        tokenizer_config_file = path / "tokenizer_config.json"

        if not config_file.is_file():
            msg = f"MolmoAct2 checkpoint at {path} is missing config.json."
            raise FileNotFoundError(msg)

        if not norm_stats_file.is_file():
            msg = f"MolmoAct2 checkpoint at {path} is missing norm_stats.json."
            raise FileNotFoundError(msg)

        if not tokenizer_config_file.is_file():
            msg = f"MolmoAct2 checkpoint at {path} is missing tokenizer_config.json."
            raise FileNotFoundError(msg)

        weights_file = path / "model.safetensors"
        if not weights_file.is_file():
            weights_file = path / "model.safetensors.index.json"

        if not weights_file.is_file():
            msg = (
                f"MolmoAct2 checkpoint at {path} must contain "
                "model.safetensors or model.safetensors.index.json."
            )
            raise FileNotFoundError(msg)

        # Parse config_file.
        with config_file.open(encoding="utf-8") as f:
            hf_config = json.load(f)
            if not isinstance(hf_config, dict):
                msg = f"MolmoAct2 config at {config_file} is not a valid JSON object."
                raise TypeError(msg)

        # Parse norm_stats_file.
        with norm_stats_file.open(encoding="utf-8") as f:
            norm_stats_config = json.load(f)
            if not isinstance(norm_stats_config, dict):
                msg = f"MolmoAct2 norm stats at {norm_stats_file} is not a valid JSON object."
                raise TypeError(msg)

        with tokenizer_config_file.open(encoding="utf-8") as f:
            tokenizer_config = json.load(f)
            if not isinstance(tokenizer_config, dict):
                msg = f"MolmoAct2 tokenizer config at {tokenizer_config_file} is not a valid JSON object."
                raise TypeError(msg)

        return hf_config, norm_stats_config, tokenizer_config, weights_file

    def forward(self, batch: Any) -> Any:  # ruff: ignore[ANN401]
        """Run a forward pass for the policy.

        The runtime path is intentionally left unimplemented while init wiring is developed.
        """
        del batch
        msg = "Forward pass is not implemented."
        raise NotImplementedError(msg)

    def predict_action_chunk(self, batch: Any) -> Any:  # ruff: ignore[ANN401]
        """Predict an action chunk for policy inference.

        The runtime path is intentionally left unimplemented while init wiring is developed.
        """
        del batch
        msg = "Action prediction is not implemented."
        raise NotImplementedError(msg)
