# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 physicalai ``Model`` frontend.

Owns the backbone and preprocessing, and exposes the physicalai inference API.
Weight loading targets the backbone directly so checkpoint keys are preserved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

import torch
import torch.nn.functional as F  # noqa: N812
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor

from physicalai.data.constants import ACTION
from physicalai.data.observation import Feature, FeatureType
from physicalai.policies.base import Model

from .backbone import MolmoAct2ForConditionalGeneration, make_molmoact2_backbone

if TYPE_CHECKING:
    from physicalai.policies.molmoact2.config import MolmoAct2Config


# Linear-module leaves inside the VLM (text transformer + vision backbone) that
# LoRA adapters are attached to by default. Names match the physicalai
# re-implementation's module attributes (text: ``att_proj``/``attn_out``/
# ``ff_proj``/``ff_out``; vision: ``wq``/``wk``/``wv``/``wo``/``w1``/``w2``/
# ``w3``/``patch_embedding``).
_VLM_LORA_LINEAR_LEAVES = "att_proj|attn_out|ff_proj|ff_out|wq|wk|wv|wo|w1|w2|w3|patch_embedding"

# Linear-module leaves inside the action expert. The time embedding is an
# ``nn.Sequential`` whose linears sit at indices 1 and 3.
_ACTION_EXPERT_LORA_LINEAR_LEAVES = (
    r"time_embed\.(1|3)|"
    r"action_embed|"
    r"context_k_proj|context_v_proj|"
    r"blocks\.\d+\.self_attn\.(qkv|out_proj)|"
    r"blocks\.\d+\.cross_attn\.(q_proj|out_proj)|"
    r"blocks\.\d+\.mlp\.(up_proj|gate_proj|down_proj)|"
    r"blocks\.\d+\.modulation\.linear|"
    r"final_layer\.(modulation\.linear|linear)"
)


def _lora_target_modules(*, enable_action_expert: bool) -> str:
    """Build the LoRA ``target_modules`` regex for the MolmoAct2 backbone.

    The backbone root is :class:`MolmoAct2ForConditionalGeneration`, so the
    parameter-name prefix for the inner backbone is ``model.`` (i.e.
    ``model.transformer.*``, ``model.vision_backbone.*`` and, when enabled,
    ``model.action_expert.*``).

    Args:
        enable_action_expert: Whether to also adapt the action-expert linears.

    Returns:
        A regex string matched against the fully-qualified module names.
    """
    vlm_targets = rf"model\.(transformer|vision_backbone)\.(?:.*\.)?({_VLM_LORA_LINEAR_LEAVES})$"
    if not enable_action_expert:
        return vlm_targets
    return f"({vlm_targets}|model\\.action_expert\\.(?:{_ACTION_EXPERT_LORA_LINEAR_LEAVES})$)"


_SAFE_WEIGHTS_NAME = "model.safetensors"
_SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"


def _env_action_dim(output_features: tuple[Feature, ...]) -> int:
    """Return the environment action dimension from the output features."""
    for feature in output_features:
        if feature.ftype == FeatureType.ACTION and feature.shape:
            return int(feature.shape[0])
    return 0


# Keys of the fully-prepared, traceable tensors the backbone consumes. All
# value-dependent host prep runs in the preprocessor, so the model graph only
# sees these fixed-shape tensors (keeping it exportable).
_MODEL_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "images",
    "token_pooling",
    "action_dim_is_pad",
)


def _masked_action_mse(
    predicted: Tensor,
    target: Tensor,
    *,
    action_horizon_is_pad: Tensor | None,
    action_dim_is_pad: Tensor | None,
) -> Tensor:
    """Mean squared error over valid action steps and dimensions only.

    Returns:
        loss of continous action compared to target actions.
    """
    loss = F.mse_loss(predicted, target, reduction="none")
    mask = torch.ones_like(loss, dtype=torch.bool)
    horizon_axis = -2
    action_dim_axis = -1
    if action_horizon_is_pad is not None:
        valid_horizon = (~action_horizon_is_pad.to(device=loss.device, dtype=torch.bool)).view(
            loss.shape[0],
            *([1] * (loss.ndim - 3)),
            loss.shape[horizon_axis],
            1,
        )
        mask = mask & valid_horizon  # noqa: PLR6104
    if action_dim_is_pad is not None:
        valid_dim = (~action_dim_is_pad.to(device=loss.device, dtype=torch.bool)).view(
            loss.shape[0],
            *([1] * (loss.ndim - 3)),
            1,
            loss.shape[action_dim_axis],
        )
        mask = mask & valid_dim  # noqa: PLR6104
    valid = mask.to(loss.dtype)
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def _masked_flow_loss(
    predicted: Tensor,
    target: Tensor,
    *,
    action_horizon_is_pad: Tensor | None,
    action_dim_is_pad: Tensor | None,
) -> Tensor:
    """Flow-matching MSE over valid action steps and dimensions only.

    ``predicted``/``target`` have shape ``(batch, num_flow_timesteps,
    horizon, max_action_dim)``. The per-example masks broadcast over the
    flow-timestep axis.

    Returns:
        Flow matching MSE loss.
    """
    return _masked_action_mse(
        predicted,
        target,
        action_horizon_is_pad=action_horizon_is_pad,
        action_dim_is_pad=action_dim_is_pad,
    )


def _strict_load_safetensors_weights(model: torch.nn.Module, checkpoint_location: str) -> None:
    """Load safetensors weights into ``model``, verifying exact key correspondence.

    Raises:
        RuntimeError: If the checkpoint keys do not exactly match the model.
        FileNotFoundError: If no safetensors checkpoint is found.
    """
    checkpoint_dir = Path(checkpoint_location)
    index_path = checkpoint_dir / _SAFE_WEIGHTS_INDEX_NAME
    single_file_path = checkpoint_dir / _SAFE_WEIGHTS_NAME

    if index_path.is_file():
        with index_path.open(encoding="utf-8") as index_file:
            weight_map = json.load(index_file)["weight_map"]
        missing = sorted(set(model.state_dict()) - set(weight_map))
        unexpected = sorted(set(weight_map) - set(model.state_dict()))
        if missing or unexpected:
            msg = f"MolmoAct2 checkpoint keys mismatch. Missing: {missing[:6]} Unexpected: {unexpected[:6]}"
            raise RuntimeError(msg)
        for shard in sorted(set(weight_map.values())):
            state_dict = load_safetensors_file(str(checkpoint_dir / shard), device="cpu")
            model.load_state_dict(state_dict, strict=False)
        return

    if single_file_path.is_file():
        model.load_state_dict(load_safetensors_file(str(single_file_path), device="cpu"), strict=True)
        return

    msg = f"No safetensors checkpoint found at {checkpoint_location!r}."
    raise FileNotFoundError(msg)


class MolmoAct2Model(Model):
    """Inference frontend wrapping :class:`MolmoAct2ForConditionalGeneration`."""

    def __init__(self, config: MolmoAct2Config) -> None:
        """Build the backbone."""
        super().__init__()
        self._train_action_expert_only = config.train_action_expert_only
        self._enable_lora_action_expert = config.enable_lora_action_expert
        self._lora_rank = config.lora_rank
        self._lora_alpha = config.lora_alpha
        self._lora_dropout = config.lora_dropout
        self._lora_bias = config.lora_bias
        self._add_action_expert = config.add_action_expert
        self._chunk_size = config.chunk_size
        self._n_action_steps = config.n_action_steps
        self._use_random_input_noise = config.use_random_input_noise
        self._mask_action_dim_padding = config.mask_action_dim_padding
        self._output_features = tuple(config.output_features or [])
        self.backbone = MolmoAct2ForConditionalGeneration(
            model=make_molmoact2_backbone(config),
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
        )

        if config.gradient_checkpointing:
            self.gradient_checkpointing_enable()

        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            compile_mode = "default"
            self.predict_action_chunk = torch.compile(self.predict_action_chunk, mode=compile_mode)  # type: ignore[method-assign]
            self.forward = torch.compile(self.forward, mode=compile_mode)  # type: ignore[method-assign]

    def train(self, mode: bool = True) -> MolmoAct2Model:  # noqa: FBT001, FBT002
        """Set training mode, keeping a frozen VLM in eval when action-expert-only.

        Mirrors the reference MolmoAct2 training recipe: when
        ``config.train_action_expert_only`` is set, the VLM (text transformer +
        vision backbone) is frozen (``requires_grad=False``) but, without this
        override, ``nn.Module.train()`` would still recursively flip it into
        train mode (e.g. enabling any dropout it contains), producing
        stochastic conditioning context during training that never matches the
        deterministic ``eval()`` context seen at inference. Only the action
        expert should toggle with ``mode``; the rest of the backbone always
        stays in ``eval()``.

        Returns:
            ``self``, matching :meth:`nn.Module.train`.
        """
        super().train(mode)
        if self._train_action_expert_only:
            self._for_cond_gen.eval()
            action_expert = getattr(self._for_cond_gen.model, "action_expert", None)
            if action_expert is not None:
                action_expert.train(mode)
        return self

    def load_pretrained_weights(self, checkpoint_location: str) -> None:
        """Load safetensors weights into the backbone (strict key match)."""
        _strict_load_safetensors_weights(self.backbone, checkpoint_location)

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing on the text, vision and action submodules.

        Sets ``gradient_checkpointing=True`` on the inner backbone's text
        transformer, vision backbone, and (when present) action expert so each
        per-layer / per-block forward is recomputed during the backward pass,
        trading compute for a smaller activation-memory footprint during
        training.
        """
        backbone = self._for_cond_gen.model
        backbone.transformer.gradient_checkpointing = True
        backbone.vision_backbone.gradient_checkpointing = True
        if backbone.action_expert is not None:
            backbone.action_expert.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing on all submodules."""
        backbone = self._for_cond_gen.model
        backbone.transformer.gradient_checkpointing = False
        backbone.vision_backbone.gradient_checkpointing = False
        if backbone.action_expert is not None:
            backbone.action_expert.gradient_checkpointing = False

    @property
    def _for_cond_gen(self) -> MolmoAct2ForConditionalGeneration:
        """The unwrapped :class:`MolmoAct2ForConditionalGeneration`.

        When LoRA adapters are applied, ``self.backbone`` is a PEFT wrapper
        whose ``base_model.model`` is the original checkpoint-root module.
        This accessor returns that underlying module so the
        ``.model.generate_actions_from_inputs(...)`` /
        ``.model.predict_flow_velocity(...)`` call paths keep working both
        before and after LoRA is applied.
        """
        backbone = self.backbone
        base_model = getattr(backbone, "base_model", None)
        if base_model is not None and hasattr(base_model, "model"):
            return base_model.model  # type: ignore[no-any-return]
        return backbone  # type: ignore[return-value]

    def apply_lora_adapters(self) -> None:
        """Apply LoRA adapters to the backbone via ``peft.get_peft_model``.

        Wraps :attr:`backbone` (:class:`MolmoAct2ForConditionalGeneration`)
        with PEFT LoRA layers targeting the VLM linears (text transformer +
        vision backbone) and, when ``config.enable_lora_action_expert`` is
        set, the action-expert linears. All non-LoRA parameters are frozen.
        When the action expert is not covered by LoRA, its parameters are
        re-unfrozen so full fine-tuning of the action expert continues.

        Raises:
            ImportError: If the ``peft`` package is not installed.
        """
        try:
            from peft import LoraConfig, get_peft_model  # noqa: PLC0415
        except ImportError as e:
            msg = "MolmoAct2 LoRA requires peft. Install with: pip install 'physicalai-train[molmoact2]'"
            raise ImportError(msg) from e

        target_modules = _lora_target_modules(
            enable_action_expert=self._enable_lora_action_expert,
        )
        lora_config = LoraConfig(
            r=self._lora_rank,
            lora_alpha=self._lora_alpha,
            lora_dropout=self._lora_dropout,
            target_modules=target_modules,
            bias=self._lora_bias,
        )

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone = get_peft_model(self.backbone, lora_config)  # type: ignore[assignment, arg-type]  # pyrefly: ignore[bad-assignment]
        if not self._enable_lora_action_expert:
            self._unfreeze_action_expert_parameters()
        self.train(self.training)

    def _unfreeze_action_expert_parameters(self) -> None:
        """Re-enable gradients for the action-expert parameters after LoRA.

        Raises:
            RuntimeError: If no action-expert parameters were found.
        """
        if not self._add_action_expert:
            msg = "enable_lora_action_expert=False, but no action_expert parameters were found."
            raise RuntimeError(msg)
        trainable = 0
        for name, param in self.backbone.named_parameters():
            if "action_expert" in name:
                param.requires_grad = True
                trainable += param.numel()
        if trainable == 0:
            msg = "enable_lora_action_expert=False, but no action_expert parameters were found."
            raise RuntimeError(msg)

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Any],
        *,
        sample_noise: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Generate an action chunk from a preprocessed inference batch.

        Args:
            batch: Preprocessed inference batch of backbone-ready model inputs.
            sample_noise: Start flow matching from sampled Gaussian noise instead
                of zeros. Defaults to ``config.use_random_input_noise``.
            generator: Optional RNG used when ``sample_noise`` is enabled.

        Returns:
            Actions of shape ``(batch, n_action_steps, env_action_dim)``.
        """
        model_inputs = {key: batch[key] for key in _MODEL_INPUT_KEYS if key in batch}
        actions = self._for_cond_gen.model.generate_actions_from_inputs(
            **model_inputs,
            action_horizon=int(self._chunk_size),
            sample_noise=self._use_random_input_noise if sample_noise is None else sample_noise,
            generator=generator,
        )
        env_action_dim = _env_action_dim(self._output_features) or actions.shape[-1]
        return actions[:, : int(self._n_action_steps), :env_action_dim].to(torch.float32)

    def forward(self, batch: dict[str, Any]) -> Tensor | tuple[Tensor, dict[str, Tensor | float]]:
        """Run the model.

        Returns:
            Training mode: ``(loss, metrics)``. Inference mode: action chunk.
        """
        if self.training:
            return self.compute_loss(batch)
        return self.predict_action_chunk(batch)

    @override
    def compute_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Continuous flow-matching training loss.

        Returns:
            The scalar loss and a metrics dict.
        """
        predicted, target = self._for_cond_gen.model.predict_flow_velocity(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            token_type_ids=batch.get("token_type_ids"),
            images=batch.get("images"),
            token_pooling=batch.get("token_pooling"),
            actions=batch[ACTION],
            action_dim_is_pad=batch.get("action_dim_is_pad"),
            freeze_encoder=self._train_action_expert_only,
        )
        loss = _masked_flow_loss(
            predicted,
            target,
            action_horizon_is_pad=batch.get("action_horizon_is_pad"),
            action_dim_is_pad=batch.get("action_dim_is_pad") if self._mask_action_dim_padding else None,
        )
        value = loss.detach()
        return loss, {"action_flow_loss": value, "loss": value}

    @override
    @torch.no_grad()
    def compute_val_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Validation MSE between predicted and ground-truth action chunks.

        Returns:
            The scalar loss and a metrics dict.
        """
        gt_actions = batch[ACTION]
        predicted = self.predict_action_chunk(batch)
        horizon = min(int(gt_actions.shape[1]), int(predicted.shape[1]))
        action_dim = min(int(gt_actions.shape[2]), int(predicted.shape[2]))
        target = gt_actions[:, :horizon, :action_dim].to(device=predicted.device, dtype=predicted.dtype)
        action_horizon_is_pad = batch.get("action_horizon_is_pad")
        if action_horizon_is_pad is not None:
            action_horizon_is_pad = action_horizon_is_pad[:, :horizon]
        action_dim_is_pad = batch.get("action_dim_is_pad")
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad[:, :action_dim]
        loss = _masked_action_mse(
            predicted[:, :horizon, :action_dim],
            target,
            action_horizon_is_pad=action_horizon_is_pad,
            action_dim_is_pad=action_dim_is_pad if self._mask_action_dim_padding else None,
        )
        return loss, {"loss": float(loss.detach().float())}

    def freeze_to_action_expert(self) -> None:
        """Freeze every parameter except the action expert (memory-lean fine-tuning).

        Raises:
            RuntimeError: if no action expert can be found in params.
        """
        trainable = 0
        for name, param in self.named_parameters():
            is_action_expert = "action_expert" in name
            param.requires_grad_(is_action_expert)
            trainable += param.numel() if is_action_expert else 0
        if trainable == 0:
            msg = "train_action_expert_only=True, but no action_expert parameters were found."
            raise RuntimeError(msg)

    @property
    def reward_delta_indices(self) -> None:
        """Reward deltas, unused because rewards are not model inputs."""
        return None

    @property
    def action_delta_indices(self) -> list[int] | None:
        """Future action indices required to train the configured action chunk."""
        return list(range(int(self._chunk_size)))

    @property
    def observation_delta_indices(self) -> None:
        """Observation deltas, unused because preprocessing expects one current frame."""
        return None
