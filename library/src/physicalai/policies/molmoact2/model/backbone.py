# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 assembly and continuous action generation.

``MolmoAct2ForConditionalGeneration`` is the checkpoint root: it owns the
``model`` backbone (text + vision + action expert) and the ``lm_head``. Weight
keys are ``model.transformer.*``, ``model.vision_backbone.*``,
``model.action_expert.*`` and ``lm_head.weight``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from .action_expert import ActionExpert
from .text import KVState, MolmoAct2TextModel
from .vision import MolmoAct2VisionBackbone

if TYPE_CHECKING:
    from physicalai.policies.molmoact2.config import MolmoAct2Config


class MolmoAct2Backbone(nn.Module):
    """Text + vision + action expert. Checkpoint prefix: ``model.*``."""

    def __init__(self, config: MolmoAct2Config) -> None:
        """Build the text decoder, vision backbone and (optional) action expert."""
        super().__init__()
        self.config = config
        self.transformer = MolmoAct2TextModel(config.text_config)
        self.vision_backbone = MolmoAct2VisionBackbone(config.vit_config, config.adapter_config)
        self.action_expert: ActionExpert | None = None
        if config.add_action_expert and config.action_expert_config is not None:
            self.action_expert = ActionExpert(
                config.action_expert_config,
                llm_kv_dim=config.text_config.num_key_value_heads * config.text_config.head_dim,
                llm_num_layers=config.text_config.num_hidden_layers,
            )

    def _require_action_expert(self) -> ActionExpert:
        """Return the action expert or raise if the checkpoint lacks one."""
        if self.action_expert is None:
            msg = "This MolmoAct2 checkpoint does not include an action expert."
            raise RuntimeError(msg)
        return self.action_expert

    def build_input_embeddings(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor | None,
        token_pooling: torch.Tensor | None,
    ) -> torch.Tensor:
        """Embed tokens and add projected image features at image-patch positions."""
        input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
        embeddings = self.transformer.wte(input_ids)
        if images is None:
            return embeddings

        image_features = self.vision_backbone(images, token_pooling).to(embeddings.dtype)
        is_image_patch = (input_ids == self.config.image_patch_id).reshape(-1)
        flat = embeddings.reshape(-1, embeddings.shape[-1]).clone()
        flat[is_image_patch] += image_features
        return flat.reshape_as(embeddings)

    def _build_attention_bias(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build an additive causal bias; image tokens attend bidirectionally."""
        batch_size, seq_len = inputs_embeds.shape[:2]
        device, dtype = inputs_embeds.device, inputs_embeds.dtype

        valid = (
            torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
            if attention_mask is None
            else attention_mask.to(device=device, dtype=torch.bool)
        )
        causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        causal = causal[None, None].expand(batch_size, 1, -1, -1)
        if token_type_ids is not None:
            image_mask = token_type_ids.to(device=device, dtype=torch.bool)
            can_attend_back = image_mask[:, None, :, None] & image_mask[:, None, None, :]
            causal = causal | can_attend_back
        allowed = valid[:, None, None, :] & causal
        return torch.where(allowed, 0.0, torch.finfo(dtype).min).to(dtype)

    def _encoder_attention_mask(
        self, input_ids: torch.Tensor | None, attention_mask: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Boolean mask of text positions the action expert may cross-attend to."""
        if attention_mask is not None:
            return attention_mask.to(dtype=torch.bool)
        if input_ids is not None:
            return input_ids != -1
        return None

    @staticmethod
    def _kv_to_sequence(cache: torch.Tensor) -> torch.Tensor:
        """Flatten ``(batch, heads, seq, head_dim)`` KV to ``(batch, seq, heads * head_dim)``."""
        batch, heads, seq_len, head_dim = cache.shape
        return cache.permute(0, 2, 1, 3).reshape(batch, seq_len, heads * head_dim)

    def _mask_action_dims(self, tensor: torch.Tensor, action_dim_is_pad: torch.Tensor | None) -> torch.Tensor:
        """Zero out padded action dimensions when configured to do so."""
        if not self.config.mask_action_dim_padding or action_dim_is_pad is None:
            return tensor
        valid = (~action_dim_is_pad.to(device=tensor.device, dtype=torch.bool))[:, None, :]
        return tensor * valid

    def encode(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
        images: torch.Tensor | None,
        token_pooling: torch.Tensor | None,
    ) -> list[KVState]:
        """Run the vision+text encoder and return per-layer KV states."""
        inputs_embeds = self.build_input_embeddings(input_ids, images, token_pooling)
        attention_bias = self._build_attention_bias(inputs_embeds, attention_mask, token_type_ids)
        _, kv_states = self.transformer(inputs_embeds, attention_bias=attention_bias)
        return kv_states

    @torch.no_grad()
    def generate_actions_from_inputs(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        token_pooling: torch.Tensor | None = None,
        action_dim_is_pad: torch.Tensor | None = None,
        action_horizon: int,
        num_steps: int | None = None,
        sample_noise: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Continuous flow-matching action generation.

        Encodes text+vision, collects per-layer KV, then Euler-integrates the
        action expert's velocity field. Integration starts from zeros by default
        (deterministic, export-friendly); set ``sample_noise`` to start from a
        sampled Gaussian vector instead.
        """
        action_expert = self._require_action_expert()
        kv_states = self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            images=images,
            token_pooling=token_pooling,
        )
        encoder_kv = [(self._kv_to_sequence(k), self._kv_to_sequence(v)) for k, v in kv_states]
        encoder_mask = self._encoder_attention_mask(input_ids, attention_mask)

        steps = int(num_steps or self.config.flow_matching_num_steps)
        if steps <= 0:
            msg = f"num_steps must be >= 1, got {steps}."
            raise ValueError(msg)

        batch_size = encoder_kv[0][0].shape[0]
        device = encoder_kv[0][0].device
        dtype = action_expert.action_embed.weight.dtype
        trajectory_shape = (batch_size, action_horizon, self.config.max_action_dim)
        if sample_noise:
            trajectory = torch.randn(*trajectory_shape, device=device, dtype=dtype, generator=generator)
            trajectory = self._mask_action_dims(trajectory, action_dim_is_pad)
        else:
            trajectory = torch.zeros(*trajectory_shape, device=device, dtype=dtype)

        context = action_expert.prepare_context(
            encoder_kv_states=encoder_kv,
            encoder_attention_mask=encoder_mask,
            batch_size=batch_size,
            seq_len=action_horizon,
            device=device,
            dtype=dtype,
        )

        dt = 1.0 / steps
        for step in range(steps):
            timestep = torch.full((batch_size,), step / steps, device=device, dtype=dtype)
            velocity = action_expert.forward_with_context(trajectory, timestep, context=context)
            velocity = self._mask_action_dims(velocity, action_dim_is_pad)
            trajectory = trajectory + dt * velocity
            trajectory = self._mask_action_dims(trajectory, action_dim_is_pad)
        return trajectory


class MolmoAct2ForConditionalGeneration(nn.Module):
    """Checkpoint root module: ``model`` backbone + ``lm_head``."""

    def __init__(self, config: MolmoAct2Config) -> None:
        """Build the backbone and the language-model head."""
        super().__init__()
        self.config = config
        self.model = MolmoAct2Backbone(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )

    @torch.no_grad()
    def generate_actions_from_inputs(self, **kwargs: Any) -> torch.Tensor:
        """Delegate continuous action generation to the backbone."""
        return self.model.generate_actions_from_inputs(**kwargs)
