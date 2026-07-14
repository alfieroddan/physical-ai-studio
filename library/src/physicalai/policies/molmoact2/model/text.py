# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 text decoder backbone.

Checkpoint key prefix: ``model.transformer.*``. A pre-norm, GQA transformer
decoder with QK-norm and rotary embeddings. For continuous action generation
the decoder returns both the final hidden states and the per-layer key/value
states (post-rotary, before GQA repeat) that the action expert cross-attends to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from transformers.activations import ACT2FN

if TYPE_CHECKING:
    from physicalai.policies.molmoact2.config import MolmoAct2TextConfig

KVState = tuple[torch.Tensor, torch.Tensor]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension onto the first."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand grouped key/value heads to the number of query heads."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MolmoAct2RMSNorm(nn.Module):
    """RMS normalization with a learnable weight (float32 reduction)."""

    def __init__(self, size: int, eps: float = 1e-6) -> None:
        """Build the norm weight."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` over its last dimension and rescale."""
        out_dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(out_dtype)


class MolmoAct2RotaryEmbedding(nn.Module):
    """Default rotary embedding; ``inv_freq`` is a persistent checkpoint buffer."""

    inv_freq: torch.Tensor

    def __init__(self, config: MolmoAct2TextConfig) -> None:
        """Precompute inverse frequencies from ``rope_theta`` and ``head_dim``."""
        super().__init__()
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` of shape ``(batch, seq_len, head_dim)``."""
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        positions = position_ids[:, None, :].float()
        freqs = (inv_freq @ positions).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class MolmoAct2Attention(nn.Module):
    """Grouped-query self-attention with fused QKV and QK-norm."""

    def __init__(self, config: MolmoAct2TextConfig) -> None:
        """Build the fused QKV projection, output projection and QK norms."""
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim

        self.fused_dims = (
            self.num_heads * self.head_dim,
            self.num_key_value_heads * self.head_dim,
            self.num_key_value_heads * self.head_dim,
        )
        self.att_proj = nn.Linear(config.hidden_size, sum(self.fused_dims), bias=config.qkv_bias)
        self.attn_out = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm: MolmoAct2RMSNorm | None = None
        self.k_norm: MolmoAct2RMSNorm | None = None
        if config.use_qk_norm:
            if config.qk_norm_type != "qwen3":
                msg = f"Only qk_norm_type='qwen3' is supported, got {config.qk_norm_type!r}."
                raise NotImplementedError(msg)
            self.q_norm = MolmoAct2RMSNorm(self.head_dim, eps=config.layer_norm_eps)
            self.k_norm = MolmoAct2RMSNorm(self.head_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_bias: torch.Tensor | None,
    ) -> tuple[torch.Tensor, KVState]:
        """Attend and also return the post-rotary key/value states."""
        input_shape = hidden_states.shape[:-1]
        head_shape = (*input_shape, -1, self.head_dim)

        query, key, value = self.att_proj(hidden_states).split(self.fused_dims, dim=-1)
        query = query.view(head_shape)
        key = key.view(head_shape)
        value = value.view(head_shape).transpose(1, 2)
        if self.q_norm is not None and self.k_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        attn = F.scaled_dot_product_attention(
            query,
            repeat_kv(key, self.num_key_value_groups),
            repeat_kv(value, self.num_key_value_groups),
            attn_mask=attention_bias,
            is_causal=attention_bias is None,
        )
        attn = attn.transpose(1, 2).reshape(*input_shape, -1)
        return self.attn_out(attn), (key, value)


class LanguageModelMLP(nn.Module):
    """SwiGLU feed-forward with a fused gate/up projection."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str) -> None:
        """Build the fused ``ff_proj`` and output ``ff_out`` projections."""
        super().__init__()
        self.ff_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.ff_out = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated feed-forward transform."""
        x, gate = self.ff_proj(x).chunk(2, dim=-1)
        return self.ff_out(self.act(gate) * x)


class MolmoAct2DecoderLayer(nn.Module):
    """Pre-norm transformer decoder layer."""

    def __init__(self, config: MolmoAct2TextConfig) -> None:
        """Build attention, feed-forward and their norms."""
        super().__init__()
        self.self_attn = MolmoAct2Attention(config)
        self.attn_norm = MolmoAct2RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = LanguageModelMLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.ff_norm = MolmoAct2RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_bias: torch.Tensor | None,
    ) -> tuple[torch.Tensor, KVState]:
        """Run attention and feed-forward, returning hidden states and KV."""
        attn_out, kv_state = self.self_attn(self.attn_norm(hidden_states), position_embeddings, attention_bias)
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.ff_norm(hidden_states))
        return hidden_states, kv_state


class MolmoAct2Embedding(nn.Module):
    """Token embedding split into base vocab and additional (new) tokens."""

    def __init__(self, num_embeddings: int, num_new_embeddings: int, features: int) -> None:
        """Build the base and additional embedding tables."""
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(num_embeddings, features))
        self.new_embedding = nn.Parameter(torch.zeros(num_new_embeddings, features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Look up token embeddings across the combined table."""
        return F.embedding(x, torch.cat([self.embedding, self.new_embedding], dim=0))


class MolmoAct2TextModel(nn.Module):
    """Decoder-only transformer producing hidden states and per-layer KV."""

    def __init__(self, config: MolmoAct2TextConfig) -> None:
        """Build the embeddings, decoder blocks, final norm and rotary embedding."""
        super().__init__()
        if config.norm_after:
            msg = "MolmoAct2 inference only supports norm_after=False."
            raise NotImplementedError(msg)
        self.config = config
        self.wte = MolmoAct2Embedding(config.vocab_size, config.additional_vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([MolmoAct2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = MolmoAct2RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.rotary_emb = MolmoAct2RotaryEmbedding(config)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[KVState]]:
        """Run the decoder over input embeddings.

        Args:
            inputs_embeds: Token embeddings ``(batch, seq_len, hidden)``.
            attention_bias: Additive attention bias ``(batch, 1, seq_len, seq_len)``
                or ``None`` for plain causal attention.
            position_ids: Position ids ``(batch, seq_len)``; defaults to ``arange``.

        Returns:
            Final hidden states and a list of per-layer ``(key, value)`` states.
        """
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        kv_states: list[KVState] = []
        for block in self.blocks:
            hidden_states, kv_state = block(hidden_states, position_embeddings, attention_bias)
            kv_states.append(kv_state)
        return self.ln_f(hidden_states), kv_states
