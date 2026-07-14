# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 vision backbone (ViT encoder + pooling adapter).

Checkpoint key prefix: ``model.vision_backbone.*``. The backbone encodes image
crops with a SigLIP-style ViT, concatenates features from a few selected ViT
layers, attention-pools them to the adapter width, and projects to the text
hidden size.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from transformers.activations import ACT2FN

if TYPE_CHECKING:
    from physicalai.policies.molmoact2.config import MolmoAct2AdapterConfig, MolmoAct2VitConfig


class VisionMultiHeadAttention(nn.Module):
    """Multi-head (optionally grouped) attention used by the ViT and the pooler.

    A single SDPA path replaces the upstream eager/sdpa/flash branches. When
    ``float32_attention`` is set, q/k/v are promoted to float32 for the scaled
    dot-product to match the reference numerics.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        input_dim: int | None = None,
        use_bias: bool = True,
        float32_attention: bool = True,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
    ) -> None:
        """Build the q/k/v/o projections."""
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.float32_attention = float32_attention
        self.attention_dropout = attention_dropout

        input_dim = input_dim or hidden_size
        self.wq = nn.Linear(input_dim, num_heads * head_dim, bias=use_bias)
        self.wk = nn.Linear(input_dim, num_key_value_heads * head_dim, bias=use_bias)
        self.wv = nn.Linear(input_dim, num_key_value_heads * head_dim, bias=use_bias)
        self.wo = nn.Linear(num_heads * head_dim, hidden_size)
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(
        self,
        inputs_q: torch.Tensor,
        inputs_kv: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend ``inputs_q`` over ``inputs_kv`` (defaults to self-attention).

        Returns:
            Multi head attention for vision stack.
        """
        inputs_kv = inputs_q if inputs_kv is None else inputs_kv
        batch, q_len, _ = inputs_q.shape

        q = self.wq(inputs_q).view(batch, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(inputs_kv).view(batch, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.wv(inputs_kv).view(batch, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        out_dtype = q.dtype
        if self.float32_attention:
            q, k, v = q.float(), k.float(), v.float()

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).reshape(batch, q_len, self.num_heads * self.head_dim).to(out_dtype)
        return self.residual_dropout(self.wo(attn))


class VisionMLP(nn.Module):
    """Two-layer feed-forward block for a ViT layer."""

    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        """Build the ``w1``/``w2`` projections and activation."""
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=True)
        self.act = ACT2FN[hidden_act]
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward transform.

        Returns:
            Output of feedforward MLP in Vit Layer.
        """
        return self.w2(self.act(self.w1(x)))


class VisionBlock(nn.Module):
    """Pre-norm ViT transformer block."""

    def __init__(self, config: MolmoAct2VitConfig) -> None:
        """Build attention, feed-forward and their norms."""
        super().__init__()
        self.attention = VisionMultiHeadAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            float32_attention=config.float32_attention,
            attention_dropout=config.attention_dropout,
            residual_dropout=config.residual_dropout,
        )
        self.feed_forward = VisionMLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run attention and feed-forward with residual connections.

        Returns:
            Output of ViT transformer block.
        """
        x = x + self.attention(self.attention_norm(x))  # noqa: PLR6104
        return x + self.feed_forward(self.ffn_norm(x))


class VisionBlockCollection(nn.Module):
    """Ordered stack of ViT blocks (checkpoint key: ``transformer.resblocks``)."""

    def __init__(self, config: MolmoAct2VitConfig) -> None:
        """Build ``num_hidden_layers`` residual blocks."""
        super().__init__()
        self.resblocks = nn.ModuleList([VisionBlock(config) for _ in range(config.num_hidden_layers)])


class VisionTransformer(nn.Module):
    """Patch embedding + positional embedding + block stack."""

    def __init__(self, config: MolmoAct2VitConfig) -> None:
        """Build patch/positional embeddings and the block stack."""
        super().__init__()
        self.config = config
        self.positional_embedding = nn.Parameter(torch.zeros(config.image_num_pos, config.hidden_size))
        self.patch_embedding = nn.Linear(config.image_patch_size**2 * 3, config.hidden_size, bias=True)
        self.transformer = VisionBlockCollection(config)

    def add_pos_emb(self, x: torch.Tensor, patch_num: tuple[int, int]) -> torch.Tensor:
        """Add (bicubic-resized if needed) positional embeddings to patches.

        Returns:
            Embeddings with position embeddings added.
        """
        side = int(math.sqrt(self.positional_embedding.shape[0]))
        pos_emb = self.positional_embedding.reshape(side, side, -1)
        if pos_emb.shape[0] != patch_num[0] or pos_emb.shape[1] != patch_num[1]:
            pos_emb = pos_emb.permute(2, 0, 1).unsqueeze(0)
            pos_emb = F.interpolate(
                pos_emb,
                size=patch_num,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            pos_emb = pos_emb.squeeze(0).permute(1, 2, 0)
        pos_emb = pos_emb.reshape(-1, pos_emb.shape[-1])
        return x + pos_emb[None].to(x.dtype)


class ImageProjectorMLP(nn.Module):
    """Gated MLP projecting pooled ViT features to the text hidden size."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, hidden_act: str) -> None:
        """Build the gated ``w1``/``w2``/``w3`` projections."""
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.w3 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.act = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated projection.

        Returns:
            Output of image projection linear layer.
        """
        return self.w2(self.act(self.w1(x)) * self.w3(x))


class MolmoAct2VisionBackbone(nn.Module):
    """Encode crops into pooled image features aligned to the text hidden size."""

    def __init__(self, vit_config: MolmoAct2VitConfig, adapter_config: MolmoAct2AdapterConfig) -> None:
        """Build the (truncated) ViT, the attention pooler and the projector."""
        super().__init__()
        self.vit_config = vit_config
        self.adapter_config = adapter_config

        self.vit_layers = [
            layer if layer >= 0 else layer + vit_config.num_hidden_layers for layer in adapter_config.vit_layers
        ]
        # Only build up to the deepest ViT layer we actually read from.
        last_layer_needed = max(self.vit_layers) + 1
        if last_layer_needed < vit_config.num_hidden_layers:
            vit_config = deepcopy(vit_config)
            vit_config.num_hidden_layers = last_layer_needed
        self.image_vit = VisionTransformer(vit_config)

        pool_dim = self.vit_config.hidden_size * len(adapter_config.vit_layers)
        self.image_pooling_2d = VisionMultiHeadAttention(
            hidden_size=adapter_config.hidden_size,
            num_heads=adapter_config.num_attention_heads,
            num_key_value_heads=adapter_config.num_key_value_heads,
            head_dim=adapter_config.head_dim,
            input_dim=pool_dim,
            float32_attention=adapter_config.float32_attention,
            attention_dropout=adapter_config.attention_dropout,
            residual_dropout=adapter_config.residual_dropout,
        )
        self.image_projector = ImageProjectorMLP(
            adapter_config.hidden_size,
            adapter_config.intermediate_size,
            adapter_config.text_hidden_size,
            adapter_config.hidden_act,
        )
        self.image_feature_dropout = nn.Dropout(adapter_config.image_feature_dropout)

    @property
    def dtype(self) -> torch.dtype:
        """Parameter dtype of the vision backbone."""
        return self.image_vit.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        """Parameter device of the vision backbone."""
        return self.image_vit.patch_embedding.weight.device

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode crops and concatenate features from the selected ViT layers.

        Args:
            images: Patchified crops of shape ``(batch, num_crops, num_patches, patch_dim)``.

        Returns:
            Features of shape ``(batch, num_crops, num_patches, hidden * num_selected_layers)``.
        """
        batch_size, num_crops, num_patches, _ = images.shape
        x = images.view(batch_size * num_crops, num_patches, -1)
        x = self.image_vit.patch_embedding(x)
        x = self.image_vit.add_pos_emb(x, self.image_vit.config.image_num_patch)

        needed = set(self.vit_layers)
        selected: dict[int, torch.Tensor] = {}
        for layer_idx, block in enumerate(self.image_vit.transformer.resblocks):
            x = block(x)
            if layer_idx in needed:
                selected[layer_idx] = x

        features = torch.cat([selected[layer] for layer in self.vit_layers], dim=-1)
        return features.view(batch_size, num_crops, num_patches, -1)

    def forward(self, images: torch.Tensor, pooled_patches_idx: torch.Tensor) -> torch.Tensor:
        """Encode, attention-pool and project crops into text-space features.

        Args:
            images: Crops of shape ``(batch, num_crops, num_patches, patch_dim)``.
            pooled_patches_idx: Per-pooled-token patch indices ``(batch, num_tokens, pool_size)``;
                negative entries mark padding.

        Returns:
            Projected features ``(num_valid_tokens, text_hidden_size)``.
        """
        batch_size = images.shape[0]
        images = images.to(dtype=self.dtype)
        image_features = self.image_feature_dropout(self.encode_image(images))
        dim = image_features.shape[-1]

        valid = pooled_patches_idx >= 0
        valid_token = torch.any(valid, dim=-1)

        batch_idx = torch.arange(batch_size, device=pooled_patches_idx.device).view(batch_size, 1, 1)
        batch_idx = batch_idx.expand(-1, pooled_patches_idx.shape[1], pooled_patches_idx.shape[2])
        to_pool = image_features.reshape(batch_size, -1, dim)[batch_idx, pooled_patches_idx.clamp_min(0)]
        to_pool = to_pool * valid.to(self.dtype)[..., None]  # noqa: PLR6104
        to_pool = to_pool.reshape(-1, pooled_patches_idx.shape[-1], dim)

        if self.adapter_config.pooling_attention_mask:
            attn_mask = valid.reshape(-1, 1, 1, valid.shape[-1])
            denom = valid.view(-1, to_pool.shape[-2]).sum(-1).clamp_min(1).to(to_pool.dtype)
            query = to_pool.sum(-2, keepdim=True) / denom[:, None, None]
        else:
            attn_mask = None
            query = to_pool.mean(-2, keepdim=True)

        pooled = self.image_pooling_2d(query, to_pool, attn_mask=attn_mask)
        pooled = pooled.reshape(batch_size, -1, pooled.shape[-1])
        pooled = self.image_projector(pooled)
        return pooled.view(-1, pooled.shape[-1])[valid_token.flatten()]
