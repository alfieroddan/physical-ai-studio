# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MolmoAct2 model components.

All tests use a tiny config fixture so component construction stays cheap; the
full 7B backbone is never built.
"""

from __future__ import annotations

import pytest
import torch

from physicalai.policies.molmoact2.config import MolmoAct2Config
from physicalai.policies.molmoact2.model.action_expert import (
    ActionExpert,
    ActionExpertContext,
    ActionExpertFinalLayer,
    ActionExpertMLP,
    ActionExpertModulation,
    ActionExpertRMSNorm,
    ActionExpertRotaryEmbedding,
    ActionExpertSelfAttention,
    SinusoidalTimeEmbedding,
    _modulate,
    _round_up_multiple,
)
from physicalai.policies.molmoact2.model.backbone import (
    MolmoAct2Backbone,
    MolmoAct2ForConditionalGeneration,
    _sample_beta_timesteps,
)
from physicalai.policies.molmoact2.model.text import (
    MolmoAct2Attention,
    MolmoAct2Embedding,
    MolmoAct2RMSNorm,
    MolmoAct2RotaryEmbedding,
    MolmoAct2TextModel,
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
)
from physicalai.policies.molmoact2.model.vision import MolmoAct2VisionBackbone
from physicalai.policies.molmoact2.model.wrapper import MolmoAct2Model, _masked_action_mse


class TestRoundUpMultiple:
    def test_rounds_up(self) -> None:
        assert _round_up_multiple(10, 16) == 16
        assert _round_up_multiple(16, 16) == 16
        assert _round_up_multiple(17, 16) == 32

    def test_zero_multiple_returns_value(self) -> None:
        assert _round_up_multiple(7, 0) == 7

    def test_negative_multiple_returns_value(self) -> None:
        assert _round_up_multiple(7, -1) == 7


class TestModulate:
    def test_identity_scale_no_shift(self) -> None:
        x = torch.ones(2, 3, 4)
        shift = torch.zeros(2, 4)
        scale = torch.zeros(2, 4)
        torch.testing.assert_close(_modulate(x, shift, scale), x)

    def test_shift_only(self) -> None:
        x = torch.zeros(2, 3, 4)
        shift = torch.full((2, 4), 2.5)
        scale = torch.zeros(2, 4)
        out = _modulate(x, shift, scale)
        assert float(out.max()) == 2.5


class TestRotaryHelpers:
    def test_rotate_half_shape(self) -> None:
        x = torch.arange(8.0).view(1, 1, 8)
        out = rotate_half(x)
        assert out.shape == x.shape

    def test_apply_rotary_preserves_shape(self) -> None:
        q = torch.randn(1, 4, 2, 16)
        k = torch.randn(1, 4, 2, 16)
        cos = torch.randn(1, 2, 16)
        sin = torch.randn(1, 2, 16)
        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_repeat_kv_no_rep(self) -> None:
        x = torch.randn(2, 3, 4, 8)
        torch.testing.assert_close(repeat_kv(x, 1), x)

    def test_repeat_kv_expands(self) -> None:
        x = torch.randn(2, 2, 4, 8)
        out = repeat_kv(x, 3)
        assert out.shape == (2, 6, 4, 8)


class TestTextRMSNorm:
    def test_preserves_shape(self) -> None:
        norm = MolmoAct2RMSNorm(size=16)
        x = torch.randn(2, 5, 16)
        out = norm(x)
        assert out.shape == x.shape

    def test_zero_input_zero_output(self) -> None:
        norm = MolmoAct2RMSNorm(size=8, eps=1e-6)
        x = torch.zeros(1, 4, 8)
        torch.testing.assert_close(norm(x), x)


class TestRotaryEmbedding:
    def test_cos_sin_shapes(self, tiny_molmoact2_config: MolmoAct2Config) -> None:
        rope = MolmoAct2RotaryEmbedding(tiny_molmoact2_config.text_config)
        x = torch.zeros(2, 4, 64)
        positions = torch.arange(4).unsqueeze(0).expand(2, 4)
        cos, sin = rope(x, positions)
        assert cos.shape == (2, 4, 16)
        assert sin.shape == (2, 4, 16)


class TestAttention:
    def test_returns_hidden_and_kv(
        self, tiny_molmoact2_config: MolmoAct2Config
    ) -> None:
        config = tiny_molmoact2_config.text_config
        attn = MolmoAct2Attention(config)
        hidden = torch.randn(2, 5, config.hidden_size)
        cos = torch.randn(2, 5, config.head_dim)
        sin = torch.randn(2, 5, config.head_dim)
        out, kv = attn(hidden, (cos, sin), attention_bias=None)
        assert out.shape == (2, 5, config.hidden_size)
        assert isinstance(kv, tuple)
        assert len(kv) == 2
        assert kv[0].shape == (2, config.num_key_value_heads, 5, config.head_dim)


class TestEmbedding:
    def test_lookup_shape(self) -> None:
        embedding = MolmoAct2Embedding(
            num_embeddings=100, num_new_embeddings=10, features=16
        )
        tokens = torch.randint(0, 110, (2, 7))
        out = embedding(tokens)
        assert out.shape == (2, 7, 16)

    def test_new_tokens_available(self) -> None:
        embedding = MolmoAct2Embedding(
            num_embeddings=100, num_new_embeddings=10, features=16
        )
        tokens = torch.tensor([[105]])
        out = embedding(tokens)
        assert out.shape == (1, 1, 16)


class TestTextModel:
    def test_forward_shape(self, tiny_molmoact2_config: MolmoAct2Config) -> None:
        config = tiny_molmoact2_config.text_config
        model = MolmoAct2TextModel(config)
        inputs_embeds = torch.randn(2, 6, config.hidden_size)
        hidden, kv_states = model(inputs_embeds)
        assert hidden.shape == (2, 6, config.hidden_size)
        assert len(kv_states) == config.num_hidden_layers

    def test_norm_after_unsupported(
        self, tiny_molmoact2_config: MolmoAct2Config
    ) -> None:
        config = tiny_molmoact2_config.text_config
        config.norm_after = True
        with pytest.raises(NotImplementedError, match="norm_after=False"):
            MolmoAct2TextModel(config)


class TestVisionBackbone:
    def test_constructs_with_tiny_config(
        self, tiny_molmoact2_config: MolmoAct2Config
    ) -> None:
        backbone = MolmoAct2VisionBackbone(
            tiny_molmoact2_config.vit_config, tiny_molmoact2_config.adapter_config
        )
        assert backbone is not None


class TestActionExpertRMSNorm:
    def test_no_weight_when_no_affine(self) -> None:
        norm = ActionExpertRMSNorm(16)
        assert norm.weight is None
        x = torch.randn(2, 4, 16)
        assert norm(x).shape == x.shape

    def test_weight_when_affine(self) -> None:
        norm = ActionExpertRMSNorm(16, elementwise_affine=True)
        assert norm.weight is not None


class TestActionExpertRotaryEmbedding:
    def test_odd_head_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="even head_dim"):
            ActionExpertRotaryEmbedding(head_dim=7)

    def test_build_cache_shape(self) -> None:
        rope = ActionExpertRotaryEmbedding(head_dim=16)
        cos, sin = rope.build_cache(seq_len=4, device=torch.device("cpu"), dtype=torch.float32)
        assert cos.shape == (1, 1, 4, 8)
        assert sin.shape == (1, 1, 4, 8)


class TestSinusoidalTimeEmbedding:
    def test_even_dim(self) -> None:
        embed = SinusoidalTimeEmbedding(dim=16)
        out = embed(torch.tensor([0.1, 0.5]))
        assert out.shape == (2, 16)

    def test_odd_dim_pads(self) -> None:
        embed = SinusoidalTimeEmbedding(dim=17)
        out = embed(torch.tensor([0.1]))
        assert out.shape == (1, 17)


class TestActionExpertSelfAttention:
    def test_forward_shape(self) -> None:
        attn = ActionExpertSelfAttention(hidden_size=64, num_heads=4, qk_norm=True, qk_norm_eps=1e-6, rope=True)
        x = torch.randn(2, 5, 64)
        out = attn(x, attn_mask=None, is_causal=True, rope_cache=None)
        assert out.shape == (2, 5, 64)


class TestActionExpertMLP:
    def test_forward_shape(self) -> None:
        mlp = ActionExpertMLP(hidden_size=64, mlp_ratio=2.0, multiple_of=16)
        x = torch.randn(2, 3, 64)
        assert mlp(x).shape == (2, 3, 64)


class TestActionExpertModulation:
    def test_forward_shape(self) -> None:
        mod = ActionExpertModulation(hidden_size=64, num_chunks=9)
        cond = torch.randn(2, 64)
        assert mod(cond).shape == (2, 9 * 64)


class TestActionExpertFinalLayer:
    def test_forward_shape(self) -> None:
        layer = ActionExpertFinalLayer(hidden_size=64, output_dim=6)
        x = torch.randn(2, 4, 64)
        cond = torch.randn(2, 64)
        assert layer(x, cond).shape == (2, 4, 6)


class TestActionExpert:
    def test_constructs(
        self, tiny_molmoact2_config: MolmoAct2Config
    ) -> None:
        config = tiny_molmoact2_config.action_expert_config
        assert config is not None
        expert = ActionExpert(
            config,
            llm_kv_dim=tiny_molmoact2_config.num_key_value_heads * tiny_molmoact2_config.head_dim,
            llm_num_layers=tiny_molmoact2_config.text_config.num_hidden_layers,
        )
        assert expert.num_heads == config.num_heads

    def test_raises_when_layer_count_mismatch(
        self, tiny_molmoact2_config: MolmoAct2Config
    ) -> None:
        config = tiny_molmoact2_config.action_expert_config
        assert config is not None
        with pytest.raises(ValueError, match="one block per text layer"):
            ActionExpert(config, llm_kv_dim=64, llm_num_layers=99)

    def test_prepare_context_and_denoise(
        self, tiny_molmoact2_config: MolmoAct2Config
    ) -> None:
        config = tiny_molmoact2_config.action_expert_config
        assert config is not None
        expert = ActionExpert(
            config,
            llm_kv_dim=tiny_molmoact2_config.num_key_value_heads * tiny_molmoact2_config.head_dim,
            llm_num_layers=tiny_molmoact2_config.text_config.num_hidden_layers,
        )
        batch = 2
        seq_len = 8
        kv_dim = tiny_molmoact2_config.num_key_value_heads * tiny_molmoact2_config.head_dim
        kv_states = [
            (torch.randn(batch, seq_len, kv_dim), torch.randn(batch, seq_len, kv_dim))
            for _ in range(tiny_molmoact2_config.text_config.num_hidden_layers)
        ]
        mask = torch.ones(batch, seq_len, dtype=torch.bool)
        context = expert.prepare_context(
            encoder_kv_states=kv_states,
            encoder_attention_mask=mask,
            seq_len=config.max_action_horizon,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert isinstance(context, ActionExpertContext)
        assert len(context.kv_contexts) == config.num_layers

        actions = torch.randn(batch, config.max_action_horizon, config.max_action_dim)
        timesteps = torch.rand(batch)
        velocity = expert.forward_with_context(actions, timesteps, context=context)
        assert velocity.shape == (batch, config.max_action_horizon, config.max_action_dim)


class TestMolmoAct2Backbone:
    def test_constructs_with_tiny_config(
        self, tiny_molmoact2_config: MolmoAct2Config
    ) -> None:
        backbone = MolmoAct2Backbone(tiny_molmoact2_config)
        assert backbone.transformer is not None
        assert backbone.vision_backbone is not None
        assert backbone.action_expert is not None

    def test_no_action_expert_when_disabled(
        self, tiny_molmoact2_config: MolmoAct2Config
    ) -> None:
        tiny_molmoact2_config.add_action_expert = False
        backbone = MolmoAct2Backbone(tiny_molmoact2_config)
        assert backbone.action_expert is None


peft = pytest.importorskip("peft", reason="peft is required for LoRA tests")


class TestMolmoAct2DeltaIndices:
    """Dataset delta indices requested by the MolmoAct2 model frontend."""

    def test_requests_action_chunk_only(self, tiny_molmoact2_config: MolmoAct2Config) -> None:
        tiny_molmoact2_config.chunk_size = 10
        model = MolmoAct2Model(tiny_molmoact2_config)

        assert model.action_delta_indices == list(range(10))
        assert model.observation_delta_indices is None
        assert model.reward_delta_indices is None


class TestMaskedActionMse:
    def test_horizon_padding_is_excluded_from_loss_and_gradients(self) -> None:
        predicted = torch.tensor([[[[2.0], [10.0]]]], requires_grad=True)
        target = torch.zeros_like(predicted)

        loss = _masked_action_mse(
            predicted,
            target,
            action_horizon_is_pad=torch.tensor([[False, True]]),
            action_dim_is_pad=None,
        )

        torch.testing.assert_close(loss, torch.tensor(4.0))
        loss.backward()
        torch.testing.assert_close(predicted.grad, torch.tensor([[[[4.0], [0.0]]]]))

    def test_denominator_counts_valid_steps_and_dimensions_only(self) -> None:
        predicted = torch.tensor([[[[2.0, 100.0], [50.0, 50.0]]]])
        target = torch.zeros_like(predicted)

        loss = _masked_action_mse(
            predicted,
            target,
            action_horizon_is_pad=torch.tensor([[False, True]]),
            action_dim_is_pad=torch.tensor([[False, True]]),
        )

        torch.testing.assert_close(loss, torch.tensor(4.0))

    def test_fully_padded_actions_are_finite_and_zero(self) -> None:
        predicted = torch.ones(1, 1, 2, 1, requires_grad=True)
        target = torch.zeros_like(predicted)

        loss = _masked_action_mse(
            predicted,
            target,
            action_horizon_is_pad=torch.tensor([[True, True]]),
            action_dim_is_pad=None,
        )

        torch.testing.assert_close(loss, torch.tensor(0.0))
        loss.backward()
        torch.testing.assert_close(predicted.grad, torch.zeros_like(predicted))

    def test_without_masks_matches_plain_mean_mse(self) -> None:
        predicted = torch.tensor([[[2.0], [4.0]]])
        target = torch.zeros_like(predicted)

        loss = _masked_action_mse(
            predicted,
            target,
            action_horizon_is_pad=None,
            action_dim_is_pad=None,
        )

        torch.testing.assert_close(loss, torch.tensor(10.0))


class TestMolmoAct2LoRA:
    """LoRA adapter application on the MolmoAct2 model frontend."""

    def _make_model(self, config: MolmoAct2Config) -> MolmoAct2Model:
        config.use_lora = False
        config.train_action_expert_only = False
        return MolmoAct2Model(config)

    def test_apply_lora_wraps_backbone(
        self, tiny_molmoact2_config: MolmoAct2Config
    ) -> None:
        model = self._make_model(tiny_molmoact2_config)
        original_backbone = model.backbone
        model.apply_lora_adapters()
        # PEFT wrapper replaces the backbone module.
        assert model.backbone is not original_backbone
        assert hasattr(model.backbone, "base_model")

    def test_apply_lora_creates_trainable_lora_params(
        self, tiny_molmoact2_config: MolmoAct2Config
    ) -> None:
        model = self._make_model(tiny_molmoact2_config)
        model.apply_lora_adapters()
        lora_params = [
            name
            for name, param in model.named_parameters()
            if param.requires_grad and "lora_" in name
        ]
        assert lora_params, "expected LoRA parameters to be trainable"

    def test_apply_lora_freezes_non_lora_vlm_params(
        self, tiny_molmoact2_config: MolmoAct2Config
    ) -> None:
        model = self._make_model(tiny_molmoact2_config)
        model.apply_lora_adapters()
        for name, param in model.named_parameters():
            if "lora_" in name:
                continue
            if "action_expert" in name:
                # Action expert stays trainable when enable_lora_action_expert is False.
                assert param.requires_grad, f"action_expert param should be trainable: {name}"
            else:
                assert not param.requires_grad, f"VLM param should be frozen: {name}"

    def test_apply_lora_unfreezes_action_expert_when_not_targeted(
        self, tiny_molmoact2_config: MolmoAct2Config
    ) -> None:
        model = self._make_model(tiny_molmoact2_config)
        model.apply_lora_adapters()
        trainable_action = [
            name
            for name, param in model.named_parameters()
            if param.requires_grad and "action_expert" in name
        ]
        assert trainable_action, "action_expert params should remain trainable"

    def test_enable_lora_action_expert_freezes_non_lora_action_expert(
        self, tiny_molmoact2_config: MolmoAct2Config
    ) -> None:
        tiny_molmoact2_config.enable_lora_action_expert = True
        model = self._make_model(tiny_molmoact2_config)
        model.apply_lora_adapters()
        for name, param in model.named_parameters():
            if "lora_" in name:
                assert param.requires_grad, f"lora param should be trainable: {name}"
            else:
                assert not param.requires_grad, f"non-lora param should be frozen: {name}"

    def test_apply_lora_raises_without_peft(
        self, tiny_molmoact2_config: MolmoAct2Config, monkeypatch
    ) -> None:
        model = self._make_model(tiny_molmoact2_config)
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "peft":
                msg = "peft not installed"
                raise ImportError(msg)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        with pytest.raises(ImportError, match="MolmoAct2 LoRA requires peft"):
            model.apply_lora_adapters()

    def test_for_cond_gen_accessor_returns_inner_module(
        self, tiny_molmoact2_config: MolmoAct2Config
    ) -> None:
        model = self._make_model(tiny_molmoact2_config)
        pre = model._for_cond_gen
        assert isinstance(pre, MolmoAct2ForConditionalGeneration)
        model.apply_lora_adapters()
        post = model._for_cond_gen
        assert isinstance(post, MolmoAct2ForConditionalGeneration)
        # The accessor follows the PEFT wrapper to the same underlying instance.
        assert post is pre


class TestMolmoAct2GradientCheckpointing:
    """Gradient-checkpointing enable/disable on the MolmoAct2 model frontend."""

    def _make_model(self, config: MolmoAct2Config) -> MolmoAct2Model:
        config.gradient_checkpointing = False
        return MolmoAct2Model(config)

    def test_enable_sets_submodule_flags(self, tiny_molmoact2_config: MolmoAct2Config) -> None:
        model = self._make_model(tiny_molmoact2_config)
        backbone = model._for_cond_gen.model
        assert backbone.transformer.gradient_checkpointing is False
        assert backbone.vision_backbone.gradient_checkpointing is False
        assert backbone.action_expert.gradient_checkpointing is False
        model.gradient_checkpointing_enable()
        assert backbone.transformer.gradient_checkpointing is True
        assert backbone.vision_backbone.gradient_checkpointing is True
        assert backbone.action_expert.gradient_checkpointing is True

    def test_enable_forces_eager_attention(self, tiny_molmoact2_config: MolmoAct2Config) -> None:
        model = self._make_model(tiny_molmoact2_config)
        model.gradient_checkpointing_enable()
        backbone = model._for_cond_gen.model
        assert backbone.transformer.config._attn_implementation == "eager"
        assert backbone.vision_backbone.image_vit.config._attn_implementation == "eager"

    def test_disable_clears_submodule_flags(self, tiny_molmoact2_config: MolmoAct2Config) -> None:
        model = self._make_model(tiny_molmoact2_config)
        model.gradient_checkpointing_enable()
        model.gradient_checkpointing_disable()
        backbone = model._for_cond_gen.model
        assert backbone.transformer.gradient_checkpointing is False
        assert backbone.vision_backbone.gradient_checkpointing is False
        assert backbone.action_expert.gradient_checkpointing is False

    def test_config_flag_enables_at_construction(
        self, tiny_molmoact2_config: MolmoAct2Config
    ) -> None:
        tiny_molmoact2_config.gradient_checkpointing = True
        model = MolmoAct2Model(tiny_molmoact2_config)
        backbone = model._for_cond_gen.model
        assert backbone.transformer.gradient_checkpointing is True
        assert backbone.vision_backbone.gradient_checkpointing is True
        assert backbone.action_expert.gradient_checkpointing is True

    def test_checkpointed_text_forward_matches_eager(
        self, tiny_molmoact2_config: MolmoAct2Config
    ) -> None:
        torch.manual_seed(0)
        model_eager = self._make_model(tiny_molmoact2_config)
        torch.manual_seed(0)
        model_ckpt = self._make_model(tiny_molmoact2_config)
        model_ckpt.gradient_checkpointing_enable()

        config = tiny_molmoact2_config.text_config
        inputs_embeds = torch.randn(2, 4, config.hidden_size, requires_grad=True)
        attention_bias = None

        model_eager.eval()
        model_ckpt.eval()
        eager_out, eager_kv = model_eager._for_cond_gen.model.transformer(inputs_embeds, attention_bias)
        ckpt_out, ckpt_kv = model_ckpt._for_cond_gen.model.transformer(inputs_embeds, attention_bias)
        torch.testing.assert_close(ckpt_out, eager_out)
        assert len(ckpt_kv) == len(eager_kv)

    def test_checkpointed_action_expert_forward_backward_runs(
        self, tiny_molmoact2_config: MolmoAct2Config
    ) -> None:
        config = tiny_molmoact2_config.action_expert_config
        assert config is not None
        expert = ActionExpert(
            config,
            llm_kv_dim=tiny_molmoact2_config.num_key_value_heads * tiny_molmoact2_config.head_dim,
            llm_num_layers=tiny_molmoact2_config.text_config.num_hidden_layers,
        )
        expert.gradient_checkpointing = True
        expert.train()
        batch = 2
        seq_len = config.max_action_horizon
        kv_states = [
            (torch.randn(batch, seq_len, config.hidden_size), torch.randn(batch, seq_len, config.hidden_size))
            for _ in range(tiny_molmoact2_config.text_config.num_hidden_layers)
        ]
        mask = torch.ones(batch, seq_len, dtype=torch.bool)
        context = expert.prepare_context(
            encoder_kv_states=kv_states,
            encoder_attention_mask=mask,
            seq_len=seq_len,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        actions = torch.randn(batch, seq_len, config.max_action_dim, requires_grad=True)
        timesteps = torch.rand(batch)
        velocity = expert.forward_with_context(actions, timesteps, context=context)
        loss = velocity.sum()
        loss.backward()
        assert actions.grad is not None


class TestSampleBetaTimesteps:
    def test_shape_and_bounds(self) -> None:
        timesteps = _sample_beta_timesteps(
            batch_size=4,
            device=torch.device("cpu"),
            cutoff=1.0,
            time_offset=0.0,
            time_scale=1.0,
            alpha=1.0,
            beta=1.5,
        )
        assert timesteps.shape == (4,)
        assert float(timesteps.min()) >= 0.0
        assert float(timesteps.max()) <= 1.0

    def test_zero_scale_returns_offset(self) -> None:
        timesteps = _sample_beta_timesteps(
            batch_size=3,
            device=torch.device("cpu"),
            cutoff=0.5,
            time_offset=0.5,
            time_scale=0.0,
            alpha=1.0,
            beta=1.0,
        )
        torch.testing.assert_close(timesteps, torch.full((3,), 0.5))
