# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the diffusion action head."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from physicalai.policies.components import DiffusionActionHead
from physicalai.policies.components.action_heads import Context, make_betas

CHUNK_SIZE, ACTION_DIM, CONTEXT_DIM, NUM_TRAIN_TIMESTEPS = 4, 3, 8, 100


class ToyDiffusionHead(DiffusionActionHead):
    """Diffusion head with a single linear layer as the denoiser."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(CHUNK_SIZE, ACTION_DIM, num_train_timesteps=NUM_TRAIN_TIMESTEPS, **kwargs)
        self.net = nn.Linear(ACTION_DIM + CONTEXT_DIM + 1, ACTION_DIM)

    def denoise(self, x_t: torch.Tensor, t: torch.Tensor, context: Context) -> torch.Tensor:
        pooled = context["tokens"].mean(dim=1, keepdim=True).expand(-1, x_t.shape[1], -1)
        time = (t.to(x_t.dtype) / self.num_train_timesteps)[:, None, None].expand(-1, x_t.shape[1], 1)
        return self.net(torch.cat([x_t, pooled, time], dim=-1))


@pytest.fixture
def context() -> Context:
    """Context with a batch of 2 and 6 tokens."""
    return {"tokens": torch.randn(2, 6, CONTEXT_DIM)}


def _diffusers_sample(
    head: ToyDiffusionHead, scheduler, context: Context, noise: torch.Tensor, **step_kwargs
) -> torch.Tensor:  # noqa: ANN001, ANN003
    x = noise
    for t in scheduler.timesteps:
        output = head.denoise(x, t.expand(x.shape[0]), context)
        x = scheduler.step(output, t, x, **step_kwargs).prev_sample
    return x


class TestSchedule:
    """Tests for the noise schedule."""

    @pytest.mark.parametrize("schedule", ["linear", "scaled_linear", "squaredcos_cap_v2"])
    def test_betas_match_diffusers(self, schedule: str) -> None:
        """Test beta schedules use the diffusers definitions."""
        diffusers = pytest.importorskip("diffusers")
        expected = diffusers.DDPMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS, beta_schedule=schedule).betas
        torch.testing.assert_close(make_betas(schedule, NUM_TRAIN_TIMESTEPS), expected)

    def test_timesteps(self) -> None:
        """Test the schedule runs from noise to the clean-data sentinel -1."""
        head = ToyDiffusionHead()
        timesteps = head.timesteps(10, device=torch.device("cpu"), dtype=torch.float32)
        assert timesteps.tolist() == [90, 80, 70, 60, 50, 40, 30, 20, 10, 0, -1]
        assert head.timesteps(NUM_TRAIN_TIMESTEPS, torch.device("cpu"), torch.float32)[0] == NUM_TRAIN_TIMESTEPS - 1

    def test_invalid_arguments(self) -> None:
        """Test out-of-range arguments are rejected."""
        with pytest.raises(ValueError, match="num_steps"):
            ToyDiffusionHead(num_inference_steps=NUM_TRAIN_TIMESTEPS + 1)
        with pytest.raises(ValueError, match="prediction_type"):
            ToyDiffusionHead(prediction_type="v_prediction")
        with pytest.raises(ValueError, match="beta_schedule"):
            ToyDiffusionHead(beta_schedule="sigmoid")


class TestDiffusionActionHead:
    """Tests for training and sampling."""

    def test_add_noise_matches_diffusers(self) -> None:
        """Test the forward process matches diffusers."""
        diffusers = pytest.importorskip("diffusers")
        scheduler = diffusers.DDPMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS, beta_schedule="squaredcos_cap_v2")
        head = ToyDiffusionHead()
        actions, noise, t = (
            torch.randn(5, CHUNK_SIZE, ACTION_DIM),
            torch.randn(5, CHUNK_SIZE, ACTION_DIM),
            torch.tensor([0, 1, 50, 98, 99]),
        )
        torch.testing.assert_close(head.add_noise(actions, noise, t), scheduler.add_noise(actions, noise, t))

    @pytest.mark.parametrize("num_steps", [NUM_TRAIN_TIMESTEPS, 10])
    @pytest.mark.parametrize("prediction_type", ["epsilon", "sample"])
    def test_ddpm_matches_diffusers(self, context: Context, num_steps: int, prediction_type: str) -> None:
        """Test eta = 1 reproduces diffusers' DDPMScheduler, including sample clipping."""
        diffusers = pytest.importorskip("diffusers")
        head = ToyDiffusionHead(num_inference_steps=num_steps, prediction_type=prediction_type, eta=1.0)
        scheduler = diffusers.DDPMScheduler(
            num_train_timesteps=NUM_TRAIN_TIMESTEPS,
            beta_schedule="squaredcos_cap_v2",
            prediction_type=prediction_type,
            clip_sample=True,
        )
        scheduler.set_timesteps(num_steps)
        noise = torch.randn(2, CHUNK_SIZE, ACTION_DIM)

        torch.manual_seed(0)
        expected = _diffusers_sample(head, scheduler, context, noise)
        torch.manual_seed(0)
        actual = head.sample(context, noise=noise)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("prediction_type", ["epsilon", "sample"])
    def test_ddim_matches_diffusers(self, context: Context, prediction_type: str) -> None:
        """Test eta = 0 reproduces diffusers' DDIMScheduler and is deterministic."""
        diffusers = pytest.importorskip("diffusers")
        head = ToyDiffusionHead(num_inference_steps=10, prediction_type=prediction_type, eta=0.0)
        scheduler = diffusers.DDIMScheduler(
            num_train_timesteps=NUM_TRAIN_TIMESTEPS,
            beta_schedule="squaredcos_cap_v2",
            prediction_type=prediction_type,
            clip_sample=True,
        )
        scheduler.set_timesteps(10)
        noise = torch.randn(2, CHUNK_SIZE, ACTION_DIM)

        expected = _diffusers_sample(head, scheduler, context, noise, eta=0.0, use_clipped_model_output=True)
        torch.testing.assert_close(head.sample(context, noise=noise), expected, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(head.sample(context, noise=noise), head.sample(context, noise=noise))

    def test_sample_is_clipped(self, context: Context) -> None:
        """Test sampled actions stay within the clip range."""
        head = ToyDiffusionHead(num_inference_steps=10, clip_sample_range=0.5)
        assert head.sample(context).abs().max() <= 0.5 + 1e-6

    @pytest.mark.parametrize("prediction_type", ["epsilon", "sample"])
    def test_compute_loss(self, context: Context, prediction_type: str) -> None:
        """Test the loss is unreduced and backpropagates into the denoiser."""
        head = ToyDiffusionHead(prediction_type=prediction_type)
        losses = head.compute_loss(torch.rand(2, CHUNK_SIZE, ACTION_DIM) * 2 - 1, context)
        assert losses.shape == (2, CHUNK_SIZE, ACTION_DIM)
        losses.mean().backward()
        assert head.net.weight.grad is not None

    def test_schedule_not_in_state_dict(self) -> None:
        """Test the schedule buffers stay out of checkpoints."""
        assert set(ToyDiffusionHead().state_dict()) == {"net.weight", "net.bias"}

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_graph_replay_cuda(self) -> None:
        """Test graph replay matches eager sampling, reuses graphs and only runs at inference."""
        head = ToyDiffusionHead(num_inference_steps=10, eta=0.0, use_random_input_noise=False).cuda().eval()
        head.enable_graph_replay()
        contexts = [{"tokens": torch.randn(2, 6, CONTEXT_DIM, device="cuda")} for _ in range(2)]
        with torch.inference_mode():
            for context in contexts:
                torch.testing.assert_close(head.sample(context), head._sample(context, None, 10))  # noqa: SLF001
            assert len(head._graphs) == 1  # noqa: SLF001
            head.sample(contexts[0], num_steps=5)
            assert len(head._graphs) == 2  # noqa: SLF001

        head.train()
        with torch.inference_mode():
            head.sample(contexts[0], num_steps=3)
        assert len(head._graphs) == 2  # noqa: SLF001

        head.double()
        assert head._graphs == {}  # noqa: SLF001

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_graph_replay_ddpm_draws_new_noise(self) -> None:
        """Test a replayed DDPM graph draws fresh noise on every call."""
        head = ToyDiffusionHead(num_inference_steps=10).cuda().eval()
        head.enable_graph_replay()
        context = {"tokens": torch.randn(2, 6, CONTEXT_DIM, device="cuda")}
        with torch.inference_mode():
            assert not torch.allclose(head.sample(context), head.sample(context))

    def test_graph_replay_falls_back_on_cpu(self, context: Context) -> None:
        """Test enable_graph_replay leaves CPU sampling eager and unchanged."""
        head = ToyDiffusionHead(num_inference_steps=10, eta=0.0, use_random_input_noise=False).eval()
        expected = head.sample(context)
        head.enable_graph_replay()
        with torch.inference_mode():
            torch.testing.assert_close(head.sample(context), expected)
        assert head._graphs == {}  # noqa: SLF001

    def test_zero_input_noise(self, context: Context) -> None:
        """Test use_random_input_noise=False starts from zeros, so DDIM sampling is deterministic."""
        head = ToyDiffusionHead(num_inference_steps=10, eta=0.0, use_random_input_noise=False)
        zeros = torch.zeros(2, CHUNK_SIZE, ACTION_DIM)
        torch.testing.assert_close(head.sample(context), head.sample(context, noise=zeros))

    @pytest.mark.parametrize("eta", [0.0, 1.0])
    @pytest.mark.parametrize("prediction_type", ["epsilon", "sample"])
    def test_bfloat16(self, context: Context, prediction_type: str, eta: float) -> None:
        """Test a bfloat16 head samples in bfloat16 and stays finite near t = 0, where 1 - alpha_bar is tiny."""
        head = ToyDiffusionHead(num_inference_steps=NUM_TRAIN_TIMESTEPS, prediction_type=prediction_type, eta=eta)
        head = head.to(torch.bfloat16)
        assert head.sqrt_alpha_bar.dtype == head.sqrt_one_minus_alpha_bar.dtype == torch.bfloat16
        assert head.sqrt_one_minus_alpha_bar[1] > 0

        actions = head.sample({"tokens": context["tokens"].to(torch.bfloat16)})
        assert actions.dtype == torch.bfloat16
        assert torch.isfinite(actions).all()
