# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MolmoAct2 optimizer."""

from __future__ import annotations

import math

import pytest
import torch

from physicalai.policies.molmoact2.optimizer import MolmoAct2AdamW, molmoact2_cosine_with_warmup_scheduler


def test_scheduler_keeps_configured_clock_for_short_runs() -> None:
    first = torch.nn.Parameter(torch.tensor([1.0]))
    second = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW(
        [
            {"params": [first], "lr": 5e-5},
            {"params": [second], "lr": 5e-6},
        ],
    )
    scheduler = molmoact2_cosine_with_warmup_scheduler(
        optimizer,
        peak_lr=5e-5,
        decay_lr=1e-6,
        num_warmup_steps=200,
        num_decay_steps=24_000,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5 / 200)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(5e-6 / 200)
    for _ in range(199):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(5e-6)

    for _ in range(2_800):
        optimizer.step()
        scheduler.step()
    cosine_step = 3_000 - 200
    expected_multiplier = 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * cosine_step / 23_800))
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5 * expected_multiplier)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(5e-6 * expected_multiplier)


def test_scheduler_stays_at_floor_after_decay() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=5e-5)
    scheduler = molmoact2_cosine_with_warmup_scheduler(
        optimizer,
        peak_lr=5e-5,
        decay_lr=1e-6,
        num_warmup_steps=2,
        num_decay_steps=5,
    )

    for _ in range(10):
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-6)


def test_updates_float32_parameters() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([1.0])
    optimizer = MolmoAct2AdamW([parameter], lr=0.1, group_grad_clip_norm=1.0)

    optimizer.step()

    assert parameter.item() < 1.0


def test_clips_each_parameter_group_independently() -> None:
    first = torch.nn.Parameter(torch.zeros(1))
    second = torch.nn.Parameter(torch.zeros(1))
    first.grad = torch.tensor([10.0])
    second.grad = torch.tensor([20.0])
    optimizer = MolmoAct2AdamW(
        [{"params": [first]}, {"params": [second]}],
        lr=0.1,
        group_grad_clip_norm=2.0,
    )

    optimizer._clip_grad_groups()

    torch.testing.assert_close(first.grad, torch.tensor([2.0]))
    torch.testing.assert_close(second.grad, torch.tensor([2.0]))


def test_bfloat16_updates_keep_compensation() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    parameter.grad = torch.tensor([1.0], dtype=torch.bfloat16)
    optimizer = MolmoAct2AdamW([parameter], lr=0.1, group_grad_clip_norm=1.0)

    optimizer.step()

    assert "compensation" in optimizer.state[parameter]
    assert optimizer.state[parameter]["compensation"].dtype == torch.bfloat16


def test_nonfinite_gradient_skips_update_and_clears_grad() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([float("inf")])
    optimizer = MolmoAct2AdamW([parameter], lr=0.1, group_grad_clip_norm=1.0)

    optimizer.step()

    torch.testing.assert_close(parameter, torch.tensor([1.0]))
    assert parameter.grad is None
    assert not optimizer.state[parameter]
