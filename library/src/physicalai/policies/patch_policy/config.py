# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Patch Policy Config."""

from dataclasses import dataclass

from physicalai.config import Config


@dataclass(frozen=True)
class PatchPolicyConfig(Config):
    """Configuration for the Patch Policy."""

    input_features: list[str] | None = None
    output_features: list[str] | None = None
    n_action_steps: int = 50

    # Image encoder arguments
    encoder_name: str = "webssl"

    # Goal args
    use_goal_image: bool = False
