# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from .config import PatchPolicyConfig


def make_policy_processors(config: PatchPolicyConfig) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Make policy processors.
 
    Returns:
        A tuple of (preprocessor, postprocessor).
    """
    del config
    preprocessor = torch.nn.Identity()
    postprocessor = torch.nn.Identity()
    return preprocessor, postprocessor
