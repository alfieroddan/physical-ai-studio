# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 model implementation."""

from torch import nn


class MolmoAct2Backbone(nn.Module):
    """MolmoAct2 base contains the vision backbone and language backbone.

    The forward pass is split between:
        - MolmoAct2TextModel
        - MolmoAct2VisionBackbone
    """
    pass