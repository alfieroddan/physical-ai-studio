# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 model package.

This package contains model-owned components:
- backbone architecture
- torch frontend/inference boundary
- image and video preprocessing attached to model execution
"""

from .backbones import MolmoAct2ForConditionalGeneration
from .image import MolmoAct2ImageProcessor, MolmoAct2ImagesOptions
from .model import MolmoAct2Model
from .video import MolmoAct2VideoProcessor, MolmoAct2VideoProcessorOptions

__all__ = [
    "MolmoAct2ForConditionalGeneration",
    "MolmoAct2ImageProcessor",
    "MolmoAct2ImagesOptions",
    "MolmoAct2Model",
    "MolmoAct2VideoProcessor",
    "MolmoAct2VideoProcessorOptions",
]
