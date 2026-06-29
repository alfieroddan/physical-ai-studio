# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 processor package with composable preprocessing components."""

from .factory import make_molmoact2_preprocessors
from .image import MolmoAct2ImageProcessor, MolmoAct2ImagesOptions
from .loader import load_molmoact2_processor
from .postprocessor import MolmoAct2Postprocessor
from .preprocessor import MolmoAct2Preprocessor
from .preprocess_steps import (
    ActionExtractor,
    ActionPadder,
    FeatureBatchNormalizer,
    PromptPack,
    PreprocessBatchBundle,
    RobotPromptEncoder,
    StateTaskImageExtractor,
)
from .processor import MolmoAct2Processor, MolmoAct2ProcessorOptions
from .video import MolmoAct2VideoProcessor, MolmoAct2VideoProcessorOptions

__all__ = [
    "ActionExtractor",
    "ActionPadder",
    "FeatureBatchNormalizer",
    "MolmoAct2ImageProcessor",
    "MolmoAct2Postprocessor",
    "MolmoAct2Preprocessor",
    "MolmoAct2Processor",
    "MolmoAct2ProcessorOptions",
    "MolmoAct2ImagesOptions",
    "MolmoAct2VideoProcessor",
    "MolmoAct2VideoProcessorOptions",
    "PreprocessBatchBundle",
    "PromptPack",
    "RobotPromptEncoder",
    "StateTaskImageExtractor",
    "load_molmoact2_processor",
    "make_molmoact2_preprocessors",
]
