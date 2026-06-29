# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 processor package with composable preprocessing components."""

from .factory import make_molmoact2_preprocessors
from .image_processing_local import MolmoAct2ImageProcessor, MolmoAct2ImagesKwargs
from .local_processor import load_molmoact2_processor_from_pretrained
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
from .processing_local import MolmoAct2Processor, MolmoAct2ProcessorKwargs
from .video_processing_local import MolmoAct2VideoProcessor, MolmoAct2VideoProcessorKwargs

__all__ = [
    "ActionExtractor",
    "ActionPadder",
    "FeatureBatchNormalizer",
    "MolmoAct2ImageProcessor",
    "MolmoAct2Postprocessor",
    "MolmoAct2Preprocessor",
    "MolmoAct2Processor",
    "MolmoAct2ProcessorKwargs",
    "MolmoAct2ImagesKwargs",
    "MolmoAct2VideoProcessor",
    "MolmoAct2VideoProcessorKwargs",
    "PreprocessBatchBundle",
    "PromptPack",
    "RobotPromptEncoder",
    "StateTaskImageExtractor",
    "load_molmoact2_processor_from_pretrained",
    "make_molmoact2_preprocessors",
]
