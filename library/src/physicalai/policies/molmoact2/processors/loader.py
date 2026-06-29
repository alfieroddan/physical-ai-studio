# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Local MolmoAct2 processor loader scoped under the processors package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from transformers import Qwen2Tokenizer

from ..config import MolmoAct2ProcessorConfig
from .image import MolmoAct2ImageProcessor
from .processor import MolmoAct2Processor
from .video import MolmoAct2VideoProcessor


def load_molmoact2_processor(
    tokenizer_name_or_path: str | Path,
    processor_config: MolmoAct2ProcessorConfig | dict[str, Any] | None = None,
) -> MolmoAct2Processor:
    """Load MolmoAct2 processor from checkpoint tokenizer and processor config data."""
    if processor_config is None:
        processor_config = MolmoAct2ProcessorConfig()
    elif isinstance(processor_config, dict):
        processor_config = MolmoAct2ProcessorConfig.from_dict(processor_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(
        str(tokenizer_name_or_path),
        local_files_only=True,
    )

    image_processor_config = processor_config.image_processor
    image_processor = MolmoAct2ImageProcessor(
        size=image_processor_config.size,
        image_mean=image_processor_config.image_mean,
        image_std=image_processor_config.image_std,
        do_convert_rgb=image_processor_config.do_convert_rgb,
        max_crops=image_processor_config.max_crops,
        overlap_margins=image_processor_config.overlap_margins,
        crop_mode=image_processor_config.crop_mode,
        patch_size=image_processor_config.patch_size,
        pooling_size=image_processor_config.pooling_size,
    )

    video_processor_config = processor_config.video_processor
    video_processor = MolmoAct2VideoProcessor(
        size=video_processor_config.size,
        image_mean=video_processor_config.image_mean,
        image_std=video_processor_config.image_std,
        do_convert_rgb=video_processor_config.do_convert_rgb,
        patch_size=video_processor_config.patch_size,
        pooling_size=video_processor_config.pooling_size,
        do_sample_frames=video_processor_config.do_sample_frames,
        frame_sample_mode=video_processor_config.frame_sample_mode,
        max_fps=int(video_processor_config.max_fps),
        sampling_fps=video_processor_config.sampling_fps,
    )

    return MolmoAct2Processor(
        image_processor=image_processor,
        video_processor=video_processor,
        tokenizer=tokenizer,
        chat_template=processor_config.chat_template,
        image_use_col_tokens=processor_config.image_use_col_tokens,
        use_single_crop_col_tokens=processor_config.use_single_crop_col_tokens,
        use_single_crop_start_token=processor_config.use_single_crop_start_token,
        video_use_col_tokens=processor_config.video_use_col_tokens,
        use_frame_special_tokens=processor_config.use_frame_special_tokens,
    )
