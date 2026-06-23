# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Local MolmoAct2 processor that doesn't depend on AutoProcessor or trust_remote_code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import Qwen2Tokenizer

from physicalai.policies.molmoact2.image_processing_local import MolmoAct2ImageProcessor
from physicalai.policies.molmoact2.processing_local import MolmoAct2Processor
from physicalai.policies.molmoact2.video_processing_local import MolmoAct2VideoProcessor


def load_molmoact2_processor_from_pretrained(
    tokenizer_path: str | Path,
    processor_config: dict[str, Any] | None = None,
) -> MolmoAct2Processor:
    """Load MolmoAct2 processor from checkpoint tokenizer and processor config data.

    Args:
        tokenizer_path: Local checkpoint directory containing the extended tokenizer vocab.
        processor_config: Optional pre-loaded processor config dict.

    Returns:
        Initialized MolmoAct2Processor instance.

    """
    tokenizer = Qwen2Tokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
    )

    processor_config = processor_config or {}

    # Load image processor
    image_processor_config = processor_config.get("image_processor", {})
    image_processor = MolmoAct2ImageProcessor(
        size=image_processor_config.get("size", {"height": 378, "width": 378}),
        image_mean=image_processor_config.get("image_mean", [0.5, 0.5, 0.5]),
        image_std=image_processor_config.get("image_std", [0.5, 0.5, 0.5]),
        do_convert_rgb=image_processor_config.get("do_convert_rgb", True),
        max_crops=image_processor_config.get("max_crops", 8),
        overlap_margins=image_processor_config.get("overlap_margins", [4, 4]),
        crop_mode=image_processor_config.get("crop_mode", "overlap-and-resize-c2"),
        patch_size=image_processor_config.get("patch_size", 14),
        pooling_size=image_processor_config.get("pooling_size", [2, 2]),
    )

    # Load video processor
    video_processor_config = processor_config.get("video_processor", {})
    video_processor = MolmoAct2VideoProcessor(
        size=video_processor_config.get("size", {"height": 378, "width": 378}),
        image_mean=video_processor_config.get("image_mean", [0.5, 0.5, 0.5]),
        image_std=video_processor_config.get("image_std", [0.5, 0.5, 0.5]),
        do_convert_rgb=video_processor_config.get("do_convert_rgb", True),
        patch_size=video_processor_config.get("patch_size", 14),
        pooling_size=video_processor_config.get("pooling_size", [3, 3]),
        do_sample_frames=video_processor_config.get("do_sample_frames", True),
        frame_sample_mode=video_processor_config.get("frame_sample_mode", "uniform_last_frame"),
        max_fps=video_processor_config.get("max_fps", 2),
        sampling_fps=video_processor_config.get("sampling_fps", 2),
    )

    # Create processor
    processor = MolmoAct2Processor(
        image_processor=image_processor,
        video_processor=video_processor,
        tokenizer=tokenizer,
        chat_template=processor_config.get("chat_template"),
        image_use_col_tokens=processor_config.get("image_use_col_tokens", True),
        use_single_crop_col_tokens=processor_config.get("use_single_crop_col_tokens"),
        use_single_crop_start_token=processor_config.get("use_single_crop_start_token", True),
        video_use_col_tokens=processor_config.get("video_use_col_tokens", False),
        use_frame_special_tokens=processor_config.get("use_frame_special_tokens", True),
    )

    return processor
