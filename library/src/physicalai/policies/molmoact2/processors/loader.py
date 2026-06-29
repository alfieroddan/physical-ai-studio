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
from .processor import MolmoAct2Processor


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

    return MolmoAct2Processor(
        tokenizer=tokenizer,
        chat_template=processor_config.chat_template,
    )
