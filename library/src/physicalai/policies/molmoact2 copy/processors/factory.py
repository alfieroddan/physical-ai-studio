# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory helpers for MolmoAct2 preprocessors."""

from __future__ import annotations

from typing import Any

from .postprocessor import MolmoAct2Postprocessor
from .preprocessor import MolmoAct2Preprocessor


def make_molmoact2_preprocessors(config: Any) -> tuple[MolmoAct2Preprocessor, MolmoAct2Postprocessor]:
    preprocessor = MolmoAct2Preprocessor(config=config)
    postprocessor = MolmoAct2Postprocessor(config=config)
    return preprocessor, postprocessor
