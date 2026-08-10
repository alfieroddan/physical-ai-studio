# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory helpers for MolmoAct2 preprocessors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .postprocessor import MolmoAct2Postprocessor
from .preprocessor import MolmoAct2Preprocessor

if TYPE_CHECKING:
    from physicalai.policies import MolmoAct2Config


def make_molmoact2_preprocessors(config: MolmoAct2Config) -> tuple[MolmoAct2Preprocessor, MolmoAct2Postprocessor]:
    """Factory method to load docstring.

    Args:
        config: MolmoAct2 config describing model.

    Returns:
        pre and post processor.

    Raises:
        ValueError: if output or input features are None.
    """
    if (config.input_features is None) or (config.output_features is None):
        msg = "Input and output features must be set; please initialize the model first."
        raise ValueError(msg)

    preprocessor = MolmoAct2Preprocessor(config=config)
    postprocessor = MolmoAct2Postprocessor(
        output_features=config.output_features,
        normalization_mode=config.normalization_mode,
        adapt_to_so101=config.adapt_to_so101,
        joint_signs=config.joint_signs,
        joint_offsets=config.joint_offsets,
    )
    return preprocessor, postprocessor
