# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 model package.

Clean, inference-focused re-implementation of MolmoAct2. Components are split
one-per-file for readability:

- :mod:`vision`         vision transformer + pooling adapter
- :mod:`text`           text decoder backbone
- :mod:`action_expert`  flow-matching action expert
- :mod:`backbone`       assembly + action generation entrypoint
- :mod:`wrapper`        physicalai ``Model`` frontend
- :mod:`image`/:mod:`video`  PyTorch preprocessing
"""

from .backbone import MolmoAct2Backbone, MolmoAct2ForConditionalGeneration
from .wrapper import MolmoAct2Model

__all__ = [
    "MolmoAct2Backbone",
    "MolmoAct2ForConditionalGeneration",
    "MolmoAct2Model",
]
