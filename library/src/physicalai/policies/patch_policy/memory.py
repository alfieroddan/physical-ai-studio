# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Temporal embedding memory for Patch Policy inference."""

from __future__ import annotations

from collections import deque

import torch
from torch import Tensor


class EmbeddingMemoryBank:
    """Retain a fixed window of encoded observations during inference.

    The bank stores one ``[B, P, E]`` embedding per inference call. Explicit
    temporal windows bypass the bank, which keeps dataset training and
    validation stateless.

    Args:
        n_obs_steps: Number of observation embeddings in the returned window.
    """

    def __init__(self, n_obs_steps: int) -> None:
        """Initialize an empty bounded observation history.

        Raises:
            ValueError: If the requested history length is not positive.
        """
        if n_obs_steps <= 0:
            message = "n_obs_steps must be positive."
            raise ValueError(message)
        self.n_obs_steps = n_obs_steps
        self._history: deque[Tensor] = deque(maxlen=n_obs_steps)

    def __call__(self, embeddings: Tensor) -> Tensor:
        """Return an explicit window or append one current embedding.

        Args:
            embeddings: Encoded observations shaped ``[B, T, P, E]``.

        Returns:
            Embeddings shaped ``[B, n_obs_steps, P, E]``.

        Raises:
            ValueError: If the temporal length is neither one nor the complete
                configured window, or if an inference stream changes shape.
        """
        if embeddings.ndim != 4:  # noqa: PLR2004
            message = f"Expected embeddings with shape [B, T, P, E], got {tuple(embeddings.shape)}."
            raise ValueError(message)
        if embeddings.shape[1] == self.n_obs_steps:
            return embeddings
        if embeddings.shape[1] != 1:
            message = (
                f"Expected one current observation or {self.n_obs_steps} observation steps, got {embeddings.shape[1]}."
            )
            raise ValueError(message)

        current = embeddings[:, 0]
        if self._history and (
            self._history[0].shape != current.shape
            or self._history[0].device != current.device
            or self._history[0].dtype != current.dtype
        ):
            message = "Observation embedding shape, device, and dtype must remain constant until reset()."
            raise ValueError(message)

        self._history.append(current.detach())
        padding = [self._history[0] for _ in range(self.n_obs_steps - len(self._history))]
        return torch.stack([*padding, *self._history], dim=1)

    def reset(self) -> None:
        """Clear all retained observation embeddings."""
        self._history.clear()


__all__ = ["EmbeddingMemoryBank"]
