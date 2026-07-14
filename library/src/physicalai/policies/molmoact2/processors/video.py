# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch video preprocessing for MolmoAct2.

Samples frames per video, patchifies each frame with the ``resize`` image
processor, and concatenates them into model-ready video tensors. Implemented
entirely in PyTorch (no NumPy).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from physicalai.policies.molmoact2.config import MolmoAct2ImageProcessorConfig

from .image import MolmoAct2ImageProcessor

if TYPE_CHECKING:
    from physicalai.policies.molmoact2.config import MolmoAct2VideoProcessorConfig


class MolmoAct2VideoProcessor:
    """Sample and patchify video frames into model-ready tensors.

    Each video's frames are resized to a single crop each and concatenated.
    Per-video pooling indices are offset by the running frame patch count so
    they index into that video's stacked frames.

    Output keys (concatenated over the input videos):

    - ``pixel_values_videos``: ``(total_frames, num_patches, patch_dim)``.
    - ``video_token_pooling``: ``(total_frames * num_pooled, pool_area)``.
    - ``video_grids``: ``(num_videos, 3)`` rows ``[num_frames, pooled_h, pooled_w]``.
    """

    def __init__(self, config: MolmoAct2VideoProcessorConfig) -> None:
        """Build the per-frame ``resize`` image processor and frame sampler."""
        self.num_frames = int(config.num_frames)
        self.do_sample_frames = bool(config.do_sample_frames)
        self.image_processor = MolmoAct2ImageProcessor(
            MolmoAct2ImageProcessorConfig(
                crop_mode="resize",
                size=dict(config.size),
                image_mean=list(config.image_mean),
                image_std=list(config.image_std),
                patch_size=int(config.patch_size),
                pooling_size=list(config.pooling_size),
            ),
        )

    def __call__(self, videos: torch.Tensor | list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Process one or more ``(T, 3, H, W)`` videos into model-ready tensors."""
        video_list = self._as_video_list(videos)

        pixel_values: list[torch.Tensor] = []
        pooling: list[torch.Tensor] = []
        grids: list[list[int]] = []
        for frames in video_list:
            frames = self._sample_frames(frames)
            encoded = self.image_processor(frames)
            num_frames = frames.shape[0]
            num_patches = encoded["pixel_values"].shape[1]

            pool = encoded["image_token_pooling"].reshape(num_frames, -1, encoded["image_token_pooling"].shape[-1])
            offsets = (torch.arange(num_frames, device=pool.device) * num_patches).view(num_frames, 1, 1)
            pool = torch.where(pool >= 0, pool + offsets, pool)

            pixel_values.append(encoded["pixel_values"])
            pooling.append(pool.reshape(-1, pool.shape[-1]))
            grids.append([num_frames, int(encoded["image_grids"][0, 0]), int(encoded["image_grids"][0, 1])])

        return {
            "pixel_values_videos": torch.cat(pixel_values, dim=0),
            "video_token_pooling": torch.cat(pooling, dim=0),
            "video_grids": torch.tensor(grids, dtype=torch.int64, device=video_list[0].device),
        }

    @staticmethod
    def _as_video_list(videos: torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
        """Normalize input into a list of ``(T, 3, H, W)`` frame tensors."""
        if isinstance(videos, list):
            frame_lists = videos
        elif videos.ndim == 5:
            frame_lists = list(videos)
        elif videos.ndim == 4:
            frame_lists = [videos]
        else:
            msg = f"Expected videos of shape (B, T, 3, H, W) or (T, 3, H, W), got {tuple(videos.shape)}."
            raise ValueError(msg)
        for frames in frame_lists:
            if frames.ndim != 4 or frames.shape[1] != 3:
                msg = f"Expected video frames of shape (T, 3, H, W), got {tuple(frames.shape)}."
                raise ValueError(msg)
        return frame_lists

    def _sample_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Uniformly sample ``num_frames`` frames (including the last)."""
        total = frames.shape[0]
        if not self.do_sample_frames or total <= self.num_frames:
            return frames
        indices = torch.linspace(0, total - 1, self.num_frames).to(torch.int64)
        return frames[indices]
