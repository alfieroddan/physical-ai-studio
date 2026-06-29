# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torch-native video processor implementation for MolmoAct2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
import torch

from .image import MolmoAct2ImageProcessor


@dataclass
class VideoMetadata:
    timestamps: np.ndarray


class MolmoAct2VideoProcessorOptions(TypedDict, total=False):
    return_metadata: bool
    num_frames: int
    do_sample_frames: bool


def _to_video_frames(video: Any) -> torch.Tensor:
    if isinstance(video, dict) and "frames" in video:
        frames = video["frames"]
    else:
        frames = video

    if not torch.is_tensor(frames):
        raise TypeError(f"Expected torch video tensor in TCHW format, got {type(frames)}")
    frames_t = frames

    if frames_t.ndim != 4:
        raise ValueError(f"Expected video frames of shape [T, C, H, W], got {tuple(frames_t.shape)}")
    if frames_t.shape[1] != 3:
        raise ValueError(f"Expected TCHW with 3 channels, got {tuple(frames_t.shape)}")
    return frames_t


class MolmoAct2VideoProcessor:
    """Video processor built from per-frame MolmoAct2 image preprocessing."""

    model_input_names = ["pixel_values_videos", "video_token_pooling", "video_grids"]

    def __init__(
        self,
        size: dict[str, int] | None = None,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_convert_rgb: bool = True,
        patch_size: int = 14,
        pooling_size: list[int] | None = None,
        do_sample_frames: bool = True,
        frame_sample_mode: str = "uniform_last_frame",
        max_fps: int = 2,
        sampling_fps: int = 2,
        **_: object,
    ) -> None:
        self.image_processor = MolmoAct2ImageProcessor(
            size=size,
            image_mean=image_mean,
            image_std=image_std,
            do_convert_rgb=do_convert_rgb,
            crop_mode="resize",
            max_crops=1,
            patch_size=patch_size,
            pooling_size=pooling_size if pooling_size is not None else [3, 3],
        )
        self.do_sample_frames = do_sample_frames
        self.frame_sample_mode = frame_sample_mode
        self.max_fps = max_fps
        self.sampling_fps = sampling_fps

    @staticmethod
    def _sample_frame_indices(total_frames: int, num_frames: int) -> np.ndarray:
        if total_frames <= 0:
            return np.asarray([], dtype=np.int64)
        if total_frames <= num_frames:
            return np.arange(total_frames, dtype=np.int64)
        return np.linspace(0, total_frames - 1, num=num_frames, endpoint=True).astype(np.int64)

    def preprocess(
        self,
        videos: torch.Tensor | list[torch.Tensor] | list[Any],
        return_tensors: str | None = None,
        return_metadata: bool = True,
        do_sample_frames: bool | None = None,
        num_frames: int = 8,
        **_: object,
    ) -> dict[str, Any]:
        if not isinstance(videos, list):
            if torch.is_tensor(videos) and videos.ndim == 5:
                videos = [videos[i] for i in range(videos.shape[0])]
            else:
                videos = [videos]

        video_patch_batches: list[torch.Tensor] = []
        video_pooling_batches: list[torch.Tensor] = []
        video_grids: list[list[int]] = []
        metadata: list[VideoMetadata] = []

        for video in videos:
            frames = _to_video_frames(video)
            should_sample = self.do_sample_frames if do_sample_frames is None else bool(do_sample_frames)
            frame_indices = (
                self._sample_frame_indices(frames.shape[0], int(num_frames))
                if should_sample
                else np.arange(frames.shape[0], dtype=np.int64)
            )
            sampled = frames[torch.as_tensor(frame_indices, dtype=torch.int64)]

            frame_pixel_values: list[torch.Tensor] = []
            frame_pooling: list[torch.Tensor] = []
            pooled_h = 0
            pooled_w = 0
            frame_offset = 0
            for frame in sampled:
                encoded = self.image_processor(frame.unsqueeze(0), return_tensors="pt")
                pixel_values = encoded["pixel_values"]
                pooling = encoded["image_token_pooling"].to(dtype=torch.int64)
                grid = encoded["image_grids"].to(dtype=torch.int64)

                frame_pixel_values.append(pixel_values)
                frame_pooling.append(torch.where(pooling >= 0, pooling + frame_offset, torch.full_like(pooling, -1)))
                pooled_h = int(grid[0, 0].item())
                pooled_w = int(grid[0, 1].item())
                frame_offset += pixel_values.shape[0] * pixel_values.shape[1]

            if frame_pixel_values:
                video_patch_batches.append(torch.cat(frame_pixel_values, dim=0))
                video_pooling_batches.append(torch.cat(frame_pooling, dim=0))
            video_grids.append([len(sampled), pooled_h, pooled_w])
            metadata.append(VideoMetadata(timestamps=np.asarray(frame_indices, dtype=np.float32)))

        pixel_values_videos = (
            torch.cat(video_patch_batches, dim=0)
            if video_patch_batches
            else torch.zeros((0, 0, 0), dtype=torch.float32)
        )
        video_token_pooling = (
            torch.cat(video_pooling_batches, dim=0)
            if video_pooling_batches
            else torch.zeros((0, 9), dtype=torch.int64)
        )
        video_grids_arr = torch.as_tensor(video_grids, dtype=torch.int64)

        data: dict[str, Any] = {
            "pixel_values_videos": pixel_values_videos,
            "video_token_pooling": video_token_pooling,
            "video_grids": video_grids_arr,
        }
        if return_metadata:
            data["video_metadata"] = metadata

        if return_tensors not in {None, "pt"}:
            raise ValueError(f"Unsupported return_tensors={return_tensors!r}; only 'pt' is supported.")
        return data

    __call__ = preprocess


__all__ = ["MolmoAct2VideoProcessor", "MolmoAct2VideoProcessorOptions"]
