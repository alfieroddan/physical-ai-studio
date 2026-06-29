# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Clean local video processor implementation for MolmoAct2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import Unpack
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.video_utils import VideoInput

from .image import MolmoAct2ImageProcessor


@dataclass
class VideoMetadata:
	timestamps: np.ndarray


class MolmoAct2VideoProcessorKwargs(TypedDict, total=False):
	return_metadata: bool
	num_frames: int


def _to_video_frames(video: Any) -> np.ndarray:
	if isinstance(video, dict) and "frames" in video:
		frames = np.asarray(video["frames"])
	else:
		frames = np.asarray(video)
	if frames.ndim != 4:
		raise ValueError(f"Expected video frames of shape [T, H, W, C], got {frames.shape}.")
	if frames.shape[-1] not in {1, 3, 4} and frames.shape[1] in {1, 3, 4}:
		frames = np.moveaxis(frames, 1, -1)
	return frames


class MolmoAct2VideoProcessor(BaseVideoProcessor):
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
		**kwargs: Unpack[MolmoAct2VideoProcessorKwargs],
	) -> None:
		super().__init__(**kwargs)
		self.image_processor = MolmoAct2ImageProcessor(
			size=size,
			image_mean=image_mean,
			image_std=image_std,
			do_convert_rgb=do_convert_rgb,
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
		return np.linspace(0, total_frames - 1, num=num_frames, dtype=np.int64)

	def preprocess(
		self,
		videos: VideoInput,
		return_tensors: str | None = None,
		return_metadata: bool = True,
		num_frames: int = 8,
		**kwargs,
	) -> BatchFeature:
		del kwargs
		if not isinstance(videos, list):
			videos = [videos]

		video_patch_batches: list[np.ndarray] = []
		video_pooling_batches: list[np.ndarray] = []
		video_grids: list[list[int]] = []
		metadata: list[VideoMetadata] = []

		for video in videos:
			frames = _to_video_frames(video)
			frame_indices = self._sample_frame_indices(frames.shape[0], int(num_frames))
			sampled = frames[frame_indices]

			frame_pixel_values: list[np.ndarray] = []
			frame_pooling: list[np.ndarray] = []
			pooled_per_frame = 0
			frame_offset = 0
			for frame in sampled:
				encoded = self.image_processor(frame, return_tensors="np")
				pixel_values = np.asarray(encoded["pixel_values"])
				pooling = np.asarray(encoded["image_token_pooling"], dtype=np.int64)
				grid = np.asarray(encoded["image_grids"], dtype=np.int64)

				frame_pixel_values.append(pixel_values)
				frame_pooling.append(np.where(pooling >= 0, pooling + frame_offset, -1))
				pooled_per_frame = int(grid[0, 0] * grid[0, 1])
				frame_offset += pixel_values.shape[0] * pixel_values.shape[1]

			if frame_pixel_values:
				video_patch_batches.append(np.concatenate(frame_pixel_values, axis=0))
				video_pooling_batches.append(np.concatenate(frame_pooling, axis=0))
			video_grids.append([len(sampled), int(np.sqrt(pooled_per_frame)), int(np.sqrt(pooled_per_frame))])
			timestamps = np.asarray(frame_indices, dtype=np.float32)
			metadata.append(VideoMetadata(timestamps=timestamps))

		pixel_values_videos = (
			np.concatenate(video_patch_batches, axis=0) if video_patch_batches else np.zeros((0, 0, 0), dtype=np.float32)
		)
		video_token_pooling = (
			np.concatenate(video_pooling_batches, axis=0) if video_pooling_batches else np.zeros((0, 9), dtype=np.int64)
		)
		video_grids_arr = np.asarray(video_grids, dtype=np.int64)

		data: dict[str, Any] = {
			"pixel_values_videos": pixel_values_videos,
			"video_token_pooling": video_token_pooling,
			"video_grids": video_grids_arr,
		}
		if return_metadata:
			data["video_metadata"] = metadata
		return BatchFeature(data=data, tensor_type=return_tensors)

	__call__ = preprocess


__all__ = ["MolmoAct2VideoProcessor", "MolmoAct2VideoProcessorKwargs"]
