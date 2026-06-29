# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Clean local image processor implementation for MolmoAct2."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor, get_size_dict
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    PILImageResampling,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
)
from transformers.utils import TensorType


class MolmoAct2ImagesKwargs(TypedDict, total=False):
    return_metadata: bool


def _to_rgb_channels_last(image: ImageInput) -> np.ndarray:
    arr = to_numpy_array(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got {arr.shape}.")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.size > 0 and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    chw = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    resized = F.interpolate(chw.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)
    return resized.squeeze(0).permute(1, 2, 0).numpy()


def _normalize(image: np.ndarray, image_mean: list[float], image_std: list[float]) -> np.ndarray:
    mean = np.asarray(image_mean, dtype=np.float32)[None, None, :]
    std = np.asarray(image_std, dtype=np.float32)[None, None, :]
    return (image.astype(np.float32) - mean) / std


def _patchify(image: np.ndarray, patch_size: int) -> np.ndarray:
    h, w, c = image.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"Image size {(h, w)} must be divisible by patch_size={patch_size}.")
    h_patches = h // patch_size
    w_patches = w // patch_size
    arr = image.reshape(h_patches, patch_size, w_patches, patch_size, c)
    arr = arr.transpose(0, 2, 1, 3, 4)
    return arr.reshape(h_patches * w_patches, patch_size * patch_size * c)


def _pooling_indices(h_patches: int, w_patches: int, pool_h: int, pool_w: int) -> tuple[np.ndarray, int, int]:
    idx = np.arange(h_patches * w_patches, dtype=np.int64).reshape(h_patches, w_patches)
    h_pad = (pool_h - (h_patches % pool_h)) % pool_h
    w_pad = (pool_w - (w_patches % pool_w)) % pool_w
    idx = np.pad(
        idx,
        [[h_pad // 2, h_pad - (h_pad // 2)], [w_pad // 2, w_pad - (w_pad // 2)]],
        mode="constant",
        constant_values=-1,
    )
    h_groups = idx.shape[0] // pool_h
    w_groups = idx.shape[1] // pool_w
    grouped = idx.reshape(h_groups, pool_h, w_groups, pool_w).transpose(0, 2, 1, 3).reshape(-1, pool_h * pool_w)
    return grouped, h_groups, w_groups


class MolmoAct2ImageProcessor(BaseImageProcessor):
    """Image processor producing patch tensors and pooling metadata for MolmoAct2."""

    model_input_names = ["pixel_values", "image_token_pooling", "image_grids", "image_num_crops"]

    def __init__(
        self,
        size: dict[str, int] | None = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_convert_rgb: bool = True,
        max_crops: int = 8,
        overlap_margins: list[int] | None = None,
        crop_mode: str = "resize",
        patch_size: int = 14,
        pooling_size: list[int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.size = get_size_dict(size if size is not None else {"height": 378, "width": 378})
        self.resample = resample
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.do_convert_rgb = do_convert_rgb
        self.max_crops = max_crops
        self.overlap_margins = overlap_margins if overlap_margins is not None else [4, 4]
        self.crop_mode = crop_mode
        self.patch_size = patch_size
        self.pooling_size = pooling_size if pooling_size is not None else [2, 2]

    def preprocess(
        self,
        images: ImageInput,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        del kwargs
        if not valid_images(images):
            raise ValueError("Invalid image input provided to MolmoAct2ImageProcessor.")

        image_list = make_flat_list_of_images(images)
        patch_batches: list[np.ndarray] = []
        pooling_batches: list[np.ndarray] = []
        grids: list[list[int]] = []
        image_num_crops: list[int] = []

        target_h = int(self.size["height"])
        target_w = int(self.size["width"])
        pool_h, pool_w = int(self.pooling_size[0]), int(self.pooling_size[1])

        for image in image_list:
            arr = _to_rgb_channels_last(image)
            resized = _resize(arr, target_h, target_w)
            normalized = _normalize(resized, list(self.image_mean), list(self.image_std))

            patches = _patchify(normalized, int(self.patch_size))[None, ...]
            h_patches = target_h // int(self.patch_size)
            w_patches = target_w // int(self.patch_size)
            pooling_idx, h_groups, w_groups = _pooling_indices(h_patches, w_patches, pool_h, pool_w)

            patch_batches.append(patches)
            pooling_batches.append(pooling_idx)
            grids.append([h_groups, w_groups, 0, 0])
            image_num_crops.append(1)

        pixel_values = (
            np.concatenate(patch_batches, axis=0)
            if patch_batches
            else np.zeros((0, 0, 0), dtype=np.float32)
        )
        image_token_pooling = (
            np.concatenate(pooling_batches, axis=0) if pooling_batches else np.zeros((0, pool_h * pool_w), dtype=np.int64)
        )
        image_grids = np.asarray(grids, dtype=np.int64)
        image_num_crops_arr = np.asarray(image_num_crops, dtype=np.int64)

        return BatchFeature(
            data={
                "pixel_values": pixel_values,
                "image_token_pooling": image_token_pooling,
                "image_grids": image_grids,
                "image_num_crops": image_num_crops_arr,
            },
            tensor_type=return_tensors,
        )

    __call__ = preprocess


__all__ = ["MolmoAct2ImageProcessor", "MolmoAct2ImagesKwargs"]
