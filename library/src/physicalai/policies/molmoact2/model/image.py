# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torch-input image processor implementation for MolmoAct2."""

from __future__ import annotations

from typing import TypedDict

import einops
import numpy as np
import torch
import torchvision.transforms


class MolmoAct2ImagesOptions(TypedDict, total=False):
    return_metadata: bool


def _normalize_image(image: np.ndarray, image_mean: list[float], image_std: list[float]) -> np.ndarray:
    if np.allclose(image_mean, [0.5, 0.5, 0.5]) and np.allclose(image_std, [0.5, 0.5, 0.5]):
        return image * np.asarray(2.0, dtype=np.float32) - np.asarray(1.0, dtype=np.float32)
    image = image.astype(np.float32)
    image -= np.array(image_mean, dtype=np.float32)[None, None, :]
    image /= np.array(image_std, dtype=np.float32)[None, None, :]
    return image


def _resize_image(image: np.ndarray, desired_output_size: list[int]) -> np.ndarray:
    chw = torch.from_numpy(image).permute(2, 0, 1)
    dtype = chw.dtype
    if torch.is_floating_point(chw):
        in_min, in_max = 0.0, 1.0
        resized = torchvision.transforms.Resize(desired_output_size, antialias=False)(chw)
        resized = torch.clip(resized, 0.0, 1.0).to(dtype)
    else:
        if chw.dtype != torch.uint8:
            raise ValueError(f"Expected uint8 or float image tensor, got {chw.dtype}")
        in_min, in_max = 0.0, 255.0
        resized = torchvision.transforms.Resize(desired_output_size, antialias=False)(chw)
        resized = torch.clip(resized, 0, 255).to(dtype)

    resized = resized.to(torch.float32)
    resized = (resized - in_min) / (in_max - in_min)
    return resized.permute(1, 2, 0).numpy()


def _select_tiling(h: int, w: int, patch_size: int, max_num_crops: int) -> np.ndarray:
    tilings: list[tuple[int, int]] = []
    for i in range(1, max_num_crops + 1):
        for j in range(1, max_num_crops + 1):
            if i * j <= max_num_crops:
                tilings.append((i, j))
    tilings.sort(key=lambda x: (x[0] * x[1], x[0]))
    candidate_tilings = np.array(tilings, dtype=np.int32)
    candidate_resolutions = candidate_tilings * patch_size

    original_size = np.array([h, w], dtype=np.float32)
    with np.errstate(divide="ignore"):
        required_scale = candidate_resolutions.astype(np.float32) / original_size[None, :]
    required_scale = np.min(required_scale, axis=-1, keepdims=True)
    if np.all(required_scale < 1):
        ix = int(np.argmax(required_scale))
    else:
        required_scale = np.where(required_scale < 1.0, 1e10, required_scale)
        ix = int(np.argmin(required_scale))
    return candidate_tilings[ix]


def _build_resized_image(
    image: np.ndarray,
    base_image_input_size: list[int],
    image_mean: list[float],
    image_std: list[float],
    image_patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    resized = _resize_image(image, base_image_input_size)
    resized = _normalize_image(resized, image_mean, image_std)
    if resized.ndim == 3:
        resized = np.expand_dims(resized, 0)
    crop_patch_w = base_image_input_size[1] // image_patch_size
    crop_patch_h = base_image_input_size[0] // image_patch_size
    resize_idx = np.arange(crop_patch_w * crop_patch_h).reshape([crop_patch_h, crop_patch_w])
    return resized, resize_idx


def _build_overlapping_crops(
    image: np.ndarray,
    max_crops: int,
    overlap_margins: list[int],
    base_image_input_size: list[int],
    image_mean: list[float],
    image_std: list[float],
    image_patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    left_margin, right_margin = overlap_margins
    total_margin_pixels = image_patch_size * (right_margin + left_margin)
    crop_patches = base_image_input_size[0] // image_patch_size
    crop_window_patches = crop_patches - (right_margin + left_margin)
    crop_window_size = crop_window_patches * image_patch_size
    crop_patch_w = base_image_input_size[1] // image_patch_size
    crop_patch_h = base_image_input_size[0] // image_patch_size

    original_image_h, original_image_w = image.shape[:2]
    crop_size = base_image_input_size[0]

    tiling = _select_tiling(
        original_image_h - total_margin_pixels,
        original_image_w - total_margin_pixels,
        crop_window_size,
        max_crops,
    )

    src = _resize_image(
        image,
        [tiling[0] * crop_window_size + total_margin_pixels, tiling[1] * crop_window_size + total_margin_pixels],
    )
    src = _normalize_image(src, image_mean, image_std)

    n_crops = int(tiling[0] * tiling[1])
    crop_arr = np.zeros([n_crops, crop_size, crop_size, 3], dtype=src.dtype)
    patch_idx_arr = np.zeros([n_crops, crop_patch_h, crop_patch_w], dtype=np.int32)

    on_crop = 0
    for i in range(int(tiling[0])):
        y0 = i * crop_window_size
        for j in range(int(tiling[1])):
            x0 = j * crop_window_size
            crop_arr[on_crop] = src[y0 : y0 + crop_size, x0 : x0 + crop_size]
            patch_idx = np.arange(crop_patch_w * crop_patch_h).reshape(crop_patch_h, crop_patch_w)
            patch_idx += on_crop * crop_patch_h * crop_patch_w

            if i != 0:
                patch_idx[:left_margin, :] = -1
            if j != 0:
                patch_idx[:, :left_margin] = -1
            if i != int(tiling[0]) - 1:
                patch_idx[-right_margin:, :] = -1
            if j != int(tiling[1]) - 1:
                patch_idx[:, -right_margin:] = -1
            patch_idx_arr[on_crop] = patch_idx
            on_crop += 1

    patch_idx_arr = np.reshape(
        patch_idx_arr,
        [int(tiling[0]), int(tiling[1]), crop_patch_h, crop_patch_w],
    )
    patch_idx_arr = np.transpose(patch_idx_arr, [0, 2, 1, 3])
    patch_idx_arr = np.reshape(patch_idx_arr, [-1])

    patch_idx_arr = patch_idx_arr[patch_idx_arr >= 0].reshape(
        src.shape[0] // image_patch_size,
        src.shape[1] // image_patch_size,
    )
    return crop_arr, patch_idx_arr


def _batch_pixels_to_patches(array: np.ndarray, patch_size: int) -> np.ndarray:
    n_crops, h, w, c = array.shape
    h_patches = h // patch_size
    w_patches = w // patch_size
    array = np.reshape(array, [n_crops, h_patches, patch_size, w_patches, patch_size, c])
    array = np.transpose(array, [0, 1, 3, 2, 4, 5])
    array = np.reshape(array, [n_crops, h_patches * w_patches, patch_size * patch_size * c])
    return array


def _arange_for_pooling(idx_arr: np.ndarray, pool_h: int, pool_w: int) -> np.ndarray:
    h_pad = pool_h * ((idx_arr.shape[0] + pool_h - 1) // pool_h) - idx_arr.shape[0]
    w_pad = pool_w * ((idx_arr.shape[1] + pool_w - 1) // pool_w) - idx_arr.shape[1]
    idx_arr = np.pad(
        idx_arr,
        [[h_pad // 2, (h_pad + 1) // 2], [w_pad // 2, (w_pad + 1) // 2]],
        mode="constant",
        constant_values=-1,
    )
    return einops.rearrange(idx_arr, "(h dh) (w dw) -> h w (dh dw)", dh=pool_h, dw=pool_w)


def _image_to_patches_and_grids(
    image: np.ndarray,
    max_crops: int,
    overlap_margins: list[int],
    base_image_input_size: list[int],
    image_mean: list[float],
    image_std: list[float],
    image_patch_size: int,
    image_pooling_w: int,
    image_pooling_h: int,
    crop_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    crop_patch_w = base_image_input_size[1] // image_patch_size
    crop_patch_h = base_image_input_size[0] // image_patch_size

    if crop_mode == "resize":
        resized, resize_idx = _build_resized_image(
            image,
            base_image_input_size,
            image_mean,
            image_std,
            image_patch_size,
        )
        resize_idx = _arange_for_pooling(resize_idx, image_pooling_h, image_pooling_w)
        resized_h, resized_w = resize_idx.shape[:2]
        resize_idx = resize_idx.reshape([-1, image_pooling_h * image_pooling_w])
        image_grid = [np.array([resized_h, resized_w, 0, 0])]
        return np.stack(image_grid, 0), _batch_pixels_to_patches(resized, image_patch_size), resize_idx

    if crop_mode not in {"overlap-and-resize-c2", "overlap-and-resize"}:
        raise ValueError(f"Unsupported MolmoAct2 image crop_mode {crop_mode!r}.")

    crop_arr, patch_idx_arr = _build_overlapping_crops(
        image,
        max_crops,
        overlap_margins,
        base_image_input_size,
        image_mean,
        image_std,
        image_patch_size,
    )
    pooling_idx = _arange_for_pooling(patch_idx_arr, image_pooling_h, image_pooling_w)
    h, w = pooling_idx.shape[:2]
    pooling_idx = pooling_idx.reshape([-1, image_pooling_h * image_pooling_w])

    resized, resize_idx = _build_resized_image(
        image,
        base_image_input_size,
        image_mean,
        image_std,
        image_patch_size,
    )
    crop_arr = np.concatenate([resized, crop_arr], 0)

    resize_idx = _arange_for_pooling(resize_idx, image_pooling_h, image_pooling_w)
    resized_h, resized_w = resize_idx.shape[:2]
    resize_idx = resize_idx.reshape([-1, image_pooling_h * image_pooling_w])

    pooling_idx = np.where(pooling_idx >= 0, pooling_idx + crop_patch_h * crop_patch_w, -1)
    pooling_idx = np.concatenate([resize_idx, pooling_idx])
    image_grid = [np.array([resized_h, resized_w, h, w])]

    return np.stack(image_grid, 0), _batch_pixels_to_patches(crop_arr, image_patch_size), pooling_idx


def _to_hwc_uint8(images_bchw: torch.Tensor) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for image in images_bchw:
        img = image
        if img.dtype.is_floating_point:
            if float(img.max().item()) <= 1.0:
                img = img * 255.0
            img = img.clamp(0, 255).to(torch.uint8)
        elif img.dtype != torch.uint8:
            img = img.clamp(0, 255).to(torch.uint8)
        out.append(img.permute(1, 2, 0).cpu().numpy())
    return out


class MolmoAct2ImageProcessor:
    """Image processor producing patch tensors and pooling metadata for MolmoAct2."""

    model_input_names = ["pixel_values", "image_token_pooling", "image_grids", "image_num_crops"]

    def __init__(
        self,
        size: dict[str, int] | None = None,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_convert_rgb: bool = True,
        max_crops: int = 8,
        overlap_margins: list[int] | None = None,
        crop_mode: str = "overlap-and-resize-c2",
        patch_size: int = 14,
        pooling_size: list[int] | None = None,
        **_: object,
    ) -> None:
        self.size = size if size is not None else {"height": 378, "width": 378}
        self.image_mean = list(image_mean) if image_mean is not None else [0.5, 0.5, 0.5]
        self.image_std = list(image_std) if image_std is not None else [0.5, 0.5, 0.5]
        self.do_convert_rgb = do_convert_rgb
        self.max_crops = int(max_crops)
        self.overlap_margins = overlap_margins if overlap_margins is not None else [4, 4]
        self.crop_mode = crop_mode
        self.patch_size = int(patch_size)
        self.pooling_size = pooling_size if pooling_size is not None else [2, 2]

    def __call__(self, images: torch.Tensor, return_tensors: str | None = "pt", **_: object) -> dict[str, torch.Tensor]:
        if not torch.is_tensor(images):
            raise TypeError(f"Expected torch.Tensor in BCHW format, got {type(images)}")
        if images.ndim != 4:
            raise ValueError(f"Expected BCHW image tensor, got shape {tuple(images.shape)}")
        if images.shape[1] != 3:
            raise ValueError(f"Expected BCHW with 3 channels, got shape {tuple(images.shape)}")
        if return_tensors not in {None, "pt"}:
            raise ValueError(f"Unsupported return_tensors={return_tensors!r}; only 'pt' is supported.")

        image_list = _to_hwc_uint8(images)

        patch_batches: list[np.ndarray] = []
        pooling_batches: list[np.ndarray] = []
        grids: list[np.ndarray] = []
        image_num_crops: list[int] = []

        base_image_input_size = [int(self.size["height"]), int(self.size["width"])]
        pool_h, pool_w = int(self.pooling_size[0]), int(self.pooling_size[1])

        for image in image_list:
            image_grid, crops, pooled_idx = _image_to_patches_and_grids(
                image,
                self.max_crops,
                self.overlap_margins,
                base_image_input_size,
                self.image_mean,
                self.image_std,
                self.patch_size,
                pool_w,
                pool_h,
                self.crop_mode,
            )
            patch_batches.append(crops)
            pooling_batches.append(pooled_idx)
            grids.append(image_grid)
            image_num_crops.append(int(crops.shape[0]))

        pixel_values = np.concatenate(patch_batches, 0) if patch_batches else np.zeros((0, 0, 0), dtype=np.float32)
        image_token_pooling = (
            np.concatenate(pooling_batches, 0) if pooling_batches else np.zeros((0, pool_h * pool_w), dtype=np.int64)
        )
        image_grids = np.concatenate(grids, 0) if grids else np.zeros((0, 4), dtype=np.int64)
        image_num_crops_arr = np.asarray(image_num_crops, dtype=np.int64)

        return {
            "pixel_values": torch.as_tensor(pixel_values),
            "image_token_pooling": torch.as_tensor(image_token_pooling, dtype=torch.int64),
            "image_grids": torch.as_tensor(image_grids, dtype=torch.int64),
            "image_num_crops": torch.as_tensor(image_num_crops_arr, dtype=torch.int64),
        }


__all__ = ["MolmoAct2ImageProcessor", "MolmoAct2ImagesOptions"]
