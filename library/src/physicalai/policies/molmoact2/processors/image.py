# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch image patchification for MolmoAct2.

Turns already-resized images into patchified crops, pooling indices and image
grids consumed by the vision backbone. Resizing lives in the preprocessor;
this stage only normalizes and patchifies so it stays export-friendly (no
data-dependent host syncs).

Only ``crop_mode="resize"`` is implemented: each image is a single crop. The
pooling layout is static, so it is precomputed once at construction.
"""

from __future__ import annotations

import torch

_IMAGE_NDIM = 4
_NUM_CHANNELS = 3


class MolmoAct2ImageProcessor:
    """Normalize and patchify pre-resized images into model-ready crops.

    Output keys (all concatenated over the ``M`` input images):

    - ``pixel_values``: ``(M, num_patches, patch_dim)`` patch pixels.
    - ``image_token_pooling``: ``(M * num_pooled, pool_area)`` patch indices per
      pooled token (``-1`` marks padding); indices are local to each image.
    - ``image_grids``: ``(M, 4)`` rows ``[pooled_h, pooled_w, 0, 0]``.
    - ``image_num_crops``: ``(M,)`` crop counts (always ``1`` for ``resize``).
    """

    def __init__(
        self,
        *,
        crop_mode: str,
        size: dict[str, int],
        patch_size: int,
        pooling_size: list[int],
        image_mean: list[float],
        image_std: list[float],
    ) -> None:
        """Read declared image settings and precompute pooling."""
        self.crop_mode = str(crop_mode)
        self.height = int(size["height"])
        self.width = int(size["width"])
        self.patch_size = int(patch_size)
        self.pool_h, self.pool_w = (int(pooling_size[0]), int(pooling_size[1]))
        self.image_mean = list(image_mean)
        self.image_std = list(image_std)

        patch_h = self.height // self.patch_size
        patch_w = self.width // self.patch_size
        pooling, self.pooled_h, self.pooled_w = self._pooling_indices(patch_h, patch_w)
        # Static pooling layout, precomputed once (kept on CPU; moved lazily).
        self._pooling = pooling

    def __call__(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Process a batch of ``(M, 3, H, W)`` images into model-ready tensors.

        Returns:
            A dict with ``pixel_values``, ``image_token_pooling``, ``image_grids``
            and ``image_num_crops``.

        Raises:
            ValueError: If ``images`` is not a 3-channel BCHW tensor.
            NotImplementedError: If ``crop_mode`` is not ``"resize"``.
        """
        if images.ndim != _IMAGE_NDIM or images.shape[1] != _NUM_CHANNELS:
            msg = f"Expected images of shape (M, 3, H, W), got {tuple(images.shape)}."
            raise ValueError(msg)
        if self.crop_mode != "resize":
            msg = f"MolmoAct2ImageProcessor only supports crop_mode='resize', got {self.crop_mode!r}."
            raise NotImplementedError(msg)

        num_images = images.shape[0]
        device = images.device
        pixel_values = self._patchify(self._normalize(images))

        pooling = self._pooling.to(device)
        image_token_pooling = pooling.unsqueeze(0).expand(num_images, -1, -1).reshape(-1, pooling.shape[-1])
        grid_row = torch.tensor([self.pooled_h, self.pooled_w, 0, 0], dtype=torch.int64, device=device)
        image_grids = grid_row.unsqueeze(0).expand(num_images, -1).contiguous()
        image_num_crops = torch.ones(num_images, dtype=torch.int64, device=device)

        return {
            "pixel_values": pixel_values,
            "image_token_pooling": image_token_pooling,
            "image_grids": image_grids,
            "image_num_crops": image_num_crops,
        }

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        """Apply channel mean/std normalization.

        Returns:
            The normalized pixels (``[-1, 1]`` for the default 0.5/0.5 stats).

        Raises:
            ValueError: If the image dtype is not float16 or float32.
        """
        if images.dtype not in {torch.float16, torch.float32}:
            msg = f"Expected images of dtype float16 or float32, got {images.dtype}."
            raise ValueError(msg)
        mean = torch.tensor(self.image_mean, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
        std = torch.tensor(self.image_std, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
        return (images - mean) / std

    def _patchify(self, pixels: torch.Tensor) -> torch.Tensor:
        """Split ``(M, C, H, W)`` pixels into ``(M, num_patches, patch_dim)``.

        Returns:
            Patch pixels ordered ``(patch_row, patch_col, channel)`` per patch.
        """
        num_images, channels, height, width = pixels.shape
        patch = self.patch_size
        pixels = pixels.permute(0, 2, 3, 1)  # (M, H, W, C)
        pixels = pixels.reshape(num_images, height // patch, patch, width // patch, patch, channels)
        pixels = pixels.permute(0, 1, 3, 2, 4, 5)
        return pixels.reshape(num_images, (height // patch) * (width // patch), patch * patch * channels)

    def _pooling_indices(self, patch_h: int, patch_w: int) -> tuple[torch.Tensor, int, int]:
        """Build patch indices grouped per pooled token, padding ragged edges with ``-1``.

        Returns:
            ``(pooling, pooled_h, pooled_w)`` where ``pooling`` has shape
            ``(pooled_h * pooled_w, pool_h * pool_w)``.
        """
        idx = torch.arange(patch_h * patch_w, dtype=torch.int64).reshape(patch_h, patch_w)

        pooled_h = (patch_h + self.pool_h - 1) // self.pool_h
        pooled_w = (patch_w + self.pool_w - 1) // self.pool_w
        pad_h = pooled_h * self.pool_h - patch_h
        pad_w = pooled_w * self.pool_w - patch_w
        # Pad bottom/right (centred like the reference) with -1 sentinels.
        idx = torch.nn.functional.pad(
            idx,
            (pad_w // 2, (pad_w + 1) // 2, pad_h // 2, (pad_h + 1) // 2),
            value=-1,
        )

        # (pooled_h * pool_h, pooled_w * pool_w) -> (pooled_h * pooled_w, pool_h * pool_w)
        idx = idx.reshape(pooled_h, self.pool_h, pooled_w, self.pool_w)
        idx = idx.permute(0, 2, 1, 3).reshape(pooled_h * pooled_w, self.pool_h * self.pool_w)
        return idx, pooled_h, pooled_w
