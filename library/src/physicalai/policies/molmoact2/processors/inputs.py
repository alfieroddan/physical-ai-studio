# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Build MolmoAct2 model inputs from a preprocessed observation batch.

Turns the preprocessor output (tokenized prompt with ``<|image|>`` placeholders
and packed ``IMAGES``) into the tensors the backbone consumes: patchified pixel
values, pooling indices, image grids, token type ids, and prompt token ids with
each ``<|image|>`` placeholder expanded into the concrete image-token layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from physicalai.data.constants import IMAGES, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.data.observation import FeatureType

if TYPE_CHECKING:
    from physicalai.policies.molmoact2.config import MolmoAct2Config

    from .image import MolmoAct2ImageProcessor


def _env_action_dim(config: MolmoAct2Config) -> int:
    """Return the environment action dimension from the output features.

    Returns:
        The first dimension of the configured action feature, or ``0`` if no
        action feature is present.
    """
    for feature in config.output_features or []:
        if feature.ftype == FeatureType.ACTION and feature.shape:
            return int(feature.shape[0])
    return 0


def _default_action_dim_is_pad(config: MolmoAct2Config, *, batch_size: int, device: torch.device) -> torch.Tensor:
    """Mark action dimensions beyond the environment action dim as padding.

    Returns:
        Boolean tensor of shape ``(batch_size, max_action_dim)`` that is ``False``
        for the first ``env_action_dim`` columns and ``True`` for the rest.
    """
    action_dim_is_pad = torch.ones((batch_size, int(config.max_action_dim)), dtype=torch.bool, device=device)
    env_action_dim = _env_action_dim(config)
    if env_action_dim > 0:
        action_dim_is_pad[:, :env_action_dim] = False
    return action_dim_is_pad


def _image_token_ids(config: MolmoAct2Config) -> list[int]:
    """List the token ids that mark image content (for token type ids).

    Returns:
        List of integer token ids defined on ``config`` that mark image
        content; ``None``-valued entries are skipped.
    """
    ids = [
        config.image_patch_id,
        config.image_col_id,
        config.image_start_token_id,
        config.low_res_image_start_token_id,
        config.frame_start_token_id,
        config.image_end_token_id,
        config.frame_end_token_id,
        config.image_low_res_id,
    ]
    return [int(token_id) for token_id in ids if token_id is not None]


def _build_token_type_ids(
    config: MolmoAct2Config,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor | None:
    """Mark image tokens (1) vs. text tokens (0), respecting the attention mask.

    Returns:
        Long tensor matching ``input_ids`` shape marking image tokens, or
        ``None`` if no image token ids are defined on ``config``.
    """
    image_token_ids = _image_token_ids(config)
    if not image_token_ids:
        return None
    token_set = torch.as_tensor(image_token_ids, device=input_ids.device, dtype=input_ids.dtype)
    is_image = (input_ids.unsqueeze(-1) == token_set.view(1, 1, -1)).any(dim=-1).to(torch.long)
    return is_image * attention_mask.to(torch.long)


def _image_token_ids_for_grid(config: MolmoAct2Config, grid: torch.Tensor) -> list[int]:  # noqa: PLR0914
    """Expand a single image grid into its sequence of image token ids.

    Returns:
        Ordered list of token ids representing one image token grid
        (low-resolution crops followed by high-resolution crops), bracketed
        by start/end tokens.

    Raises:
        ValueError: If required image token ids (``image_patch_id``,
            ``image_start_token_id``, ``image_end_token_id``) are unset on
            ``config``.
    """
    resized_h, resized_w, height, width = (int(x) for x in grid.tolist())

    if config.image_patch_id is None or config.image_start_token_id is None or config.image_end_token_id is None:
        msg = "image_patch_id, image_start_token_id, and image_end_token_id must be configured"
        raise ValueError(msg)
    image_patch_id = int(config.image_patch_id)
    image_start_token_id = int(config.image_start_token_id)
    image_end_token_id = int(config.image_end_token_id)
    image_col_id = None if config.image_col_id is None else int(config.image_col_id)
    low_res_start_id = (
        int(config.low_res_image_start_token_id)
        if config.low_res_image_start_token_id is not None
        else image_start_token_id
    )

    processor_config = config.processor_config
    image_use_col_tokens = bool(processor_config.image_use_col_tokens) if processor_config is not None else True
    use_single_crop_col_tokens = (
        image_use_col_tokens
        if processor_config is None or processor_config.use_single_crop_col_tokens is None
        else bool(processor_config.use_single_crop_col_tokens)
    )
    use_single_crop_start_token = (
        bool(processor_config.use_single_crop_start_token) if processor_config is not None else True
    )

    def make_rows(num_rows: int, num_cols: int, *, use_col: bool) -> list[int]:
        row = [image_patch_id] * num_cols
        if use_col and image_col_id is not None:
            row += [image_col_id]
        return row * num_rows

    if height == 0 or width == 0:
        return [
            image_start_token_id,
            *make_rows(resized_h, resized_w, use_col=use_single_crop_col_tokens),
            image_end_token_id,
        ]

    high_res = [
        image_start_token_id,
        *make_rows(height, width, use_col=image_use_col_tokens),
        image_end_token_id,
    ]
    low_start = low_res_start_id if use_single_crop_start_token else image_start_token_id
    low_res = [
        low_start,
        *make_rows(resized_h, resized_w, use_col=use_single_crop_col_tokens),
        image_end_token_id,
    ]
    return [*low_res, *high_res]


def _expand_image_placeholders(
    *,
    config: MolmoAct2Config,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grids: torch.Tensor,
    image_placeholder_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Replace each ``<image>`` placeholder with its expanded image token ids.

    Returns:
        ``(out_ids, out_mask, token_type_ids)`` tensors with the placeholder
        tokens replaced and padded rows masked.

    Raises:
        ValueError: If there are too few ``image_grids`` to expand every
            ``<image>`` placeholder in ``input_ids``.
    """
    if int(image_grids.shape[0]) == 0:
        return input_ids, attention_mask, _build_token_type_ids(config, input_ids, attention_mask)

    pad_values = input_ids[attention_mask == 0]
    pad_token_id = int(pad_values[0]) if int(pad_values.numel()) > 0 else 0

    expanded_rows: list[list[int]] = []
    grid_idx = 0
    for batch_idx in range(int(input_ids.shape[0])):
        valid = attention_mask[batch_idx].to(torch.bool)
        expanded: list[int] = []
        for token in input_ids[batch_idx][valid].tolist():
            token_int = int(token)
            if token_int == image_placeholder_token_id:
                if grid_idx >= int(image_grids.shape[0]):
                    msg = "Not enough image grids to expand all <|image|> placeholders."
                    raise ValueError(msg)
                expanded.extend(_image_token_ids_for_grid(config, image_grids[grid_idx]))
                grid_idx += 1
            else:
                expanded.append(token_int)
        expanded_rows.append(expanded)

    max_len = max((len(row) for row in expanded_rows), default=1)
    out_ids = torch.full((len(expanded_rows), max_len), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
    out_mask = torch.zeros((len(expanded_rows), max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    for batch_idx, row in enumerate(expanded_rows):
        if not row:
            continue
        row_tensor = torch.as_tensor(row, dtype=input_ids.dtype, device=input_ids.device)
        out_ids[batch_idx, : row_tensor.numel()] = row_tensor
        out_mask[batch_idx, : row_tensor.numel()] = 1

    return out_ids, out_mask, _build_token_type_ids(config, out_ids, out_mask)


def build_batched_images(  # noqa: PLR0914
    config: MolmoAct2Config,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    image_token_pooling: torch.Tensor,
    image_grids: torch.Tensor,
    image_num_crops: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Regroup per-image crops/pooling into per-example padded tensors.

    The image processor emits crops and pooling indices concatenated over all
    images. This reassembles them per batch element (inferring the
    image-to-example mapping from ``image_end`` tokens) and offsets pooling
    indices into each example's stacked crop patches. Runs host-side (it uses
    data-dependent Python control flow), keeping the exported model graph clean.

    Returns:
        ``(images, token_pooling)`` of shapes ``(N, max_crops, n_patches, pixels)``
        and ``(N, max_pooled, pool_area)``.

    Raises:
        ValueError: If the number of ``image_end`` tokens does not match the
            number of image grids supplied in ``image_grids``.
    """
    counts = (input_ids == config.image_end_token_id).sum(1)  # images per example  # pyrefly: ignore[missing-attribute]
    num_images = int(image_grids.shape[0])
    if int(counts.sum()) != num_images:
        msg = f"image_end tokens ({int(counts.sum())}) do not match image grids ({num_images})."
        raise ValueError(msg)

    num_examples = counts.shape[0]
    device = input_ids.device
    _n_crops, n_patches, pixels_per_patch = pixel_values.shape

    pooled_per_image = (image_grids[:, :2].prod(1) + image_grids[:, 2:].prod(1)).to(image_num_crops.dtype)
    example_for_image = torch.arange(num_examples, device=device).repeat_interleave(counts)
    crops_per_example = torch.zeros(num_examples, dtype=image_num_crops.dtype, device=device)
    crops_per_example.index_add_(0, example_for_image, image_num_crops)
    pooled_per_example = torch.zeros(num_examples, dtype=pooled_per_image.dtype, device=device)
    pooled_per_example.index_add_(0, example_for_image, pooled_per_image)
    patches_per_image = image_num_crops * n_patches

    max_crops = int(crops_per_example.max())
    images = torch.full(
        (num_examples, max_crops, n_patches, pixels_per_patch),
        -1.0,
        dtype=pixel_values.dtype,
        device=pixel_values.device,
    )
    max_pooled = int(pooled_per_example.max())
    token_pooling = torch.full(
        (num_examples, max_pooled, image_token_pooling.shape[-1]),
        -1,
        dtype=image_token_pooling.dtype,
        device=image_token_pooling.device,
    )

    crop_offset = 0
    pooled_offset = 0
    image_offset = 0
    for example_idx in range(num_examples):
        num_example_images = int(counts[example_idx])
        num_example_crops = int(crops_per_example[example_idx])
        images[example_idx, :num_example_crops] = pixel_values[crop_offset : crop_offset + num_example_crops]

        example_pooling = image_token_pooling[
            pooled_offset : pooled_offset + int(pooled_per_example[example_idx])
        ].clone()
        patch_offset = 0
        row = 0
        for local_image in range(num_example_images):
            num_pooled = int(pooled_per_image[image_offset + local_image])
            block = example_pooling[row : row + num_pooled]
            example_pooling[row : row + num_pooled] = torch.where(block >= 0, block + patch_offset, block)
            patch_offset += int(patches_per_image[image_offset + local_image])
            row += num_pooled
        token_pooling[example_idx, : example_pooling.shape[0]] = example_pooling

        crop_offset += num_example_crops
        pooled_offset += int(pooled_per_example[example_idx])
        image_offset += num_example_images

    return images, token_pooling


def build_model_inputs(  # noqa: PLR0914
    batch: dict[str, Any],
    *,
    config: MolmoAct2Config,
    image_processor: MolmoAct2ImageProcessor,
) -> dict[str, torch.Tensor]:
    """Assemble backbone-ready model inputs from a preprocessed batch.

    Args:
        batch: Preprocessed batch with ``TOKENIZED_PROMPT`` (with ``<image>``
            placeholders) and packed, already-resized ``IMAGES`` of shape
            ``(N, B, C, H, W)``.
        config: The MolmoAct2 configuration.
        image_processor: The PyTorch image (patchify) processor.

    Returns:
        A dict with ``input_ids``, ``attention_mask``, ``token_type_ids``,
        ``images`` (per-example batched crops), ``token_pooling`` and
        ``action_dim_is_pad``.

    Raises:
        ValueError: If ``images`` is not a 5-D ``(N, B, C, H, W)`` tensor, or
            if ``config.image_placeholder_token_id`` is unset.
    """
    input_ids = batch[TOKENIZED_PROMPT]
    attention_mask = batch.get(TOKENIZED_PROMPT_MASK)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    images = batch[IMAGES]
    if images.ndim != 5:  # noqa: PLR2004
        msg = f"Expected packed images of shape (N, B, C, H, W), got {tuple(images.shape)}."
        raise ValueError(msg)
    num_images, batch_size, channels, height, width = images.shape
    flat_images = images.permute(1, 0, 2, 3, 4).reshape(batch_size * num_images, channels, height, width)

    image_out = image_processor(flat_images)
    pixel_values = image_out["pixel_values"].to(input_ids.device)
    image_token_pooling = image_out["image_token_pooling"].to(input_ids.device)
    image_grids = image_out["image_grids"].to(input_ids.device)
    image_num_crops = image_out["image_num_crops"].to(input_ids.device)

    if config.image_placeholder_token_id is None:
        msg = "MolmoAct2 config is missing image_placeholder_token_id for placeholder expansion."
        raise ValueError(msg)
    input_ids, attention_mask, token_type_ids = _expand_image_placeholders(
        config=config,
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_grids=image_grids,
        image_placeholder_token_id=int(config.image_placeholder_token_id),
    )

    batched_images, token_pooling = build_batched_images(
        config,
        input_ids,
        pixel_values,
        image_token_pooling,
        image_grids,
        image_num_crops,
    )
    action_dim_is_pad = _default_action_dim_is_pad(config, batch_size=batch_size, device=input_ids.device)

    model_inputs: dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "images": batched_images,
        "token_pooling": token_pooling,
        "action_dim_is_pad": action_dim_is_pad,
    }
    if token_type_ids is not None:
        model_inputs["token_type_ids"] = token_type_ids
    return model_inputs
