# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 model implementation."""

# isort: skip_file
# ruff: noqa: D,I001

import json
import os
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor
from tqdm import tqdm

from physicalai.data.constants import IMAGES, IMAGE_MASKS, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK, STATE
from physicalai.data.observation import FeatureType
from physicalai.policies.base import Model
from physicalai.policies.molmoact2.action_tokenizer import UniversalActionProcessor
from physicalai.policies.molmoact2.config import MolmoAct2Config

from .backbones import MolmoAct2ForConditionalGeneration
from .image import MolmoAct2ImageProcessor
from .video import MolmoAct2VideoProcessor

SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"


def _validate_local_pooling_indices(inputs: dict[str, Any]) -> None:
    if "pixel_values" in inputs and "image_token_pooling" in inputs:
        pixel_values = inputs["pixel_values"]
        image_token_pooling = inputs["image_token_pooling"]
        if torch.is_tensor(pixel_values) and pixel_values.ndim == 3 and torch.is_tensor(image_token_pooling):
            n_patches = int(pixel_values.shape[1])
            valid = image_token_pooling >= 0
            if torch.any(valid):
                max_idx = int(image_token_pooling[valid].max().item())
                if max_idx >= n_patches:
                    raise ValueError(
                        "image_token_pooling contains out-of-range indices for per-image local patch IDs: "
                        f"max_idx={max_idx}, n_patches={n_patches}."
                    )

    if "pixel_values_videos" in inputs and "video_token_pooling" in inputs:
        pixel_values_videos = inputs["pixel_values_videos"]
        video_token_pooling = inputs["video_token_pooling"]
        if (
            torch.is_tensor(pixel_values_videos)
            and pixel_values_videos.ndim == 3
            and torch.is_tensor(video_token_pooling)
        ):
            n_frame_patches_total = int(pixel_values_videos.shape[0] * pixel_values_videos.shape[1])
            valid = video_token_pooling >= 0
            if torch.any(valid):
                max_idx = int(video_token_pooling[valid].max().item())
                if max_idx >= n_frame_patches_total:
                    msg = f"video_token_pooling contains out-of-range indices for local frame patch IDs: max_idx={max_idx}, total_patches={n_frame_patches_total}."
                    raise ValueError(msg)


def _build_image_processor(config: MolmoAct2Config) -> MolmoAct2ImageProcessor | None:
    processor_config = config.processor_config
    if processor_config is None:
        return None
    image_cfg = processor_config.image_processor
    return MolmoAct2ImageProcessor(
        size=image_cfg.size,
        image_mean=image_cfg.image_mean,
        image_std=image_cfg.image_std,
        do_convert_rgb=image_cfg.do_convert_rgb,
        max_crops=image_cfg.max_crops,
        overlap_margins=image_cfg.overlap_margins,
        crop_mode=image_cfg.crop_mode,
        patch_size=image_cfg.patch_size,
        pooling_size=image_cfg.pooling_size,
    )


def _build_video_processor(config: MolmoAct2Config) -> MolmoAct2VideoProcessor | None:
    processor_config = config.processor_config
    if processor_config is None:
        return None
    video_cfg = processor_config.video_processor
    return MolmoAct2VideoProcessor(
        size=video_cfg.size,
        image_mean=video_cfg.image_mean,
        image_std=video_cfg.image_std,
        do_convert_rgb=video_cfg.do_convert_rgb,
        patch_size=video_cfg.patch_size,
        pooling_size=video_cfg.pooling_size,
        do_sample_frames=video_cfg.do_sample_frames,
        frame_sample_mode=video_cfg.frame_sample_mode,
        max_fps=int(video_cfg.max_fps),
        sampling_fps=video_cfg.sampling_fps,
    )


def _flatten_images(images: torch.Tensor | None, image_masks: torch.Tensor | None) -> torch.Tensor | None:
    if images is None:
        return None
    if images.ndim == 4:
        images = images.unsqueeze(0)
    if images.ndim != 5:
        msg = f"MolmoAct2 expected images with shape [N, B, C, H, W], got {tuple(images.shape)}."
        raise ValueError(msg)

    num_images, batch_size = int(images.shape[0]), int(images.shape[1])
    if image_masks is None:
        image_masks = torch.ones((num_images, batch_size), dtype=torch.bool, device=images.device)
    elif image_masks.ndim == 1:
        image_masks = image_masks.unsqueeze(0)

    flat_images: list[torch.Tensor] = []
    valid_masks = image_masks.to(device=images.device, dtype=torch.bool)
    for batch_idx in range(batch_size):
        for image_idx in range(num_images):
            if bool(valid_masks[image_idx, batch_idx].item()):
                flat_images.append(images[image_idx, batch_idx])

    if not flat_images:
        return None
    return torch.stack(flat_images, dim=0)


def _default_action_dim_is_pad(config: MolmoAct2Config, *, batch_size: int, device: torch.device) -> torch.Tensor:
    action_dim_is_pad = torch.ones((batch_size, int(config.max_action_dim)), dtype=torch.bool, device=device)
    env_action_dim = _env_action_dim(config)
    if env_action_dim > 0:
        action_dim_is_pad[:, :env_action_dim] = False
    return action_dim_is_pad


def _to_device(value: torch.Tensor | None, *, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    return torch.as_tensor(value, device=device)


def _resolved_image_placeholder_token_id(config: MolmoAct2Config) -> int | None:
    if config.image_placeholder_token_id is None:
        return None
    return int(config.image_placeholder_token_id)


def _build_token_type_ids(
    config: MolmoAct2Config, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor | None:
    image_token_ids = [
        config.image_patch_id,
        config.image_col_id,
        config.image_start_token_id,
        config.low_res_image_start_token_id,
        config.frame_start_token_id,
        config.image_end_token_id,
        config.frame_end_token_id,
        config.image_low_res_id,
    ]
    image_token_ids = [int(x) for x in image_token_ids if x is not None]
    if not image_token_ids:
        return None

    token_set = torch.as_tensor(image_token_ids, device=input_ids.device, dtype=input_ids.dtype)
    token_type_ids = (input_ids.unsqueeze(-1) == token_set.view(1, 1, -1)).any(dim=-1).to(dtype=torch.long)
    return token_type_ids * attention_mask.to(dtype=torch.long)


def _image_token_ids_for_grid(config: MolmoAct2Config, grid: torch.Tensor) -> list[int]:
    if grid.ndim != 1 or int(grid.numel()) != 4:
        msg = f"Expected image grid shape (4,), got {tuple(grid.shape)}"
        raise ValueError(msg)

    if config.image_patch_id is None or config.image_start_token_id is None or config.image_end_token_id is None:
        msg = "MolmoAct2 config must define image_patch_id/image_start_token_id/image_end_token_id."
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
    resized_h, resized_w, height, width = [int(x) for x in grid.tolist()]

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
        row = [image_patch_id] * int(num_cols)
        if use_col and image_col_id is not None:
            row = row + [image_col_id]
        return row * int(num_rows)

    if height == 0 or width == 0:
        return (
            [image_start_token_id]
            + make_rows(resized_h, resized_w, use_col=use_single_crop_col_tokens)
            + [
                image_end_token_id,
            ]
        )

    high_res = [image_start_token_id] + make_rows(height, width, use_col=image_use_col_tokens) + [image_end_token_id]
    low_start = low_res_start_id if use_single_crop_start_token else image_start_token_id
    low_res = [low_start] + make_rows(resized_h, resized_w, use_col=use_single_crop_col_tokens) + [image_end_token_id]
    return low_res + high_res


def _expand_image_placeholders(
    *,
    config: MolmoAct2Config,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grids: torch.Tensor,
    image_placeholder_token_id: int,
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if int(image_grids.shape[0]) == 0:
        return input_ids, attention_mask, _build_token_type_ids(config, input_ids, attention_mask), labels

    pad_values = input_ids[attention_mask == 0]
    pad_token_id = int(pad_values[0].item()) if int(pad_values.numel()) > 0 else 0

    expanded_per_example: list[list[int]] = []
    expanded_labels_per_example: list[list[int]] | None = [] if labels is not None else None
    grid_idx = 0
    batch_size = int(input_ids.shape[0])

    for b_idx in range(batch_size):
        valid_mask_bool = attention_mask[b_idx].to(dtype=torch.bool)
        valid_ids = input_ids[b_idx][valid_mask_bool]
        valid_labels = labels[b_idx][valid_mask_bool] if labels is not None else None
        expanded_ids: list[int] = []
        expanded_labels: list[int] | None = [] if labels is not None else None
        for pos, token in enumerate(valid_ids.tolist()):
            token_int = int(token)
            if token_int == image_placeholder_token_id:
                if grid_idx >= int(image_grids.shape[0]):
                    msg = "Not enough image grids to expand all <|image|> placeholders in input_ids."
                    raise ValueError(msg)
                image_token_ids = _image_token_ids_for_grid(config, image_grids[grid_idx])
                expanded_ids.extend(image_token_ids)
                if expanded_labels is not None:
                    expanded_labels.extend([-100] * len(image_token_ids))
                grid_idx += 1
            else:
                expanded_ids.append(token_int)
                if expanded_labels is not None and valid_labels is not None:
                    expanded_labels.append(int(valid_labels[pos].item()))

        expanded_per_example.append(expanded_ids)
        if expanded_labels_per_example is not None:
            expanded_labels_per_example.append(expanded_labels)

    max_len = max((len(tokens) for tokens in expanded_per_example), default=1)
    out_ids = torch.full((batch_size, max_len), fill_value=pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
    out_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    out_labels: torch.Tensor | None = None
    if labels is not None:
        out_labels = torch.full((batch_size, max_len), fill_value=-100, dtype=labels.dtype, device=labels.device)

    for b_idx, tokens in enumerate(expanded_per_example):
        if not tokens:
            continue
        token_tensor = torch.as_tensor(tokens, dtype=input_ids.dtype, device=input_ids.device)
        out_ids[b_idx, : token_tensor.numel()] = token_tensor
        out_mask[b_idx, : token_tensor.numel()] = 1
        if out_labels is not None and expanded_labels_per_example is not None:
            label_tensor = torch.as_tensor(expanded_labels_per_example[b_idx], dtype=labels.dtype, device=labels.device)
            out_labels[b_idx, : label_tensor.numel()] = label_tensor

    token_type_ids = _build_token_type_ids(config, out_ids, out_mask)
    return out_ids, out_mask, token_type_ids, out_labels


def _build_model_inputs_from_batch(
    *,
    batch: dict[str, Any],
    config: MolmoAct2Config,
    device: torch.device,
    image_processor: MolmoAct2ImageProcessor | None,
    video_processor: MolmoAct2VideoProcessor | None,
    include_labels: bool,
) -> dict[str, Tensor]:
    input_ids = torch.as_tensor(batch[TOKENIZED_PROMPT], device=device)
    attention_mask = _to_device(batch.get(TOKENIZED_PROMPT_MASK), device=device)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    labels = _to_device(batch.get("labels"), device=device) if include_labels else None
    token_type_ids = _to_device(batch.get("token_type_ids"), device=device)

    pixel_values = _to_device(batch.get("pixel_values"), device=device)
    image_token_pooling = _to_device(batch.get("image_token_pooling"), device=device)
    image_grids = _to_device(batch.get("image_grids"), device=device)
    image_num_crops = _to_device(batch.get("image_num_crops"), device=device)

    if pixel_values is None and image_processor is not None:
        images = _to_device(batch.get(IMAGES), device=device)
        image_masks = _to_device(batch.get(IMAGE_MASKS), device=device)
        flat_images = _flatten_images(images, image_masks)
        if flat_images is not None:
            image_out = image_processor(flat_images, return_tensors="pt")
            pixel_values = _to_device(image_out.get("pixel_values"), device=device)
            image_token_pooling = _to_device(image_out.get("image_token_pooling"), device=device)
            image_grids = _to_device(image_out.get("image_grids"), device=device)
            image_num_crops = _to_device(image_out.get("image_num_crops"), device=device)

    image_placeholder_token_id = _resolved_image_placeholder_token_id(config)
    if image_grids is not None and image_placeholder_token_id is not None:
        input_ids, attention_mask, rebuilt_token_type_ids, labels = _expand_image_placeholders(
            config=config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grids=image_grids,
            image_placeholder_token_id=image_placeholder_token_id,
            labels=labels,
        )
        token_type_ids = rebuilt_token_type_ids

    pixel_values_videos = _to_device(batch.get("pixel_values_videos"), device=device)
    video_token_pooling = _to_device(batch.get("video_token_pooling"), device=device)
    video_grids = _to_device(batch.get("video_grids"), device=device)
    videos_btchw = _to_device(batch.get("videos_btchw"), device=device)
    if pixel_values_videos is None and videos_btchw is not None and video_processor is not None:
        video_out = video_processor(videos_btchw, return_tensors="pt", return_metadata=False)
        pixel_values_videos = _to_device(video_out.get("pixel_values_videos"), device=device)
        video_token_pooling = _to_device(video_out.get("video_token_pooling"), device=device)
        video_grids = _to_device(video_out.get("video_grids"), device=device)

    action_dim_is_pad = _to_device(batch.get("action_dim_is_pad"), device=device)
    if action_dim_is_pad is None:
        action_dim_is_pad = _default_action_dim_is_pad(config, batch_size=int(input_ids.shape[0]), device=device)
    else:
        action_dim_is_pad = action_dim_is_pad.to(dtype=torch.bool)

    model_inputs: dict[str, Tensor] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "action_dim_is_pad": action_dim_is_pad,
    }

    optional = {
        "token_type_ids": token_type_ids,
        "pixel_values": pixel_values,
        "image_token_pooling": image_token_pooling,
        "image_grids": image_grids,
        "image_num_crops": image_num_crops,
        "pixel_values_videos": pixel_values_videos,
        "video_token_pooling": video_token_pooling,
        "video_grids": video_grids,
        "labels": labels,
    }
    for key, value in optional.items():
        if value is not None:
            model_inputs[key] = value

    _validate_local_pooling_indices(model_inputs)
    return model_inputs


def _strict_load_safetensors_weights(model: torch.nn.Module, checkpoint_location: str) -> None:
    """Load safetensors weights into a model, strictly verifying key correspondence.

    Supports both sharded (index JSON) and single-file checkpoints. Raises
    clearly if the checkpoint does not exist or does not match the model.

    Args:
        model: The PyTorch module to load weights into.
        checkpoint_location: Directory containing the safetensors checkpoint.

    Raises:
        FileNotFoundError: If neither a sharded index nor a single weights file
            is found at ``checkpoint_location``.
        RuntimeError: If the checkpoint keys do not exactly match the model keys.
    """
    index_path = os.path.join(checkpoint_location, SAFE_WEIGHTS_INDEX_NAME)
    single_file_path = os.path.join(checkpoint_location, SAFE_WEIGHTS_NAME)

    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as file_obj:
            index = json.load(file_obj)
        weight_map = index["weight_map"]
        loaded_keys = set(weight_map)
        model_keys = set(model.state_dict())
        missing_keys = sorted(model_keys - loaded_keys)
        unexpected_keys = sorted(loaded_keys - model_keys)
        if missing_keys or unexpected_keys:
            message = ["MolmoAct2 safetensors do not match the local model implementation."]
            if missing_keys:
                message.append(f"Missing keys: {missing_keys[:8]}")
            if unexpected_keys:
                message.append(f"Unexpected keys: {unexpected_keys[:8]}")
            raise RuntimeError(" ".join(message))
        shards = sorted(set(weight_map.values()))
        for shard_file in tqdm(shards, desc="Loading MolmoAct2 weights", unit="shard"):
            state_dict = load_safetensors_file(os.path.join(checkpoint_location, shard_file), device="cpu")
            model.load_state_dict(state_dict, strict=False)
            del state_dict
        return

    if os.path.isfile(single_file_path):
        print(f"Loading MolmoAct2 weights from {single_file_path} ...")
        state_dict = load_safetensors_file(single_file_path, device="cpu")
        model.load_state_dict(state_dict, strict=True)
        print("MolmoAct2 weights loaded.")
        return

    raise FileNotFoundError(
        f"No safetensors checkpoint found at '{checkpoint_location}'. "
        f"Expected '{SAFE_WEIGHTS_NAME}' or '{SAFE_WEIGHTS_INDEX_NAME}'."
    )


def _env_action_dim(config: MolmoAct2Config) -> int:
    if config.output_features is None:
        msg = f"Output action dimension can't be determined, config has no output features. Config Output Features: {config.output_features}"
        raise ValueError(msg)
    action_feature = next((f for f in config.output_features if f.ftype == FeatureType.ACTION), None)
    if action_feature is None or action_feature.shape is None:
        return 0
    return int(action_feature.shape[0])


class MolmoAct2Model(Model):
    """Wrapper for MolmoAct2ForConditionalGeneration using physicalai config.

    This model handles both training and inference modes:

    - Training: Computes supervised losses using the backbone.
    - Inference: Generates predicted action chunks.

    Weight loading is intentionally separated from construction. Call
    :meth:`load_pretrained_weights` explicitly after instantiation when a
    pretrained checkpoint is available.
    """

    def __init__(self, config: MolmoAct2Config) -> None:
        """Initialize the MolmoAct2 model wrapper.

        Constructs the backbone architecture from ``config`` but does **not**
        load any weights. Call :meth:`load_pretrained_weights` separately to
        load a pretrained checkpoint.

        Args:
            config: MolmoAct2Config instance with all model components defined.
        """
        super().__init__()
        self.config = config
        self.backbone = MolmoAct2ForConditionalGeneration(config)
        self.image_processor = _build_image_processor(config)
        self.video_processor = _build_video_processor(config)
        self._action_tokenizer: UniversalActionProcessor | None = None

    def enable_torch_compile(self, *, mode: str = "default") -> None:
        """Compile the full MolmoAct2 backbone with torch.compile.

        Compile only the tensor-heavy action generation entrypoint used by
        inference. This avoids wrapper/module-level graph breaks and tends to
        produce lower steady-state latency than compiling broad module scopes.
        """
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            msg = "compile_model=true requires torch.compile, but this PyTorch build does not provide it."
            raise RuntimeError(msg)
        torch.set_float32_matmul_precision("high")
        self.backbone.model.generate_actions_from_inputs = compile_fn(  # type: ignore[method-assign]
            self.backbone.model.generate_actions_from_inputs,
            mode=mode,
        )

    def load_pretrained_weights(self, checkpoint_location: str) -> None:
        """Load pretrained safetensors weights from a checkpoint directory.

        Args:
            checkpoint_location: Path to a directory containing either
                ``model.safetensors`` or ``model.safetensors.index.json``.
        """
        _strict_load_safetensors_weights(self.backbone, checkpoint_location)

    # TODO
    @property
    def action_delta_indices(self) -> list | None:
        """Return action delta indices if this wrapper defines them."""
        return None

    # TODO
    @property
    def observation_delta_indices(self) -> list | None:
        """Return observation delta indices if this wrapper defines them."""
        return None

    # TODO
    @property
    def reward_delta_indices(self) -> list | None:
        """Return reward delta indices if this wrapper defines them."""
        return None

    # TODO: refactor to more reasonable api
    @property
    def _inner_model(self):
        return self.backbone.model

    def compute_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        """Compute mode-aware MolmoAct2 training loss."""
        # TODO
        return None, None

    @torch.no_grad()
    def compute_val_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        # TODO
        return None, {"loss": None}

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]] | Tensor:
        """Run forward pass in training or inference mode.

        Args:
            batch: Input batch dictionary.

        Returns:
            In training mode, a tuple of (loss_tensor, metrics_dict).
            In inference mode, the predicted action tensor.
        """
        if self.training:
            return self.compute_loss(batch)
        return self.predict_action_chunk(batch)

    def prepare_graph_inputs(self, batch: dict[str, Any]) -> dict[str, Any]:
        target_device = next(self.backbone.parameters()).device
        include_labels = bool(self.training)
        return _build_model_inputs_from_batch(
            batch=batch,
            config=self.config,
            device=target_device,
            image_processor=self.image_processor,
            video_processor=self.video_processor,
            include_labels=include_labels,
        )

    @property
    def exported_torch_module(self) -> torch.nn.Module:
        """Torch module used for export-time inference graph tracing."""
        return self._inner_model

    def predict_action_chunk(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """Generate an action chunk directly from a preprocessed inference batch."""
        model_inputs = self.prepare_graph_inputs(batch)

        actions = self._inner_model.generate_actions_from_inputs(
            **model_inputs,
            action_horizon=int(self.config.n_action_steps),
        )

        env_action_dim = _env_action_dim(self.config)
        if env_action_dim > 0:
            actions = actions[..., :env_action_dim]

        n_action_steps = int(self.config.n_action_steps)
        if n_action_steps > 0:
            actions = actions[:, :n_action_steps]

        return {"actions": actions}
