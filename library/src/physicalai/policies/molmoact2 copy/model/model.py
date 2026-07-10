# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 model implementation."""

import json
import os
from typing import Any

import torch
import torch.nn.functional as nn_functional
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor
from tqdm import tqdm

from physicalai.data.constants import ACTION, IMAGES, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.data.observation import FeatureType
from physicalai.policies.base import Model
from physicalai.policies.molmoact2.action_tokenizer import UniversalActionProcessor
from physicalai.policies.molmoact2.config import MolmoAct2Config

from .backbones import MolmoAct2ForConditionalGeneration
from .image import MolmoAct2ImageProcessor
from .video import MolmoAct2VideoProcessor

SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"


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


def _default_action_dim_is_pad(config: MolmoAct2Config, *, batch_size: int, device: torch.device) -> torch.Tensor:
    action_dim_is_pad = torch.ones((batch_size, int(config.max_action_dim)), dtype=torch.bool, device=device)
    env_action_dim = _env_action_dim(config)
    if env_action_dim > 0:
        action_dim_is_pad[:, :env_action_dim] = False
    return action_dim_is_pad


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


def _build_discrete_labels_from_input_ids(
    config: MolmoAct2Config,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    action_start_id = config.action_start_token_id
    action_end_id = config.action_end_token_id
    if action_start_id is None or action_end_id is None:
        msg = "MolmoAct2 discrete labels require action_start_token_id/action_end_token_id in config."
        raise ValueError(msg)

    eos_token_id = config.eos_token_id
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0] if eos_token_id else None
    eos_token_id = None if eos_token_id is None else int(eos_token_id)

    labels = torch.full_like(input_ids, -100)
    start_id = int(action_start_id)
    end_id = int(action_end_id)

    for batch_idx in range(input_ids.shape[0]):
        valid = attention_mask[batch_idx].to(dtype=torch.bool)
        row = input_ids[batch_idx]
        starts = (row == start_id).nonzero(as_tuple=False).flatten().tolist()
        ends = (row == end_id).nonzero(as_tuple=False).flatten().tolist()
        end_ptr = 0
        for start in starts:
            while end_ptr < len(ends) and ends[end_ptr] < start:
                end_ptr += 1
            if end_ptr >= len(ends):
                msg = "Found <action_start> without matching <action_end> in MolmoAct2 labels."
                raise ValueError(msg)
            end = int(ends[end_ptr])
            label_end = end + 1
            if eos_token_id is not None and label_end < int(row.shape[0]) and int(row[label_end]) == eos_token_id:
                label_end += 1
            labels[batch_idx, start:label_end] = row[start:label_end]
            end_ptr += 1
        if not starts:
            msg = "No discrete action span found in MolmoAct2 training text."
            raise ValueError(msg)
        labels[batch_idx] = torch.where(valid, labels[batch_idx], torch.full_like(labels[batch_idx], -100))

    return labels


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
        if expanded_labels_per_example is not None and expanded_labels is not None:
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
            label_tensor = torch.as_tensor(
                expanded_labels_per_example[b_idx],
                dtype=out_labels.dtype,
                device=out_labels.device,
            )
            out_labels[b_idx, : label_tensor.numel()] = label_tensor

    token_type_ids = _build_token_type_ids(config, out_ids, out_mask)
    return out_ids, out_mask, token_type_ids, out_labels


def _build_model_inputs_from_batch(
    *,
    batch: dict[str, Any],
    config: MolmoAct2Config,
    image_processor: MolmoAct2ImageProcessor | None,
    video_processor: MolmoAct2VideoProcessor | None,
    include_labels: bool,
) -> dict[str, Tensor]:
    # Text modality: prompt tokens and mask.
    input_ids = batch[TOKENIZED_PROMPT]
    prompt_mask = batch.get(TOKENIZED_PROMPT_MASK)
    attention_mask = prompt_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    labels = batch["labels"] if include_labels and "labels" in batch else None
    if include_labels and labels is None and config.action_mode in {"discrete", "both"}:
        labels = _build_discrete_labels_from_input_ids(
            config,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    # Image modality: preprocess raw images into model-ready image tensors.
    if image_processor is None:
        msg = "MolmoAct2 image processor is required for image-based action prediction."
        raise ValueError(msg)

    images = batch[IMAGES]
    if images.ndim != 5:
        msg = f"MolmoAct2 expected preprocessed images with shape [N, B, C, H, W], got {tuple(images.shape)}."
        raise ValueError(msg)

    num_images, batch_size, channels, height, width = images.shape
    # Flatten [N, B, C, H, W] -> [N*B, C, H, W] for processor input.
    flat_images = images.permute(1, 0, 2, 3, 4).reshape(batch_size * num_images, channels, height, width)

    image_out = image_processor(flat_images, return_tensors="pt")
    pixel_values = image_out["pixel_values"].to(input_ids.device)
    image_token_pooling = image_out["image_token_pooling"].to(input_ids.device)
    image_grids = image_out["image_grids"].to(input_ids.device)
    image_num_crops = image_out["image_num_crops"].to(input_ids.device)

    # Prompt/Image fusion: expand image placeholders and rebuild token types.
    token_type_ids = _build_token_type_ids(config, input_ids, attention_mask)
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

    # Video modality: preprocess videos when provided, or consume preprocessed video tensors.
    pixel_values_videos = batch.get("pixel_values_videos")
    video_token_pooling = batch.get("video_token_pooling")
    video_grids = batch.get("video_grids")
    videos_btchw = batch.get("videos_btchw")
    if video_processor is not None and videos_btchw is not None:
        video_out = video_processor(videos_btchw, return_tensors="pt", return_metadata=False)
        pixel_values_videos = video_out.get("pixel_values_videos")
        video_token_pooling = video_out.get("video_token_pooling")
        video_grids = video_out.get("video_grids")
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.to(input_ids.device)
        if video_token_pooling is not None:
            video_token_pooling = video_token_pooling.to(input_ids.device)
        if video_grids is not None:
            video_grids = video_grids.to(input_ids.device)

    # Action modality: build per-dimension padding mask from environment action dims.
    action_dim_is_pad = _default_action_dim_is_pad(
        config,
        batch_size=int(input_ids.shape[0]),
        device=input_ids.device,
    )

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
    model_inputs.update({key: value for key, value in optional.items() if value is not None})

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


def _sample_beta_timesteps(
    *,
    batch_size: int,
    device: torch.device,
    cutoff: float,
    time_offset: float,
    time_scale: float,
    alpha: float,
    beta: float,
) -> Tensor:
    if cutoff < time_offset:
        msg = f"flow-matching cutoff must be >= time_offset, got {cutoff} < {time_offset}"
        raise ValueError(msg)
    if time_scale <= 0:
        msg = f"flow-matching time_scale must be > 0, got {time_scale}"
        raise ValueError(msg)
    upper = min(cutoff, time_offset + time_scale)
    dist = torch.distributions.Beta(
        torch.tensor(alpha, device=device),
        torch.tensor(beta, device=device),
    )
    samples = dist.sample((batch_size,))
    scale = upper - time_offset
    if scale == 0:
        return torch.full((batch_size,), time_offset, device=device, dtype=samples.dtype)
    return time_offset + scale * samples


def _masked_loss_mean(
    loss: Tensor,
    *,
    action_horizon_is_pad: Tensor | None,
    action_dim_is_pad: Tensor | None,
) -> Tensor:
    mask = torch.ones_like(loss, dtype=torch.bool)

    if action_horizon_is_pad is not None:
        horizon_mask = ~action_horizon_is_pad.to(device=loss.device, dtype=torch.bool)
        horizon_view = horizon_mask.view(
            horizon_mask.shape[0],
            *([1] * max(loss.ndim - 3, 0)),
            horizon_mask.shape[1],
            1,
        )
        mask = mask & horizon_view

    if action_dim_is_pad is not None:
        dim_mask = ~action_dim_is_pad.to(device=loss.device, dtype=torch.bool)
        dim_view = dim_mask.view(
            dim_mask.shape[0],
            *([1] * max(loss.ndim - 2, 0)),
            dim_mask.shape[1],
        )
        mask = mask & dim_view

    valid = mask.to(dtype=loss.dtype)
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


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
        torch.set_float32_matmul_precision("high")
        self.backbone.model.generate_actions_from_inputs = torch.compile(  # type: ignore[method-assign]
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

    @property
    def action_tokenizer(self) -> UniversalActionProcessor:
        if self._action_tokenizer is not None:
            return self._action_tokenizer
        tokenizer_path = str(getattr(self.config, "discrete_action_tokenizer", "")).strip()
        if not tokenizer_path:
            msg = "config.discrete_action_tokenizer is required for discrete MolmoAct2 training."
            raise ValueError(msg)
        self._action_tokenizer = UniversalActionProcessor.from_pretrained_local(tokenizer_path)
        return self._action_tokenizer

    def _discrete_loss(
        self,
        *,
        labels: Tensor,
        hidden_states: Tensor | None,
        softmax_auxiliary_loss: bool,
        softmax_auxiliary_loss_scale: float,
    ) -> tuple[Tensor, Tensor | None]:
        if hidden_states is None:
            msg = "MolmoAct2 backbone did not return last_hidden_state."
            raise RuntimeError(msg)

        ignore_index = -100
        shift_labels = nn_functional.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
        valid_positions = shift_labels != ignore_index
        if not bool(valid_positions.any()):
            msg = "MolmoAct2 discrete training labels contain no valid action tokens."
            raise RuntimeError(msg)

        hidden_size = hidden_states.shape[-1]
        selected_hidden = hidden_states.reshape(-1, hidden_size)[valid_positions.reshape(-1)]
        selected_labels = shift_labels.reshape(-1)[valid_positions.reshape(-1)].to(device=hidden_states.device)
        logits = nn_functional.linear(selected_hidden, self.backbone.lm_head.weight).float()
        log_z = logits.logsumexp(dim=-1)
        target_logits = logits.gather(dim=-1, index=selected_labels[:, None]).squeeze(-1)
        ce_loss = (log_z - target_logits).mean()

        if not softmax_auxiliary_loss:
            return ce_loss, None

        z_loss = float(softmax_auxiliary_loss_scale) * log_z.pow(2).mean()
        return ce_loss, z_loss

    def _continuous_loss(
        self,
        *,
        batch: dict[str, Any],
        model_inputs: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor | None]:
        flow_loss, hidden_states = self._compute_flow_matching_loss_joint_per_layer(
            batch=batch,
            model_inputs=model_inputs,
        )
        return flow_loss, hidden_states

    def _encoder_attention_mask_for_action_expert(
        self,
        *,
        input_ids: Tensor | None,
        attention_mask: Tensor | None,
    ) -> Tensor | None:
        if attention_mask is not None:
            return self._inner_model._get_encoder_attention_mask(input_ids, attention_mask)
        if input_ids is not None:
            return input_ids != -1
        return None

    def _prepare_flow_matching_tensors(
        self,
        *,
        actions: Tensor,
        action_dim_is_pad: Tensor | None,
        timesteps: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        action_expert = self._inner_model._require_action_expert()
        action_dtype = next(action_expert.parameters()).dtype
        actions = actions.to(dtype=action_dtype)
        batch_size = int(actions.shape[0])
        device = actions.device

        if timesteps is None:
            timesteps = _sample_beta_timesteps(
                batch_size=batch_size,
                device=device,
                cutoff=self.config.flow_matching_cutoff,
                time_offset=self.config.flow_matching_time_offset,
                time_scale=self.config.flow_matching_time_scale,
                alpha=self.config.flow_matching_beta_alpha,
                beta=self.config.flow_matching_beta_beta,
            ).to(dtype=action_dtype)
        else:
            timesteps = timesteps.to(device=device, dtype=action_dtype)
            if tuple(timesteps.shape) != (batch_size,):
                msg = f"flow timesteps must have shape {(batch_size,)}, got {tuple(timesteps.shape)}."
                raise ValueError(msg)

        if self.config.mask_action_dim_padding:
            actions = self._inner_model._mask_action_dim_tensor(
                actions,
                action_dim_is_pad=action_dim_is_pad,
                enabled=True,
            )

        expected_noise_shape = tuple(actions.shape)
        if noise is None:
            noise = torch.randn(*expected_noise_shape, device=device, dtype=actions.dtype)
        else:
            noise = noise.to(device=device, dtype=actions.dtype)
            if tuple(noise.shape) != expected_noise_shape:
                msg = f"flow noise must have shape {expected_noise_shape}, got {tuple(noise.shape)}."
                raise ValueError(msg)

        if self.config.mask_action_dim_padding:
            noise = self._inner_model._mask_action_dim_tensor(
                noise,
                action_dim_is_pad=action_dim_is_pad,
                enabled=True,
            )

        t_broadcast = timesteps.view(batch_size, 1, 1)
        xt = (1.0 - t_broadcast) * noise + t_broadcast * actions
        target_velocity = actions - noise
        return actions, timesteps, xt, target_velocity

    def _prepare_joint_training_backbone_inputs(
        self,
        model_inputs: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        backbone_model = self._inner_model
        input_ids = model_inputs.get("input_ids")
        if input_ids is None:
            msg = "MolmoAct2 training requires input_ids."
            raise ValueError(msg)

        images, token_pooling = backbone_model.merge_visual_inputs(
            input_ids=input_ids,
            pixel_values=model_inputs.get("pixel_values"),
            image_token_pooling=model_inputs.get("image_token_pooling"),
            image_grids=model_inputs.get("image_grids"),
            image_num_crops=model_inputs.get("image_num_crops"),
            pixel_values_videos=model_inputs.get("pixel_values_videos"),
            video_token_pooling=model_inputs.get("video_token_pooling"),
            video_grids=model_inputs.get("video_grids"),
        )
        inputs_embeds, _ = backbone_model.build_input_embeddings(input_ids, images, token_pooling)

        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = model_inputs.get("position_ids")
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask_mapping = backbone_model._build_native_attention_bias(
            inputs_embeds=inputs_embeds,
            attention_mask=model_inputs.get("attention_mask"),
            token_type_ids=model_inputs.get("token_type_ids"),
            past_key_values=None,
        )
        return inputs_embeds, causal_mask_mapping, position_ids, cache_position

    @staticmethod
    def _decoder_layer_kv_outputs(layer_outputs: tuple[Any, ...], *, output_attentions: bool) -> tuple[Tensor, Tensor]:
        output_idx = 2 if output_attentions else 1
        return layer_outputs[output_idx], layer_outputs[output_idx + 1]

    @staticmethod
    def _action_time_conditioning(action_expert: torch.nn.Module, timesteps: Tensor) -> Tensor:
        return action_expert._time_conditioning(timesteps)

    def _compute_flow_matching_loss_joint_per_layer(
        self,
        *,
        batch: dict[str, Any],
        model_inputs: dict[str, Tensor],
        timesteps: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        backbone_model = self._inner_model
        transformer = backbone_model.transformer
        action_expert = backbone_model._require_action_expert()
        actions = batch.get(ACTION)
        if actions is None:
            msg = "MolmoAct2 training requires padded action targets in the preprocessed batch."
            raise ValueError(msg)

        actions, timesteps, xt, target_velocity = self._prepare_flow_matching_tensors(
            actions=actions,
            action_dim_is_pad=batch.get("action_dim_is_pad"),
            timesteps=timesteps,
            noise=noise,
        )
        batch_size = int(actions.shape[0])
        device = actions.device

        hidden_states, causal_mask_mapping, position_ids, cache_position = self._prepare_joint_training_backbone_inputs(
            model_inputs,
        )
        if hidden_states.shape[0] != batch_size:
            msg = f"Backbone batch size {hidden_states.shape[0]} does not match action batch size {batch_size}."
            raise ValueError(msg)

        encoder_attention_mask = self._encoder_attention_mask_for_action_expert(
            input_ids=model_inputs.get("input_ids"),
            attention_mask=model_inputs.get("attention_mask"),
        )
        action_attention_mask = None
        if batch.get("action_horizon_is_pad") is not None:
            action_attention_mask = ~batch["action_horizon_is_pad"].to(device=device, dtype=torch.bool)

        valid_action = None
        if action_attention_mask is not None:
            valid_action = action_attention_mask.to(device=device, dtype=actions.dtype).unsqueeze(-1)

        rope_cache = None
        if len(action_expert.blocks) > 0 and action_expert.blocks[0].self_attn.rope is not None:
            rope_cache = action_expert.blocks[0].self_attn.rope.build_cache(
                seq_len=actions.shape[1],
                device=device,
                dtype=actions.dtype,
            )

        cross_mask = action_expert._build_cross_attention_mask(
            encoder_attention_mask,
            batch_size,
            actions.dtype,
        )
        self_mask = action_expert._build_self_attention_mask(
            action_attention_mask,
            actions.shape[1],
            device,
            actions.dtype,
        )

        conditioning = self._action_time_conditioning(action_expert, timesteps)
        action_hidden = action_expert.action_embed(xt)
        if valid_action is not None:
            action_hidden = action_hidden * valid_action

        if transformer.config.rope_scaling_layers is not None:
            position_embeddings_mapping = {
                "default": transformer.rotary_embs["default"](hidden_states, position_ids),
                "scaling": transformer.rotary_embs["scaling"](hidden_states, position_ids),
            }
        else:
            position_embeddings = transformer.rotary_emb(hidden_states, position_ids)

        for layer_idx in range(int(transformer.config.num_hidden_layers)):
            decoder_block = transformer.blocks[layer_idx]
            action_block = action_expert.blocks[layer_idx]
            if transformer.config.rope_scaling_layers is not None:
                position_embeddings_i = (
                    position_embeddings_mapping["scaling"]
                    if layer_idx in transformer.config.rope_scaling_layers
                    else position_embeddings_mapping["default"]
                )
            else:
                position_embeddings_i = position_embeddings

            layer_outputs = decoder_block(
                hidden_states,
                position_embeddings=position_embeddings_i,
                attention_mask=causal_mask_mapping,
                position_ids=position_ids,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
                collect_layer_kv_states=True,
            )
            hidden_states = layer_outputs[0]
            key_states, value_states = self._decoder_layer_kv_outputs(layer_outputs, output_attentions=False)
            key_states = backbone_model._cache_to_sequence(key_states)
            value_states = backbone_model._cache_to_sequence(value_states)

            k_ctx = action_expert._project_kv_tensor(key_states, action_expert.context_k_proj)
            v_ctx = action_expert._project_kv_tensor(value_states, action_expert.context_v_proj)
            k_norm = action_block.cross_attn.k_norm
            if k_norm is not None:
                k_ctx = k_norm(k_ctx.transpose(1, 2)).transpose(1, 2)

            action_hidden = action_block(
                action_hidden,
                conditioning,
                cross_kv=(k_ctx, v_ctx),
                self_attn_mask=self_mask,
                attn_mask=cross_mask,
                is_causal=action_expert.config.causal_attn,
                modulation=None,
                rope_cache=rope_cache,
            )
            if valid_action is not None:
                action_hidden = action_hidden * valid_action

        hidden_states = transformer.ln_f(hidden_states)
        pred_velocity = action_expert.final_layer(action_hidden, conditioning)
        if valid_action is not None:
            pred_velocity = pred_velocity * valid_action

        loss = nn_functional.mse_loss(pred_velocity, target_velocity, reduction="none")
        loss = _masked_loss_mean(
            loss,
            action_horizon_is_pad=batch.get("action_horizon_is_pad"),
            action_dim_is_pad=batch.get("action_dim_is_pad") if self.config.mask_action_dim_padding else None,
        )
        return loss, hidden_states

    def _resolved_action_dim(self, batch: dict[str, Any], gt_actions: Tensor | None = None) -> int:
        action_dim_is_pad = batch.get("action_dim_is_pad")
        if action_dim_is_pad is not None:
            valid_counts = (~action_dim_is_pad.to(dtype=torch.bool)).sum(dim=-1)
            if bool((valid_counts == valid_counts[0]).all()) and int(valid_counts[0]) > 0:
                return int(valid_counts[0])
        config_dim = _env_action_dim(self.config)
        if config_dim > 0:
            return int(config_dim)
        if gt_actions is not None:
            return int(gt_actions.shape[-1])
        return int(self.config.max_action_dim)

    def _discrete_generation_max_steps(self, action_horizon: int) -> int:
        return max(1, int(action_horizon) * 16)

    def _resolve_predict_action_mode(self, requested_mode: str | None) -> str:
        training_mode = str(getattr(self.config, "action_mode", "continuous"))
        if requested_mode is None:
            if training_mode == "both":
                msg = (
                    "MolmoAct2 inference requires predict_action_mode to be set explicitly "
                    "to either 'continuous' or 'discrete' when action_mode='both'."
                )
                raise ValueError(msg)
            requested_mode = training_mode

        resolved_mode = str(requested_mode)
        if resolved_mode not in {"continuous", "discrete"}:
            msg = "predict_action_mode must be either 'continuous' or 'discrete'."
            raise ValueError(msg)
        if resolved_mode == "continuous" and training_mode == "discrete":
            msg = "MolmoAct2 action_mode='discrete' checkpoint cannot run continuous inference."
            raise ValueError(msg)
        if resolved_mode == "discrete" and training_mode == "continuous":
            msg = "MolmoAct2 action_mode='continuous' checkpoint cannot run discrete inference."
            raise ValueError(msg)
        return resolved_mode

    def _generate_discrete_actions_from_inputs(
        self,
        *,
        model_inputs: dict[str, Tensor],
        action_dim: int,
        action_horizon: int,
    ) -> Tensor:
        backbone_inputs = {
            key: value for key, value in model_inputs.items() if key not in ("action_dim_is_pad", "labels")
        }
        prefill_output = self.backbone(
            **backbone_inputs,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        generated_token_ids = self.backbone._continue_discrete_generation_from_output(
            prefill_output,
            past_key_values=prefill_output.past_key_values,
            attention_mask=backbone_inputs.get("attention_mask"),
            end_token_id=self.backbone._require_eos_token_id(),
            max_steps=self._discrete_generation_max_steps(action_horizon),
        )
        return self.backbone._decode_discrete_action_chunk(
            generated_token_ids,
            action_tokenizer=self.action_tokenizer,
            action_dim=action_dim,
            action_horizon=action_horizon,
        )

    def compute_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        """Compute mode-aware MolmoAct2 training loss."""
        # Convert batch to model expectation.
        model_inputs = self.prepare_graph_inputs(batch)
        losses = []
        metrics = {}

        # Discrete action supervision.
        if self.config.action_mode == "discrete":
            labels = model_inputs.get("labels")
            if labels is None:
                msg = "MolmoAct2 discrete training requires labels."
                raise RuntimeError(msg)
            outputs = self._inner_model(
                **model_inputs,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )
            discrete_ce_loss, discrete_z_loss = self._discrete_loss(
                labels=labels,
                hidden_states=outputs.last_hidden_state,
                softmax_auxiliary_loss=bool(getattr(self.config, "softmax_auxiliary_loss", False)),
                softmax_auxiliary_loss_scale=float(getattr(self.config, "softmax_auxiliary_loss_scale", 0.0)),
            )
            discrete_loss = discrete_ce_loss if discrete_z_loss is None else discrete_ce_loss + discrete_z_loss
            losses.append(discrete_loss)
            metrics["discrete_ce_loss"] = discrete_ce_loss.item()
            if discrete_z_loss is not None:
                metrics["discrete_z_loss"] = discrete_z_loss.item()

        # Continuous action supervision.
        elif self.config.action_mode == "continuous":
            flow_loss, _ = self._continuous_loss(batch=batch, model_inputs=model_inputs)
            losses.append(flow_loss)
            metrics["action_flow_loss"] = flow_loss.item()

        # Joint continuous + discrete action supervision.
        else:
            # Keep tokenizer loading on the training-loss path only.
            labels = model_inputs.get("labels")
            if labels is None:
                msg = "MolmoAct2 joint training requires labels for the discrete loss."
                raise RuntimeError(msg)

            flow_loss, hidden_states = self._continuous_loss(batch=batch, model_inputs=model_inputs)

            if hidden_states is None:
                outputs = self._inner_model(
                    **model_inputs,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                hidden_states = outputs.last_hidden_state

            discrete_ce_loss, discrete_z_loss = self._discrete_loss(
                labels=labels,
                hidden_states=hidden_states,
                softmax_auxiliary_loss=bool(getattr(self.config, "softmax_auxiliary_loss", False)),
                softmax_auxiliary_loss_scale=float(getattr(self.config, "softmax_auxiliary_loss_scale", 0.0)),
            )
            discrete_loss = discrete_ce_loss if discrete_z_loss is None else discrete_ce_loss + discrete_z_loss

            losses.append(discrete_loss)
            metrics["discrete_ce_loss"] = discrete_ce_loss.item()
            if discrete_z_loss is not None:
                metrics["discrete_z_loss"] = discrete_z_loss.item()

            losses.append(flow_loss)
            metrics["action_flow_loss"] = flow_loss.item()

        loss = torch.stack(losses).sum(dim=0)
        metrics["loss"] = loss.item()
        return loss, metrics

    @torch.no_grad()
    def compute_val_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        """Compute validation MSE between predicted and ground-truth actions."""
        gt_actions = batch.get(ACTION)
        if gt_actions is None:
            gt_actions = batch.get("actions")
        if gt_actions is None:
            msg = "MolmoAct2 validation requires ground-truth actions in the batch."
            raise ValueError(msg)

        predicted = self.predict_action_chunk(batch)["actions"]

        min_horizon = min(int(gt_actions.shape[1]), int(predicted.shape[1]))
        min_action_dim = min(int(gt_actions.shape[2]), int(predicted.shape[2]))
        gt_trimmed = gt_actions[:, :min_horizon, :min_action_dim].to(device=predicted.device, dtype=predicted.dtype)
        pred_trimmed = predicted[:, :min_horizon, :min_action_dim]

        loss = nn_functional.mse_loss(pred_trimmed, gt_trimmed)
        return loss, {"loss": float(loss.detach().float().item())}

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
        include_labels = bool(self.training)
        return _build_model_inputs_from_batch(
            batch=batch,
            config=self.config,
            image_processor=self.image_processor,
            video_processor=self.video_processor,
            include_labels=include_labels,
        )

    @property
    def exported_torch_module(self) -> torch.nn.Module:
        """Torch module used for export-time inference graph tracing."""
        return self._inner_model

    def predict_action_chunk(
        self,
        batch: dict[str, Any],
        predict_action_mode: str | None = None,
    ) -> dict[str, Tensor]:
        """Generate an action chunk directly from a preprocessed inference batch."""
        model_inputs = self.prepare_graph_inputs(batch)
        inference_action_mode = self._resolve_predict_action_mode(predict_action_mode)

        if inference_action_mode == "discrete":
            actions = self._generate_discrete_actions_from_inputs(
                model_inputs=model_inputs,
                action_dim=self._resolved_action_dim(batch),
                action_horizon=int(self.config.n_action_steps),
            )
        else:
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
