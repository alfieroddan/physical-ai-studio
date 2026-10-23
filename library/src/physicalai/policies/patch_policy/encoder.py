# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pretrained visual encoders used by Patch Policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from types import ModuleType

    from .config import EncoderOutput, FeatureEncoderName

_TIMM_MODELS: dict[FeatureEncoderName, str] = {
    "dinov2": "vit_small_patch14_dinov2.lvd142m",
}
_WEBSSL_REPO = "facebook/webssl-dino300m-full2b-224"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _import_timm() -> ModuleType:
    """Import the optional timm dependency.

    Returns:
        The imported timm module.

    Raises:
        ImportError: If the Patch Policy optional dependencies are not installed.
    """
    try:
        import timm  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("Patch Policy requires timm. Install physicalai-train[patch_policy].") from exc
    return timm


class TimmFeatureEncoder(nn.Module):
    """Encode arbitrary leading image dimensions into patch tokens.

    Args:
        name: Supported timm encoder name.
        output: Encoder representation to return for each image.
        pretrained: Whether to load pretrained backbone weights.
        frozen: Whether to freeze the backbone in evaluation mode.
        image_size: Input image height and width.
    """

    mean: Tensor
    std: Tensor
    backbone: nn.Module

    def __init__(
        self,
        name: FeatureEncoderName,
        output: EncoderOutput,
        *,
        pretrained: bool = True,
        frozen: bool = True,
        image_size: tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()
        self.output = output
        self.frozen = frozen
        self.name = name
        if name == "dinov2":
            backbone = _import_timm().create_model(
                _TIMM_MODELS[name],
                pretrained=pretrained,
                num_classes=0,
                img_size=image_size,
            )
            self.backbone = backbone
            self.output_dim = int(backbone.num_features)
            patch_embed = getattr(backbone, "patch_embed", None)
            self.num_tokens = int(getattr(patch_embed, "num_patches", 1)) if output == "patches" else 1
        elif name == "webssl":
            try:
                from transformers import Dinov2Config, Dinov2Model  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError("WebSSL requires transformers. Install physicalai-train[patch_policy].") from exc

            config = Dinov2Config.from_pretrained(_WEBSSL_REPO)
            backbone = Dinov2Model.from_pretrained(_WEBSSL_REPO) if pretrained else Dinov2Model(config)
            self.backbone = backbone
            self.output_dim = int(config.hidden_size)
            patch_size = config.patch_size
            if isinstance(patch_size, int):
                patch_size = (patch_size, patch_size)
            self.num_tokens = (
                (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1]) if output == "patches" else 1
            )
        else:
            try:
                from torchvision.models import ResNet18_Weights, resnet18  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError("ResNet18 requires torchvision. Install physicalai-train[patch_policy].") from exc

            model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            self.backbone = nn.Sequential(*list(model.children())[:-1])
            self.output_dim = int(model.fc.in_features)
            self.num_tokens = 1

        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN)[None, :, None, None])
        self.register_buffer("std", torch.tensor(_IMAGENET_STD)[None, :, None, None])
        if frozen:
            self.requires_grad_(requires_grad=False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> TimmFeatureEncoder:  # noqa: FBT001, FBT002
        """Set training mode while keeping a frozen backbone in evaluation mode.

        Args:
            mode: Whether to enable training mode for trainable modules.

        Returns:
            This encoder.
        """
        super().train(mode=mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def _tokens(self, images: Tensor) -> Tensor:
        """Extract the configured token representation from flat images.

        Args:
            images: Image tensor with shape ``[N, C, H, W]``.

        Returns:
            Encoded image tokens with shape ``[N, P, E]``.

        Raises:
            TypeError: If the backbone returns an unsupported value.
            ValueError: If the returned feature tensor has an unsupported rank.
        """
        if self.name == "webssl":
            features = self.backbone(
                pixel_values=images,
                interpolate_pos_encoding=images.shape[-2:] != (224, 224),
            ).last_hidden_state
            if self.output == "cls":
                return features[:, :1]
            patches = features[:, 1:]
            return patches if self.output == "patches" else patches.mean(dim=1, keepdim=True)

        if isinstance(self.backbone, nn.Sequential):
            features = self.backbone(images)
        else:
            forward_features = getattr(self.backbone, "forward_features", None)
            if not callable(forward_features):
                raise TypeError(f"Encoder {self.name!r} does not expose forward_features().")
            features = forward_features(images)
        if isinstance(features, dict):
            feature_key = "x_norm_clstoken" if self.output == "cls" else "x_norm_patchtokens"
            features = features.get(feature_key, features.get("last_hidden_state"))
        if not isinstance(features, Tensor):
            raise TypeError("timm forward_features must return a tensor or a supported tensor dictionary.")
        if features.ndim == 4:
            features = features.flatten(2).transpose(1, 2)
        elif features.ndim == 2:
            features = features[:, None]
        elif features.ndim != 3:
            raise ValueError(f"Unsupported timm feature shape {tuple(features.shape)}.")

        prefix = int(getattr(self.backbone, "num_prefix_tokens", 0))
        patches = features[:, prefix:] if prefix and features.shape[1] > prefix else features
        if self.output == "patches":
            return patches
        if self.output == "cls" and prefix:
            return features[:, :1]
        return patches.mean(dim=1, keepdim=True)

    def forward(self, images: Tensor) -> Tensor:
        """Encode images while preserving their leading dimensions.

        Args:
            images: Images with shape ``[..., C, H, W]``.

        Returns:
            Tokens with shape ``[..., patches, embedding]``.

        Raises:
            ValueError: If the input does not contain channel and spatial dimensions.
        """
        if images.ndim < 4:
            raise ValueError("Expected images with shape [..., C, H, W].")
        leading = images.shape[:-3]
        flat = images.reshape(-1, *images.shape[-3:])
        tokens = self._tokens(flat)
        return tokens.reshape(*leading, *tokens.shape[-2:])
