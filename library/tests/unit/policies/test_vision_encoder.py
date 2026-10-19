# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the generic vision encoder."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torchvision
from PIL import Image
from torch import nn

from physicalai.policies.components import VisionEncoder

RESNET_LAYERS = ["layer1", "layer2", "layer3", "layer4"]


@pytest.fixture
def resnet() -> nn.Module:
    """Randomly initialised torchvision ResNet18."""
    return torchvision.models.resnet18(weights=None)


@pytest.fixture
def vit() -> nn.Module:
    """Randomly initialised timm ViT-Tiny (patch 16, 192 dims, 1 CLS token)."""
    import timm  # noqa: PLC0415

    return timm.create_model("vit_tiny_patch16_224", pretrained=False)


class TestConvolutional:
    """Tests for convolutional backbones."""

    def test_multi_layer_avg_concat(self, resnet: nn.Module) -> None:
        """Test all four stages avg pooled and concatenated into one vector."""
        encoder = VisionEncoder(resnet, layers=RESNET_LAYERS, pooling="avg", aggregate="concat")
        assert encoder(torch.rand(2, 3, 64, 64)).shape == (2, 64 + 128 + 256 + 512)

    def test_multi_layer_dict(self, resnet: nn.Module) -> None:
        """Test that no aggregation returns feature maps keyed by layer name."""
        encoder = VisionEncoder(resnet, layers=RESNET_LAYERS)
        features = encoder(torch.rand(2, 3, 64, 64))
        assert list(features) == RESNET_LAYERS
        assert features["layer4"].shape == (2, 512, 2, 2)

    def test_feature_map_concat_resizes(self, resnet: nn.Module) -> None:
        """Test feature maps are resized to the largest spatial size before concatenating."""
        encoder = VisionEncoder(resnet, layers=["layer1", "layer2"], aggregate="concat")
        assert encoder(torch.rand(1, 3, 64, 64)).shape == (1, 64 + 128, 16, 16)

    def test_cls_pooling_raises(self, resnet: nn.Module) -> None:
        """Test CLS pooling is rejected for feature maps."""
        encoder = VisionEncoder(resnet, layers=["layer4"], pooling="cls")
        with pytest.raises(ValueError, match="CLS pooling"):
            encoder(torch.rand(1, 3, 64, 64))

    def test_unknown_layer_raises(self, resnet: nn.Module) -> None:
        """Test an unknown layer name suggests close matches."""
        with pytest.raises(ValueError, match="layer4"):
            VisionEncoder(resnet, layers=["layer5"])


class TestTransformer:
    """Tests for transformer backbones."""

    @pytest.mark.parametrize(
        ("pooling", "shape"),
        [("cls", (2, 192)), ("avg", (2, 192)), ("none", (2, 196, 192))],
    )
    def test_pooling(self, vit: nn.Module, pooling: str, shape: tuple[int, ...]) -> None:
        """Test CLS, averaged and patch token outputs."""
        encoder = VisionEncoder(vit, pooling=pooling)
        assert encoder(torch.rand(2, 3, 224, 224)).shape == shape

    def test_timm_name(self) -> None:
        """Test creating the backbone from a timm model name."""
        encoder = VisionEncoder(
            "vit_tiny_patch16_224", pretrained=False, layers=["blocks.5", "blocks.11"], pooling="cls", aggregate="mean"
        )
        assert encoder(torch.rand(1, 3, 224, 224)).shape == (1, 192)
        assert encoder.feature_dims == {"blocks.5": 192, "blocks.11": 192}


class TestModule:
    """Tests for shared module behaviour."""

    @pytest.mark.parametrize("pooling", ["none", "avg", "max"])
    def test_feature_dims_uses_configured_input_size(self, pooling: str) -> None:
        """Test fixed-size vector backbones preserve outputs and feature dimensions."""
        backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 48, 4))
        encoder = VisionEncoder(
            backbone, input_size=(3, 32, 48), pooling=pooling, normalize=False, num_prefix_tokens=1
        )
        images = torch.rand(2, 3, 32, 48)

        torch.testing.assert_close(encoder(images), backbone(images))
        assert encoder.feature_dims == {"output": 4}

    def test_cls_pooling_rejects_vectors(self) -> None:
        """Test already-pooled outputs cannot provide a CLS token."""
        backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 48, 4))
        encoder = VisionEncoder(backbone, input_size=(3, 32, 48), pooling="cls")

        with pytest.raises(ValueError, match="already-pooled vectors"):
            encoder(torch.rand(2, 3, 32, 48))
        with pytest.raises(ValueError, match="already-pooled vectors"):
            _ = encoder.feature_dims

    def test_feature_dims_without_parameters(self) -> None:
        """Test parameter-less backbones can infer feature dimensions."""
        encoder = VisionEncoder(nn.Identity())

        assert encoder.feature_dims == {"output": 3}

    def test_feature_dims_preserves_training_modes_and_buffers(self, resnet: nn.Module) -> None:
        """Test small-input probing leaves BatchNorm statistics and mixed modes unchanged."""
        encoder = VisionEncoder(resnet, layers=["layer4"], frozen=False, input_size=(3, 32, 32))
        resnet.layer1.eval()
        training_modes = [module.training for module in resnet.modules()]
        buffers = {name: buffer.clone() for name, buffer in resnet.named_buffers()}

        assert encoder.feature_dims == {"layer4": 512}

        assert [module.training for module in resnet.modules()] == training_modes
        for name, buffer in resnet.named_buffers():
            torch.testing.assert_close(buffer, buffers[name])

    def test_feature_dims_restores_training_modes_on_error(self) -> None:
        """Test a failing dummy forward restores every submodule's original mode."""
        backbone = nn.Sequential(nn.BatchNorm2d(3), nn.Flatten(), nn.Linear(1, 4))
        encoder = VisionEncoder(backbone, frozen=False, input_size=(3, 32, 32))
        backbone[1].eval()
        training_modes = [module.training for module in backbone.modules()]
        buffers = {name: buffer.clone() for name, buffer in backbone.named_buffers()}

        with pytest.raises(RuntimeError, match="mat1 and mat2"):
            _ = encoder.feature_dims

        assert [module.training for module in backbone.modules()] == training_modes
        for name, buffer in backbone.named_buffers():
            torch.testing.assert_close(buffer, buffers[name])
        assert encoder._feature_dims is None

    @pytest.mark.parametrize("buffer_dtype", [None, torch.float64])
    def test_encode_image_without_parameters(self, buffer_dtype: torch.dtype | None) -> None:
        """Test parameter-less backbones use a buffer or CPU/float32 defaults."""
        backbone = nn.Identity()
        if buffer_dtype is not None:
            backbone.register_buffer("reference", torch.zeros(1, dtype=buffer_dtype))
        encoder = VisionEncoder(backbone, normalize=False)
        image = torch.rand(3, 32, 32)

        features = encoder.encode_image(image)

        expected = image.unsqueeze(0).to(dtype=buffer_dtype if buffer_dtype is not None else torch.float32)
        torch.testing.assert_close(features, expected)

    def test_encode_image_formats(self, resnet: nn.Module) -> None:
        """Test PIL images, HWC arrays and CHW tensors give the same features."""
        encoder = VisionEncoder(resnet, layers=["layer4"], pooling="avg")
        array = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)

        from_pil = encoder.encode_image(Image.fromarray(array))
        from_numpy = encoder.encode_image(array)
        from_tensor = encoder.encode_image(torch.from_numpy(array).permute(2, 0, 1))

        assert from_pil.shape == (1, 512)
        torch.testing.assert_close(from_pil, from_numpy)
        torch.testing.assert_close(from_pil, from_tensor)

    def test_encode_image_batch_and_resize(self, resnet: nn.Module) -> None:
        """Test a list of differently sized images is resized and batched."""
        encoder = VisionEncoder(resnet, layers=["layer4"], pooling="avg")
        images = [torch.rand(3, 32, 32), torch.rand(3, 48, 40)]
        assert encoder.encode_image(images, size=64).shape == (2, 512)
        assert encoder.encode_image(torch.rand(32, 48, 3), channels_last=True).shape == (1, 512)

    def test_frozen(self, resnet: nn.Module) -> None:
        """Test a frozen backbone has no trainable parameters and stays in eval mode."""
        encoder = VisionEncoder(resnet, frozen=True).train()
        assert not any(p.requires_grad for p in encoder.parameters())
        assert not encoder.model.training
