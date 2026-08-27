# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for MolmoAct2 preprocessing and postprocessing."""

import pytest
import torch

from physicalai.data.observation import ACTION, IMAGES, STATE, TASK, Feature, FeatureType, NormalizationParameters
from physicalai.policies.molmoact2 import MolmoAct2Config
from physicalai.policies.molmoact2.processors import (
    MolmoAct2Postprocessor,
    MolmoAct2Preprocessor,
    make_molmoact2_preprocessors,
)
from physicalai.policies.molmoact2.processors.image import MolmoAct2ImageProcessor
from physicalai.policies.molmoact2.processors.normalization import MolmoAct2NormalizeTransform
from physicalai.policies.molmoact2.processors.preprocess_steps import (
    ActionPadder,
    ImagePacker,
    PreprocessBatchBundle,
    RobotPromptEncoder,
    StateTaskImageExtractor,
)


def test_factory_builds_matched_processors(tiny_molmoact2_config: MolmoAct2Config) -> None:
    preprocessor, postprocessor = make_molmoact2_preprocessors(tiny_molmoact2_config)

    assert isinstance(preprocessor, MolmoAct2Preprocessor)
    assert isinstance(postprocessor, MolmoAct2Postprocessor)


def test_factory_requires_resolved_features(tiny_molmoact2_config: MolmoAct2Config) -> None:
    tiny_molmoact2_config.input_features = None

    with pytest.raises(ValueError, match="features must be set"):
        make_molmoact2_preprocessors(tiny_molmoact2_config)


def test_normalization_round_trip() -> None:
    feature = Feature(
        name=ACTION,
        ftype=FeatureType.ACTION,
        shape=(2,),
        normalization_data=NormalizationParameters(q01=[0.0, -2.0], q99=[2.0, 2.0]),
    )
    normalizer = MolmoAct2NormalizeTransform(input_features=[], output_features=[feature])
    denormalizer = MolmoAct2NormalizeTransform(input_features=[], output_features=[feature], inverse=True)
    action = torch.tensor([[[0.5, 1.0]]])

    normalized = normalizer({ACTION: action})[ACTION]
    restored = denormalizer({ACTION: normalized})[ACTION]

    torch.testing.assert_close(restored, action)


def test_extractor_accepts_flattened_observations() -> None:
    extractor = StateTaskImageExtractor(image_keys=["front"])
    image = torch.zeros(2, 3, 8, 8)

    bundle = extractor.extract({STATE: torch.zeros(2, 4), TASK: "Pick block.", f"{IMAGES}.front": image})

    assert bundle.tasks == ["pick block", "pick block"]
    assert len(bundle.images_by_example) == 2
    assert bundle.images_by_example[0][0].shape == (3, 8, 8)


def test_prompt_encoder_includes_state_and_image_tokens() -> None:
    encoder = RobotPromptEncoder(
        num_state_tokens=16,
        setup_type="tabletop",
        control_mode="joint",
        add_setup_tokens=True,
        add_control_tokens=True,
    )
    bundle = PreprocessBatchBundle(
        state=torch.zeros(1, 2),
        tasks=["pick block"],
        images_by_example=[[torch.zeros(3, 8, 8)]],
    )

    prompt = encoder.encode(bundle).prompt_texts[0]

    assert "pick block" in prompt
    assert "<|image|>" in prompt
    assert "<state_start>" in prompt


def test_image_processing_and_packing_shapes() -> None:
    packer = ImagePacker(image_size=(28, 28))
    packed, mask = packer([[torch.zeros(3, 28, 28)], [torch.ones(3, 28, 28)]])
    processor = MolmoAct2ImageProcessor(
        crop_mode="resize",
        size={"height": 28, "width": 28},
        patch_size=14,
        pooling_size=[2, 2],
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    )
    processed = processor(packed[0])

    assert packed.shape == (1, 2, 3, 28, 28)
    assert mask.tolist() == [[True, True]]
    assert processed["pixel_values"].shape == (2, 4, 14 * 14 * 3)


def test_action_padder_returns_values_and_masks() -> None:
    padded, horizon_mask, dim_mask = ActionPadder(max_action_dim=4)(
        torch.tensor([[[2.0, -2.0]]]),
    )

    torch.testing.assert_close(padded, torch.tensor([[[1.0, -1.0, 0.0, 0.0]]]))
    assert horizon_mask.tolist() == [[False]]
    assert dim_mask.tolist() == [[False, False, True, True]]


def test_postprocessor_clamps_and_denormalizes() -> None:
    feature = Feature(
        name=ACTION,
        ftype=FeatureType.ACTION,
        shape=(1,),
        normalization_data=NormalizationParameters(q01=[0.0], q99=[2.0]),
    )
    postprocessor = MolmoAct2Postprocessor(output_features=[feature])

    result = postprocessor({ACTION: torch.tensor([[[-2.0], [2.0]]])})[ACTION]

    torch.testing.assert_close(result, torch.tensor([[[0.0], [2.0]]]))


def test_invalid_processor_inputs_raise() -> None:
    with pytest.raises(ValueError, match="state tensor"):
        StateTaskImageExtractor(image_keys=[]).extract({TASK: "task"})
    with pytest.raises(ValueError, match="action tensor"):
        MolmoAct2Postprocessor(output_features=[])({})