# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MolmoAct2 preprocessing components."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from physicalai.data.constants import EXTRA
from physicalai.data.observation import ACTION, IMAGES, STATE, TASK, FeatureType
from physicalai.policies.molmoact2.config import (
    MOLMOACT2_IMAGE_PLACEHOLDER_TOKEN_ID,
    MolmoAct2Config,
)
from physicalai.policies.molmoact2.processors.factory import make_molmoact2_preprocessors
from physicalai.policies.molmoact2.processors.image import MolmoAct2ImageProcessor
from physicalai.policies.molmoact2.processors.inputs import (
    _expand_image_placeholders,
    _image_token_ids,
    _image_token_ids_for_grid,
    build_model_inputs,
)
from physicalai.policies.molmoact2.processors.joint_transform import JointFrameTransform
from physicalai.policies.molmoact2.processors.postprocessor import MolmoAct2Postprocessor
from physicalai.policies.molmoact2.processors.preprocess_steps import (
    ActionExtractor,
    ActionPadder,
    ImagePacker,
    ImageResizeNormalizer,
    MolmoAct2NormalizeTransform,
    RobotPromptEncoder,
    StateTaskImageExtractor,
)
from physicalai.policies.molmoact2.processors.preprocessor import MolmoAct2Preprocessor
from physicalai.policies.molmoact2.processors.utils import (
    build_discrete_state_string,
    build_robot_text,
    normalize_text,
)


class TestNormalizeText:
    def test_strips_and_lowercases(self) -> None:
        assert normalize_text("  Pick Up The Block  ") == "pick up the block"

    def test_strips_task_prefix(self) -> None:
        assert normalize_text("Task: pick up the block") == "pick up the block"

    def test_strips_instruction_prefix(self) -> None:
        assert normalize_text("the task is to pick up") == "pick up"

    def test_empty_returns_empty(self) -> None:
        assert normalize_text("") == ""
        assert normalize_text(None) == ""

    def test_strips_trailing_punctuation(self) -> None:
        assert normalize_text("pick up.") == "pick up"


class TestBuildDiscreteStateString:
    def test_wraps_in_state_tokens(self) -> None:
        state = torch.tensor([-1.0, -1.0])
        text = build_discrete_state_string(state, num_state_tokens=10)
        assert text.startswith("<state_start>")
        assert text.endswith("<state_end>")
        assert "<state_0>" in text

    def test_clamps_and_rounds(self) -> None:
        state = torch.tensor([-2.0, 2.0])
        text = build_discrete_state_string(state, num_state_tokens=3)
        assert "<state_0>" in text
        assert "<state_2>" in text

    def test_raises_on_non_positive_tokens(self) -> None:
        with pytest.raises(ValueError, match="num_state_tokens"):
            build_discrete_state_string(torch.zeros(2), num_state_tokens=0)


class TestBuildRobotText:
    def test_single_image_prefix(self) -> None:
        text = build_robot_text(
            task="pick up",
            discrete_state_string="",
            setup_type="",
            control_mode="",
            add_setup_tokens=False,
            add_control_tokens=False,
            num_images=1,
        )
        assert text.startswith("<|image|>")

    def test_multi_image_prefix(self) -> None:
        text = build_robot_text(
            task="task",
            discrete_state_string="",
            setup_type="",
            control_mode="",
            add_setup_tokens=False,
            add_control_tokens=False,
            num_images=2,
        )
        assert "Image 1<|image|>" in text
        assert "Image 2<|image|>" in text

    def test_no_image_prefix(self) -> None:
        text = build_robot_text(
            task="task",
            discrete_state_string="",
            setup_type="",
            control_mode="",
            add_setup_tokens=False,
            add_control_tokens=False,
            num_images=0,
        )
        assert "<|image|>" not in text

    def test_setup_tokens_wrapped(self) -> None:
        text = build_robot_text(
            task="task",
            discrete_state_string="",
            setup_type="tabletop",
            control_mode="",
            add_setup_tokens=True,
            add_control_tokens=False,
            num_images=0,
        )
        assert "<setup_start>tabletop<setup_end>" in text

    def test_control_tokens_wrapped(self) -> None:
        text = build_robot_text(
            task="task",
            discrete_state_string="",
            setup_type="",
            control_mode="joint",
            add_setup_tokens=False,
            add_control_tokens=True,
            num_images=0,
        )
        assert "<control_start>joint<control_end>" in text

    def test_state_clause_included(self) -> None:
        text = build_robot_text(
            task="task",
            discrete_state_string="<state_start><state_0><state_end>",
            setup_type="",
            control_mode="",
            add_setup_tokens=False,
            add_control_tokens=False,
            num_images=0,
        )
        assert "current state of the robot" in text


class TestJointFrameTransform:
    def test_to_checkpoint_applies_sign_offset(self) -> None:
        transform = JointFrameTransform(joint_signs=[1.0, -1.0], joint_offsets=[0.0, 90.0])
        values = torch.tensor([[1.0, 2.0, 3.0]])
        out = transform.to_checkpoint(values)
        assert float(out[0, 0]) == 1.0
        assert float(out[0, 1]) == pytest.approx(-2.0 + 90.0)
        assert float(out[0, 2]) == 3.0

    def test_to_robot_inverts(self) -> None:
        transform = JointFrameTransform(joint_signs=[1.0, -1.0], joint_offsets=[0.0, 90.0])
        values = torch.tensor([[1.0, 88.0]])
        to_ckpt = transform.to_checkpoint(values)
        back = transform.to_robot(to_ckpt)
        torch.testing.assert_close(back, values)

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="joint_signs"):
            JointFrameTransform(joint_signs=[1.0], joint_offsets=[0.0, 1.0])


class TestActionPadder:
    def test_pads_last_axis(self) -> None:
        padder = ActionPadder(max_action_dim=8)
        action = torch.randn(2, 3, 4)
        padded, horizon_pad, dim_pad = padder(action)
        assert padded.shape == (2, 3, 8)
        assert padded[..., :4].allclose(action.clamp(-1.0, 1.0))
        assert torch.equal(dim_pad[..., :4], torch.zeros(2, 4, dtype=torch.bool))
        assert torch.equal(dim_pad[..., 4:], torch.ones(2, 4, dtype=torch.bool))

    def test_promotes_2d_to_3d(self) -> None:
        padder = ActionPadder(max_action_dim=4)
        action = torch.randn(2, 2)
        padded, _, _ = padder(action)
        assert padded.shape == (2, 1, 4)

    def test_preserves_horizon_padding_mask(self) -> None:
        padder = ActionPadder(max_action_dim=4)
        action = torch.randn(2, 3, 2)
        expected = torch.tensor([[False, False, True], [False, True, True]])
        _, horizon_pad, _ = padder(action, expected)
        assert torch.equal(horizon_pad, expected)

    def test_rejects_mismatched_horizon_padding_mask(self) -> None:
        padder = ActionPadder(max_action_dim=4)
        with pytest.raises(ValueError, match="action_horizon_is_pad must match"):
            padder(torch.randn(2, 3, 2), torch.zeros(2, 2, dtype=torch.bool))

    def test_preprocessor_reads_canonical_lerobot_padding_key(self) -> None:
        preprocessor = object.__new__(MolmoAct2Preprocessor)
        torch.nn.Module.__init__(preprocessor)
        preprocessor._action_extractor = ActionExtractor()
        preprocessor._action_padder = ActionPadder(max_action_dim=4)
        expected = torch.tensor([[False, False, True]])

        result = MolmoAct2Preprocessor._preprocess_action(
            preprocessor,
            {ACTION: torch.zeros(1, 3, 2), f"{EXTRA}.action_is_pad": expected},
        )

        assert torch.equal(result["action_horizon_is_pad"], expected)

    def test_preprocessor_does_not_read_typoed_padding_key(self) -> None:
        preprocessor = object.__new__(MolmoAct2Preprocessor)
        torch.nn.Module.__init__(preprocessor)
        preprocessor._action_extractor = ActionExtractor()
        preprocessor._action_padder = ActionPadder(max_action_dim=4)

        result = MolmoAct2Preprocessor._preprocess_action(
            preprocessor,
            {ACTION: torch.zeros(1, 3, 2), f"{EXTRA}.actions_id_pad": torch.ones(1, 3, dtype=torch.bool)},
        )

        assert not result["action_horizon_is_pad"].any()

    def test_raises_when_dim_exceeds_max(self) -> None:
        padder = ActionPadder(max_action_dim=2)
        with pytest.raises(ValueError, match="exceeds max_action_dim"):
            padder(torch.randn(1, 1, 4))

    def test_raises_on_wrong_ndim(self) -> None:
        padder = ActionPadder(max_action_dim=2)
        with pytest.raises(ValueError, match="action shape"):
            padder(torch.randn(1, 1, 1, 1))


class TestImageResizeNormalizer:
    def test_resize_normalize_uint8(self) -> None:
        normalizer = ImageResizeNormalizer(image_size=(14, 14))
        img = torch.randint(0, 256, (3, 28, 28), dtype=torch.uint8)
        out = normalizer([[img], [img]])
        assert out[0][0].shape == (3, 14, 14)
        assert out[0][0].dtype == torch.float32
        assert float(out[0][0].max()) <= 1.0
        assert float(out[0][0].min()) >= 0.0

    def test_resize_normalize_float01(self) -> None:
        normalizer = ImageResizeNormalizer(image_size=(14, 14))
        img = torch.rand(3, 28, 28)
        out = normalizer([[img]])
        assert out[0][0].shape == (3, 14, 14)
        assert out[0][0].dtype == torch.float32
        assert float(out[0][0].max()) <= 1.0

    def test_resize_normalize_float255(self) -> None:
        normalizer = ImageResizeNormalizer(image_size=(14, 14))
        img = torch.rand(3, 28, 28) * 255.0
        out = normalizer([[img]])
        assert out[0][0].shape == (3, 14, 14)
        assert out[0][0].dtype == torch.float32
        assert float(out[0][0].max()) <= 1.0 + 1e-5

    def test_empty_input_returns_empty(self) -> None:
        normalizer = ImageResizeNormalizer(image_size=(14, 14))
        assert normalizer([]) == []

    def test_preserves_nested_structure(self) -> None:
        normalizer = ImageResizeNormalizer(image_size=(14, 14))
        img1 = torch.randint(0, 256, (3, 28, 28), dtype=torch.uint8)
        img2 = torch.randint(0, 256, (3, 28, 28), dtype=torch.uint8)
        out = normalizer([[img1, img2], [img1, img2]])
        assert len(out) == 2
        assert len(out[0]) == 2
        assert len(out[1]) == 2


class TestImagePacker:
    def test_pack_shapes(self) -> None:
        packer = ImagePacker(image_size=(14, 14))
        img = torch.zeros(3, 14, 14, dtype=torch.float32)
        images = [[img], [img]]
        out_images, out_masks = packer(images)
        assert out_images.shape == (1, 2, 3, 14, 14)
        assert out_masks.shape == (1, 2)
        assert out_masks.dtype == torch.bool

    def test_empty_batch(self) -> None:
        packer = ImagePacker(image_size=(14, 14))
        out_images, out_masks = packer([])
        assert out_images.shape == (0, 0, 3, 14, 14)
        assert out_masks.shape == (0, 0)

    def test_inconsistent_image_count_raises(self) -> None:
        packer = ImagePacker(image_size=(14, 14))
        img = torch.zeros(3, 14, 14, dtype=torch.float32)
        images = [[img], [img, img]]
        with pytest.raises(ValueError, match="consistent number of images"):
            packer(images)


class TestStateTaskImageExtractor:
    def test_extracts_state_task_images(self) -> None:
        extractor = StateTaskImageExtractor(image_keys=["image"])
        batch = {
            STATE: torch.zeros(2, 4),
            TASK: ["do thing", "do other"],
            f"{IMAGES}.image": torch.zeros(2, 3, 14, 14),
        }
        bundle = extractor.extract(batch)
        assert bundle.state.shape == (2, 4)
        assert len(bundle.tasks) == 2
        assert len(bundle.images_by_example) == 2
        assert len(bundle.images_by_example[0]) == 1

    def test_missing_state_raises(self) -> None:
        extractor = StateTaskImageExtractor(image_keys=[])
        with pytest.raises(ValueError, match="state tensor"):
            extractor.extract({TASK: "task", IMAGES: {}})

    def test_missing_images_raises(self) -> None:
        extractor = StateTaskImageExtractor(image_keys=["missing"])
        with pytest.raises(ValueError, match="image tensors"):
            extractor.extract({STATE: torch.zeros(1, 2)})

    def test_task_broadcast(self) -> None:
        extractor = StateTaskImageExtractor(image_keys=[])
        batch = {
            STATE: torch.zeros(3, 2),
            TASK: "shared task",
            f"{IMAGES}.x": torch.zeros(3, 3, 14, 14),
        }
        bundle = extractor.extract(batch)
        assert bundle.tasks == ["shared task", "shared task", "shared task"]


class TestRobotPromptEncoder:
    def test_encode_builds_prompt_per_example(self) -> None:
        encoder = RobotPromptEncoder(
            num_state_tokens=8,
            setup_type="tabletop",
            control_mode="joint",
            add_setup_tokens=True,
            add_control_tokens=True,
        )
        bundle = StateTaskImageExtractor(image_keys=["cam"]).extract(
            {
                STATE: torch.zeros(2, 4),
                TASK: "pick",
                f"{IMAGES}.cam": torch.zeros(2, 3, 14, 14),
            }
        )
        pack = encoder.encode(bundle)
        assert len(pack.prompt_texts) == 2
        assert all("<state_start>" in t for t in pack.prompt_texts)
        assert all("<control_start>joint<control_end>" in t for t in pack.prompt_texts)


class TestMolmoAct2NormalizeTransform:
    def test_normalizes_state_bounds(
        self, molmoact2_features: tuple[list, list]
    ) -> None:
        inputs, outputs = molmoact2_features
        normalizer = MolmoAct2NormalizeTransform(input_features=inputs, output_features=outputs)
        batch = {STATE: torch.full((1, 6), 2.0)}
        out = normalizer(batch)
        assert out[STATE].shape == (1, 6)

    def test_identity_without_norm_stats(self) -> None:
        from physicalai.data.observation import Feature, FeatureType

        inputs = [Feature(name="state", ftype=FeatureType.STATE, shape=(4,))]
        outputs = [Feature(name="action", ftype=FeatureType.ACTION, shape=(4,))]
        normalizer = MolmoAct2NormalizeTransform(input_features=inputs, output_features=outputs)
        batch = {STATE: torch.full((1, 4), 2.0)}
        out = normalizer(batch)
        torch.testing.assert_close(out[STATE], batch[STATE])


class TestMolmoAct2NormalizeMask:
    """Per-dimension mask support in :class:`MolmoAct2NormalizeTransform`."""

    def _masked_features(self) -> tuple[list, list]:
        from physicalai.data.observation import Feature, FeatureType, NormalizationParameters

        # state dims: [0,1,2] masked (normalized), [3,4] pass-through, [5] masked
        state_mask = [True, True, True, False, False, True]
        action_mask = [True, True, False, False, True, True]
        inputs = [
            Feature(
                name="state",
                ftype=FeatureType.STATE,
                shape=(6,),
                normalization_data=NormalizationParameters(
                    mean=[0.0] * 6,
                    std=[1.0] * 6,
                    q01=[0.0] * 6,
                    q99=[2.0] * 6,
                    mask=state_mask,
                ),
            ),
        ]
        outputs = [
            Feature(
                name="action",
                ftype=FeatureType.ACTION,
                shape=(6,),
                normalization_data=NormalizationParameters(
                    mean=[0.0] * 6,
                    std=[1.0] * 6,
                    q01=[0.0] * 6,
                    q99=[2.0] * 6,
                    mask=action_mask,
                ),
            ),
        ]
        return inputs, outputs

    def test_masked_dims_normalized_unmasked_unchanged(self) -> None:
        inputs, outputs = self._masked_features()
        normalizer = MolmoAct2NormalizeTransform(input_features=inputs, output_features=outputs)
        state = torch.full((1, 6), 2.0)
        action = torch.full((1, 6), 0.5)
        batch = {STATE: state.clone(), ACTION: action.clone()}
        out = normalizer(batch)

        state_mask = torch.tensor([True, True, True, False, False, True])
        action_mask = torch.tensor([True, True, False, False, True, True])

        # quantile norm with q01=0, q99=2: x -> 2*(x-0)/2 - 1 = x - 1
        expected_state = torch.where(state_mask, torch.full((6,), 1.0), state.squeeze(0)).unsqueeze(0)
        torch.testing.assert_close(out[STATE], expected_state)
        expected_action = torch.where(action_mask, torch.full((6,), -0.5), action.squeeze(0)).unsqueeze(0)
        torch.testing.assert_close(out[ACTION], expected_action)

    def test_no_mask_when_field_absent(self) -> None:
        from physicalai.data.observation import Feature, FeatureType, NormalizationParameters

        inputs = [
            Feature(
                name="state",
                ftype=FeatureType.STATE,
                shape=(6,),
                normalization_data=NormalizationParameters(
                    q01=[0.0] * 6,
                    q99=[2.0] * 6,
                ),
            ),
        ]
        outputs = [
            Feature(
                name="action",
                ftype=FeatureType.ACTION,
                shape=(6,),
                normalization_data=NormalizationParameters(
                    q01=[0.0] * 6,
                    q99=[2.0] * 6,
                ),
            ),
        ]
        normalizer = MolmoAct2NormalizeTransform(input_features=inputs, output_features=outputs)
        out = normalizer({STATE: torch.full((1, 6), 2.0)})
        torch.testing.assert_close(out[STATE], torch.full((1, 6), 1.0))


class TestMolmoAct2ImageProcessor:
    def test_output_shapes(self) -> None:
        processor = MolmoAct2ImageProcessor(
            crop_mode="resize",
            size={"height": 28, "width": 28},
            patch_size=14,
            pooling_size=[2, 2],
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )
        images = torch.zeros(2, 3, 28, 28)
        out = processor(images)
        assert "pixel_values" in out
        assert "image_token_pooling" in out
        assert "image_grids" in out
        assert "image_num_crops" in out
        assert out["pixel_values"].shape[0] == 2
        assert out["image_grids"].shape == (2, 4)
        assert int(out["image_num_crops"].min()) == 1


class TestImageTokenIdsForGrid:
    def _config(self) -> MolmoAct2Config:
        return MolmoAct2Config(
            image_default_input_size=(28, 28),
            image_patch_size=14,
            image_start_token_id=10,
            low_res_image_start_token_id=10,
            image_end_token_id=12,
            image_patch_id=11,
            image_col_id=13,
        )

    def test_low_res_only_when_no_high_res(self) -> None:
        config = self._config()
        grid = torch.tensor([2, 2, 0, 0])
        ids = _image_token_ids_for_grid(config, grid)
        assert ids[0] == 10
        assert ids[-1] == 12

    def test_concatenates_low_and_high_res(self) -> None:
        config = self._config()
        grid = torch.tensor([2, 2, 3, 3])
        ids = _image_token_ids_for_grid(config, grid)
        assert ids[0] == 10
        assert ids[-1] == 12
        assert ids.count(11) == 2 * 2 + 3 * 3

    def test_missing_image_token_ids_raises(self) -> None:
        config = MolmoAct2Config(image_start_token_id=None, image_end_token_id=12, image_patch_id=11)
        with pytest.raises(ValueError, match="must be configured"):
            _image_token_ids_for_grid(config, torch.tensor([2, 2, 0, 0]))


class TestExpandImagePlaceholders:
    def _config(self) -> MolmoAct2Config:
        config = MolmoAct2Config(
            image_start_token_id=10,
            image_end_token_id=12,
            image_patch_id=11,
            image_col_id=13,
            image_placeholder_token_id=MOLMOACT2_IMAGE_PLACEHOLDER_TOKEN_ID,
        )
        return config

    def test_no_grids_returns_input(self) -> None:
        config = self._config()
        input_ids = torch.tensor([[5, 6, 7]])
        attention_mask = torch.tensor([[1, 1, 1]])
        grids = torch.empty((0, 4))
        out_ids, out_mask, token_types = _expand_image_placeholders(
            config=config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grids=grids,
            image_placeholder_token_id=MOLMOACT2_IMAGE_PLACEHOLDER_TOKEN_ID,
        )
        torch.testing.assert_close(out_ids, input_ids)
        torch.testing.assert_close(out_mask, attention_mask)
        assert token_types is not None

    def test_expands_placeholder_tokens(self) -> None:
        config = self._config()
        placeholder = MOLMOACT2_IMAGE_PLACEHOLDER_TOKEN_ID
        input_ids = torch.tensor([[placeholder, 5]])
        attention_mask = torch.ones_like(input_ids)
        grids = torch.tensor([[2, 2, 0, 0]])
        out_ids, out_mask, _ = _expand_image_placeholders(
            config=config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grids=grids,
            image_placeholder_token_id=placeholder,
        )
        assert out_ids.shape[0] == 1
        assert int(out_ids[0, 0]) == 10
        expanded = out_ids.tolist()[0]
        assert int(expanded[expanded.index(12)]) == 12
        assert int(expanded[-1]) == 5

    def test_preserves_masked_tokenizer_padding(self) -> None:
        config = self._config()
        placeholder = MOLMOACT2_IMAGE_PLACEHOLDER_TOKEN_ID
        input_ids = torch.tensor([[placeholder, 5, 0, 0]])
        attention_mask = torch.tensor([[1, 1, 0, 0]])
        grids = torch.tensor([[2, 2, 0, 0]])

        out_ids, out_mask, _ = _expand_image_placeholders(
            config=config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grids=grids,
            image_placeholder_token_id=placeholder,
        )

        assert out_ids.shape == (1, 9)
        assert out_mask.shape == (1, 9)
        assert int(out_mask.sum()) == 7

    def test_raises_when_too_few_grids(self) -> None:
        config = self._config()
        placeholder = MOLMOACT2_IMAGE_PLACEHOLDER_TOKEN_ID
        input_ids = torch.tensor([[placeholder, placeholder, 5]])
        attention_mask = torch.ones_like(input_ids)
        grids = torch.tensor([[2, 2, 0, 0]])
        with pytest.raises(ValueError, match="Not enough image grids"):
            _expand_image_placeholders(
                config=config,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grids=grids,
                image_placeholder_token_id=placeholder,
            )


class TestMakeMolmoact2Preprocessors:
    def test_uses_configured_tokenizer_padding(
        self,
        tiny_molmoact2_config: MolmoAct2Config,
        molmoact2_features: tuple[list, list],
    ) -> None:
        inputs, outputs = molmoact2_features
        tiny_molmoact2_config.input_features = inputs
        tiny_molmoact2_config.output_features = outputs
        tiny_molmoact2_config.tokenizer_padding = "longest"

        preprocessor, _ = make_molmoact2_preprocessors(tiny_molmoact2_config)

        assert preprocessor._tokenizers.padding == "longest"

    def test_returns_pre_and_post(
        self,
        tiny_molmoact2_config: MolmoAct2Config,
        molmoact2_features: tuple[list, list],
    ) -> None:
        inputs, outputs = molmoact2_features
        config = tiny_molmoact2_config
        config.input_features = inputs
        config.output_features = outputs
        preprocessor, postprocessor = make_molmoact2_preprocessors(config)
        assert isinstance(preprocessor, MolmoAct2Preprocessor)
        assert isinstance(postprocessor, MolmoAct2Postprocessor)

    def test_raises_when_features_none(
        self, tiny_molmoact2_config: MolmoAct2Config
    ) -> None:
        tiny_molmoact2_config.input_features = None
        tiny_molmoact2_config.output_features = None
        with pytest.raises(ValueError, match="Input and output features must be set"):
            make_molmoact2_preprocessors(tiny_molmoact2_config)


class TestPostprocessor:
    def test_identity_without_features(self) -> None:
        postprocessor = MolmoAct2Postprocessor(output_features=[])
        action = torch.zeros(2, 3, 4)
        result = postprocessor({ACTION: action})
        assert result[ACTION].shape == (2, 3, 4)

    def test_missing_action_raises(self) -> None:
        postprocessor = MolmoAct2Postprocessor(output_features=[])
        with pytest.raises(ValueError, match="action tensor"):
            postprocessor({"other": torch.zeros(1)})

    def test_clamps_to_unit_range(
        self, molmoact2_features: tuple[list, list]
    ) -> None:
        _, outputs = molmoact2_features
        postprocessor = MolmoAct2Postprocessor(output_features=outputs)
        action = torch.full((1, 2, 6), 5.0)
        result = postprocessor({ACTION: action})
        assert float(result[ACTION].max()) <= 1.0

    def test_masked_dims_denormalized_unmasked_pass_through(self) -> None:
        from physicalai.data.observation import Feature, FeatureType, NormalizationParameters

        # action dims: [0,1,4,5] masked (denormalized), [2,3] pass-through
        action_mask = [True, True, False, False, True, True]
        outputs = [
            Feature(
                name="action",
                ftype=FeatureType.ACTION,
                shape=(6,),
                normalization_data=NormalizationParameters(
                    q01=[0.0] * 6,
                    q99=[2.0] * 6,
                    mask=action_mask,
                ),
            ),
        ]
        postprocessor = MolmoAct2Postprocessor(output_features=outputs)
        # inverse quantile with q01=0, q99=2: y -> (y+1)*2/2 + 0 = y + 1. clamped -1.0 -> 0.0
        action = torch.full((1, 2, 6), -1.0)
        result = postprocessor({ACTION: action})
        mask = torch.tensor(action_mask).view(1, 1, 6)
        # masked dims denormalized (-1 -> 0); unmasked dims clamped & passed through (-1)
        expected = torch.where(mask, torch.full((1, 2, 6), 0.0), torch.full((1, 2, 6), -1.0))
        torch.testing.assert_close(result[ACTION], expected)

    def test_roundtrip_with_masked_normalizer(self) -> None:
        from physicalai.data.observation import Feature, FeatureType, NormalizationParameters

        action_mask = [True, True, False, False, True, True]
        inputs: list = []
        outputs = [
            Feature(
                name="action",
                ftype=FeatureType.ACTION,
                shape=(6,),
                normalization_data=NormalizationParameters(
                    q01=[-2.0] * 6,
                    q99=[2.0] * 6,
                    mask=action_mask,
                ),
            ),
        ]
        normalizer = MolmoAct2NormalizeTransform(input_features=inputs, output_features=outputs)
        postprocessor = MolmoAct2Postprocessor(output_features=outputs)
        # unmasked dims stay within [-1, 1] so clamp is a no-op and they survive;
        # masked dims round-trip exactly through the quantile transform.
        raw = torch.tensor([[0.5, -0.5, 0.3, -0.3, 0.25, -0.25]])
        normalized = normalizer({ACTION: raw.clone()})[ACTION]
        denormalized = postprocessor({ACTION: normalized})[ACTION]
        torch.testing.assert_close(denormalized, raw)
