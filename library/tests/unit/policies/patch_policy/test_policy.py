# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for the modular Patch Policy pipeline."""

from __future__ import annotations

import warnings
from dataclasses import fields, replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import lightning as L  # noqa: N812
import pytest
import torch

from physicalai.data import DataModule, Dataset, Feature, FeatureType, NormalizationParameters, Observation
from physicalai.policies.patch_policy import PatchPolicy
from physicalai.policies.patch_policy.encoder import TimmFeatureEncoder
from physicalai.policies.patch_policy.heads import PatchBehaviorTransformerHead
from physicalai.policies.patch_policy.model import PatchPolicyModel
from physicalai.policies.patch_policy.policy import _copy_normalization_data
from physicalai.policies.patch_policy.processors import PatchPolicyPostprocessor, PatchPolicyPreprocessor
from physicalai.policies.utils import in_episode_bound, reduce_losses
from physicalai.train import Trainer


def features() -> tuple[list[Feature], list[Feature]]:
    """Return a minimal visual/action contract."""
    return (
        [Feature(name="front", ftype=FeatureType.VISUAL, shape=(3, 32, 32))],
        [Feature(name="action", ftype=FeatureType.ACTION, shape=(2,))],
    )


def make_policy(action_head: str, *, use_random_input_noise: bool = True) -> PatchPolicy:
    """Construct a small policy without downloading pretrained weights."""
    inputs, outputs = features()
    return PatchPolicy(
        input_features=inputs,
        output_features=outputs,
        encoder_name="resnet18",
        encoder_output="average",
        encoder_pretrained=False,
        action_head=action_head,  # type: ignore[arg-type]
        n_obs_steps=2,
        chunk_size=4,
        n_action_steps=2,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        num_train_timesteps=10,
        num_inference_steps=2,
        use_random_input_noise=use_random_input_noise,
    )


def batch() -> Observation:
    """Return a synthetic two-step visual batch."""
    return Observation(
        images={"front": torch.rand(2, 2, 3, 32, 32)},
        action=torch.rand(2, 4, 2) * 2 - 1,
    )


@pytest.mark.parametrize("flatten", [False, True])
def test_preprocessor_preserves_action_padding(flatten: bool) -> None:
    inputs, outputs = features()
    processor = PatchPolicyPreprocessor(inputs, outputs, (4, 4), n_obs_steps=2)
    observation = batch()
    padding = torch.tensor([[False, False, True, True], [True, True, True, True]])
    observation.extra = {"action_is_pad": padding}

    processed = processor(observation.to_dict(flatten=flatten))

    torch.testing.assert_close(processed["extra.action_is_pad"], padding)
    torch.testing.assert_close(in_episode_bound(processed), ~padding)
    losses = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0], [100.0, 100.0]]]).expand(2, -1, -1)
    torch.testing.assert_close(reduce_losses(losses, in_episode_bound(processed)), torch.tensor(2.0))


@pytest.mark.parametrize("all_padded", [False, True])
def test_behavior_transformer_padding_ignores_targets_and_gradients(all_padded: bool) -> None:
    torch.manual_seed(0)
    head = PatchBehaviorTransformerHead(
        8,
        2,
        4,
        n_obs_steps=1,
        n_patches=1,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        latent_dim=4,
        num_embeddings=4,
        groups=2,
    ).eval()
    head.trunk.requires_grad_(False)
    valid_steps = torch.tensor(
        [[False, False, False, False] if all_padded else [True, True, False, False], [False, False, False, False]],
    )
    actions = torch.randn(2, 4, 2, requires_grad=True)
    changed = torch.where(valid_steps.unsqueeze(-1), actions.detach(), 100.0).requires_grad_()
    context = {"tokens": torch.zeros(2, 1, 8)}

    first_losses = head.compute_loss(actions, context, valid_steps=valid_steps)
    first_loss = reduce_losses(first_losses, valid_steps)
    first_loss.backward()
    first_gradients = {
        name: parameter.grad.clone() for name, parameter in head.named_parameters() if parameter.grad is not None
    }
    head.zero_grad(set_to_none=True)
    second_loss = reduce_losses(head.compute_loss(changed, context, valid_steps=valid_steps), valid_steps)
    second_loss.backward()

    torch.testing.assert_close(first_loss, second_loss)
    assert torch.isfinite(first_loss)
    assert actions.grad is not None
    assert changed.grad is not None
    torch.testing.assert_close(actions.grad, changed.grad)
    torch.testing.assert_close(actions.grad[~valid_steps], torch.zeros_like(actions.grad[~valid_steps]))
    for name, parameter in head.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, first_gradients[name])
            if all_padded:
                torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter.grad))
    torch.testing.assert_close(first_losses[~valid_steps], torch.zeros_like(first_losses[~valid_steps]))
    if all_padded:
        torch.testing.assert_close(first_loss, torch.tensor(0.0))


def test_behavior_transformer_unpadded_loss_is_unchanged() -> None:
    head = PatchBehaviorTransformerHead(
        8,
        2,
        4,
        n_obs_steps=1,
        n_patches=1,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        latent_dim=4,
        num_embeddings=4,
        groups=2,
    ).eval()
    actions = torch.randn(2, 4, 2)
    context = {"tokens": torch.zeros(2, 1, 8)}

    without_mask = head.compute_loss(actions, context)
    with_mask = head.compute_loss(actions, context, valid_steps=torch.ones(2, 4, dtype=torch.bool))

    torch.testing.assert_close(with_mask, without_mask)


@pytest.mark.parametrize("action_head", ["diffusion", "behavior_transformer"])
def test_model_training_and_validation_use_action_padding(action_head: str, monkeypatch: pytest.MonkeyPatch) -> None:
    policy = make_policy(action_head)
    model = policy.model
    assert isinstance(model, PatchPolicyModel)
    observation = batch()
    padding = torch.tensor([[False, False, True, True], [True, True, True, True]])
    observation.extra = {"action_is_pad": padding}
    prepared = policy._prepare_batch(observation, require_actions=True)
    losses = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0], [100.0, 100.0]]]).expand(2, -1, -1)
    compute_loss = Mock(return_value=losses)
    monkeypatch.setattr(model, "_context", Mock(return_value={}))
    monkeypatch.setattr(model.action_head, "compute_loss", compute_loss)
    monkeypatch.setattr(model.action_head, "sample", Mock(return_value=prepared["action"] + losses.sqrt()))

    training_loss, _ = model.compute_loss(prepared)
    validation_loss, _ = model.compute_val_loss(prepared)

    torch.testing.assert_close(training_loss, torch.tensor(2.0))
    torch.testing.assert_close(validation_loss, torch.tensor(2.0))
    if action_head == "behavior_transformer":
        torch.testing.assert_close(compute_loss.call_args.kwargs["valid_steps"], ~padding)
    else:
        assert not compute_loss.call_args.kwargs


def test_processors_warn_when_action_normalization_data_is_missing() -> None:
    _, outputs = features()

    with pytest.warns(UserWarning, match="normalization"):
        PatchPolicyPreprocessor([], outputs, (32, 32))
    with pytest.warns(UserWarning, match="de-normalization"):
        PatchPolicyPostprocessor(outputs, n_action_steps=1)


@torch.no_grad()
def test_processors_normalize_images_and_concatenated_actions() -> None:
    """Camera features match by name regardless of mapping order."""
    inputs = [
        Feature(
            name="front",
            ftype=FeatureType.VISUAL,
            shape=(3, 4, 4),
            normalization_data=NormalizationParameters(mean=[0.5, 1.0, 1.5], std=[0.5, 1.0, 0.5]),
        ),
        Feature(
            name="wrist",
            ftype=FeatureType.VISUAL,
            shape=(3, 4, 4),
            normalization_data=NormalizationParameters(mean=[1.0, 1.0, 1.0], std=[2.0, 2.0, 2.0]),
        ),
    ]
    outputs = [
        Feature(
            name="action.arm",
            ftype=FeatureType.ACTION,
            shape=(2,),
            normalization_data=NormalizationParameters(mean=[1.0, 2.0], std=[2.0, 4.0]),
        ),
        Feature(
            name="action.gripper",
            ftype=FeatureType.ACTION,
            shape=(1,),
            normalization_data=NormalizationParameters(mean=[3.0], std=[0.5]),
        ),
    ]
    preprocessor = PatchPolicyPreprocessor(inputs, outputs, (4, 4))
    postprocessor = PatchPolicyPostprocessor(outputs, n_action_steps=1)
    raw_actions = torch.tensor([[[3.0, 6.0, 3.5], [1.0, 2.0, 3.0]]])

    processed = preprocessor(
        {
            "images": {
                "wrist": torch.full((1, 1, 3, 8, 8), 3.0),
                "front": torch.cat(
                    [torch.full((1, 1, 1, 8, 8), value) for value in (0.5, 1.0, 2.0)],
                    dim=2,
                ),
            },
            "action": raw_actions,
        },
    )
    flattened = preprocessor(
        {
            "images.wrist": torch.full((1, 1, 3, 8, 8), 3.0),
            "images.front": torch.cat(
                [torch.full((1, 1, 1, 8, 8), value) for value in (0.5, 1.0, 2.0)],
                dim=2,
            ),
            "action": raw_actions,
        },
    )

    assert processed["images"]["front"].shape == (1, 1, 3, 4, 4)
    assert processed["images"]["wrist"].shape == (1, 1, 3, 4, 4)
    torch.testing.assert_close(processed["images"]["front"][0, 0, :, 0, 0], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(processed["images"]["wrist"][0, 0, :, 0, 0], torch.ones(3))
    assert processed["action"].shape == (1, 2, 3)
    torch.testing.assert_close(processed["action"], torch.tensor([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]]))
    torch.testing.assert_close(flattened["images"]["front"], processed["images"]["front"])
    torch.testing.assert_close(flattened["images"]["wrist"], processed["images"]["wrist"])
    torch.testing.assert_close(flattened["action"], processed["action"])
    torch.testing.assert_close(postprocessor(processed["action"]), raw_actions[:, :1])


def test_preprocessor_adds_singleton_time_axis_to_single_frame() -> None:
    """Single frames gain a time axis without processor-owned history or padding."""
    feature = Feature(name="front", ftype=FeatureType.VISUAL, shape=(3, 4, 4))
    preprocessor = PatchPolicyPreprocessor([feature], [], (4, 4), n_obs_steps=3)
    frame = torch.full((1, 3, 4, 4), 5.0)

    processed = preprocessor({"images": {"front": frame}})["images"]

    assert processed.shape == (1, 1, 3, 4, 4)
    torch.testing.assert_close(processed[0, 0, :, 0, 0], torch.full((3,), 5.0))


def test_preprocessor_uses_mean_std_feature_statistics() -> None:
    """Patch Policy applies mean/std normalization when feature statistics exist."""
    feature = Feature(
        name="front",
        ftype=FeatureType.VISUAL,
        shape=(3, 4, 4),
        normalization_data=NormalizationParameters(mean=[1.0, 1.0, 1.0], std=[2.0, 2.0, 2.0]),
    )
    preprocessor = PatchPolicyPreprocessor([feature], [], (4, 4))
    image = torch.full((1, 3, 4, 4), 3.0)

    processed = preprocessor({"images": {"front": image}})["images"]["front"]

    torch.testing.assert_close(processed, torch.ones_like(image[:, None]))


def test_preprocessor_scales_uint8_before_image_normalization() -> None:
    """Uint8 pixels are converted to [0, 1] before ImageNet normalization."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    feature = Feature(
        name="front",
        ftype=FeatureType.VISUAL,
        shape=(3, 4, 4),
        normalization_data=NormalizationParameters(mean=mean, std=std),
    )
    preprocessor = PatchPolicyPreprocessor([feature], [], (4, 4))
    image = torch.full((1, 3, 4, 4), 255, dtype=torch.uint8)

    processed = preprocessor({"images": {"front": image}})["images"]["front"]

    torch.testing.assert_close(processed[0, 0, :, 0, 0], (1 - torch.tensor(mean)) / torch.tensor(std))


def test_policy_attaches_encoder_stats_and_preserves_dataset_action_stats() -> None:
    """Visual features use encoder stats while action features keep dataset stats."""
    policy = make_policy("diffusion")
    visual = policy.config.input_features[0]
    assert visual.normalization_data is not None
    assert visual.normalization_data.mean == pytest.approx([0.485, 0.456, 0.406])
    assert visual.normalization_data.std == pytest.approx([0.229, 0.224, 0.225])

    action_stats = NormalizationParameters(mean=[1.0, 2.0], std=[0.5, 1.5])
    action_feature = Feature(name="action", ftype=FeatureType.ACTION, shape=(2,))
    dataset_feature = Feature(
        name="action",
        ftype=FeatureType.ACTION,
        shape=(2,),
        normalization_data=action_stats,
    )

    resolved = _copy_normalization_data([action_feature], [dataset_feature])

    assert resolved[0].normalization_data == action_stats


def test_setup_resolves_features_from_training_dataset() -> None:
    inputs, outputs = features()
    action_stats = NormalizationParameters(mean=[1.0, 2.0], std=[0.5, 1.5])
    dataset_outputs = [replace(outputs[0], normalization_data=action_stats)]
    train_dataset = Mock(spec=Dataset)
    train_dataset.observation_features = {feature.name: feature for feature in inputs}
    train_dataset.action_features = {feature.name: feature for feature in dataset_outputs}
    trainer = Mock()
    trainer.datamodule.train_dataset = train_dataset
    policy = PatchPolicy(
        encoder_name="resnet18",
        encoder_output="average",
        encoder_pretrained=False,
    )
    policy._trainer = trainer

    policy.setup("fit")

    assert policy.model is None
    assert policy.config is None
    assert policy._preprocessor is None
    assert policy._postprocessor is None
    assert policy.input_features == inputs
    assert policy.output_features == dataset_outputs

    policy.configure_model()

    assert policy.config is not None
    assert isinstance(policy.model, PatchPolicyModel)
    assert policy.input_features == policy.config.input_features
    assert policy.output_features == dataset_outputs
    assert policy.output_features[0].normalization_data == action_stats


def test_configure_model_is_idempotent() -> None:
    policy = make_policy("diffusion")
    model = policy.model
    config = policy.config
    preprocessor = policy._preprocessor
    postprocessor = policy._postprocessor

    policy.configure_model()

    assert policy.model is model
    assert policy.config is config
    assert policy._preprocessor is preprocessor
    assert policy._postprocessor is postprocessor


def test_configure_model_defers_until_features_are_resolved() -> None:
    policy = PatchPolicy()

    policy.configure_model()

    assert policy.model is None
    assert policy.config is None
    assert policy._preprocessor is None
    assert policy._postprocessor is None
    with pytest.raises(RuntimeError, match="features and model are resolved"):
        policy.configure_optimizers()


@pytest.mark.parametrize("dataset_stats", [None, NormalizationParameters(mean=[10.0, 20.0], std=[2.0, 2.0])])
def test_setup_replaces_existing_action_statistics(dataset_stats: NormalizationParameters | None) -> None:
    policy = make_policy("diffusion")
    assert policy.output_features is not None
    old_stats = NormalizationParameters(mean=[1.0, 2.0], std=[1.0, 1.0])
    policy.output_features = [replace(policy.output_features[0], normalization_data=old_stats)]
    policy.configure_model()
    model = policy.model
    inputs, outputs = features()
    train_dataset = Mock(spec=Dataset)
    train_dataset.observation_features = {feature.name: feature for feature in inputs}
    train_dataset.action_features = {"action": replace(outputs[0], normalization_data=dataset_stats)}
    trainer = Mock()
    trainer.datamodule.train_dataset = train_dataset
    policy.trainer = trainer

    with pytest.warns(UserWarning, match="Replacing normalization statistics"):
        policy.setup("fit")
    policy.configure_model()

    assert policy.model is model
    assert policy.config is not None
    assert policy.config.output_features[0].normalization_data == dataset_stats
    assert policy._preprocessor is not None
    assert policy._postprocessor is not None
    assert policy._preprocessor.output_features == policy.config.output_features
    assert policy._postprocessor.output_features == policy.config.output_features
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        policy.setup("fit")
        policy.configure_model()
    assert not captured


def test_configure_model_refreshes_processors_without_rebuilding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = make_policy("diffusion")
    model = policy.model
    assert model is not None
    preprocessor = policy._preprocessor
    postprocessor = policy._postprocessor
    monkeypatch.setattr(PatchPolicyModel, "from_config", Mock(side_effect=AssertionError("Model was rebuilt.")))
    inputs, outputs = features()
    action_stats = NormalizationParameters(mean=[1.0, 2.0], std=[0.5, 1.5])
    dataset_outputs = [replace(outputs[0], normalization_data=action_stats)]
    train_dataset = Mock(spec=Dataset)
    train_dataset.observation_features = {feature.name: feature for feature in inputs}
    train_dataset.action_features = {feature.name: feature for feature in dataset_outputs}
    trainer = Mock()
    trainer.datamodule.train_dataset = train_dataset
    policy._trainer = trainer

    policy.setup("fit")

    assert policy._preprocessor is preprocessor
    assert policy._postprocessor is postprocessor

    policy.configure_model()

    assert policy.model is model
    assert policy.config is not None
    assert policy.config.output_features == dataset_outputs
    assert policy._preprocessor is not None
    assert policy._postprocessor is not None
    assert policy._preprocessor.image_size == model.encoder.image_size
    assert policy._preprocessor is not preprocessor
    assert policy._postprocessor is not postprocessor
    assert policy.input_features is not None
    assert policy.input_features[0].normalization_data is not None
    assert policy.input_features[0].normalization_data.mean == pytest.approx([0.485, 0.456, 0.406])

    preprocessor = policy._preprocessor
    postprocessor = policy._postprocessor
    policy.configure_model()
    assert policy.model is model
    assert policy._preprocessor is preprocessor
    assert policy._postprocessor is postprocessor


@pytest.mark.parametrize("feature_kind", ["input", "output"])
def test_setup_rejects_mismatched_explicit_features(feature_kind: str) -> None:
    inputs, outputs = features()
    train_dataset = Mock(spec=Dataset)
    train_dataset.observation_features = {feature.name: feature for feature in inputs}
    train_dataset.action_features = {feature.name: feature for feature in outputs}
    trainer = Mock()
    trainer.datamodule.train_dataset = train_dataset
    if feature_kind == "input":
        policy = PatchPolicy(input_features=[replace(inputs[0], shape=(3, 16, 16))])
        error = "input_features"
    else:
        policy = PatchPolicy(output_features=[replace(outputs[0], shape=(3,))])
        error = "output_features"
    policy._trainer = trainer

    with pytest.raises(ValueError, match=error):
        policy.setup("fit")


def test_encoder_does_not_apply_image_normalization_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    """The encoder passes processor-normalized inputs directly to timm."""
    model = make_policy("diffusion").model
    assert isinstance(model, PatchPolicyModel)
    images = torch.rand(1, 1, 3, 32, 32)
    received: list[torch.Tensor] = []

    def capture(value: torch.Tensor) -> torch.Tensor:
        received.append(value)
        return torch.zeros(value.shape[0], model.encoder.output_dim, 1, 1)

    monkeypatch.setattr(model.encoder.backbone, "forward", capture)
    model.encoder(images)

    torch.testing.assert_close(received[0], images.reshape(-1, 3, 32, 32))


def test_preprocessor_requires_matching_temporal_length() -> None:
    """Explicit temporal images stay 5D and must match n_obs_steps."""
    feature = Feature(name="front", ftype=FeatureType.VISUAL, shape=(3, 4, 4))
    preprocessor = PatchPolicyPreprocessor([feature], [], (4, 4), n_obs_steps=3)
    images = torch.zeros(1, 3, 3, 4, 4)

    processed = preprocessor({"images": {"front": images}})["images"]

    assert processed["front"].shape == (1, 3, 3, 4, 4)

    with pytest.raises(ValueError, match="Expected 3 observation timestep"):
        preprocessor({"images": {"front": torch.zeros(1, 2, 3, 4, 4)}})


def test_select_action_does_not_track_queued_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    """The policy does not preprocess or retain observations while actions are queued."""
    policy = make_policy("diffusion")
    policy.eval()
    captured: list[torch.Tensor] = []

    def predict(batch: dict[str, Any]) -> torch.Tensor:
        captured.append(batch["images"]["front"].clone())
        return torch.zeros(1, 4, 2)

    assert policy.model is not None
    monkeypatch.setattr(policy.model, "predict_action_chunk", predict)
    for value in (1.0, 2.0, 3.0):
        policy.select_action(Observation(images={"front": torch.full((1, 3, 32, 32), value)}))

    assert len(captured) == 2
    assert captured[0].shape == (1, 1, 3, *policy.model.encoder.image_size)
    assert captured[1].shape == (1, 1, 3, *policy.model.encoder.image_size)
    torch.testing.assert_close(captured[0][0, 0, 0, 0, 0], torch.tensor(1.0))
    torch.testing.assert_close(captured[1][0, 0, 0, 0, 0], torch.tensor(3.0))

    policy.reset()
    policy.select_action(Observation(images={"front": torch.full((1, 3, 32, 32), 4.0)}))
    torch.testing.assert_close(captured[2][0, 0, 0, 0, 0], torch.tensor(4.0))


@torch.no_grad()
def test_diffusion_shapes() -> None:
    """Diffusion computes a scalar loss and an execution chunk."""
    policy = make_policy("diffusion")
    policy.train()
    loss, _ = policy(batch())
    assert loss.shape == ()
    policy.eval()
    assert policy.predict_action_chunk(batch()).shape == (2, 2, 2)


@torch.no_grad()
def test_behavior_transformer_shapes() -> None:
    """Behavior Transformer computes a scalar loss and an execution chunk."""
    policy = make_policy("behavior_transformer")
    policy.train()
    loss, _ = policy(batch())
    assert loss.shape == ()
    policy.eval()
    assert policy.predict_action_chunk(batch()).shape == (2, 2, 2)


def test_temporal_indices() -> None:
    """The model requests exactly its configured observation and action windows."""
    model = make_policy("diffusion").model
    assert model is not None
    assert model.observation_delta_indices == [-1, 0]
    assert model.action_delta_indices == [0, 1, 2, 3]


def test_from_config_reconstructs_explicit_values() -> None:
    """Construction from config preserves every model setting without an argument dictionary."""
    original = make_policy("diffusion", use_random_input_noise=False)
    config = replace(original.config, eta=0.0, clip_sample_range=0.5, use_random_input_noise=False)
    restored = PatchPolicy.from_config(config)

    assert restored.config == config
    assert not hasattr(restored, "_config_values")
    assert isinstance(restored.model, PatchPolicyModel)
    assert getattr(restored.model.action_head, "eta") == 0.0
    assert getattr(restored.model.action_head, "clip_sample_range") == 0.5
    assert getattr(restored.model.action_head, "use_random_input_noise") is False


def test_random_input_noise_propagates_from_policy() -> None:
    """The public policy option reaches the resolved config and diffusion head."""
    policy = make_policy("diffusion", use_random_input_noise=False)

    assert policy.config.use_random_input_noise is False
    assert isinstance(policy.model, PatchPolicyModel)
    assert getattr(policy.model.action_head, "use_random_input_noise") is False


@torch.no_grad()
def test_dinov2_uses_native_image_size() -> None:
    """DINOv2 supplies the resolution used by preprocessing and patch counts."""
    config = replace(
        make_policy("diffusion").config,
        encoder_name="dinov2",
        encoder_output="patches",
    )
    policy = PatchPolicy.from_config(config)
    model = policy.model
    assert model is not None
    image_size = model.encoder.image_size
    assert image_size == model.encoder.backbone.patch_embed.img_size
    assert policy._preprocessor.image_size == image_size
    assert "image_size" not in policy.hparams
    assert "image_size" not in policy.config.to_dict()
    expected_tokens = (image_size[0] // 14) * (image_size[1] // 14)

    assert model.encoder.num_tokens == expected_tokens
    tokens = model.encoder(torch.rand(1, 1, 1, 3, *image_size))
    assert tokens.shape == (1, 1, 1, expected_tokens, model.encoder.output_dim)


def test_resnet18_uses_fallback_image_size() -> None:
    policy = make_policy("diffusion")

    assert policy.model.encoder.image_size == (224, 224)
    assert policy._preprocessor.image_size == (224, 224)
    prepared = policy._prepare_batch(batch(), require_actions=False)
    assert prepared["images"]["front"].shape[-2:] == (224, 224)


@pytest.mark.parametrize("image_size", [224, (112, 224)])
def test_webssl_uses_config_image_size(image_size: int | tuple[int, int], monkeypatch: pytest.MonkeyPatch) -> None:
    from transformers import Dinov2Config  # noqa: PLC0415

    config = Dinov2Config(
        image_size=image_size,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
    )
    monkeypatch.setattr(Dinov2Config, "from_pretrained", Mock(return_value=config))

    encoder = TimmFeatureEncoder("webssl", "patches", pretrained=False)

    expected_size = (image_size, image_size) if isinstance(image_size, int) else image_size
    assert encoder.image_size == expected_size
    assert encoder.num_tokens == (expected_size[0] // config.patch_size) * (expected_size[1] // config.patch_size)


def test_checkpoint_hooks_restore_complete_config() -> None:
    """Checkpoint hooks rebuild a lazy policy from its complete serialized config."""
    original = make_policy("diffusion", use_random_input_noise=False)
    checkpoint: dict[str, object] = {}
    original.on_save_checkpoint(checkpoint)

    restored = PatchPolicy(n_action_steps=original.config.n_action_steps)
    restored.on_load_checkpoint(checkpoint)

    assert restored.config == original.config
    assert isinstance(restored.model, PatchPolicyModel)
    assert restored.config is not None
    for field in fields(restored.config):
        if field.name in restored.hparams:
            assert getattr(restored, field.name) == getattr(restored.config, field.name)
            assert restored.hparams[field.name] == getattr(restored.config, field.name)


@pytest.mark.parametrize("processor_name", ["_preprocessor", "_postprocessor"])
def test_checkpoint_repairs_existing_model_processors(processor_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    policy = make_policy("diffusion")
    checkpoint: dict[str, Any] = {}
    policy.on_save_checkpoint(checkpoint)
    model = policy.model
    setattr(policy, processor_name, None)
    monkeypatch.setattr(PatchPolicyModel, "from_config", Mock(side_effect=AssertionError("Model was rebuilt.")))

    policy.on_load_checkpoint(checkpoint)

    assert policy.model is model
    assert policy._preprocessor is not None
    assert policy._postprocessor is not None


def test_checkpoint_refreshes_statistics_without_rebuilding_model() -> None:
    policy = make_policy("diffusion")
    model = policy.model
    assert policy.config is not None
    stats = NormalizationParameters(mean=[10.0, 20.0], std=[2.0, 3.0])
    config = replace(
        policy.config,
        output_features=[replace(policy.config.output_features[0], normalization_data=stats)],
    )

    policy.on_load_checkpoint({"policy_config": config.to_dict()})

    assert policy.model is model
    assert policy.config == config
    assert policy.output_features == config.output_features
    assert policy._preprocessor is not None
    assert policy._postprocessor is not None
    normalized = policy._preprocessor.normalize_actions(torch.tensor([[[12.0, 23.0]]]))
    torch.testing.assert_close(normalized, torch.ones_like(normalized))
    torch.testing.assert_close(policy._postprocessor(normalized), torch.tensor([[[12.0, 23.0]]]))


def test_checkpoint_rejects_incompatible_existing_model() -> None:
    policy = make_policy("diffusion")
    model = policy.model
    assert policy.config is not None
    config = policy.config

    with pytest.raises(ValueError, match="does not match"):
        policy.on_load_checkpoint({"policy_config": replace(config, hidden_dim=64).to_dict()})

    assert policy.model is model
    assert policy.config is config


def test_load_from_checkpoint_round_trip(tmp_path: Path) -> None:
    """Lightning checkpoint loading reconstructs modules before restoring their tensors."""
    original = make_policy("diffusion", use_random_input_noise=False)
    checkpoint = {
        "state_dict": original.state_dict(),
        "hyper_parameters": dict(original.hparams),
        "pytorch-lightning_version": L.__version__,
    }
    original.on_save_checkpoint(checkpoint)
    path = tmp_path / "patch_policy.ckpt"
    # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch  # Test-only checkpoint is generated from trusted local state.
    torch.save(checkpoint, path)

    restored = PatchPolicy.load_from_checkpoint(path)

    assert restored.config == original.config
    assert set(restored.state_dict()) == set(original.state_dict())
    assert restored.config.use_random_input_noise is False
    for name, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


@pytest.mark.parametrize("initialization", ["features", "dataset", "checkpoint"])
def test_configure_model_initialization_paths(initialization: str, dummy_dataset: Any, tmp_path: Path) -> None:
    dataset = dummy_dataset(num_samples=2)
    dataset.delta_indices = {"action": [0, 1, 2, 3]}
    inputs = list(dataset.observation_features.values())
    outputs = list(dataset.action_features.values())
    kwargs: dict[str, Any] = {
        "encoder_name": "resnet18",
        "encoder_output": "average",
        "encoder_pretrained": False,
        "action_head": "diffusion",
        "n_obs_steps": 1,
        "chunk_size": 4,
        "n_action_steps": 2,
        "hidden_dim": 32,
        "num_layers": 1,
        "num_heads": 4,
        "num_train_timesteps": 10,
        "num_inference_steps": 2,
    }
    policy = PatchPolicy(
        input_features=None if initialization == "dataset" else inputs,
        output_features=None if initialization == "dataset" else outputs,
        **kwargs,
    )
    if initialization == "checkpoint":
        checkpoint = {
            "state_dict": policy.state_dict(),
            "hyper_parameters": dict(policy.hparams),
            "pytorch-lightning_version": L.__version__,
        }
        policy.on_save_checkpoint(checkpoint)
        path = tmp_path / "fit.ckpt"
        torch.save(checkpoint, path)
        policy = PatchPolicy.load_from_checkpoint(path)
    model = policy.model
    datamodule = DataModule(dataset, train_batch_size=2, num_workers=0, persistent_workers=False)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(policy, datamodule=datamodule)

    assert trainer.global_step == 0
    assert policy.model is not None
    if model is not None:
        assert policy.model is model
    assert policy.config is not None
    assert policy.input_features == policy.config.input_features
    assert policy.output_features == policy.config.output_features
    assert policy._preprocessor is not None
    assert policy._postprocessor is not None
    assert policy._preprocessor.output_features == policy.config.output_features
    assert policy._postprocessor.output_features == policy.config.output_features
