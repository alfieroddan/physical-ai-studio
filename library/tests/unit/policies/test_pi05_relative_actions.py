# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Relative π0.5 training and export tests without network weights."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from datasets import Dataset

from physicalai.data import Observation
from physicalai.data.lerobot.dataset import _LeRobotDatasetAdapter  # noqa: PLC2701
from physicalai.inference import InferenceModel
from physicalai.inference.manifest import Manifest, ModelSpec, PolicySpec
from physicalai.policies.mixins.peft import is_lora_injected
from physicalai.policies.pi05 import Pi05
from physicalai.policies.pi05.preprocessor import Pi05Preprocessor

if TYPE_CHECKING:
    from pathlib import Path


def test_relative_stats_use_only_episode_contained_chunks() -> None:
    """Use full chunks in a single episode and leave grippers absolute."""
    names = ["elbow.pos", "gripper.pos"]
    features = {key: {"names": names, "shape": (2,)} for key in ("action", "observation.state")}
    rows = Dataset.from_dict({
        "episode_index": [0, 0, 1, 1],
        "observation.state": [[10., 50.], [11., 51.], [20., 60.], [21., 61.]],
        "action": [[12., 1.], [14., 2.], [21., 3.], [25., 4.]],
    })
    dataset = _LeRobotDatasetAdapter.from_lerobot(SimpleNamespace(hf_dataset=rows, features=features))
    stats = dataset.relative_action_stats(2, ("gripper",))
    assert stats["relative_action_mask"] == (True, False)
    assert stats["mean"] == pytest.approx([3., 2.5])
    features["observation.state"] = {"names": names[::-1], "shape": (2,)}
    with pytest.raises(ValueError, match="same order"):
        dataset.relative_action_stats(2, ("gripper",))


def test_relative_train_and_torch_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the relative config and normalization through a Torch export and reload."""

    class FixedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(1, 1)

        @staticmethod
        def get_default_peft_targets() -> str:
            return "projection"

        @staticmethod
        def forward(batch: dict[str, torch.Tensor]) -> torch.Tensor:
            return batch["action"][..., :2]

        def compute_val_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
            return self(batch)

        @staticmethod
        def predict_action_chunk(batch: dict[str, torch.Tensor]) -> torch.Tensor:
            return batch["state"].new_tensor([[[0.5, -0.5], [1.5, 0.]]])

    def initialize(self: Pi05, stats: dict, _weights_file: Path | None = None) -> None:
        self._dataset_stats = stats
        self.model = FixedModel()
        if self.config.use_lora:
            self._inject_lora()
        self._preprocessor, self._postprocessor = self._make_preprocessors(stats)

    monkeypatch.setattr(Pi05, "_initialize_model", initialize)
    monkeypatch.setattr(
        Pi05Preprocessor, "_tokenize",
        lambda _self, texts: (torch.zeros(len(texts), 1, dtype=torch.long), torch.ones(len(texts), 1, dtype=torch.bool)),
    )
    stats = {
        "action": {"name": "action", "shape": (2,), "mean": [1., 2.], "std": [2., 2.],
                   "relative_action_mask": (True, False)},
        "observation.state": {"name": "observation.state", "shape": (2,), "mean": [9., 49.], "std": [2., 2.]},
    }
    policy = Pi05(chunk_size=2, n_action_steps=1, use_relative_actions=True, lora_enabled=True,
                  normalization_mode="MEAN_STD", dataset_stats=stats)
    with pytest.raises(ValueError, match="does not support Real-Time Chunking"):
        policy.rtc_enabled = True

    image = torch.zeros(1, 3, 224, 224)
    state = torch.tensor([[10., 50.]])
    absolute = torch.tensor([[[12., 1.], [14., 2.]]])
    batch = Observation(state=state, action=absolute, images={"left": image})
    normalized = torch.tensor([[[0.5, -0.5], [1.5, 0.]]])
    torch.testing.assert_close(policy(batch), normalized)
    policy.eval()
    torch.testing.assert_close(policy.compute_val_loss(batch), normalized)
    torch.testing.assert_close(policy.predict_action_chunk(batch), absolute)

    openvino = policy.extra_export_args["openvino"]
    assert [spec.type for spec in openvino.postprocessors_specs] == [
        "denormalize", "absolute_actions", "action_chunk_trimmer",
    ]
    policy.export(tmp_path, backend="torch")
    assert is_lora_injected(policy.model)
    model = InferenceModel(tmp_path, backend="torch", device="cpu")
    inputs = {"state": state.numpy(), "images.left": image.numpy()}
    np.testing.assert_array_equal(model.predict_action_chunk(inputs), [[12., 1.]])
    np.testing.assert_array_equal(model.predict_action_chunk({**inputs, "state": state.numpy() + 10}), [[22., 1.]])


def test_runtime_openvino_uses_pi05_relative_postprocessors(tmp_path: Path) -> None:
    """Run the exported processor specs with a small OpenVINO graph on CPU."""
    ov = pytest.importorskip("openvino")
    policy = Pi05(use_relative_actions=True, chunk_size=2, n_action_steps=1, normalization_mode="MEAN_STD")
    policy._dataset_stats = {
        "action": {"mean": [1., 2.], "std": [2., 2.], "relative_action_mask": (True, False)},
        "observation.state": {"mean": [10., 50.], "std": [2., 2.]},
    }
    specs = policy.extra_export_args["openvino"]
    state = ov.opset13.parameter([1, 2], np.float32, name="state")
    state.output(0).set_names({"state"})
    zeros = ov.opset13.multiply(state, ov.opset13.constant(np.float32(0)))
    offsets = ov.opset13.constant(np.array([[[0.5, -0.5], [1.5, 0.]]], np.float32))
    actions = ov.opset13.add(ov.opset13.unsqueeze(zeros, ov.opset13.constant([1])), offsets)
    actions.output(0).set_names({"action"})
    ov.save_model(ov.Model([actions], [state]), tmp_path / "pi05.xml")
    Manifest(
        policy=PolicySpec(name="pi05"),
        model=ModelSpec(
            artifacts={"openvino": "pi05.xml"},
            preprocessors=specs.preprocessors_specs[:1],  # State normalization is enough for the stand-in graph.
            postprocessors=specs.postprocessors_specs,
        ),
    ).save(tmp_path / "manifest.json")

    model = InferenceModel(tmp_path, backend="openvino", device="CPU")
    first = model.predict_action_chunk({"state": np.array([[10., 50.]], np.float32)})
    second = model.predict_action_chunk({"state": np.array([[20., 60.]], np.float32)})
    np.testing.assert_array_equal(first, [[12., 1.]])
    np.testing.assert_array_equal(second, [[22., 1.]])
