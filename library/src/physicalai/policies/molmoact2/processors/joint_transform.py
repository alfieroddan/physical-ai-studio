# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SO-100/101 joint frame transform for MolmoAct2.

The released MolmoAct2-SO100_101 checkpoint was trained with the pre-#777
LeRobot joint calibration. Newer LeRobot data uses a different convention, so
joint observations/actions must be mapped into the checkpoint convention on the
way in and back to the robot convention on the way out.

- Robot -> checkpoint:  ``x_ckpt = sign * x_robot + offset``
- Checkpoint -> robot:  ``x_robot = sign * (x_ckpt - offset)``

``sign`` is +/-1 (so ``1 / sign == sign``). The transform touches only the
leading joint dimensions; any trailing dimensions pass through unchanged. For
SO-101 the defaults flip ``shoulder_lift`` and shift ``shoulder_lift`` /
``elbow_flex`` by 90 degrees, matching the LeRobot backward-compatibility guide.
"""

from __future__ import annotations

import torch


class JointFrameTransform:
    """Map joint values between the robot and checkpoint calibration frames."""

    def __init__(self, joint_signs: list[float], joint_offsets: list[float]) -> None:
        """Store per-joint signs and offsets.

        Raises:
            ValueError: If ``joint_signs`` and ``joint_offsets`` differ in length.
        """
        if len(joint_signs) != len(joint_offsets):
            msg = f"joint_signs ({len(joint_signs)}) and joint_offsets ({len(joint_offsets)}) must match."
            raise ValueError(msg)
        self.num_joints = len(joint_signs)
        self._signs = torch.tensor(joint_signs, dtype=torch.float32)
        self._offsets = torch.tensor(joint_offsets, dtype=torch.float32)

    def to_checkpoint(self, values: torch.Tensor) -> torch.Tensor:
        """Map robot-frame joints to the checkpoint frame.

        Returns:
            ``values`` with the leading joint dims mapped ``sign * x + offset``.
        """
        return self._apply(values, inverse=False)

    def to_robot(self, values: torch.Tensor) -> torch.Tensor:
        """Map checkpoint-frame joints back to the robot frame.

        Returns:
            ``values`` with the leading joint dims mapped ``sign * (x - offset)``.
        """
        return self._apply(values, inverse=True)

    def _apply(self, values: torch.Tensor, *, inverse: bool) -> torch.Tensor:
        """Apply the (inverse) affine joint transform to the leading joint dims.

        Returns:
            A new tensor with the leading joint dimensions transformed.
        """
        num_joints = min(self.num_joints, values.shape[-1])
        signs = self._signs[:num_joints].to(device=values.device, dtype=values.dtype)
        offsets = self._offsets[:num_joints].to(device=values.device, dtype=values.dtype)

        out = values.clone()
        joints = values[..., :num_joints]
        out[..., :num_joints] = signs * (joints - offsets) if inverse else signs * joints + offsets
        return out
