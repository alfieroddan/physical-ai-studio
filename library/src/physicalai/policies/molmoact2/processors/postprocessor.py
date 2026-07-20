# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 postprocessor."""

from __future__ import annotations

import torch
from torch import nn

from physicalai.data.observation import ACTION, Feature, FeatureType
from physicalai.policies.utils.features import feature_by_type
from physicalai.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType

from .joint_transform import JointFrameTransform


class _MolmoAct2DenormalizeTransform(FeatureNormalizeTransform):
    """Inverse :class:`FeatureNormalizeTransform` with MolmoAct2 mask support.

    The shared :class:`FeatureNormalizeTransform` ignores any ``mask`` carried
    on :class:`NormalizationParameters`. MolmoAct2 actions may carry a
    per-dimension mask selecting which dims are quantile-normalized; for the
    masked-out dims the denormalized value must equal the (clamped) model
    output, so the mask is applied here on top of the base inverse transform.
    """

    def __init__(
        self,
        features: dict[str, Feature],
        norm_map: dict[FeatureType, NormalizationType],
    ) -> None:
        """Build the inverse transform and attach optional mask buffers.

        Args:
            features: Mapping of feature name -> Feature.
            norm_map: Mapping of feature type -> normalization mode.
        """
        super().__init__(features, norm_map, inverse=True)
        for name, feature in features.items():
            norm_mode = norm_map.get(feature.ftype, NormalizationType.IDENTITY)  # type: ignore[arg-type]
            if norm_mode is not NormalizationType.QUANTILES:
                continue
            norm_data = feature.normalization_data
            if norm_data is None or norm_data.mask is None:
                continue
            buffer = self.buffers_lookup.get(name)
            if buffer is None:
                continue
            shape = feature.shape if feature.shape is not None else ()
            mask_tensor = torch.tensor(norm_data.mask, dtype=torch.float32).view(shape)
            buffer["mask"] = nn.Parameter(mask_tensor, requires_grad=False)

    @staticmethod
    def _apply_normalization(
        batch: dict,
        key: str,
        norm_mode: NormalizationType,
        buffer: nn.ParameterDict,
        *,
        inverse: bool,
    ) -> None:
        """Denormalize, honouring an optional per-dim mask for quantiles.

        For ``QUANTILES`` features that carry a ``mask`` buffer, only the masked
        dimensions are denormalized; the rest pass through unchanged. Every
        other mode (and unmasked quantiles) is delegated to the base class so
        MolmoAct2 stays numerically aligned with the shared implementation.

        Args:
            batch: Input batch, modified in place.
            key: Batch key to denormalize.
            norm_mode: Normalization mode for this feature.
            buffer: Buffer ``ParameterDict`` (may contain a ``"mask"`` entry).
            inverse: Whether to apply the inverse transformation.
        """
        if batch[key] is None:
            return
        feature_mask = buffer.get("mask") if hasattr(buffer, "get") else None
        if norm_mode is NormalizationType.QUANTILES and feature_mask is not None:
            q01 = buffer["q01"]
            q99 = buffer["q99"]
            denom = q99 - q01
            denom = torch.where(
                denom == 0,
                torch.tensor(1e-8, device=denom.device, dtype=denom.dtype),
                denom,
            )
            transformed = (batch[key] + 1.0) * denom / 2.0 + q01 if inverse else 2.0 * (batch[key] - q01) / denom - 1.0
            mask_bool = feature_mask.bool()
            for _ in range(batch[key].ndim - mask_bool.ndim):
                mask_bool = mask_bool.unsqueeze(0)
            mask_bool = mask_bool.expand_as(batch[key])
            batch[key] = torch.where(mask_bool, transformed, batch[key])
            return
        FeatureNormalizeTransform._apply_normalization(  # noqa: SLF001
            batch,
            key,
            norm_mode,
            buffer,
            inverse=inverse,
        )


class MolmoAct2Postprocessor(torch.nn.Module):
    """Convert normalized MolmoAct2 outputs back into action space."""

    def __init__(
        self,
        *,
        output_features: list[Feature],
        normalization_mode: str = "QUANTILES",
        adapt_to_so101: bool = False,
        joint_signs: list[float] | None = None,
        joint_offsets: list[float] | None = None,
    ) -> None:
        """Initialize MolmoAct2 postprocessor.

        Args:
            output_features: Output feature definitions.
            normalization_mode: Normalization mode for action denormalization.
            adapt_to_so101: Map actions from the checkpoint frame back to the SO-101
                robot frame after denormalization.
            joint_signs: Per-joint signs for the frame transform.
            joint_offsets: Per-joint offsets for the frame transform.
        """
        super().__init__()
        action_feature = feature_by_type(output_features, FeatureType.ACTION)
        self.action_name = action_feature.name if action_feature else ACTION
        features_map = {f.name: f for f in output_features if f.name}
        action_norm = (
            NormalizationType(normalization_mode)
            if action_feature is not None and action_feature.normalization_data is not None
            else NormalizationType.IDENTITY
        )
        self._denormalizer = _MolmoAct2DenormalizeTransform(
            features_map,
            {FeatureType.ACTION: action_norm},
        )
        self._adapt_to_so101 = adapt_to_so101
        self._joint_transform = JointFrameTransform(joint_signs or [], joint_offsets or []) if adapt_to_so101 else None

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Denormalize, clamp and (optionally) map actions back to the robot frame.

        Args:
            batch: Batch containing ACTION or "actions" tensor.

        Returns:
            Batch with ACTION denormalized and clamped.

        Raises:
            ValueError: If no action tensor is present in the batch.
        """
        batch = dict(batch)
        action = batch.get(ACTION, batch.get("actions"))
        if action is None:
            msg = "MolmoAct2 postprocessor expected an action tensor in outputs."
            raise ValueError(msg)

        action = action.clamp(-1.0, 1.0)
        action = self._denormalizer.to(action.device)({self.action_name: action})[self.action_name]
        if self._joint_transform is not None:
            action = self._joint_transform.to_robot(action)
        batch[ACTION] = action
        return batch
