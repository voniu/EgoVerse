"""PIPER-only residual joint head for Residual Joint Transformer.

It is intentionally a new class so mixed checkpoints cannot be confused with
PIPER-only Residual Joint Transformer checkpoints.  The architecture and physical losses remain
identical; only the normalization namespace changes from the shared 9D pose
action to the PIPER-specific 7D ``actions_joints`` action.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from egomimic.models.residual_joint_transformer import (
    ResidualJointTransformer,
)


class KeyedResidualJointTransformer(ResidualJointTransformer):
    """Residual Joint Transformer head reading the independent 7D joint normalizer."""

    action_norm_prefix = "actions_joints"

    def _validate_data_contract(self, data: dict, *, require_action: bool) -> None:
        if "state_joints" not in data:
            raise ValueError(
                "Residual Joint Transformer requires normalized state_joints"
            )
        if require_action and "action" not in data:
            raise ValueError(
                "Residual Joint Transformer training requires joint action labels"
            )
        state = data["state_joints"]
        if state.ndim < 2 or state.shape[-1] != self.joint_dims + 1:
            raise ValueError(
                "Residual Joint Transformer proprio contract mismatch: expected final dimension "
                f"{self.joint_dims + 1}, got {tuple(state.shape)}"
            )
        if not torch.isfinite(state).all():
            raise ValueError(
                "Residual Joint Transformer received non-finite normalized proprio"
            )
        # During mixed inference HPT keeps the shared 9D pose under
        # ``data['action']`` while asking this auxiliary head for 7D joints.
        # The joint label is required only for training loss computation.
        if not require_action:
            return
        actions = data["action"]
        expected = (self.action_horizon, self.joint_dims + 1)
        if actions.ndim != 3 or tuple(actions.shape[1:]) != expected:
            raise ValueError(
                f"Residual Joint Transformer action contract mismatch: expected [B,{expected[0]},"
                f"{expected[1]}], got {tuple(actions.shape)}"
            )
        if not torch.isfinite(actions).all():
            raise ValueError(
                "Residual Joint Transformer received non-finite normalized actions"
            )
        raw_actions = self._unnormalize(actions, data, self.action_norm_prefix)
        raw_gripper = raw_actions[..., self.joint_dims]
        binary_distance = torch.minimum(raw_gripper.abs(), (raw_gripper - 1.0).abs())
        if torch.max(binary_distance).item() > 2e-4:
            raise ValueError(
                "Residual Joint Transformer gripper target is not binary after independent "
                f"joint-action unnormalization: max distance="
                f"{torch.max(binary_distance).item():.6g}"
            )

    def _raw_control_targets(self, data: dict) -> torch.Tensor:
        raw = self._unnormalize(data["action"], data, self.action_norm_prefix)
        return raw.index_select(1, self._control_indices(raw.device))

    def forward(self, global_cond, data: dict | None = None) -> torch.Tensor:
        if data is None:
            raise ValueError(
                "KeyedResidualJointTransformer inference requires current proprio data"
            )
        self._validate_data_contract(data, require_action=False)
        control_joints, gripper_logits = self._raw_control_prediction(global_cond, data)
        joints = F.interpolate(
            control_joints.transpose(1, 2),
            size=self.action_horizon,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        gripper_open = (gripper_logits >= 0.0).to(joints.dtype)
        gripper = F.interpolate(
            gripper_open[:, None], size=self.action_horizon, mode="nearest"
        ).transpose(1, 2)
        raw_actions = torch.cat((joints, gripper), dim=-1)
        return self._normalize(raw_actions, data, self.action_norm_prefix)
