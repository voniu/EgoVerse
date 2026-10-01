import types

import torch

from egomimic.algo.hpt_mixed_clean_v2 import attach_keyed_normalization_metadata
from egomimic.models.residual_joint_policy_mixed_clean_v2 import (
    ResidualJointPolicyMixedCleanV2,
)


def _quantile_normalize(values, low, high):
    return 2.0 * (values - low) / (high - low + 1e-6) - 1.0


def _head_and_data():
    torch.manual_seed(3)
    head = ResidualJointPolicyMixedCleanV2(
        action_horizon=8,
        control_horizon=4,
        infer_ac_dims={"piper_left_arm": 7},
        input_dim=16,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        fk_position_loss_weight=0.0,
        fk_rotation_loss_weight=0.0,
    )
    head._fk_losses = types.MethodType(
        lambda self, prediction, target: (
            prediction.new_zeros(()),
            prediction.new_zeros(()),
        ),
        head,
    )
    batch = 2
    action_low = torch.full((8, 7), -1.0)
    action_high = torch.full((8, 7), 1.0)
    action_low[:, 6] = 0.0
    action_high[:, 6] = 1.0
    raw_action = torch.zeros(batch, 8, 7)
    raw_action[..., :6] = torch.linspace(-0.2, 0.2, 8)[None, :, None]
    raw_action[:, :4, 6] = 1.0
    action = _quantile_normalize(raw_action, action_low, action_high)

    proprio_low = torch.tensor([-1.0] * 6 + [0.0])
    proprio_high = torch.tensor([1.0] * 7)
    raw_proprio = torch.zeros(batch, 7)
    raw_proprio[:, 6] = 1.0
    proprio = _quantile_normalize(raw_proprio, proprio_low, proprio_high)
    data = {
        "action": action,
        "state_joints": proprio[:, None],
        "_actions_joints_norm_mode": "quantile",
        "_actions_joints_norm_quantile_1": action_low,
        "_actions_joints_norm_quantile_99": action_high,
        "_proprio_norm_mode": "quantile",
        "_proprio_norm_quantile_1": proprio_low,
        "_proprio_norm_quantile_99": proprio_high,
        # Deliberately incompatible shared-pose stats.  The mixed head must
        # never read these 9D tensors for a 7D joint action.
        "_action_norm_mode": "quantile",
        "_action_norm_quantile_1": torch.zeros(8, 9),
        "_action_norm_quantile_99": torch.ones(8, 9),
    }
    return head, data, torch.randn(batch, 5, 16)


def test_mixed_head_uses_keyed_joint_stats_for_loss_and_inference():
    head, data, features = _head_and_data()
    loss = head.compute_loss(features, data)
    prediction = head(features, data=data)
    assert torch.isfinite(loss)
    assert tuple(prediction.shape) == (2, 8, 7)
    assert torch.isfinite(prediction).all()
    loss.backward()
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_mixed_head_rejects_missing_joint_action_stats():
    head, data, features = _head_and_data()
    del data["_actions_joints_norm_quantile_1"]
    try:
        head.compute_loss(features, data)
    except KeyError as exc:
        assert "actions_joints" in str(exc)
    else:
        raise AssertionError("missing per-key normalization metadata was accepted")


def test_keyed_metadata_keeps_joint_shape_separate_from_shared_pose():
    data = {}
    reference = torch.zeros(2, 100, 7)
    attach_keyed_normalization_metadata(
        data,
        prefix="actions_joints",
        mode="quantile",
        stats={
            "quantile_1": torch.zeros(100, 7),
            "quantile_99": torch.ones(100, 7),
        },
        reference=reference,
    )
    assert data["_actions_joints_norm_mode"] == "quantile"
    assert tuple(data["_actions_joints_norm_quantile_1"].shape) == (100, 7)
    assert tuple(data["_actions_joints_norm_quantile_99"].shape) == (100, 7)
