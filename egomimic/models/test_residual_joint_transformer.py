import torch

from egomimic.models.residual_joint_policy_clean_v2 import (
    ResidualJointPolicyCleanV2,
)
from egomimic.models.residual_joint_transformer import ResidualJointTransformer


def _policy():
    return ResidualJointTransformer(
        action_horizon=100,
        control_horizon=30,
        infer_ac_dims={"piper_left_arm": 7},
        input_dim=16,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        fk_position_loss_weight=0.0,
        fk_rotation_loss_weight=0.0,
    )


def _data(batch=2):
    action_low = torch.tensor([-2.0] * 6 + [0.0]).repeat(100, 1)
    action_high = torch.tensor([2.0] * 6 + [1.0]).repeat(100, 1)
    proprio_low = torch.tensor([-2.0] * 6 + [0.0])
    proprio_high = torch.tensor([2.0] * 6 + [1.0])
    action = torch.zeros(batch, 100, 7)
    action[..., 6] = 1.0
    state = torch.zeros(batch, 1, 7)
    state[..., 6] = 1.0
    return {
        "action": action,
        "state_joints": state,
        "_action_norm_mode": "quantile",
        "_action_norm_quantile_1": action_low,
        "_action_norm_quantile_99": action_high,
        "_proprio_norm_mode": "quantile",
        "_proprio_norm_quantile_1": proprio_low,
        "_proprio_norm_quantile_99": proprio_high,
    }


def test_initial_prediction_is_persistence_with_binary_open_gripper():
    policy = _policy().eval()
    features = torch.randn(2, 8, 16)
    output = policy((features, "piper_left_arm"), data=_data())
    assert output.shape == (2, 100, 7)
    torch.testing.assert_close(output[..., :6], torch.zeros_like(output[..., :6]))
    torch.testing.assert_close(output[..., 6], torch.ones_like(output[..., 6]))


def test_loss_is_finite_and_reaches_decoder():
    policy = _policy().train()
    policy._fk_losses = lambda predicted, target: (
        predicted.new_zeros(()),
        predicted.new_zeros(()),
    )
    features = torch.randn(2, 8, 16, requires_grad=True)
    data = _data()
    # Create a non-trivial physical target while preserving binary gripper.
    data["action"][..., 0] = 0.2
    data["action"][:, 50:, 6] = -1.0
    loss = policy.compute_loss(features, data)
    assert torch.isfinite(loss)
    loss.backward()
    assert policy.joint_residual_head.weight.grad is not None
    assert policy.gripper_head.weight.grad is not None
    assert torch.isfinite(policy.joint_residual_head.weight.grad).all()
    assert torch.isfinite(policy.gripper_head.weight.grad).all()
    assert set(policy.last_loss_components) == {
        "residual_joint_position_loss",
        "residual_joint_delta_loss",
        "residual_joint_acceleration_loss",
        "residual_joint_jerk_loss",
        "residual_joint_fk_position_loss",
        "residual_joint_fk_rotation_loss",
        "residual_joint_gripper_bce",
        "residual_joint_total_loss",
    }


def test_action_and_proprio_normalizers_are_not_interchanged():
    policy = _policy().eval()
    data = _data(batch=1)
    data["state_joints"][..., 0] = 0.5
    current = policy._current_raw(data, data["action"])
    # Proprio range [-2, 2]: normalized 0.5 maps to +1 radian.
    torch.testing.assert_close(current[0, 0], torch.tensor(1.0), atol=1e-6, rtol=0)


def test_residual_joint_contract_identity_and_determinism():
    policy = _policy().eval()
    assert policy.contract_name == "piper_residual_joint_transformer"
    assert policy.contract_version == 2
    features = torch.randn(1, 8, 16)
    data = _data(batch=1)
    first = policy((features, "piper_left_arm"), data=data)
    second = policy((features, "piper_left_arm"), data=data)
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_pre_rename_checkpoint_import_resolves_to_current_head():
    assert ResidualJointPolicyCleanV2 is ResidualJointTransformer


def test_contract_rejects_wrong_action_width():
    policy = _policy().eval()
    data = _data(batch=1)
    data["action"] = data["action"][..., :6]
    try:
        policy((torch.randn(1, 8, 16), "piper_left_arm"), data=data)
    except ValueError as error:
        assert "action contract mismatch" in str(error)
    else:
        raise AssertionError(
            "Residual Joint Transformer accepted a 6D action without gripper"
        )


if __name__ == "__main__":
    test_initial_prediction_is_persistence_with_binary_open_gripper()
    test_loss_is_finite_and_reaches_decoder()
    test_action_and_proprio_normalizers_are_not_interchanged()
    test_residual_joint_contract_identity_and_determinism()
    test_pre_rename_checkpoint_import_resolves_to_current_head()
    test_contract_rejects_wrong_action_width()
    print("ResidualJointTransformer tests passed")
