import numpy as np
import pytest

from egomimic.robot.piper.rollout_joint import (
    apply_joint_limits,
    dummy_action_shape,
    uniform_horizon_indices,
)


def test_uniform_horizon_indices_cover_the_complete_prediction() -> None:
    indices = uniform_horizon_indices(model_horizon=100, control_horizon=30)

    assert len(indices) == 30
    assert indices[0] == 0
    assert indices[-1] == 99
    assert np.all(np.diff(indices) > 0)


def test_dummy_action_shape_follows_each_key() -> None:
    pose_stats = {"mean": np.zeros((100, 9))}
    joint_stats = {"mean": np.zeros((30, 7))}

    assert dummy_action_shape(pose_stats) == (100, 9)
    assert dummy_action_shape(joint_stats) == (30, 7)


def test_dummy_action_shape_rejects_flat_stats() -> None:
    with pytest.raises(ValueError, match="per-timestep"):
        dummy_action_shape({"mean": np.zeros(7)})


def test_apply_joint_limits_clips_small_overshoot() -> None:
    bounds = np.deg2rad(np.array([[0.0, 180.0], [-170.0, 0.0]]))
    target = np.deg2rad(np.array([-2.0, 1.0]))

    clipped, violation = apply_joint_limits(target, bounds, tolerance_deg=3.0)

    np.testing.assert_allclose(clipped, np.array([bounds[0, 0], bounds[1, 1]]))
    np.testing.assert_allclose(violation, np.array([2.0, 1.0]))


def test_apply_joint_limits_can_pass_small_overshoot() -> None:
    bounds = np.deg2rad(np.array([[0.0, 180.0], [-170.0, 0.0]]))
    target = np.deg2rad(np.array([-2.0, 1.0]))

    passed, _ = apply_joint_limits(
        target, bounds, tolerance_deg=3.0, allow_small_overshoot=True
    )

    np.testing.assert_allclose(passed, target)


def test_apply_joint_limits_rejects_large_overshoot() -> None:
    bounds = np.deg2rad(np.array([[0.0, 180.0], [-170.0, 0.0]]))
    target = np.deg2rad(np.array([-4.0, 1.0]))

    with pytest.raises(RuntimeError, match="J1 target exceeds URDF limit by 4.00deg"):
        apply_joint_limits(
            target, bounds, tolerance_deg=3.0, allow_small_overshoot=True
        )
