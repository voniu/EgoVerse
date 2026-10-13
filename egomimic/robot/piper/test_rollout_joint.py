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


class _RecordingPiper:
    def __init__(self):
        self.gripper_calls = []

    def GripperCtrl(self, *args):
        self.gripper_calls.append(args)


def test_send_gripper_uses_configured_effort() -> None:
    from types import SimpleNamespace

    from egomimic.robot.piper.robot_interface import PiperHardware

    robot = object.__new__(PiperHardware)
    robot.piper = _RecordingPiper()
    robot.controlling = True
    robot.gripper_effort = 3000
    robot.geometry = SimpleNamespace(grip_scale=1e-6)

    robot.send_gripper(0.05)

    assert robot.piper.gripper_calls == [(50000, 3000, 0x01, 0)]


def test_gripper_effort_is_validated_before_connecting() -> None:
    from egomimic.robot.piper.robot_interface import PiperHardware

    with pytest.raises(ValueError, match="gripper_effort"):
        PiperHardware(geometry=None, gripper_effort=6000)


def _ramp_chunk(start=0.0, gripper=1.0):
    chunk = np.zeros((30, 7))
    chunk[:, 0] = start + np.arange(30)  # joint 1 advances 1 unit per action step
    chunk[:, 6] = gripper
    return chunk


def test_time_ensemble_interpolates_at_wall_clock_time() -> None:
    from egomimic.robot.piper.rollout_joint import time_ensemble

    chunk = _ramp_chunk()
    chunk[10:, 6] = 0.0
    action, sources = time_ensemble([(100.0, chunk)], 100.0 + 9.5 / 30, decay=0.05)

    assert sources == 1
    assert action[0] == pytest.approx(9.5)
    assert action[6] == 1.0  # gripper uses the row in effect, not an interpolation


def test_time_ensemble_weights_newer_chunks_by_age() -> None:
    from egomimic.robot.piper.rollout_joint import time_ensemble

    old, new = _ramp_chunk(0.0), _ramp_chunk(100.0)
    now = 10.0 + 10 / 30
    action, sources = time_ensemble([(10.0, old), (now, new)], now, decay=0.05)

    w_old, w_new = np.exp(-0.05 * 10), 1.0
    expected = (w_old * 10.0 + w_new * 100.0) / (w_old + w_new)
    assert sources == 2
    assert action[0] == pytest.approx(expected)


def test_time_ensemble_ignores_expired_chunks_and_falls_back() -> None:
    from egomimic.robot.piper.rollout_joint import time_ensemble

    expired = _ramp_chunk(0.0)
    action, sources = time_ensemble([(0.0, expired)], 5.0, decay=0.05)

    assert sources == 0
    np.testing.assert_allclose(action, expired[-1])
