import numpy as np
import pytest

from egomimic.rldb.embodiment.piper_residual_joint import (
    AriaLeftPoseResidualJoint,
    PiperResidualJoint,
)


def _apply(batch, transforms):
    for transform in transforms:
        batch = transform.transform(batch)
    return batch


def _poses(length):
    pose = np.zeros((length, 7), dtype=np.float32)
    pose[:, 0] = np.linspace(0.0, 0.1, length)
    pose[:, 3] = 1.0
    return pose


def test_piper_mixed_shapes_and_binary_gripper():
    raw = np.zeros((30, 7), dtype=np.float32)
    raw[:10, 6] = 100.0
    batch = {
        "right.cmd_ee_pose": _poses(30),
        "right.cmd_joints": np.zeros((30, 6), dtype=np.float32),
        "source.action": raw,
        "right.obs_joints": np.zeros(6, dtype=np.float32),
        "right.obs_gripper": np.array([0.1], dtype=np.float32),
    }
    result = _apply(batch, PiperResidualJoint.get_transform_list(mixed=True))
    assert result["actions_cartesian_pose"].shape == (100, 9)
    assert result["actions_joints"].shape == (30, 7)
    assert result["observations.state.joints"].shape == (7,)
    assert set(np.unique(result["actions_joints"][:, 6])) <= {0.0, 1.0}


def test_piper_joint_and_gripper_labels_are_native_frames():
    raw = np.zeros((30, 7), dtype=np.float32)
    raw[1::2, 6] = 100.0  # alternate every frame: any resampling would show
    joints = np.arange(30 * 6, dtype=np.float32).reshape(30, 6)
    batch = {
        "right.cmd_ee_pose": _poses(30),
        "right.cmd_joints": joints.copy(),
        "source.action": raw,
        "right.obs_joints": np.zeros(6, dtype=np.float32),
        "right.obs_gripper": np.array([0.1], dtype=np.float32),
    }
    result = _apply(batch, PiperResidualJoint.get_transform_list(mixed=True))
    np.testing.assert_array_equal(result["actions_joints"][:, :6], joints)
    np.testing.assert_array_equal(
        result["actions_joints"][:, 6], (np.arange(30) % 2 == 1).astype(np.float32)
    )


@pytest.mark.parametrize(
    ("pose_label", "zarr_key"),
    [("cmd", "right.cmd_ee_pose_camframe"), ("obs", "right.obs_ee_pose_camframe")],
)
def test_piper_pose_label_selects_zarr_key(pose_label, zarr_key):
    mapping = PiperResidualJoint.get_keymap(mixed=True, pose_label=pose_label)
    assert mapping["right.cmd_ee_pose"]["zarr_key"] == zarr_key
    assert mapping["right.cmd_joints"]["zarr_key"] == "right.cmd_joints"


def test_piper_pose_label_rejects_unknown_value():
    with pytest.raises(ValueError, match="pose_label"):
        PiperResidualJoint.get_keymap(mixed=True, pose_label="future")


def test_aria_shared_pose_shapes():
    head = np.zeros(7, dtype=np.float32)
    head[3] = 1.0
    hand = _poses(30)
    result = _apply(
        {
            "head_pose": head,
            "left.obs_ee_pose": hand[0],
            "left.action_ee_pose": hand,
        },
        AriaLeftPoseResidualJoint.get_transform_list(),
    )
    assert result["observations.state.ee_pose"].shape == (9,)
    assert result["actions_cartesian_pose"].shape == (100, 9)
