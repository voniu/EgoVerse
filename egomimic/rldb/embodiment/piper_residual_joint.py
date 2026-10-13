"""Residual Joint Transformer representations for residual joint control and human cotraining."""

import numpy as np

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.zarr.action_chunk_transforms import (
    ConcatKeys,
    InterpolatePose,
    Transform,
    XYZWXYZ_to_XYZ6D,
)
from egomimic.robot.piper.gripper_utils import binarize_gripper_commands
from egomimic.utils.pose_utils import _matrix_to_xyzwxyz, _xyzwxyz_to_matrix

RAW_HORIZON = 30
MODEL_HORIZON = 100
# Robot label for the shared pose head: leader command or observed follower pose.
POSE_LABEL_KEYS = {
    "cmd": "right.cmd_ee_pose_camframe",
    "obs": "right.obs_ee_pose_camframe",
}


class BinaryGripperIntent(Transform):
    """Recover binary open/close intent from raw LeRobot command percentages."""

    def transform(self, batch):
        raw_action = np.asarray(batch.pop("source.action"))
        batch["right.cmd_gripper_binary"] = binarize_gripper_commands(
            raw_action[:, 6:7] / 100.0
        )
        return batch


class PiperResidualJoint(Embodiment):
    """Piper state plus calibrated joint command and binary gripper intent."""

    @classmethod
    def get_keymap(
        cls, norm_mode=False, action_horizon=RAW_HORIZON, mixed=False, pose_label="cmd"
    ):
        if pose_label not in POSE_LABEL_KEYS:
            raise ValueError(
                f"pose_label must be one of {sorted(POSE_LABEL_KEYS)}, got {pose_label!r}"
            )
        mapping = {
            cls.VIZ_IMAGE_KEY: {
                "key_type": "camera_keys",
                "zarr_key": "images.front_1",
            },
            "observations.images.right_wrist_img": {
                "key_type": "camera_keys",
                "zarr_key": "images.right_wrist",
            },
            "right.obs_joints": {
                "key_type": "proprio_keys",
                "zarr_key": "right.obs_joints",
            },
            "right.obs_gripper": {
                "key_type": "proprio_keys",
                "zarr_key": "right.obs_gripper",
            },
            "right.cmd_joints": {
                "key_type": "action_keys",
                "zarr_key": "right.cmd_joints",
                "horizon": action_horizon,
            },
            "source.action": {
                "key_type": "action_keys",
                "zarr_key": "source.action",
                "horizon": action_horizon,
            },
        }
        if mixed:
            mapping["right.cmd_ee_pose"] = {
                "key_type": "action_keys",
                "zarr_key": POSE_LABEL_KEYS[pose_label],
                "horizon": action_horizon,
            }
        if norm_mode:
            mapping = {
                key: value
                for key, value in mapping.items()
                if value["key_type"] != "camera_keys"
            }
        return mapping

    @staticmethod
    def get_transform_list(mixed=False):
        transforms = []
        if mixed:
            transforms.extend(
                [
                    InterpolatePose(
                        MODEL_HORIZON,
                        "right.cmd_ee_pose",
                        "right.cmd_ee_pose",
                        mode="xyzwxyz",
                    ),
                    XYZWXYZ_to_XYZ6D(["right.cmd_ee_pose"]),
                    ConcatKeys(
                        ["right.cmd_ee_pose"],
                        "actions_cartesian_pose",
                        delete_old_keys=True,
                    ),
                ]
            )
        # Joint and gripper labels stay at the native 30 Hz frames. Resampling to
        # MODEL_HORIZON and picking 30 rounded indices back shifts joint labels by
        # up to 0.14 frame and moves 14 of 30 gripper labels one frame early.
        transforms.extend(
            [
                BinaryGripperIntent(),
                ConcatKeys(
                    ["right.cmd_joints", "right.cmd_gripper_binary"],
                    "actions_joints",
                    delete_old_keys=True,
                ),
                ConcatKeys(
                    ["right.obs_joints", "right.obs_gripper"],
                    "observations.state.joints",
                    delete_old_keys=True,
                ),
            ]
        )
        return transforms


class CurrentCameraPoses9D(Transform):
    """Express current and future hand poses in the current Aria camera frame."""

    def transform(self, batch):
        camera_from_world = np.linalg.inv(
            _xyzwxyz_to_matrix(np.asarray(batch.pop("head_pose"))[None])[0]
        )
        observation = np.asarray(batch.pop("left.obs_ee_pose"))
        action = np.asarray(batch.pop("left.action_ee_pose"))
        observation_camera = camera_from_world @ _xyzwxyz_to_matrix(observation[None])
        action_camera = camera_from_world @ _xyzwxyz_to_matrix(action)
        batch["observations.state.ee_pose"] = _matrix_to_xyzwxyz(observation_camera)[0]
        batch["actions_cartesian_pose"] = _matrix_to_xyzwxyz(action_camera)
        return batch


class AriaLeftPoseResidualJoint(Embodiment):
    """Aria left-hand state/action in the shared 9D camera-frame pose space."""

    @classmethod
    def get_keymap(cls, norm_mode=False, action_horizon=RAW_HORIZON):
        mapping = {
            cls.VIZ_IMAGE_KEY: {
                "key_type": "camera_keys",
                "zarr_key": "images.front_1",
            },
            "left.obs_ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "left.obs_ee_pose",
            },
            "left.action_ee_pose": {
                "key_type": "action_keys",
                "zarr_key": "left.obs_ee_pose",
                "horizon": action_horizon,
            },
            "head_pose": {"key_type": "proprio_keys", "zarr_key": "head_pose"},
        }
        if norm_mode:
            mapping.pop(cls.VIZ_IMAGE_KEY)
        return mapping

    @staticmethod
    def get_transform_list():
        return [
            CurrentCameraPoses9D(),
            InterpolatePose(
                MODEL_HORIZON,
                "actions_cartesian_pose",
                "actions_cartesian_pose",
                mode="xyzwxyz",
            ),
            XYZWXYZ_to_XYZ6D(["observations.state.ee_pose", "actions_cartesian_pose"]),
        ]
