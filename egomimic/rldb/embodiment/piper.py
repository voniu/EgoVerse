"""Piper single-arm samples: camera-frame TCP xyz/ypr and gripper travel.

Position and gripper travel are meters; intrinsic ZYX angles are radians.
Native link6 axes are retained. Human tool-axis alignment is not applied here.
"""

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.zarr.action_chunk_transforms import (
    ConcatKeys,
    DropKeys,
    XYZWXYZ_to_XYZYPR,
)


class Piper(Embodiment):
    @classmethod
    def get_keymap(
        cls,
        keymap_mode="cartesian",
        norm_mode=False,
        annotation_key=None,
        action_horizon=30,
        joint_head=False,
    ):
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        mapping = super().get_keymap(keymap_mode, norm_mode, annotation_key)
        for spec in mapping.values():
            if spec["key_type"] == "action_keys":
                spec["horizon"] = action_horizon
        if joint_head:
            # The direct-control variant observes hardware joint state. The
            # Cartesian pose remains a supervised auxiliary output, not an
            # additional proprio input computed from the same joints.
            mapping.pop("right.obs_ee_pose")
            mapping.update(
                {
                    "right.obs_joints": {
                        "key_type": "proprio_keys",
                        "zarr_key": "right.obs_joints",
                    },
                    "right.cmd_joints": {
                        "key_type": "action_keys",
                        "zarr_key": "right.cmd_joints",
                        "horizon": action_horizon,
                    },
                }
            )
        return mapping

    @classmethod
    def _get_keymap(cls, keymap_mode):
        if keymap_mode != "cartesian":
            raise ValueError(f"Unsupported Piper keymap: {keymap_mode}")
        return {
            cls.VIZ_IMAGE_KEY: {
                "key_type": "camera_keys",
                "zarr_key": "images.front_1",
            },
            "observations.images.right_wrist_img": {
                "key_type": "camera_keys",
                "zarr_key": "images.right_wrist",
            },
            "right.obs_ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "right.obs_ee_pose_camframe",
            },
            "right.obs_gripper": {
                "key_type": "proprio_keys",
                "zarr_key": "right.obs_gripper",
            },
            "right.cmd_ee_pose": {
                "key_type": "action_keys",
                "zarr_key": "right.cmd_ee_pose_camframe",
            },
            "right.cmd_gripper": {
                "key_type": "action_keys",
                "zarr_key": "right.cmd_gripper",
            },
        }

    @staticmethod
    def get_transform_list(joint_head=False):
        # ZarrDataset handles JPEG decoding, last-command padding and float32 tensors.
        if joint_head:
            return [
                XYZWXYZ_to_XYZYPR(keys=["right.cmd_ee_pose"]),
                ConcatKeys(
                    ["right.cmd_ee_pose", "right.cmd_gripper"],
                    "actions_cartesian",
                    delete_old_keys=False,
                ),
                ConcatKeys(
                    ["right.obs_joints", "right.obs_gripper"],
                    "observations.state.joints",
                    delete_old_keys=True,
                ),
                ConcatKeys(
                    ["right.cmd_joints", "right.cmd_gripper"],
                    "actions_joints",
                    delete_old_keys=True,
                ),
                DropKeys(["right.cmd_ee_pose"]),
            ]
        transforms = [
            XYZWXYZ_to_XYZYPR(keys=["right.obs_ee_pose", "right.cmd_ee_pose"]),
            ConcatKeys(
                ["right.obs_ee_pose", "right.obs_gripper"],
                "observations.state.ee_pose",
                delete_old_keys=not joint_head,
            ),
            ConcatKeys(
                ["right.cmd_ee_pose", "right.cmd_gripper"],
                "actions_cartesian",
                delete_old_keys=not joint_head,
            ),
        ]
        return transforms
