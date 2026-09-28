"""Single-left-hand samples; tool-axis convention is supplied by the converter."""

import numpy as np
from scipy.spatial.transform import Rotation

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.zarr.action_chunk_transforms import Transform
from egomimic.utils.pose_utils import _xyzwxyz_to_matrix


class CurrentCameraPoses(Transform):
    """Express the entire future chunk in the camera frame at its starting time."""

    def transform(self, batch):
        camera_world = np.linalg.inv(
            _xyzwxyz_to_matrix(batch.pop("head_pose")[None])[0]
        )
        for source, target in (
            ("left.obs_ee_pose", "observations.state.ee_pose"),
            ("left.action_ee_pose", "actions_cartesian"),
        ):
            pose = batch.pop(source)
            single = pose.ndim == 1
            matrices = camera_world @ _xyzwxyz_to_matrix(pose[None] if single else pose)
            angles = Rotation.from_matrix(matrices[:, :3, :3]).as_euler("ZYX")
            value = np.column_stack((matrices[:, :3, 3], angles)).astype(np.float32)
            batch[target] = value[0] if single else value
        return batch


class AriaGen2Left(Embodiment):
    @classmethod
    def get_keymap(cls, norm_mode=False, action_horizon=30):
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
            del mapping[cls.VIZ_IMAGE_KEY]
        return mapping

    @staticmethod
    def get_transform_list():
        return [CurrentCameraPoses()]
