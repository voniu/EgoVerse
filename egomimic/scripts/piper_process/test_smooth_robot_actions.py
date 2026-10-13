import numpy as np
import pytest

from egomimic.scripts.piper_process.smooth_robot_actions import episode_base_camera


class _Group:
    def __init__(self, attrs):
        self.attrs = attrs


def test_stored_episode_extrinsics_take_precedence():
    stored = np.eye(4)
    stored[:3, 3] = [0.1, 0.2, 0.3]
    config = {
        "cameras": {
            "d435if": {
                "extrinsics": {
                    "translation_m": [9.0, 9.0, 9.0],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            }
        }
    }
    group = _Group({"extrinsics": {"right": stored.tolist()}})
    np.testing.assert_allclose(episode_base_camera(group, config), stored)


def test_calibration_extrinsics_are_the_fallback():
    config = {
        "cameras": {
            "d435if": {
                "extrinsics": {
                    "translation_m": [0.0, 0.0, 1.0],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            }
        }
    }
    matrix = episode_base_camera(_Group({}), config)
    np.testing.assert_allclose(matrix[:3, 3], [0.0, 0.0, 1.0])


def test_missing_extrinsics_fail_loudly():
    with pytest.raises(ValueError, match="extrinsics"):
        episode_base_camera(_Group({}), {"cameras": {"d435if": {}}})
