import numpy as np
import yaml

from egomimic.robot.piper.config import check_cameras, load_config
from egomimic.robot.piper.initial_pose import initial_pose
from egomimic.robot.piper.realsense_setup import TEMPLATE, fill_setup, intrinsics_record


def record(serial, fx):
    return intrinsics_record(
        {"serial": serial, "device_name": "RealSense", "firmware": "1"},
        {
            "width": 640,
            "height": 480,
            "fx": fx,
            "fy": fx,
            "ppx": 320.0,
            "ppy": 240.0,
            "model": "distortion.inverse_brown_conrady",
            "coeffs": [0.0] * 5,
        },
    )


def test_filled_template_passes_the_camera_check(tmp_path) -> None:
    setup = fill_setup(
        yaml.safe_load(TEMPLATE.read_text()),
        record("111", 600.0),
        record("222", 390.0),
        "d405_222_intrinsics_640x480.json",
    )
    path = tmp_path / "setup.yaml"
    path.write_text(yaml.safe_dump(setup))

    config = load_config(path, require_cameras=True)

    check_cameras(config, path)
    assert config["cameras"]["d435if"]["intrinsics"]["K"][0][0] == 600.0
    assert config["cameras"]["d435if"]["intrinsics"]["distortion_model"] == (
        "inverse_brown_conrady"
    )
    assert config["cameras"]["d405"]["serial"] == "222"


def test_initial_pose_is_the_mean_first_frame_in_model_units() -> None:
    states = np.array(
        [
            [1000, 2000, -3000, 0, 4000, 0, 100000],
            [3000, 2000, -3000, 0, 4000, 0, 98000],
        ]
    )

    pose = initial_pose(states, load_config(), "test")

    assert np.allclose(np.rad2deg(pose["model_joints_rad"]), [2, 2, -3, 0, 4, 0])
    assert pose["gripper_m"] == 0.099
    assert pose["joint_std_deg"][0] == 1.0
    assert pose["episodes"] == 2
