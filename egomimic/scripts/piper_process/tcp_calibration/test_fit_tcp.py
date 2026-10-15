import numpy as np

from egomimic.robot.piper.config import load_config
from egomimic.robot.piper_kinematics import PiperKinematics
from egomimic.scripts.piper_process.tcp_calibration.fit_tcp import (
    MODELS,
    ROOT,
    Problem,
)


def look_at(camera, target):
    z = (target - camera) / np.linalg.norm(target - camera)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((x, np.cross(z, x), z))
    transform[:3, 3] = camera
    return transform


def synthetic_problem(tcp_offset, joint_offsets_deg, frames=30):
    config = load_config()
    fk = PiperKinematics(ROOT / config["kinematics"]["reference_urdf_path"])
    tcp0 = np.asarray(config["tcp"]["translation_m"])
    K = np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
    T_base_cam = look_at(np.array([0.9, 0.0, 0.5]), np.array([0.3, 0.0, 0.15]))
    rng = np.random.default_rng(0)
    low = np.deg2rad([-60, 60, -110, -20, 20, -100])
    high = np.deg2rad([10, 120, -50, 100, 85, 10])
    truth = Problem(fk, tcp0 + tcp_offset, [])
    params = np.zeros(15)
    params[9:15] = np.deg2rad(joint_offsets_deg)
    rows = []
    while len(rows) < frames:
        q = rng.uniform(low, high)
        row = {"q": q, "K": K, "T_base_cam": T_base_cam, "target": None, "depth_m": 1.0}
        p_base = (
            fk.fk_from_feedback_deg(np.rad2deg(q) + joint_offsets_deg)
            @ np.r_[tcp0 + tcp_offset, 1]
        )
        if (np.linalg.inv(T_base_cam) @ p_base)[2] <= 0.1:
            continue
        row["target"] = truth.project(params, MODELS["M3"], [row])[0]
        rows.append(row)
    return fk, tcp0, rows


def test_fit_recovers_tcp_offset_and_one_joint_zero() -> None:
    tcp_offset = np.array([-0.02, 0.03, 0.01])
    joint_offsets = np.array([0, 0, 0, 8.0, 0, 0])
    fk, tcp0, rows = synthetic_problem(tcp_offset, joint_offsets)
    problem = Problem(fk, tcp0, rows, fit_joints=[4])

    params, report = problem.evaluate(MODELS["M3"], leave_one_out=False)

    assert np.allclose(params[0:3], tcp_offset, atol=2e-3)
    assert abs(np.rad2deg(params[12]) - 8.0) < 0.5
    assert np.allclose(params[[9, 10, 11, 13, 14]], 0)
    assert report["train_rms_px"] < 1.0
    _, before = problem.evaluate(MODELS["M0"], leave_one_out=False)
    assert before["train_rms_px"] > 20
