"""Smooth Piper command trajectories in a disposable Zarr dataset copy."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
import zarr
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from egomimic.robot.piper_kinematics import PiperKinematics

ROOT = Path(__file__).resolve().parents[3]
CALIBRATION = ROOT / "egomimic/robot/piper/calibration.yaml"


def transform(spec):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(spec["quaternion_xyzw"]).as_matrix()
    matrix[:3, 3] = spec["translation_m"]
    return matrix


def poses(matrices):
    quaternion = Rotation.from_matrix(matrices[:, :3, :3]).as_quat()
    return np.column_stack((matrices[:, :3, 3], quaternion[:, [3, 0, 1, 2]])).astype(
        np.float32
    )


def episode_base_camera(group, config):
    """T_base_camera that this episode's camera-frame poses were built with.

    The converter stores it in the Zarr metadata; the repository calibration no
    longer carries D435IF extrinsics because the camera moves between setups.
    """
    stored = group.attrs.get("extrinsics", {}).get("right")
    if stored is not None:
        matrix = np.asarray(stored, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"extrinsics.right must be 4x4, got {matrix.shape}")
        return matrix
    extrinsics = config.get("cameras", {}).get("d435if", {}).get("extrinsics")
    if extrinsics is None:
        raise ValueError(
            "Episode has no stored extrinsics and --calibration lacks "
            "cameras.d435if.extrinsics"
        )
    return transform(extrinsics)


def smooth(values, window):
    filtered = median_filter(values, size=(3, 1), mode="nearest")
    result = savgol_filter(filtered, window, 2, axis=0, mode="interp")
    return np.clip(result, values.min(axis=0), values.max(axis=0)).astype(np.float32)


def trajectory_metrics(values):
    velocity = np.rad2deg(np.diff(values, axis=0))
    acceleration = np.diff(velocity, axis=0)
    return {
        "step_deg": np.max(np.abs(velocity), axis=1),
        "velocity_change_deg": np.max(np.abs(acceleration), axis=1),
    }


def percentiles(values):
    return {
        str(p): float(np.percentile(values, p)) for p in (50, 90, 95, 99, 99.9, 100)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--joint-window", type=int, default=9)
    parser.add_argument("--gripper-window", type=int, default=5)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if (
        min(args.joint_window, args.gripper_window) < 3
        or args.joint_window % 2 != 1
        or args.gripper_window % 2 != 1
    ):
        parser.error("smoothing windows must be odd and at least 3")

    config = yaml.safe_load(args.calibration.read_text())
    kinematics = config["kinematics"]
    fk = PiperKinematics(ROOT / kinematics["reference_urdf_path"])
    tcp = transform(config["tcp"])

    before = {"step_deg": [], "velocity_change_deg": []}
    after = {"step_deg": [], "velocity_change_deg": []}
    changes = []
    paths = sorted(args.dataset.glob("*.zarr"))
    for index, path in enumerate(paths, 1):
        group = zarr.open_group(str(path), mode="r+")
        frames = int(group.attrs["total_frames"])
        base_camera = episode_base_camera(group, config)
        joints = np.asarray(group["right.cmd_joints"][:frames], dtype=np.float32)
        gripper = np.asarray(group["right.cmd_gripper"][:frames], dtype=np.float32)
        smoothed_joints = smooth(joints, args.joint_window)
        smoothed_gripper = smooth(gripper, args.gripper_window)

        base_command = fk.fk_from_feedback_deg(np.rad2deg(smoothed_joints)) @ tcp
        camera_command = np.linalg.inv(base_camera) @ base_command
        group["right.cmd_joints"][:frames] = smoothed_joints
        group["right.cmd_gripper"][:frames] = smoothed_gripper
        group["right.cmd_ee_pose"][:frames] = poses(base_command)
        group["right.cmd_ee_pose_camframe"][:frames] = poses(camera_command)
        group.attrs["action_smoothing"] = {
            "method": "median3+savgol",
            "joint_window": args.joint_window,
            "gripper_window": args.gripper_window,
            "polyorder": 2,
            "fps": 30,
        }

        for key, values in trajectory_metrics(joints).items():
            before[key].append(values)
        for key, values in trajectory_metrics(smoothed_joints).items():
            after[key].append(values)
        changes.append(np.abs(np.rad2deg(smoothed_joints - joints)).reshape(-1))
        print(f"[{index}/{len(paths)}] {path.name}: {frames} frames", flush=True)

    report = {
        "method": "median3+savgol",
        "episodes": len(paths),
        "joint_window": args.joint_window,
        "gripper_window": args.gripper_window,
        "polyorder": 2,
        "fps": 30,
        "before": {
            key: percentiles(np.concatenate(value)) for key, value in before.items()
        },
        "after": {
            key: percentiles(np.concatenate(value)) for key, value in after.items()
        },
        "joint_change_deg": percentiles(np.concatenate(changes)),
    }
    output = args.report or args.dataset / "action_smoothing_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
