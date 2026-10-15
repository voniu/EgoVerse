"""Recompute Piper TCP pose fields in converted Zarr episodes with a kinematic correction.

Rewrites right.{obs,cmd}_ee_pose and right.{obs,cmd}_ee_pose_camframe from the
stored joints, using FK(q_deg + joint_offsets_deg) @ Trans(tcp_link6_m) and each
episode's stored extrinsics.right. Joints, gripper and images are untouched. The
correction is recorded in attrs["pose_correction"]; an episode that already carries
one is refused, so it cannot be applied twice. Any later IK or FK on these labels
must use the same correction.

    python -m egomimic.scripts.piper_process.tcp_calibration.apply_pose_correction \
        --dataset <piper_zarr_dir> --correction correction.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from egomimic.robot.piper.config import load_config
from egomimic.robot.piper_kinematics import PiperKinematics

ROOT = Path(__file__).resolve().parents[4]


def poses(matrices):
    """[x, y, z, qw, qx, qy, qz] rows from 4x4 transforms."""
    quat = Rotation.from_matrix(matrices[:, :3, :3]).as_quat()
    return np.column_stack((matrices[:, :3, 3], quat[:, [3, 0, 1, 2]])).astype(
        np.float32
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--correction", type=Path, required=True)
    parser.add_argument(
        "--calibration", type=Path, help="Setup file (default: $PIPER_SETUP)"
    )
    args = parser.parse_args()
    correction = json.loads(args.correction.read_text())
    config = load_config(args.calibration)
    fk = PiperKinematics(ROOT / config["kinematics"]["reference_urdf_path"])
    offsets_deg = np.asarray(correction["joint_offsets_deg"], float)
    tcp = np.eye(4)
    tcp[:3, 3] = correction["tcp_link6_m"]

    episodes = sorted(args.dataset.glob("*.zarr"))
    for path in episodes:
        if "pose_correction" in zarr.open_group(str(path), mode="r").attrs:
            raise SystemExit(
                f"{path.name} already has a pose_correction; reconvert first"
            )

    shift = []
    for k, path in enumerate(episodes, 1):
        group = zarr.open_group(str(path), mode="r+")
        n = int(group.attrs["total_frames"])
        camera_base = np.linalg.inv(
            np.asarray(group.attrs["extrinsics"]["right"], float)
        )
        for side in ("obs", "cmd"):
            q = np.asarray(group[f"right.{side}_joints"][:n], float)
            base = fk.fk_from_feedback_deg(np.rad2deg(q) + offsets_deg) @ tcp
            before = np.asarray(group[f"right.{side}_ee_pose_camframe"][:n, :3])
            group[f"right.{side}_ee_pose"][:n] = poses(base)
            group[f"right.{side}_ee_pose_camframe"][:n] = poses(camera_base @ base)
            if side == "obs":
                shift.append(
                    np.linalg.norm((camera_base @ base)[:, :3, 3] - before, axis=1)
                )
        group.attrs["pose_correction"] = {
            **{
                k_: correction[k_]
                for k_ in ("name", "tcp_link6_m", "joint_offsets_deg")
            },
            "source": str(args.correction),
        }
        print(f"[{k}/{len(episodes)}] {path.name}", flush=True)

    shift = np.concatenate(shift) * 1000
    print(
        f"TCP position change mm: median {np.median(shift):.1f}, "
        f"p95 {np.percentile(shift, 95):.1f}, max {shift.max():.1f}"
    )


if __name__ == "__main__":
    main()
