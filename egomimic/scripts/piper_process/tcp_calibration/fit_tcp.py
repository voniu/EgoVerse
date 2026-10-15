"""Fit TCP, joint-zero and extrinsic corrections to labelled gripper centres.

Each label is the midpoint of the two clicked finger pads. Candidate models are
compared by leave-one-frame-out reprojection RMS, so extra parameters cannot win
by overfitting:
  M0 current: FK @ TCP of the setup, stored extrinsics, no joint offsets
  M1 + TCP offset in the link6 frame (3)
  M2 + M1 + small extrinsic correction (6)
  M3 + M1 + joint zero offsets of --fit-joints
  M4 all of the above
Writes <workdir>/fit_report.json; --write-correction stores one model as the
correction JSON read by apply_pose_correction.

    python -m egomimic.scripts.piper_process.tcp_calibration.fit_tcp \
        --dataset <piper_zarr_dir> --workdir <tcp_workdir> --calibration <setup.yaml> \
        [--fit-joints 4] [--write-correction correction.json --model M3 --name <name>]
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import zarr
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from egomimic.robot.piper.config import load_config
from egomimic.robot.piper_kinematics import PiperKinematics

ROOT = Path(__file__).resolve().parents[4]
MODELS = {
    "M0": dict(tcp=False, ext=False, joint=False),
    "M1": dict(tcp=True, ext=False, joint=False),
    "M2": dict(tcp=True, ext=True, joint=False),
    "M3": dict(tcp=True, ext=False, joint=True),
    "M4": dict(tcp=True, ext=True, joint=True),
}


class Problem:
    """Rows: dicts with q (model rad), K, T_base_cam, target (px), depth_m."""

    def __init__(self, fk, tcp0, rows, fit_joints=range(1, 7)):
        self.fk, self.tcp0, self.rows = fk, np.asarray(tcp0, float), rows
        self.joint_mask = np.isin(np.arange(1, 7), list(fit_joints))

    def project(self, params, use, rows=None):
        tcp = self.tcp0 + (params[0:3] if use["tcp"] else 0)
        d_ext = np.eye(4)
        if use["ext"]:
            d_ext[:3, :3] = Rotation.from_rotvec(params[3:6]).as_matrix()
            d_ext[:3, 3] = params[6:9]
        offsets = params[9:15] * self.joint_mask if use["joint"] else np.zeros(6)
        out = []
        for row in self.rows if rows is None else rows:
            T_base_link6 = self.fk.fk_from_feedback_deg(np.rad2deg(row["q"] + offsets))
            p_base = T_base_link6 @ np.r_[tcp, 1.0]
            p_cam = np.linalg.inv(row["T_base_cam"] @ d_ext) @ p_base
            uvw = row["K"] @ p_cam[:3]
            out.append(uvw[:2] / uvw[2])
        return np.array(out)

    def residual(self, params, use, rows):
        targets = np.array([r["target"] for r in rows])
        res = (self.project(params, use, rows) - targets).ravel()
        # weak priors keep unobservable directions near the current calibration
        prior = (
            0.5
            * np.r_[
                params[0:3] / 0.05,
                params[3:6] / np.deg2rad(5),
                params[6:9] / 0.05,
                params[9:15] / np.deg2rad(5),
            ]
        )
        active = np.r_[
            [use["tcp"]] * 3, [use["ext"]] * 6, self.joint_mask & use["joint"]
        ]
        return np.r_[res, prior[active]]

    def fit(self, use, rows=None):
        rows = self.rows if rows is None else rows
        if not any(use.values()):
            return np.zeros(15)
        return least_squares(
            self.residual, np.zeros(15), args=(use, rows), loss="huber", f_scale=5.0
        ).x

    def evaluate(self, use, leave_one_out=True):
        params = self.fit(use)
        targets = np.array([r["target"] for r in self.rows])
        err = np.linalg.norm(self.project(params, use) - targets, axis=1)
        loo = []
        if leave_one_out:
            for k, row in enumerate(self.rows):
                p = self.fit(use, self.rows[:k] + self.rows[k + 1 :])
                loo.append(
                    np.linalg.norm(self.project(p, use, [row])[0] - row["target"])
                )
        mm = (
            err
            / np.array([r["K"][0, 0] for r in self.rows])
            * np.array([r["depth_m"] for r in self.rows])
        )
        return params, {
            "train_rms_px": round(float(np.sqrt((err**2).mean())), 2),
            "loo_rms_px": round(float(np.sqrt(np.mean(np.square(loo)))), 2)
            if loo
            else None,
            "median_err_mm_at_depth": round(float(np.median(mm) * 1000), 1),
            "tcp_offset_mm": (params[0:3] * 1000).round(1).tolist()
            if use["tcp"]
            else None,
            "extrinsic_rot_deg": float(np.rad2deg(np.linalg.norm(params[3:6])))
            if use["ext"]
            else None,
            "extrinsic_trans_mm": (params[6:9] * 1000).round(1).tolist()
            if use["ext"]
            else None,
            "joint_offsets_deg": np.rad2deg(params[9:15] * self.joint_mask)
            .round(2)
            .tolist()
            if use["joint"]
            else None,
        }


def load_rows(dataset, frames_path, labels_path):
    frames = {f["id"]: f for f in json.loads(frames_path.read_text())}
    rows = []
    for fid, label in json.loads(labels_path.read_text()).items():
        if label.get("skip") or "right" not in label:
            continue
        f = frames[fid]
        group = zarr.open_group(str(dataset / f["episode"]), mode="r")
        rows.append(
            {
                "id": fid,
                "q": np.asarray(group["right.obs_joints"][f["frame"]], float),
                "K": np.asarray(group.attrs["intrinsics"]["front_1"])[:, :3],
                "T_base_cam": np.asarray(group.attrs["extrinsics"]["right"], float),
                "target": (np.asarray(label["left"]) + np.asarray(label["right"])) / 2,
                "depth_m": f["depth_m"],
            }
        )
    return rows


def correction_record(name, model, problem, params, report, labels_path):
    use = MODELS[model]
    tcp_offset = params[0:3] if use["tcp"] else np.zeros(3)
    joint_offsets = params[9:15] * problem.joint_mask if use["joint"] else np.zeros(6)
    return {
        "name": name,
        "date": datetime.date.today().isoformat(),
        "model": "T_base_tcp = FK(q_feedback_deg + joint_offsets_deg) @ Trans(tcp_link6_m)",
        "tcp_link6_m": (problem.tcp0 + tcp_offset).tolist(),
        "tcp_offset_from_nominal_mm": (tcp_offset * 1000).round(2).tolist(),
        "joint_offsets_deg": np.rad2deg(joint_offsets).tolist(),
        "fit": {
            "model": model,
            "frames": len(problem.rows),
            "train_rms_px": report[model]["train_rms_px"],
            "loo_rms_px": report[model]["loo_rms_px"],
            "before_rms_px": report["M0"]["train_rms_px"],
            "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument(
        "--calibration", type=Path, help="Setup file (default: $PIPER_SETUP)"
    )
    parser.add_argument(
        "--fit-joints",
        type=int,
        nargs="+",
        default=list(range(1, 7)),
        help="joints (1-6) whose zero offset M3/M4 may fit",
    )
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--write-correction", type=Path)
    parser.add_argument("--model", choices=sorted(MODELS), default="M3")
    parser.add_argument("--name", default="tcp_fit")
    args = parser.parse_args()
    if any(not 1 <= j <= 6 for j in args.fit_joints):
        parser.error("--fit-joints must be in 1..6")

    config = load_config(args.calibration)
    fk = PiperKinematics(ROOT / config["kinematics"]["reference_urdf_path"])
    labels_path = args.workdir / "labels.json"
    rows = load_rows(args.dataset, args.workdir / "frames.json", labels_path)
    if len(rows) < args.min_frames:
        raise SystemExit(
            f"need at least {args.min_frames} labelled frames, have {len(rows)}"
        )
    print(f"{len(rows)} labelled frames, joint offsets fitted for J{args.fit_joints}")
    problem = Problem(fk, config["tcp"]["translation_m"], rows, args.fit_joints)
    report, fitted = {}, {}
    for model, use in MODELS.items():
        fitted[model], report[model] = problem.evaluate(use)
        r = report[model]
        print(
            f"{model} train {r['train_rms_px']:6.1f}px  LOO {r['loo_rms_px'] or 0:6.1f}px"
            f"  ~{r['median_err_mm_at_depth']:5.1f} mm"
        )
    out = args.workdir / "fit_report.json"
    out.write_text(json.dumps({"fit_joints": args.fit_joints, **report}, indent=1))
    print("report ->", out)
    if args.write_correction:
        record = correction_record(
            args.name, args.model, problem, fitted[args.model], report, labels_path
        )
        args.write_correction.write_text(json.dumps(record, indent=1))
        print("correction ->", args.write_correction)


if __name__ == "__main__":
    main()
