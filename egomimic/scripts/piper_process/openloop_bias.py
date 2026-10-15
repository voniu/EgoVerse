"""Open-loop first-step bias of a Piper residual joint checkpoint.

Feeds recorded front/wrist images and joint state to the deploy policy
(`rollout_joint.JointPolicy.predict`) and compares the predicted joint chunk with
the recorded commands, mapped to TCP positions with the same FK. With a
residual head every replanning re-anchors at the current state, so a constant
first-step offset integrates on the robot; its mean and sign consistency are a
proxy to check before real-robot tests.

Phases per episode: "grasp" is the approach before the first gripper close,
"insert" the motion before the first release after it.

    python -m egomimic.scripts.piper_process.openloop_bias \
        --checkpoint run/train/checkpoints/step_step=27500.ckpt \
        --dataset /data/piper_zarr --episodes 6 7 15 22 39 42 66 \
        --output bias.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import simplejpeg
import torch
import zarr

from egomimic.robot.piper.config import load_config
from egomimic.robot.piper.rollout_joint import JointPolicy
from egomimic.robot.piper_kinematics import PiperKinematics

ROOT = Path(__file__).resolve().parents[3]


def tcp_fk(calibration_path=None):
    config = load_config(calibration_path)
    fk = PiperKinematics(ROOT / config["kinematics"]["reference_urdf_path"])
    tcp = np.r_[config["tcp"]["translation_m"], 1.0]

    def tcp_mm(q):
        q = np.rad2deg(np.atleast_2d(np.asarray(q, dtype=float)))
        return (fk.fk_from_feedback_deg(q) @ tcp)[:, :3] * 1000

    return tcp_mm


def phase_frames(binary_gripper, stride, window):
    closes = np.flatnonzero((binary_gripper[1:] == 0) & (binary_gripper[:-1] == 1)) + 1
    if not len(closes):
        return {}
    close = int(closes[0])
    opens = np.flatnonzero(
        (binary_gripper[close + 1 :] == 1) & (binary_gripper[close:-1] == 0)
    )
    phases = {"grasp": range(max(10, close - window), close - 5, stride)}
    if len(opens):
        release = int(opens[0] + close + 1)
        phases["insert"] = range(max(close + 10, release - window), release - 5, stride)
    return phases


def summarize(errors):
    errors = np.asarray(errors)
    return {
        "n": int(len(errors)),
        "mean_mm": errors.mean(0).round(2).tolist(),
        "share_negative": (errors < 0).mean(0).round(3).tolist(),
        "median_xy_mm": float(np.median(np.linalg.norm(errors[:, :2], axis=1))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Setup file (default: $PIPER_SETUP, else robot-model constants)",
    )
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument(
        "--window", type=int, default=150, help="frames before the event"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("openloop_bias needs CUDA (JointPolicy loads on cuda)")

    policy = JointPolicy(args.checkpoint)
    tcp_mm = tcp_fk(args.calibration)
    first, end = {}, {}
    for index in args.episodes:
        paths = sorted(args.dataset.glob(f"*episode_{index:03d}.zarr"))
        if len(paths) != 1:
            raise SystemExit(f"episode {index}: expected one Zarr, found {paths}")
        group = zarr.open_group(str(paths[0]), mode="r")
        n = int(group.attrs["total_frames"])
        q_obs = np.asarray(group["right.obs_joints"][:n], dtype=float)
        gripper = np.asarray(group["right.obs_gripper"][:n, 0], dtype=float)
        q_cmd = np.asarray(group["right.cmd_joints"][:n], dtype=float)
        binary = np.asarray(group["right.cmd_gripper_binary"][:n, 0])
        for phase, frames in phase_frames(binary, args.stride, args.window).items():
            for t in frames:
                images = [
                    simplejpeg.decode_jpeg(group[key][t : t + 1][0], colorspace="RGB")
                    for key in ("images.front_1", "images.right_wrist")
                ]
                state = np.r_[q_obs[t], gripper[t]].astype(np.float32)
                prediction = policy.predict(state, *images)[0][:, :6]
                horizon = min(len(prediction) - 1, n - 1 - t)
                first.setdefault(phase, []).append(
                    tcp_mm(prediction[0])[0] - tcp_mm(q_cmd[t])[0]
                )
                end.setdefault(phase, []).append(
                    tcp_mm(prediction[horizon])[0] - tcp_mm(q_cmd[t + horizon])[0]
                )
        print(f"episode {index:03d} done", flush=True)

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "step": int(policy.step),
        "episodes": args.episodes,
        "first_step": {phase: summarize(v) for phase, v in first.items()},
        "chunk_end": {phase: summarize(v) for phase, v in end.items()},
    }
    args.output.write_text(json.dumps(report, indent=2))
    for kind in ("first_step", "chunk_end"):
        for phase, item in report[kind].items():
            print(
                f"{kind:10s} {phase:6s} n={item['n']:3d} mean xyz {item['mean_mm']} mm, "
                f"share<0 {item['share_negative']}, median |xy| {item['median_xy_mm']:.1f} mm"
            )


if __name__ == "__main__":
    main()
