"""Start pose for rollouts: mean first-frame follower state of the demonstrations.

Reads observation.state (raw SDK feedback: joints in 0.001 deg, gripper in 0.001 mm)
at frame 0 of every episode in a LeRobot v3 dataset and writes the JSON read by
`rollout_joint --initial-pose`. A large spread means the demonstrations did not share
one start pose; check it before using the mean.

    python -m egomimic.robot.piper.initial_pose \
        --dataset <lerobot_dataset> --output initial_pose.json [--calibration setup.yaml]
"""

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from egomimic.robot.piper.config import load_config


def first_frame_states(dataset):
    states = {}
    for path in sorted((dataset / "data").rglob("*.parquet")):
        table = pq.read_table(
            path, columns=["episode_index", "frame_index", "observation.state"]
        ).to_pandas()
        for row in table[table.frame_index == 0].itertuples(index=False):
            states[int(row.episode_index)] = np.asarray(row[2], dtype=float)
    if not states:
        raise ValueError(f"no frame-0 observation.state under {dataset}/data")
    return np.stack([states[k] for k in sorted(states)])


def initial_pose(states, config, source):
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] != 7:
        raise ValueError("expected seven-channel Piper states")
    k = config["kinematics"]
    scale = k["recorded_sdk_feedback"]["degrees_per_raw_joint_unit"]
    grip_scale = k["recorded_sdk_feedback"]["gripper_mm_per_raw_unit"] / 1000
    raw = states.mean(0)
    joints = np.deg2rad(raw[:6] * scale) * np.asarray(
        k["feedback_to_model"]["signs"]
    ) + np.asarray(k["feedback_to_model"]["offsets_rad"])
    spread_deg = (states[:, :6] * scale).std(0)
    return {
        "captured_at": datetime.date.today().isoformat(),
        "source": source,
        "episodes": int(len(states)),
        "joint_std_deg": spread_deg.round(3).tolist(),
        "gripper_std_mm": round(float((states[:, 6] * grip_scale * 1000).std()), 2),
        "raw_feedback": raw.round(1).tolist(),
        "model_joints_rad": joints.tolist(),
        "gripper_m": round(float(raw[6] * grip_scale), 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--calibration", type=Path, help="Setup file (default: $PIPER_SETUP)"
    )
    parser.add_argument("--port", default="can_follower")
    args = parser.parse_args()
    pose = initial_pose(
        first_frame_states(args.dataset),
        load_config(args.calibration),
        f"mean first-frame follower observation.state of {args.dataset.name}",
    )
    pose["port"] = args.port
    args.output.write_text(json.dumps(pose, indent=2))
    print(
        f"{pose['episodes']} episodes, joint std (deg) {pose['joint_std_deg']}, "
        f"gripper {pose['gripper_m'] * 1000:.1f} mm -> {args.output}"
    )


if __name__ == "__main__":
    main()
