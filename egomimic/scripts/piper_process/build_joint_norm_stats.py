"""Add full-train Piper joint statistics to existing A/B pose statistics."""

import argparse
import json
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset, split_dataset_names


def read_training_arrays(dataset_dir: Path, horizon: int):
    paths = sorted(dataset_dir.glob("*.zarr"))
    train, _ = split_dataset_names([path.stem for path in paths], 0.1, seed=42)
    states, joint_actions, cartesian_actions = [], [], []
    for path in paths:
        if path.stem not in train:
            continue
        group = zarr.open_group(str(path), mode="r")
        frames = int(group.attrs["total_frames"])
        obs = np.column_stack(
            (group["right.obs_joints"][:frames], group["right.obs_gripper"][:frames])
        ).astype(np.float32)
        cmd = np.column_stack(
            (group["right.cmd_joints"][:frames], group["right.cmd_gripper"][:frames])
        ).astype(np.float32)
        pose = np.asarray(group["right.cmd_ee_pose_camframe"][:frames])
        quaternion_xyzw = pose[:, [4, 5, 6, 3]]
        ypr = Rotation.from_quat(quaternion_xyzw).as_euler("ZYX")
        cartesian = np.column_stack((pose[:, :3], ypr, cmd[:, 6])).astype(np.float32)
        indices = np.minimum(
            np.arange(frames)[:, None] + np.arange(horizon)[None, :], frames - 1
        )
        states.append(obs)
        joint_actions.append(cmd[indices])
        cartesian_actions.append(cartesian[indices])
        print(path.name, frames, flush=True)
    return (
        np.concatenate(states),
        np.concatenate(joint_actions),
        np.concatenate(cartesian_actions),
        len(train),
    )


def serializable_stats(values):
    return {
        key: np.asarray(value).tolist()
        for key, value in MultiDataset._compute_stats_for_array(values).items()
    }


def write_stats(base_path, output_path, robot_stats, frames, episodes):
    payload = json.loads(base_path.read_text())
    payload["stats"]["7"] = robot_stats
    payload["frames"] = frames
    payload["joint_stats"] = {
        "source": "all frames in the fixed seed-42 Piper training split",
        "episodes": episodes,
        "horizon": 30,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-a", type=Path, required=True)
    parser.add_argument("--base-b", type=Path, required=True)
    parser.add_argument("--output-a", type=Path, required=True)
    parser.add_argument("--output-b", type=Path, required=True)
    args = parser.parse_args()

    states, actions, cartesian_actions, episodes = read_training_arrays(
        args.dataset, horizon=30
    )
    robot_stats = {
        "observations.state.joints": serializable_stats(states),
        "actions_cartesian": serializable_stats(cartesian_actions),
        "actions_joints": serializable_stats(actions),
    }
    for output in (args.output_a, args.output_b):
        output.parent.mkdir(parents=True, exist_ok=True)
    write_stats(args.base_a, args.output_a, robot_stats, len(states), episodes)
    write_stats(args.base_b, args.output_b, robot_stats, len(states), episodes)
    print(
        f"wrote {args.output_a} and {args.output_b}: {episodes} episodes, {len(states)} frames"
    )


if __name__ == "__main__":
    main()
