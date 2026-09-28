"""Evaluate direct Piper joint heads on the fixed seven-episode robot split."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader

from egomimic.rldb.embodiment.piper import Piper
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset
from egomimic.robot.piper.robot_interface import Geometry
from egomimic.scripts.piper_process.evaluate_full import DOMAIN, load_model

JOINT_ACTION = "actions_joints"


def fk_pose(geometry, joints):
    flat = joints.reshape(-1, 6)
    feedback = (flat - geometry.offsets) / geometry.signs
    matrices = (
        geometry.fk.fk_from_feedback_deg(np.rad2deg(feedback)) @ geometry.link_tcp
    )
    quat = Rotation.from_matrix(matrices[:, :3, :3]).as_quat()
    return matrices[:, :3, 3].reshape(*joints.shape[:-1], 3), quat.reshape(
        *joints.shape[:-1], 4
    )


def episode_metrics(prediction, target, valid, geometry):
    joint_error = np.rad2deg(prediction[..., :6] - target[..., :6])
    pred_xyz, pred_quat = fk_pose(geometry, prediction[..., :6])
    true_xyz, true_quat = fk_pose(geometry, target[..., :6])
    position = np.linalg.norm(pred_xyz - true_xyz, axis=-1) * 1000
    rotation = np.rad2deg(
        (
            Rotation.from_quat(true_quat.reshape(-1, 4)).inv()
            * Rotation.from_quat(pred_quat.reshape(-1, 4))
        ).magnitude()
    ).reshape(valid.shape)

    def mean(error):
        return float(((error * valid).sum(1) / valid.sum(1)).mean())

    pred_open, true_open = prediction[..., 6] >= 0.05, target[..., 6] >= 0.05
    tp = int((pred_open & true_open & valid).sum())
    fp = int((pred_open & ~true_open & valid).sum())
    fn = int((~pred_open & true_open & valid).sum())
    lo, hi = geometry.bounds.T
    outside = ((prediction[..., :6] < lo) | (prediction[..., :6] > hi)).any(-1)
    adjacent = np.linalg.norm(np.rad2deg(np.diff(prediction[..., :6], axis=1)), axis=-1)
    adjacent_valid = valid[:, 1:] & valid[:, :-1]
    return {
        "first_joint_mae_deg": float(np.abs(joint_error[:, 0]).mean()),
        "chunk_joint_mae_deg": mean(np.abs(joint_error).mean(-1)),
        "chunk_joint_rmse_deg": float(np.sqrt(mean((joint_error**2).mean(-1)))),
        "first_fk_position_mm": float(position[:, 0].mean()),
        "chunk_fk_position_mm": mean(position),
        "chunk_fk_rotation_deg": mean(rotation),
        "chunk_gripper_mae_mm": mean(
            np.abs(prediction[..., 6] - target[..., 6]) * 1000
        ),
        "gripper_f1_at_50mm": 2 * tp / max(2 * tp + fp + fn, 1),
        "joint_limit_violation_fraction": mean(outside),
        "adjacent_joint_step_deg": float(
            (adjacent * adjacent_valid).sum() / adjacent_valid.sum()
        ),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset", type=Path, default=Path(".local_calibration/piper_all")
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path(".local_calibration/aria_gen2_aligned/cotrain_data_check.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--limit-episodes", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = json.loads(args.split.read_text())[DOMAIN]["valid_paths"]
    if args.limit_episodes is not None:
        names = names[: args.limit_episodes]
    wrapper, step = load_model(args.checkpoint)
    model, policy = wrapper.model, wrapper.model.nets["policy"]
    head = policy.heads[f"{DOMAIN}_{JOINT_ACTION}"]
    is_flow = hasattr(head, "sample_action")
    seeds = args.seeds if is_flow else [args.seeds[0]]
    geometry = Geometry()
    rows = []
    with torch.inference_mode():
        for name in names:
            ds = ZarrDataset(
                args.dataset / f"{name}.zarr",
                Piper.get_keymap(action_horizon=30, joint_head=True),
                Piper.get_transform_list(joint_head=True),
                strict=True,
            )
            n = len(ds)
            valid = np.arange(30)[None] < (n - np.arange(n))[:, None]
            targets, predictions = [], [[] for _ in seeds]
            for raw in DataLoader(ds, batch_size=args.batch_size, num_workers=0):
                targets.append(raw[JOINT_ACTION].numpy())
                normalized = model.norm_stats.normalize(raw, 7)
                batch = model.process_batch_for_training({DOMAIN: normalized})[7]
                data = model._robomimic_to_hpt_data(
                    batch,
                    model.camera_keys[7],
                    model.proprio_keys[7],
                    model.lang_keys[7],
                    model.ac_keys[7],
                    model.auxiliary_ac_keys[DOMAIN],
                )
                features, _ = policy.forward_features(DOMAIN, data)
                for index, seed in enumerate(seeds):
                    generator = torch.Generator(device="cuda").manual_seed(
                        seed * 1_000_003 + len(targets)
                    )
                    pred = (
                        head.sample_action(features, DOMAIN, generator=generator)
                        if is_flow
                        else head(features)
                    )
                    pred = model.norm_stats.unnormalize(
                        {JOINT_ACTION: pred.float()}, 7
                    )[JOINT_ACTION]
                    predictions[index].append(pred.cpu().numpy())
            target = np.concatenate(targets)
            prediction = np.stack([np.concatenate(parts) for parts in predictions])
            for index, seed in enumerate(seeds):
                rows.append(
                    dict(
                        episode=name,
                        seed=seed,
                        **episode_metrics(prediction[index], target, valid, geometry),
                    )
                )
            np.savez_compressed(
                args.output / f"{name}.npz",
                prediction=prediction,
                target=target,
                valid=valid,
                seeds=seeds,
            )
            print(name, flush=True)
    aggregate = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
        if key not in ["episode", "seed"]
    }
    report = dict(
        checkpoint=str(args.checkpoint.resolve()),
        global_step=step,
        head="flow" if is_flow else "mlp",
        episodes=names,
        frames=sum(
            len(
                ZarrDataset(
                    args.dataset / f"{name}.zarr",
                    Piper.get_keymap(joint_head=True),
                    Piper.get_transform_list(joint_head=True),
                )
            )
            for name in names
        ),
        aggregation="equal inference seeds and equal episodes; valid horizons per anchor",
        gripper_f1_threshold_m=0.05,
        aggregate=aggregate,
        per_episode_seed=rows,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
