"""Fit CPU state-only position baselines on the fixed training episodes."""

import json
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Rotation


def read_episode(root, name):
    z = zarr.open_group(str(root / f"{name}.zarr"), mode="r")
    n = z.attrs["total_frames"]
    pose = z["right.obs_ee_pose_camframe"][:n]
    angles = Rotation.from_quat(pose[:, [4, 5, 6, 3]]).as_euler("ZYX")
    state = np.c_[pose[:, :3], angles, z["right.obs_gripper"][:n]]
    return state, z["right.cmd_ee_pose_camframe"][:n, :3]


def features(state, mean, std, quadratic):
    x = (state - mean) / std
    columns = [np.ones((len(x), 1)), x]
    if quadratic:
        i, j = np.triu_indices(x.shape[1])
        columns.append(x[:, i] * x[:, j])
    return np.concatenate(columns, axis=1)


def main():
    root = Path(".local_calibration")
    split = json.loads(
        (root / "aria_gen2_aligned/cotrain_data_check.json").read_text()
    )["piper_right_arm"]
    train = [read_episode(root / "piper_all", name) for name in split["train_paths"]]
    states = np.concatenate([s for s, _ in train])
    mean, std = states.mean(0), states.std(0)
    results = {}
    for quadratic in [False, True]:
        weights = []
        for h in range(30):
            x = np.concatenate(
                [features(s[: len(s) - h], mean, std, quadratic) for s, _ in train]
            )
            y = np.concatenate([a[h:] for _, a in train])
            penalty = np.eye(x.shape[1])  # Fixed ridge strength; no validation tuning.
            penalty[0, 0] = 0
            weights.append(np.linalg.solve(x.T @ x + penalty, x.T @ y))
        rows = []
        for name in split["valid_paths"]:
            s, a = read_episode(root / "piper_all", name)
            pred = np.stack(
                [features(s, mean, std, quadratic) @ w for w in weights], axis=1
            )
            indices = np.arange(len(s))[:, None] + np.arange(30)
            valid = indices < len(s)
            target = a[np.minimum(indices, len(s) - 1)]
            saved = np.load(root / "mlp_final_validation" / f"A_{name}.npz")
            np.testing.assert_allclose(target, saved["target"][..., :3], atol=1e-7)
            np.testing.assert_array_equal(valid, saved["valid"])
            error = np.linalg.norm(pred - target, axis=-1) * 1000
            rows.append(
                dict(
                    episode=name,
                    first_position_mm=float(error[:, 0].mean()),
                    chunk_position_mm=float(
                        ((error * valid).sum(1) / valid.sum(1)).mean()
                    ),
                )
            )
        results["quadratic" if quadratic else "linear"] = dict(
            per_episode=rows,
            **{
                key: float(np.mean([r[key] for r in rows]))
                for key in ["first_position_mm", "chunk_position_mm"]
            },
        )
    report = dict(
        protocol="State-only xyz regression; train63/valid7; horizon30; no padded training targets; equal-episode validation; ridge=1 fixed before scoring. No images or validation fitting.",
        results=results,
    )
    (root / "state_baseline.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
