"""CPU diagnostics of saved predictions: geometry, timing and persistence baselines."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from egomimic.scripts.piper_process.evaluate_full import pose_errors


def average(values, valid):
    """Equal valid horizons per anchor, then equal anchors."""
    counts = valid.sum(-1)
    return float((np.where(valid, values, 0).sum(-1) / counts)[counts > 0].mean())


def project(xyz, K):
    uvw = xyz @ K.T
    return uvw[..., :2] / uvw[..., 2:]


def analyze(pred, target, valid, K):
    delta = (pred[..., :3] - target[..., :3]) * 1000
    result = {
        key: average(value, valid) for key, value in pose_errors(pred, target).items()
    }
    result["first_position_mm"] = float(np.linalg.norm(delta[:, 0], axis=-1).mean())
    for i, axis in enumerate("xyz"):
        result[f"{axis}_bias_mm"] = average(delta[..., i], valid)
        result[f"{axis}_mae_mm"] = average(abs(delta[..., i]), valid)
        result[f"{axis}_mse_mm2"] = average(delta[..., i] ** 2, valid)
    puv, tuv = project(pred[..., :3], K), project(target[..., :3], K)
    positive = (pred[..., 2] > 0) & (target[..., 2] > 0)
    inside = (
        positive
        & (puv[..., 0] >= 0)
        & (puv[..., 0] < 640)
        & (puv[..., 1] >= 0)
        & (puv[..., 1] < 480)
    )
    result["nonpositive_depth_fraction"] = average(~positive, valid)
    result["pred_outside_image_fraction"] = average(~inside, valid)
    result["pixel_error"] = average(
        np.linalg.norm(puv - tuv, axis=-1), valid & positive
    )
    target_inside = (
        (target[..., 2] > 0)
        & (tuv[..., 0] >= 0)
        & (tuv[..., 0] < 640)
        & (tuv[..., 1] >= 0)
        & (tuv[..., 1] < 480)
    )
    result["target_outside_image_fraction"] = average(~target_inside, valid)
    result["both_visible_pixel_error"] = average(
        np.linalg.norm(puv - tuv, axis=-1), valid & inside & target_inside
    )
    for name, xyz in [("prediction", pred[..., :3]), ("target", target[..., :3])]:
        result[f"{name}_adjacent_step_mm"] = average(
            np.linalg.norm(np.diff(xyz, axis=1), axis=-1) * 1000,
            valid[:, 1:] & valid[:, :-1],
        )
    # Same-anchor future chunk only. Nearest points ignore chronology and are optimistic diagnostics.
    distance = (
        np.linalg.norm(pred[:, :, None, :3] - target[:, None, :, :3], axis=-1) * 1000
    )
    result["nearest_chunk_3d_mm"] = average(
        np.where(valid[:, None, :], distance, np.inf).min(-1), valid
    )
    pdistance = np.linalg.norm(puv[:, :, None] - tuv[:, None, :], axis=-1)
    result["nearest_chunk_pixel"] = average(
        np.where(
            valid[:, None, :] & (target[:, None, :, 2] > 0), pdistance, np.inf
        ).min(-1),
        valid & positive,
    )
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--evaluation", type=Path, default=Path(".local_calibration/full_validation")
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path(".local_calibration/piper_all")
    )
    args = parser.parse_args()
    output = args.evaluation / "diagnostics"
    output.mkdir(exist_ok=True)
    manifest = json.loads((args.evaluation / "report.json").read_text())
    rows, curves, horizons = [], {}, {}
    lags = np.arange(-30, 31)
    for episode in manifest["episodes"]:
        z = zarr.open_group(str(args.dataset / (episode + ".zarr")), mode="r")
        n = z.attrs["total_frames"]
        K = np.asarray(z.attrs["intrinsics"]["front_1"])[:, :3]
        obs = np.asarray(z["right.obs_ee_pose_camframe"][:n])
        angles = Rotation.from_quat(obs[:, [4, 5, 6, 3]]).as_euler("ZYX")
        state = np.concatenate(
            [obs[:, :3], angles, z["right.obs_gripper"][:n]], axis=-1
        )
        for label in ["A", "B", "Hold state", "Hold command"]:
            saved = np.load(
                args.evaluation
                / f'{label if label in ("A", "B") else "A"}_{episode}.npz'
            )
            target, valid = saved["target"], saved["valid"]
            assert len(target) == n
            np.testing.assert_allclose(
                target[:, 0, :3], z["right.cmd_ee_pose_camframe"][:n, :3], atol=1e-7
            )
            # Verify chunk labels really are future commands, without assuming it.
            indices = np.minimum(np.arange(n)[:, None] + np.arange(30), n - 1)
            np.testing.assert_allclose(target, target[:, 0][indices], atol=1e-7)
            preds = (
                saved["prediction"]
                if label in ("A", "B")
                else np.broadcast_to(
                    (state if label == "Hold state" else target[:, 0])[None, :, None],
                    (1, n, 30, 7),
                )
            )
            for seed, pred in enumerate(preds):
                rows.append(
                    dict(
                        model=label,
                        episode=episode,
                        seed=seed,
                        **analyze(pred, target, valid, K),
                    )
                )
                error = np.linalg.norm(pred[..., :3] - target[..., :3], axis=-1) * 1000
                horizons.setdefault(label, []).append(
                    [float(error[valid[:, h], h].mean()) for h in range(30)]
                )
                # Identical interior anchors and all 30 horizons for every tested lag.
                anchors = np.arange(30, n - 59)
                curve = [
                    float(
                        np.linalg.norm(
                            pred[anchors, :, :3]
                            - target[:, 0, :3][anchors[:, None] + np.arange(30) + lag],
                            axis=-1,
                        ).mean()
                        * 1000
                    )
                    for lag in lags
                ]
                curves.setdefault(label, []).append(curve)
        print(episode, flush=True)
    aggregate = {}
    for label in curves:
        selected = [r for r in rows if r["model"] == label]
        aggregate[label] = {
            k: float(np.mean([r[k] for r in selected]))
            for k in selected[0]
            if k not in ("model", "episode", "seed")
        }
        mse = np.array([aggregate[label][f"{axis}_mse_mm2"] for axis in "xyz"])
        aggregate[label]["xyz_squared_error_share"] = (mse / mse.sum()).tolist()
        curve = np.mean(curves[label], axis=0)
        best = int(curve.argmin())
        aggregate[label]["lag_diagnostic"] = dict(
            best_lag_frames=int(lags[best]),
            zero_lag_mm=float(curve[30]),
            best_lag_mm=float(curve[best]),
            curve_mm=curve.tolist(),
        )
        aggregate[label]["position_by_horizon_mm"] = np.mean(
            horizons[label], axis=0
        ).tolist()
    report = dict(
        protocol="Equal horizons per anchor, equal anchors per episode, equal episodes and inference seeds. Baselines use the same targets and masks. Pixel metrics require positive depths; off-screen points are retained.",
        lag_protocol="delta > 0 compares prediction[t,h] to command[t+h+delta]. All lags use anchors 30 <= t < N-59 and h=0..29. Post-hoc diagnostic, not a corrected score.",
        frames=manifest["frames_per_seed_per_model"],
        aggregate=aggregate,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "per_episode_seed.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for label, values in aggregate.items():
        axes[0].plot(np.arange(30) / 30, values["position_by_horizon_mm"], label=label)
        axes[1].plot(lags / 30, values["lag_diagnostic"]["curve_mm"], label=label)
    axes[0].set(
        xlabel="Future horizon (s)",
        ylabel="Position error (mm)",
        title="Matched-time error",
    )
    axes[1].set(
        xlabel="Target time shift (s)",
        ylabel="Position error (mm)",
        title="Fixed-lag diagnostic (common interior frames)",
    )
    axes[1].axvline(0, color="gray", lw=1)
    for i, label in enumerate(["A", "B"]):
        axes[2].bar(
            np.arange(3) + i * 0.35,
            [aggregate[label][f"{a}_mae_mm"] for a in "xyz"],
            0.35,
            label=label,
        )
    axes[2].set(
        xticks=np.arange(3) + 0.175,
        xticklabels=["X right", "Y down", "Z depth"],
        ylabel="Absolute error (mm)",
        title="Camera-axis error",
    )
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "diagnostics.png", dpi=160)
    plt.close(fig)
    # Fixed example, chosen by episode order and frame index, not by error.
    episode, t = manifest["episodes"][0], 100
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for row, label in enumerate(["A", "B"]):
        data = np.load(args.evaluation / f"{label}_{episode}.npz")
        for col, axis in enumerate("XYZ"):
            ax = axes[row, col]
            ax.plot(
                np.arange(30) / 30,
                data["target"][t, :, col] * 1000,
                color="green",
                label="Demo",
            )
            for s, (color, linestyle, marker) in enumerate(
                [("#7b3294", "-", "o"), ("#0072b2", "--", "s"), ("#d55e00", ":", "^")][
                    : len(data["prediction"])
                ]
            ):
                ax.plot(
                    np.arange(30) / 30,
                    data["prediction"][s, t, :, col] * 1000,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=3,
                    linewidth=1.5,
                    alpha=0.85,
                    label=f"Prediction seed {data['seeds'][s]}",
                )
            ax.set(
                title=f"{label}: camera {axis}",
                xlabel="Future horizon (s)",
                ylabel="Position (mm)",
            )
            ax.legend()
    fig.suptitle(f"{episode}, frame {t}: temporal order exposed")
    fig.tight_layout()
    fig.savefig(output / "time_order_example.png", dpi=150)
    plt.close(fig)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
