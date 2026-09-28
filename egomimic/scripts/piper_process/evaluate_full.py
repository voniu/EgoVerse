"""Paired full-episode offline evaluation of Piper A/B, without resampling."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import simplejpeg
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader

from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.embodiment.piper import Piper
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset, split_dataset_names

DOMAIN = "piper_right_arm"
ACTION = "actions_cartesian"
HORIZON = 30


def pose_errors(prediction, target):
    """Euclidean xyz distance, SO(3) angle, and absolute opening error."""
    shape = prediction.shape[:-1]
    pred_r = Rotation.from_euler("ZYX", prediction[..., 3:6].reshape(-1, 3))
    true_r = Rotation.from_euler("ZYX", target[..., 3:6].reshape(-1, 3))
    return {
        "position_mm": np.linalg.norm(prediction[..., :3] - target[..., :3], axis=-1)
        * 1000,
        "rotation_deg": np.rad2deg((true_r.inv() * pred_r).magnitude()).reshape(shape),
        "gripper_mm": np.abs(prediction[..., 6] - target[..., 6]) * 1000,
    }


def summarize(prediction, target, valid, fm):
    result = {}
    for key, error in pose_errors(prediction, target).items():
        result["first_" + key] = float(error[:, 0].mean())
        # Equal weight per anchor frame, then per episode in the final report.
        result["chunk_" + key] = float(((error * valid).sum(1) / valid.sum(1)).mean())
    result["masked_fm_loss"] = float(((fm * valid).sum(1) / valid.sum(1)).mean())
    return result


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    params = checkpoint["hyper_parameters"]
    wrapper = ModelWrapper(
        config_tree=params["config_tree"], norm_stats_state=params["norm_stats_state"]
    )
    wrapper.load_state_dict(checkpoint["state_dict"], strict=True)
    wrapper.to("cuda").eval()
    wrapper.model.device = torch.device("cuda")
    return wrapper, checkpoint["global_step"]


def evaluate_model(label, checkpoint, paths, output, seeds, batch_size):
    wrapper, step = load_model(checkpoint)
    model = wrapper.model
    policy = model.nets["policy"]
    policy.train_mode = False
    head = policy.heads[DOMAIN]
    assert head.action_horizon == HORIZON
    is_flow = bool(policy.diffusion)
    if is_flow:
        assert head.num_inference_steps == 50
    records = []
    frame_offset = 0
    with torch.inference_mode():
        for path in paths:
            ds = ZarrDataset(
                path,
                Piper.get_keymap(action_horizon=HORIZON),
                Piper.get_transform_list(),
                strict=True,
            )
            n = len(ds)
            valid = np.arange(HORIZON)[None, :] < (n - np.arange(n))[:, None]
            truth = np.empty((n, HORIZON, 7), dtype=np.float32)
            predictions = np.empty((len(seeds), n, HORIZON, 7), dtype=np.float32)
            losses = np.empty((len(seeds), n, HORIZON), dtype=np.float32)
            cursor = 0
            for raw in DataLoader(
                ds, batch_size=batch_size, shuffle=False, num_workers=0
            ):
                target = raw[ACTION].numpy().copy()
                count = len(target)
                truth[cursor : cursor + count] = target
                # Apply cached training normalization directly: no rejection/clipping.
                normalized = model.norm_stats.normalize(raw, 7)
                restored = model.norm_stats.unnormalize(
                    {ACTION: normalized[ACTION]}, 7
                )[ACTION]
                np.testing.assert_allclose(restored.numpy(), target, atol=2e-6)
                batch = model.process_batch_for_training({DOMAIN: normalized})[7]
                batch["pad_mask"] = torch.as_tensor(
                    valid[cursor : cursor + count, :, None], device="cuda"
                )
                data = model._robomimic_to_hpt_data(
                    batch,
                    model.camera_keys[7],
                    model.proprio_keys[7],
                    model.lang_keys[7],
                    model.ac_keys[7],
                )
                # Same FP32 arithmetic for both checkpoints; no stochastic augmentation.
                features, _ = policy.forward_features(DOMAIN, data)
                for s, seed in enumerate(seeds):
                    paired_seed = seed * 1_000_003 + frame_offset + cursor
                    generator = torch.Generator(device="cuda").manual_seed(paired_seed)
                    pred = (
                        head.sample_action(features, DOMAIN, generator=generator)
                        if is_flow
                        else head(features)
                    )
                    normalized_prediction = pred.float()
                    pred = model.norm_stats.unnormalize({ACTION: pred.float()}, 7)[
                        ACTION
                    ]
                    predictions[s, cursor : cursor + count] = pred.cpu().numpy()
                    # Pair both the Gaussian noise and Beta time draws for FM loss.
                    torch.manual_seed(paired_seed + 10_000_019)
                    if is_flow:
                        velocity, velocity_target = head.predict(
                            data["action"], features
                        )
                        error = (velocity - velocity_target).square().mean(-1)
                    else:
                        error = (
                            (normalized_prediction - data["action"]).square().mean(-1)
                        )
                    losses[s, cursor : cursor + count] = error.cpu().numpy()
                cursor += count
            assert (
                cursor == n
                and np.isfinite(predictions).all()
                and np.isfinite(losses).all()
            )
            for s, seed in enumerate(seeds):
                metrics = summarize(predictions[s], truth, valid, losses[s])
                if not is_flow:
                    metrics["masked_action_mse"] = metrics.pop("masked_fm_loss")
                records.append(
                    {
                        "model": label,
                        "episode": path.stem,
                        "seed": seed,
                        "frames": n,
                        **metrics,
                    }
                )
            np.savez_compressed(
                output / f"{label}_{path.stem}.npz",
                prediction=predictions,
                target=truth,
                valid=valid,
                **({"fm_loss": losses} if is_flow else {"action_mse": losses}),
                seeds=seeds,
                frame_index=np.arange(n),
            )
            frame_offset += n
            print(
                f"{label}: {path.stem}, {n} frames, cumulative {frame_offset}",
                flush=True,
            )
    del wrapper, model, policy, head
    torch.cuda.empty_cache()
    return records, step, "flow" if is_flow else "mlp"


def projection_previews(paths, output):
    """First three episodes by name, seed 0; deterministic selection, no cherry-picking."""
    for path in paths[:3]:
        a, b = (np.load(output / f"{label}_{path.stem}.npz") for label in ("A", "B"))
        ds = ZarrDataset(
            path, Piper.get_keymap(), Piper.get_transform_list(), strict=True
        )
        writer = cv2.VideoWriter(
            str(output / f"{path.stem}_projection.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (1280, 480),
        )
        assert writer.isOpened()
        for t in range(len(ds)):
            raw = ds.episode_reader.read({"images.front_1": (t, None)})
            image = cv2.cvtColor(
                simplejpeg.decode_jpeg(raw["images.front_1"], colorspace="RGB"),
                cv2.COLOR_RGB2BGR,
            )
            K = np.asarray(ds.intrinsics["front_1"])[:, :3]
            panels = []
            for label, archive in [("A", a), ("B", b)]:
                panel = image.copy()
                mask = archive["valid"][t]
                for points, color in [
                    (archive["target"][t, mask, :3], (0, 220, 0)),
                    (archive["prediction"][0, t, mask, :3], (255, 0, 255)),
                ]:
                    pixels = points @ K.T
                    visible = points[:, 2] > 0.01
                    uv = pixels[visible, :2] / pixels[visible, 2:]
                    uv = np.rint(uv).astype(int)
                    # Draw individual points; do not connect across invisible sections.
                    for u, v in uv:
                        if 0 <= u < 640 and 0 <= v < 480:
                            cv2.circle(panel, (int(u), int(v)), 2, color, -1)
                cv2.putText(
                    panel,
                    f"{label} | frame {t} | seed 0",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    panel,
                    "Green: demo | Magenta: prediction",
                    (10, 465),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                panels.append(panel)
            writer.write(np.concatenate(panels, axis=1))
        writer.release()


def write_report(records, output, manifest):
    with (output / "per_episode_seed.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metrics = [k for k in records[0] if k.startswith(("first_", "chunk_", "masked_"))]
    aggregate = {}
    for label in ("A", "B"):
        aggregate[label] = {}
        for metric in metrics:
            per_seed = [
                np.mean(
                    [
                        r[metric]
                        for r in records
                        if r["model"] == label and r["seed"] == s
                    ]
                )
                for s in manifest["seeds"]
            ]
            aggregate[label][metric] = {
                "mean": float(np.mean(per_seed)),
                "seed_std": float(np.std(per_seed)),
                "per_seed": list(map(float, per_seed)),
            }
    manifest["aggregate_equal_episode_weight"] = aggregate
    manifest["complete"] = True
    (output / "report.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(aggregate, indent=2), flush=True)


def plot_report(output):
    import matplotlib.pyplot as plt

    report = json.loads((output / "report.json").read_text())
    scores = report["aggregate_equal_episode_weight"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), layout="constrained")
    for ax, metric, title, unit in zip(
        axes,
        ["chunk_position_mm", "chunk_rotation_deg", "chunk_gripper_mm"],
        ["TCP position", "TCP orientation", "Gripper opening"],
        ["mm", "degrees", "mm"],
    ):
        values = [scores[label][metric]["mean"] for label in ("A", "B")]
        errors = [scores[label][metric]["seed_std"] for label in ("A", "B")]
        ax.bar(["A", "B"], values, yerr=errors, capsize=5, color=["#2469b1", "#e17c24"])
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_ylim(bottom=0)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.2f}", ha="center", va="bottom")
    fig.suptitle(
        f"Full validation: valid 30-frame chunk error\nEqual episode weights; {len(report['seeds'])} inference seed(s)",
        fontsize=11,
    )
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--previews", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = json.loads(args.split.read_text())["piper_right_arm"]["valid_paths"]
    all_paths = sorted(args.dataset.glob("*.zarr"))
    _, expected = split_dataset_names(
        [p.stem for p in all_paths], valid_ratio=0.1, seed=42
    )
    assert set(names) == expected and len(names) == 7
    paths = [args.dataset / (name + ".zarr") for name in sorted(names)]
    # Verify both checkpoints use identical robot normalization, not just shape.
    stats = []
    for path in (args.a, args.b):
        c = torch.load(path, map_location="cpu", weights_only=False)
        stats.append(c["hyper_parameters"]["norm_stats_state"])
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    normalizers = [MultiDataset.from_state(s) for s in stats]
    assert normalizers[0].norm_mode == normalizers[1].norm_mode
    assert json.dumps(normalizers[0].norm_stats[7], sort_keys=True) == json.dumps(
        normalizers[1].norm_stats[7], sort_keys=True
    )
    del c, stats, normalizers
    manifest = {
        "complete": False,
        "episodes": sorted(names),
        "seeds": args.seeds,
        "batch_size": args.batch_size,
        "precision": "float32",
        "action_horizon": HORIZON,
        "inference_steps": 50,
        "resampling": False,
        "tail_padding_scored": False,
        "aggregation": "mean valid horizons per anchor, then mean anchors per episode, then equal episode mean",
        "seed_std_interpretation": "Inference noise variation, not independent training runs or confidence intervals",
        "checkpoints": {
            label: {
                "path": str(p.resolve()),
                "sha256": hashlib.file_digest(p.open("rb"), "sha256").hexdigest(),
            }
            for label, p in [("A", args.a), ("B", args.b)]
        },
    }
    records = []
    for label, path in [("A", args.a), ("B", args.b)]:
        rows, step, head_type = evaluate_model(
            label, path, paths, args.output, args.seeds, args.batch_size
        )
        records.extend(rows)
        manifest["checkpoints"][label]["global_step"] = step
        manifest["checkpoints"][label]["head_type"] = head_type
        manifest["checkpoints"][label]["inference_steps"] = (
            50 if head_type == "flow" else 0
        )
    if all(v["head_type"] == "mlp" for v in manifest["checkpoints"].values()):
        manifest["inference_steps"] = 0
        manifest["seed_std_interpretation"] = (
            "Deterministic MLP; repeated inference seeds do not provide uncertainty estimates."
        )
    manifest["frames_per_seed_per_model"] = sum(
        r["frames"] for r in records if r["model"] == "A" and r["seed"] == args.seeds[0]
    )
    assert manifest["frames_per_seed_per_model"] == 2381
    write_report(records, args.output, manifest)
    if args.previews:
        projection_previews(paths, args.output)
    plot_report(args.output)


if __name__ == "__main__":
    main()
