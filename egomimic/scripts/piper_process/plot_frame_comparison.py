"""Plot existing FM/MLP predictions against commands on an aligned frame axis."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def select(pred, mode):
    n = pred.shape[1]
    if mode == "first_action":
        frames = np.arange(n)
        return frames, pred[:, frames, 0, :3]
    if mode == "lookahead15":
        frames = np.arange(15, n)
        return frames, pred[:, frames - 15, 15, :3]
    frames = np.arange(n)
    anchor = frames // 10 * 10
    return frames, pred[:, anchor, frames - anchor, :3]


def main():
    root = Path(".local_calibration")
    sources = {"FM": root / "full_validation", "MLP": root / "mlp_final_validation"}
    reports = {
        k: json.loads((p / "report.json").read_text()) for k, p in sources.items()
    }
    assert reports["FM"]["episodes"] == reports["MLP"]["episodes"]
    output = root / "fm_mlp_frame_curves"
    output.mkdir(exist_ok=True)
    descriptions = {
        "first_action": "30 Hz inference: first action at each observed frame (h=0)",
        "lookahead15": "30 Hz inference: h=15 prediction aligned to target frame t+15 (0.5 s ahead)",
        "chunk_stride10": "3 Hz query: use first 10 actions of each 30-step chunk; no blending",
    }
    columns = [(kind, group) for kind in ["FM", "MLP"] for group in ["A", "B"]]
    rows = []
    links = []
    with PdfPages(output / "all_episodes.pdf") as pdf:
        for episode in reports["FM"]["episodes"]:
            loaded = {
                (kind, group): np.load(path / f"{group}_{episode}.npz")
                for kind, path in sources.items()
                for group in ["A", "B"]
            }
            reference = loaded["FM", "A"]
            truth = reference["target"][:, 0, :3]
            for d in loaded.values():
                for key in ["target", "valid", "frame_index"]:
                    np.testing.assert_array_equal(d[key], reference[key])
            for mode, description in descriptions.items():
                fig, axes = plt.subplots(
                    3, 4, figsize=(20, 9), sharex=True, sharey="row"
                )
                for col, (kind, group) in enumerate(columns):
                    d = loaded[kind, group]
                    frames, predictions = select(d["prediction"], mode)
                    assert d["seeds"][0] == 0
                    gt = truth[frames]
                    errors = np.linalg.norm(predictions - gt, axis=-1) * 1000
                    rows.append(
                        dict(
                            episode=episode,
                            model=kind,
                            group=group,
                            mode=mode,
                            seed0_position_mm=float(errors[0].mean()),
                            mean_inference_seeds_position_mm=float(errors.mean()),
                        )
                    )
                    for axis, coordinate in enumerate("XYZ"):
                        ax = axes[axis, col]
                        ax.plot(
                            frames,
                            gt[:, axis],
                            color="#17834b",
                            lw=1.8,
                            label="GT command",
                        )
                        color = "#8544a6" if kind == "FM" else "#db701c"
                        ax.plot(
                            frames,
                            predictions[0, :, axis],
                            color=color,
                            lw=0.85,
                            alpha=0.9,
                            label=f"{kind} prediction"
                            + (" seed 0" if kind == "FM" else ""),
                        )
                        if kind == "FM":
                            ax.fill_between(
                                frames,
                                predictions[:, :, axis].min(0),
                                predictions[:, :, axis].max(0),
                                color=color,
                                alpha=0.15,
                                label="Range: inference seeds 0/1/2",
                            )
                        ax.grid(alpha=0.2)
                        if col == 0:
                            ax.set_ylabel(f"Camera {coordinate} (m)")
                        if axis == 2:
                            ax.set_xlabel("Target frame")
                        if axis == 0:
                            step = reports[kind]["checkpoints"][group]["global_step"]
                            ax.set_title(
                                f"{kind} {group} | step {step}\nXYZ error (seed 0): {errors[0].mean():.1f} mm"
                            )
                            ax.legend(fontsize=7, loc="best")
                    csv_path = output / f"{episode}_{mode}_{kind}_{group}.csv"
                    with csv_path.open("w") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            ["frame", "gt_x_m", "gt_y_m", "gt_z_m"]
                            + [
                                f"seed{int(seed)}_{axis}_m"
                                for seed in d["seeds"]
                                for axis in "xyz"
                            ]
                        )
                        writer.writerows(
                            np.c_[
                                frames,
                                gt,
                                predictions.transpose(1, 0, 2).reshape(len(frames), -1),
                            ]
                        )
                fig.suptitle(
                    f"{episode}\n{description}\nRecorded observations; camera optical coordinates; no smoothing or IK",
                    fontsize=12,
                )
                fig.tight_layout(rect=(0, 0, 1, 0.91))
                filename = f"{episode}_{mode}.png"
                fig.savefig(output / filename, dpi=150)
                pdf.savefig(fig)
                plt.close(fig)
                links.append(
                    f'<h3>{episode}: {mode}</h3><img loading="lazy" src="{filename}" style="width:100%">'
                )
            print(episode, flush=True)
    with (output / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregates = []
    for mode in descriptions:
        for kind, group in columns:
            selected = [
                r
                for r in rows
                if r["mode"] == mode and r["model"] == kind and r["group"] == group
            ]
            aggregates.append(
                dict(
                    mode=mode,
                    model=kind,
                    group=group,
                    **{
                        k: float(np.mean([r[k] for r in selected]))
                        for k in [
                            "seed0_position_mm",
                            "mean_inference_seeds_position_mm",
                        ]
                    },
                )
            )
    (output / "report.json").write_text(
        json.dumps(
            dict(
                protocol=descriptions,
                coordinate_frame="D435IF color optical; meters",
                gt="Recorded commanded TCP via FK, not measured executed TCP",
                inference="Offline recorded observations, not closed-loop rollout; FM range is not confidence interval",
                checkpoints={k: r["checkpoints"] for k, r in reports.items()},
                aggregate_equal_episode=aggregates,
            ),
            indent=2,
        )
    )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>FM / MLP frame curves</title><h1>FM / MLP: GT and predictions</h1><p>Green: GT command. Purple: FM seed0; band: range of three inference seeds. Orange: deterministic MLP. All use recorded observations; no smoothing or IK.</p>'
        + "".join(links)
    )
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
