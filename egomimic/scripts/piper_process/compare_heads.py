"""Compare physical errors from completed FM and MLP full-validation reports."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--flow", type=Path, default=Path(".local_calibration/full_validation")
    )
    parser.add_argument("--mlp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {
        name: json.loads((path / "report.json").read_text())
        for name, path in [("FM", args.flow), ("MLP", args.mlp)]
    }
    assert all(r["complete"] for r in reports.values())
    assert reports["FM"]["episodes"] == reports["MLP"]["episodes"]
    assert (
        reports["FM"]["frames_per_seed_per_model"]
        == reports["MLP"]["frames_per_seed_per_model"]
        == 2381
    )
    assert reports["FM"]["aggregation"] == reports["MLP"]["aggregation"]
    rows = []
    for kind, root in [("FM", args.flow), ("MLP", args.mlp)]:
        report = reports[kind]
        for label in ["A", "B"]:
            adjacent = []
            for episode in report["episodes"]:
                a = np.load(args.flow / f"{label}_{episode}.npz")
                b = np.load(args.mlp / f"{label}_{episode}.npz")
                for key in ["target", "valid", "frame_index"]:
                    np.testing.assert_array_equal(a[key], b[key])
                data = a if kind == "FM" else b
                mask = data["valid"][:, 1:] & data["valid"][:, :-1]
                count = mask.sum(-1)
                delta = (
                    np.linalg.norm(
                        np.diff(data["prediction"][..., :3], axis=2), axis=-1
                    )
                    * 1000
                )
                per_anchor = (delta * mask).sum(-1)[:, count > 0] / count[count > 0]
                adjacent.append(float(per_anchor.mean()))
            metrics = {
                key: value["mean"]
                for key, value in report["aggregate_equal_episode_weight"][
                    label
                ].items()
                if key.startswith(("first_", "chunk_"))
            }
            rows.append(
                dict(
                    head=kind,
                    group=label,
                    checkpoint_step=report["checkpoints"][label]["global_step"],
                    adjacent_mm=float(np.mean(adjacent)),
                    **metrics,
                )
            )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "comparison.json").write_text(
        json.dumps(
            dict(
                rows=rows,
                note="Identical complete validation targets and masks checked. Different head structures/objectives; inference seeds are not independent training runs. All metrics are offline, not success rates.",
            ),
            indent=2,
        )
        + "\n"
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key, title in zip(
        axes,
        ["chunk_position_mm", "first_position_mm", "adjacent_mm"],
        [
            "Chunk position error",
            "First action position error",
            "Adjacent predicted action displacement",
        ],
    ):
        ax.bar(
            [r["head"] + " " + r["group"] for r in rows],
            [r[key] for r in rows],
            color=["#4682b4", "#6baed6", "#e17c24", "#fdae6b"],
        )
        ax.set(title=title, ylabel="mm")
    fig.tight_layout()
    fig.savefig(args.output / "comparison.png", dpi=160)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
