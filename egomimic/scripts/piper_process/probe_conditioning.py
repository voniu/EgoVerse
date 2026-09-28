"""Fixed-noise condition swaps and FM velocity errors on eight cached observations."""

import json
from pathlib import Path

import numpy as np
import torch

from egomimic.scripts.piper_process.evaluate_full import ACTION, DOMAIN, load_model


def main():
    torch.set_num_threads(4)
    root = Path(".local_calibration/convergence_eight")
    wrapper, _ = load_model(
        Path(".local_calibration/hpc/results/checkpoints/a/best.ckpt")
    )
    head = wrapper.model.nets["policy"].heads[DOMAIN]
    head.load_state_dict(
        torch.load(root / "fm_checkpoint.pt", map_location="cuda", weights_only=False)[
            "head"
        ]
    )
    head.eval()
    cache = torch.load(root / "inputs.pt", weights_only=False)
    features, actions = cache["features"].cuda(), cache["actions"].cuda()
    rows = []
    with torch.no_grad():
        for seed in range(10):
            noise = torch.randn(
                actions.shape,
                device="cuda",
                generator=torch.Generator(device="cuda").manual_seed(100 + seed),
            )
            predictions = {}
            for name, condition in [
                ("correct", features),
                ("reversed", features.flip(0)),
                ("mean", features.mean(0, keepdim=True).expand_as(features)),
            ]:
                pred = head.inference(noise.clone(), condition)
                predictions[name] = (
                    wrapper.model.norm_stats.unnormalize({ACTION: pred}, 7)[ACTION]
                    .cpu()
                    .numpy()
                )
                truth = (
                    cache["target"][::-1].copy()
                    if name == "reversed"
                    else cache["target"]
                )
                rows.append(
                    dict(
                        seed=seed,
                        condition=name,
                        own_target_mm=float(
                            np.linalg.norm(
                                predictions[name][..., :3] - cache["target"][..., :3],
                                axis=-1,
                            ).mean()
                            * 1000
                        ),
                        supplied_condition_target_mm=float(
                            np.linalg.norm(
                                predictions[name][..., :3] - truth[..., :3], axis=-1
                            ).mean()
                            * 1000
                        ),
                    )
                )
            for row in rows[-3:]:
                row["change_from_correct_mm"] = float(
                    np.linalg.norm(
                        predictions[row["condition"]][..., :3]
                        - predictions["correct"][..., :3],
                        axis=-1,
                    ).mean()
                    * 1000
                )
        times = []
        for t in [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]:
            errors = []
            for seed in range(10):
                noise = torch.randn(
                    actions.shape,
                    device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(200 + seed),
                )
                x = t * noise + (1 - t) * actions
                v = head.model(x, torch.full((8,), t, device="cuda"), features)
                errors.append(
                    (v - (noise - actions)).square().mean((1, 2)).cpu().numpy()
                )
            times.append(
                dict(
                    t=t,
                    velocity_mse=float(np.mean(errors)),
                    per_observation_mse=np.mean(errors, axis=0).tolist(),
                )
            )
    summary = {
        name: {
            key: float(np.mean([r[key] for r in rows if r["condition"] == name]))
            for key in [
                "own_target_mm",
                "supplied_condition_target_mm",
                "change_from_correct_mm",
            ]
        }
        for name in ["correct", "reversed", "mean"]
    }
    report = dict(
        summary=summary,
        rows=rows,
        velocity_error_by_time=times,
        note="Fixed identical initial noise across conditions. Reversed condition is scored both against original and reversed targets. Mean condition has no genuine corresponding target. Velocity probes use teacher-interpolated states, not rollout states; normalized velocity errors across time are not physical trajectory error.",
    )
    (root / "conditioning_probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
