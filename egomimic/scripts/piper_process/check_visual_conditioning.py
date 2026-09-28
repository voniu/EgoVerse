"""Measure fixed MLP sensitivity to temporally mismatched camera images."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from egomimic.rldb.embodiment.piper import Piper
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset
from egomimic.scripts.piper_process.evaluate_full import ACTION, DOMAIN, load_model


def main():
    root = Path(".local_calibration")
    reference = root / "mlp_final_validation"
    manifest = json.loads((reference / "report.json").read_text())
    rows = []
    for label in ["A", "B"]:
        wrapper, step = load_model(
            root / "hpc/mlp/checkpoints" / label.lower() / "final.ckpt"
        )
        model = wrapper.model
        policy = model.nets["policy"]
        assert not policy.diffusion
        camera_keys = model.camera_keys[7]
        assert len(camera_keys) == 2
        with torch.inference_mode():
            for name in manifest["episodes"]:
                ds = ZarrDataset(
                    root / "piper_all" / f"{name}.zarr",
                    Piper.get_keymap(action_horizon=30),
                    Piper.get_transform_list(),
                    strict=True,
                )
                n = len(ds)
                donor = Subset(ds, ((np.arange(n) + n // 2) % n).tolist())
                saved = np.load(reference / f"{label}_{name}.npz")
                predictions = {mode: [] for mode in ["original", *camera_keys, "both"]}
                for raw, alternate in zip(
                    DataLoader(ds, batch_size=32), DataLoader(donor, batch_size=32)
                ):
                    batch = model.process_batch_for_training(
                        {DOMAIN: model.norm_stats.normalize(raw, 7)}
                    )[7]
                    other = model.process_batch_for_training(
                        {DOMAIN: model.norm_stats.normalize(alternate, 7)}
                    )[7]
                    for mode in predictions:
                        changed = batch.copy()
                        for key in camera_keys:
                            if mode in [key, "both"]:
                                changed[key] = other[key]
                        data = model._robomimic_to_hpt_data(
                            changed,
                            camera_keys,
                            model.proprio_keys[7],
                            model.lang_keys[7],
                            model.ac_keys[7],
                        )
                        features, _ = policy.forward_features(DOMAIN, data)
                        pred = model.norm_stats.unnormalize(
                            {ACTION: policy.heads[DOMAIN](features).float()}, 7
                        )[ACTION]
                        predictions[mode].append(pred.cpu().numpy())
                predictions = {k: np.concatenate(v) for k, v in predictions.items()}
                np.testing.assert_allclose(
                    predictions["original"],
                    saved["prediction"][0],
                    atol=2e-5,
                    rtol=2e-5,
                )
                valid = saved["valid"]
                for mode, pred in predictions.items():
                    error = (
                        np.linalg.norm(
                            pred[..., :3] - saved["target"][..., :3], axis=-1
                        )
                        * 1000
                    )
                    shift = (
                        np.linalg.norm(
                            pred[..., :3] - predictions["original"][..., :3], axis=-1
                        )
                        * 1000
                    )
                    rows.append(
                        dict(
                            model=label,
                            episode=name,
                            mode=mode,
                            first_position_mm=float(error[:, 0].mean()),
                            chunk_position_mm=float(
                                ((error * valid).sum(1) / valid.sum(1)).mean()
                            ),
                            prediction_change_mm=float(
                                ((shift * valid).sum(1) / valid.sum(1)).mean()
                            ),
                        )
                    )
                print(label, name, flush=True)
        del wrapper, model, policy
        torch.cuda.empty_cache()
    aggregate = []
    for label in ["A", "B"]:
        for mode in predictions:
            group = [r for r in rows if r["model"] == label and r["mode"] == mode]
            aggregate.append(
                dict(
                    model=label,
                    mode=mode,
                    **{
                        k: float(np.mean([r[k] for r in group]))
                        for k in [
                            "first_position_mm",
                            "chunk_position_mm",
                            "prediction_change_mm",
                        ]
                    },
                )
            )
    report = dict(
        protocol="Frozen final10000 MLP; full seven validation episodes; donor image index=(t+floor(n/2))%n within episode. State, targets and weights unchanged. Original inference checked against saved full evaluation. This is sensitivity to mismatched inputs, not retraining ablation or deployment protocol.",
        aggregate=aggregate,
        per_episode=rows,
    )
    (root / "visual_conditioning.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
