"""Full, non-resampled robot validation for deterministic Piper HPT policies."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from egomimic.eval.eval import Eval
from egomimic.rldb.embodiment.piper import Piper
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset, split_dataset_names
from egomimic.scripts.piper_process.evaluate_full import (
    ACTION,
    DOMAIN,
    HORIZON,
    pose_errors,
)


class FullPiperEval(Eval):
    def __init__(self, dataset_dir, batch_size=32):
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
        self.trainer = None
        self.model = None
        self.datasets = None

    def on_validation_start(self):
        if self.datasets is None:
            paths = sorted(self.dataset_dir.glob("*.zarr"))
            _, valid = split_dataset_names(
                [p.stem for p in paths], valid_ratio=0.1, seed=42
            )
            self.datasets = [
                ZarrDataset(
                    p,
                    Piper.get_keymap(action_horizon=HORIZON),
                    Piper.get_transform_list(),
                    strict=True,
                )
                for p in paths
                if p.stem in valid
            ]

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        if batch_idx != 0:
            return
        model = self.model
        policy = model.nets["policy"]
        assert (
            not policy.diffusion
        ), "FullPiperEval currently selects deterministic policies only."
        policy.train_mode = False
        episodes = []
        # Match offline evaluation precision even during BF16 training.
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            for dataset in self.datasets:
                cursor = 0
                totals = dict(
                    chunk_position_mm=0.0,
                    first_position_mm=0.0,
                    chunk_rotation_deg=0.0,
                    chunk_gripper_mm=0.0,
                )
                for raw in DataLoader(
                    dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
                ):
                    target = raw[ACTION].numpy()
                    normalized = model.norm_stats.normalize(raw, 7)
                    processed = model.process_batch_for_training({DOMAIN: normalized})[
                        7
                    ]
                    data = model._robomimic_to_hpt_data(
                        processed,
                        model.camera_keys[7],
                        model.proprio_keys[7],
                        model.lang_keys[7],
                        model.ac_keys[7],
                    )
                    pred = policy.forward(DOMAIN, data)[DOMAIN]
                    pred = (
                        model.norm_stats.unnormalize({ACTION: pred.float()}, 7)[ACTION]
                        .cpu()
                        .numpy()
                    )
                    n = len(target)
                    mask = (
                        np.arange(HORIZON)[None, :]
                        < (len(dataset) - np.arange(cursor, cursor + n))[:, None]
                    )
                    errors = pose_errors(pred, target)
                    for key, error in errors.items():
                        totals["chunk_" + key] += float(
                            ((error * mask).sum(1) / mask.sum(1)).sum()
                        )
                    totals["first_position_mm"] += float(
                        errors["position_mm"][:, 0].sum()
                    )
                    cursor += n
                assert cursor == len(dataset)
                episodes.append(
                    dict(
                        episode=Path(dataset.episode_path).stem,
                        frames=cursor,
                        **{k: v / cursor for k, v in totals.items()},
                    )
                )
        metrics = {k: float(np.mean([r[k] for r in episodes])) for k in totals}
        for key, value in metrics.items():
            self.trainer.lightning_module.log(
                "Valid/robot_" + key,
                value,
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=False,
            )
        output = Path(self.trainer.default_root_dir) / "full_validation"
        output.mkdir(exist_ok=True)
        report = dict(
            step=int(self.trainer.global_step),
            frames=sum(r["frames"] for r in episodes),
            episodes=episodes,
            metrics=metrics,
            precision="float32",
            resampling=False,
            tail_padding_scored=False,
            aggregation="Equal valid horizons per anchor, anchors per episode, equal episode mean",
        )
        (output / f"step_{self.trainer.global_step}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print("[FULL_ROBOT_VALIDATION]", json.dumps(report), flush=True)

    def on_validation_end(self):
        pass
