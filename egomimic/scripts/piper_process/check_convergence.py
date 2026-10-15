"""Compare FM convergence with direct regression on identical frozen features."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import default_collate

from egomimic.rldb.embodiment.piper import Piper
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset
from egomimic.scripts.piper_process.evaluate_full import (
    ACTION,
    DOMAIN,
    load_model,
    pose_errors,
)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--samples", type=int, choices=[1, 8], default=1)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--noise-batch", type=int, default=16)
    parser.add_argument(
        "--resume-fm",
        type=Path,
        help="Load diagnostic head weights; start a fresh optimizer and cosine schedule.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--episode", type=Path, required=True, help="training-split Zarr episode"
    )
    parser.add_argument(
        "--valid-report",
        type=Path,
        required=True,
        help="JSON whose 'episodes' lists validation episodes (excluded here)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    args.output.mkdir(exist_ok=True)
    wrapper, checkpoint_step = load_model(args.checkpoint)
    model, policy = wrapper.model, wrapper.model.nets["policy"]
    policy.train_mode = False
    head = policy.heads[DOMAIN]
    if args.resume_fm:
        resume = torch.load(args.resume_fm, map_location="cuda", weights_only=False)
        head.load_state_dict(resume["head"], strict=True)
    episode = args.episode
    valid = json.loads(args.valid_report.read_text())["episodes"]
    assert episode.stem not in valid
    ds = ZarrDataset(
        episode, Piper.get_keymap(), Piper.get_transform_list(), strict=True
    )
    indices = np.linspace(0, len(ds) - 30, 8, dtype=int)
    if args.samples == 1:
        indices = indices[3:4]
    raw = default_collate([ds[int(i)] for i in indices])
    target = raw[ACTION].numpy().copy()
    normalized = model.norm_stats.normalize(raw, 7)
    batch = model.process_batch_for_training({DOMAIN: normalized})[7]
    data = model._robomimic_to_hpt_data(
        batch,
        model.camera_keys[7],
        model.proprio_keys[7],
        model.lang_keys[7],
        model.ac_keys[7],
    )
    with torch.no_grad():
        features = policy.forward_features(DOMAIN, dict(data))[0].detach()
    actions = data["action"].detach()
    restored = model.norm_stats.unnormalize({ACTION: actions}, 7)[ACTION].cpu().numpy()
    np.testing.assert_allclose(restored, target, atol=2e-6)
    torch.save(
        dict(
            features=features.cpu(),
            actions=actions.cpu(),
            target=target,
            frames=indices,
        ),
        args.output / "inputs.pt",
    )
    results = {"fm": [], "regression": []}

    def evaluate(kind, network, iteration, seeds=(0, 1, 2)):
        network.eval()
        with torch.no_grad():
            if kind == "fm":
                predictions = [
                    head.sample_action(
                        features,
                        DOMAIN,
                        generator=torch.Generator(device="cuda").manual_seed(seed),
                    )
                    for seed in seeds
                ]
            else:
                predictions = [network(features).reshape(len(features), 30, 7)]
            pred = np.stack(
                [
                    model.norm_stats.unnormalize({ACTION: p}, 7)[ACTION].cpu().numpy()
                    for p in predictions
                ]
            )
        errors = pose_errors(pred, np.broadcast_to(target, pred.shape))
        position = errors["position_mm"]
        record = dict(
            step=iteration,
            position_mm=float(position.mean()),
            position_p95_mm=float(np.percentile(position, 95)),
            rotation_deg=float(errors["rotation_deg"].mean()),
            gripper_mm=float(errors["gripper_mm"].mean()),
            adjacent_mm=float(
                np.linalg.norm(np.diff(pred[..., :3], axis=2), axis=-1).mean() * 1000
            ),
            seed_spread_mm=float(
                np.linalg.norm(pred[..., :3] - pred[..., :3].mean(0), axis=-1).mean()
                * 1000
            ),
            seeds=list(seeds) if kind == "fm" else [],
        )
        results[kind].append(record)
        suffix = "_seeds10-19" if kind == "fm" and seeds[0] == 10 else ""
        np.savez_compressed(
            args.output / f"{kind}_{iteration}{suffix}.npz",
            prediction=pred,
            target=target,
            frames=indices,
        )
        print(kind, record, flush=True)
        (args.output / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
        return record

    # Deterministic MLP diagnostic, not a drop-in architecture comparison.
    torch.manual_seed(2026)
    reg = nn.Sequential(
        nn.LayerNorm(features.shape[-1]),
        nn.Flatten(1),
        nn.Linear(features[0].numel(), 256),
        nn.GELU(),
        nn.Linear(256, 210),
    ).cuda()
    optimizer = torch.optim.AdamW(reg.parameters(), lr=3e-4, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=2000, eta_min=1e-6
    )
    for step in range(1, 2001):
        optimizer.zero_grad(set_to_none=True)
        loss = (reg(features).reshape_as(actions) - actions).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reg.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step in (200, 500, 1000, 2000):
            evaluate("regression", reg, step)
    torch.save(reg.state_dict(), args.output / "regression.pt")

    head.eval()  # Keep gradients; disable dropout for this memorization diagnostic.
    torch.manual_seed(2026)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=1e-6
    )
    assert args.noise_batch % len(features) == 0
    repeats = args.noise_batch // len(features)
    train_features = features.repeat_interleave(repeats, dim=0)
    train_actions = actions.repeat_interleave(repeats, dim=0)
    losses = []
    evaluate("fm", head, 0)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        pred, truth = head.predict(train_actions, train_features)
        loss = head.loss_fn(pred, truth)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        if step % 100 == 0:
            print(
                f"FM step {step}: loss {np.mean(losses[-100:]):.6f}, lr {scheduler.get_last_lr()[0]:.7f}",
                flush=True,
            )
        if step % 1000 == 0 or step == args.steps:
            evaluate("fm", head, step)
    # Additional unseen inference seeds test residual stochastic variation.
    final = evaluate("fm", head, args.steps, seeds=tuple(range(10, 20)))
    torch.save(
        dict(
            head=head.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            step=args.steps,
        ),
        args.output / "fm_checkpoint.pt",
    )
    report = dict(
        checkpoint_step=checkpoint_step,
        episode=episode.stem,
        frames=indices.tolist(),
        fm_train_batch=args.noise_batch,
        resumed_head=str(args.resume_fm) if args.resume_fm else None,
        optimizer_reset=True,
        fm_steps=args.steps,
        fm_lr=args.lr,
        fm_eta_min=1e-6,
        position_encoding="original learned, unchanged",
        frozen_features=True,
        dropout=False,
        precision="FP32",
        regression="LayerNorm -> flatten all 30x256 tokens -> Linear(256) -> GELU -> Linear(210); AdamW lr=3e-4 cosine to1e-6, no weight decay, 2000 steps",
        target_adjacent_mm=float(
            np.linalg.norm(np.diff(target[..., :3], axis=1), axis=-1).mean() * 1000
        ),
        results=results,
        losses=losses,
        position_goal_mm=5,
        position_goal_pass=final["position_mm"] <= 5,
        note="Training memorization only. Regression changes architecture/objective/optimization, not proof that FM is intrinsically defective. Direct regression is deterministic; zero seed spread is by construction.",
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
