"""Isolate the trained FM head on eight fixed training observations."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import default_collate

from egomimic.models.fm_policy import FMPolicy
from egomimic.rldb.embodiment.piper import Piper
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset
from egomimic.scripts.piper_process.evaluate_full import ACTION, DOMAIN, load_model


def check_flow_math():
    """Exercise production interpolation/integration against a known straight path."""

    class Oracle(torch.nn.Module):
        def forward(self, x, t, cond):
            self.x, self.t = x.detach().clone(), t.detach().clone()
            self.calls = getattr(self, "calls", 0) + 1
            return cond

    oracle = Oracle()
    head = FMPolicy(oracle, 30, {"test": 7}, num_inference_steps=50)
    actions = torch.randn(2, 30, 7)
    noise = torch.randn_like(actions)
    result = head.inference(noise, noise - actions)
    integration_error = float((result - actions).abs().max())
    calls = oracle.calls
    _, velocity = head.predict(actions, torch.zeros_like(actions))
    identity_error = float(
        (oracle.x - oracle.t[:, None, None] * velocity - actions).abs().max()
    )
    assert calls == 50 and integration_error < 1e-4 and identity_error < 1e-6
    return dict(
        integration_steps=calls,
        integration_max_abs_error=integration_error,
        interpolation_max_abs_error=identity_error,
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--position", choices=["original", "sinusoidal"], default="original"
    )
    parser.add_argument(
        "--output", type=Path, default=Path(".local_calibration/flow_check")
    )
    parser.add_argument("--samples", type=int, choices=[1, 8], default=8)
    parser.add_argument(
        "--single-repeats",
        type=int,
        default=1,
        help="Independent noise draws per update for the same single observation.",
    )
    parser.add_argument("--scope", choices=["head", "full"], default="head")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--deterministic-fit",
        action="store_true",
        help="Disable dropout, augmentation and BN running-stat updates; gradients remain enabled.",
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    output = args.output
    output.mkdir(exist_ok=True)
    wrapper, step = load_model(
        Path(".local_calibration/hpc/results/checkpoints/a/best.ckpt")
    )
    model = wrapper.model
    policy = model.nets["policy"]
    policy.train_mode = False
    head = policy.heads[DOMAIN]
    manifest = json.loads(
        Path(".local_calibration/full_validation/report.json").read_text()
    )
    path = next(
        p
        for p in sorted(Path(".local_calibration/piper_all").glob("*.zarr"))
        if p.stem not in manifest["episodes"]
    )
    ds = ZarrDataset(path, Piper.get_keymap(), Piper.get_transform_list(), strict=True)
    indices = np.linspace(0, len(ds) - 30, 8, dtype=int)
    if args.samples == 1:
        indices = np.repeat(indices[3:4], args.single_repeats)
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
        features, _ = policy.forward_features(DOMAIN, dict(data))
    features = features.detach()
    actions = data["action"].detach()
    records = []

    def evaluate(tag, steps=50):
        policy.eval()
        head.num_inference_steps = steps
        with torch.no_grad():
            eval_features = (
                policy.forward_features(DOMAIN, dict(data))[0]
                if args.scope == "full"
                else features
            )
            predictions = []
            for seed in range(3):
                gen = torch.Generator(device="cuda").manual_seed(seed)
                pred = head.sample_action(eval_features, DOMAIN, generator=gen)
                predictions.append(
                    model.norm_stats.unnormalize({ACTION: pred}, 7)[ACTION]
                    .cpu()
                    .numpy()
                )
        pred = np.stack(predictions)
        record = dict(
            tag=tag,
            steps=steps,
            position_mm=float(
                np.linalg.norm(pred[..., :3] - target[None, ..., :3], axis=-1).mean()
                * 1000
            ),
            adjacent_mm=float(
                np.linalg.norm(np.diff(pred[..., :3], axis=2), axis=-1).mean() * 1000
            ),
        )
        records.append(record)
        np.savez_compressed(
            output / (tag + ".npz"), prediction=pred, target=target, frames=indices
        )
        print(record, flush=True)
        return pred

    original = evaluate("original_50")
    evaluate("original_200", 200)
    head.num_inference_steps = 50
    with torch.no_grad():
        noise = torch.randn_like(actions)
        # Exact constant velocity along one known noise-to-data line must recover actions.
        x, time, count = noise.clone(), torch.ones(len(indices), device="cuda"), 0
        while time[0] >= 0.01:
            x = x - 0.02 * (noise - actions)
            time = time - 0.02
            count += 1
        oracle_error = float((x - actions).abs().max())
        assert count == 50 and oracle_error < 1e-4
        t = torch.full((len(indices),), 0.5, device="cuda")
        perm = torch.arange(29, -1, -1, device="cuda")
        v = head.model(noise, t, features)
        permuted = head.model(noise[:, perm], t, features)[:, perm]
        permutation_delta = float((v - permuted).square().mean().sqrt())
        condition_delta = float(
            (v - head.model(noise, t, features.flip(0))).square().mean().sqrt()
        )
    # Cache observations/trunk features; optimize only a diagnostic copy's FM head.
    if args.position == "sinusoidal":
        with torch.no_grad():
            p = head.model.pos_emb
            frequencies = torch.exp(
                torch.arange(0, p.shape[-1], 2, device=p.device)
                * (-np.log(10000.0) / p.shape[-1])
            )
            phase = torch.arange(p.shape[1], device=p.device)[:, None] * frequencies
            encoding = torch.zeros_like(p)
            encoding[0, :, 0::2] = phase.sin()
            encoding[0, :, 1::2] = phase.cos()
            p.add_(encoding)
    # Pair all training noise, time draws and dropout between diagnostic branches.
    torch.manual_seed(2026)
    fit_module = policy if args.scope == "full" else head
    optimizer = torch.optim.AdamW(fit_module.parameters(), lr=1e-4, weight_decay=1e-4)
    gradient_audit = {}
    losses = []
    for iteration in range(1, args.steps + 1):
        fit_module.train(not args.deterministic_fit)
        optimizer.zero_grad(set_to_none=True)
        train_features = (
            policy.forward_features(DOMAIN, dict(data))[0]
            if args.scope == "full"
            else features
        )
        pred, truth = head.predict(actions, train_features)
        loss = head.loss_fn(pred, truth)
        loss.backward()
        if iteration == 1:
            for name, parameter in fit_module.named_parameters():
                if parameter.grad is not None:
                    group = name.split(".")[0]
                    gradient_audit[group] = gradient_audit.get(group, 0.0) + float(
                        parameter.grad.detach().square().sum()
                    )
        torch.nn.utils.clip_grad_norm_(fit_module.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if iteration % 50 == 0:
            print(
                f"step {iteration}: FM loss mean {np.mean(losses[-50:]):.6f}",
                flush=True,
            )
        if iteration in (200, 500, 1000, args.steps):
            final = evaluate(f"overfit_{iteration}")
    report = dict(
        source_checkpoint_step=step,
        position_intervention=args.position,
        scope=args.scope,
        deterministic_fit=args.deterministic_fit,
        updates=args.steps,
        unique_observations=int(len(np.unique(indices))),
        batch_size=len(indices),
        first_step_gradient_squared_norm_by_module=gradient_audit,
        implementation_checks=check_flow_math(),
        episode=path.stem,
        frames=indices.tolist(),
        protocol=f"A checkpoint, {len(indices)} fixed TRAIN observations. Scope={args.scope}, deterministic_fit={args.deterministic_fit}. FP32 AdamW lr=1e-4, weight_decay=1e-4, gradient clip=1, {args.steps} updates. Three paired inference seeds. No validation fit; does not establish generalization. Deterministic fit keeps BN running stats fixed and disables dropout; full scope backpropagates through encoders/stems/trunk/head.",
        oracle_steps=count,
        oracle_max_abs_error=oracle_error,
        velocity_permutation_delta_rms=permutation_delta,
        velocity_condition_shuffle_delta_rms=condition_delta,
        target_adjacent_mm=float(
            np.linalg.norm(np.diff(target[..., :3], axis=1), axis=-1).mean() * 1000
        ),
        evaluations=records,
        losses=losses,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for row, (label, prediction) in enumerate(
        [("Original", original), (f"{args.scope} overfit {args.steps}", final)]
    ):
        for c, axis in enumerate("XYZ"):
            ax = axes[row, c]
            example = min(3, len(indices) - 1)
            ax.plot(target[example, :, c] * 1000, color="green", label="Demo")
            for seed, (color, style) in enumerate(
                [("#7b3294", "-"), ("#0072b2", "--"), ("#d55e00", ":")]
            ):
                ax.plot(
                    prediction[seed, example, :, c] * 1000,
                    style,
                    color=color,
                    label=f"seed {seed}",
                )
            ax.set(title=f"{label}: {axis}", xlabel="Future frame", ylabel="mm")
            ax.legend()
    fig.tight_layout()
    fig.savefig(output / "overfit.png", dpi=150)
    torch.save(fit_module.state_dict(), output / f"diagnostic_{args.scope}.pt")


if __name__ == "__main__":
    main()
