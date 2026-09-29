"""Deploy the Piper joint-action MLP; --execute enables SDK commands."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from egomimic.robot.piper.robot_interface import (
    CALIBRATION,
    ROOT,
    Geometry,
    PiperHardware,
)
from egomimic.scripts.piper_process.evaluate_full import ACTION, DOMAIN, load_model
from egomimic.scripts.piper_process.evaluate_joint_heads import JOINT_ACTION
from egomimic.utils.provenance import run_record, sha256


class JointMLPPolicy:
    def __init__(self, checkpoint):
        self.wrapper, self.step = load_model(checkpoint)
        self.model = self.wrapper.model
        self.policy = self.model.nets["policy"]
        self.head = self.policy.heads[f"{DOMAIN}_{JOINT_ACTION}"]
        if self.policy.diffusion or hasattr(self.head, "sample_action"):
            raise ValueError("This deployment expects the deterministic joint MLP head")
        self.horizon = self.head.action_horizon
        state_stats = self.model.norm_stats.norm_stats[7]["observations.state.joints"]
        if np.asarray(state_stats["mean"]).size != 7:
            raise ValueError("Checkpoint does not contain seven-channel joint state")

    @torch.inference_mode()
    def predict(self, state, front, wrist):
        raw = {
            "observations.state.joints": torch.as_tensor(state, dtype=torch.float32)[
                None
            ],
            ACTION: torch.zeros(1, self.horizon, 7),
            JOINT_ACTION: torch.zeros(1, self.horizon, 7),
        }
        for key, rgb in [("front_img_1", front), ("right_wrist_img", wrist)]:
            if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                raise ValueError("Cameras must supply 640x480 uint8 RGB images")
            raw[f"observations.images.{key}"] = (
                torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None] / 255
            )
        batch = self.model.process_batch_for_training(
            {DOMAIN: self.model.norm_stats.normalize(raw, 7)}
        )[7]
        data = self.model._robomimic_to_hpt_data(
            batch,
            self.model.camera_keys[7],
            self.model.proprio_keys[7],
            self.model.lang_keys[7],
            self.model.ac_keys[7],
            self.model.auxiliary_ac_keys[DOMAIN],
        )
        features, _ = self.policy.forward_features(DOMAIN, data)
        raw_prediction = self.head(features).float()
        prediction = raw_prediction.clamp(-1, 1)
        result = (
            self.model.norm_stats.unnormalize({JOINT_ACTION: prediction}, 7)[
                JOINT_ACTION
            ][0]
            .cpu()
            .numpy()
        )
        raw_result = (
            self.model.norm_stats.unnormalize({JOINT_ACTION: raw_prediction}, 7)[
                JOINT_ACTION
            ][0]
            .cpu()
            .numpy()
        )
        normalized_overrun = (raw_prediction[0].abs() - 1).clamp_min(0).cpu().numpy()
        if not np.isfinite([result, raw_result, normalized_overrun]).all():
            raise RuntimeError("Nonfinite model output")
        return result, raw_result, normalized_overrun


def clip_near_joint_limits(joints, bounds, tolerance_deg):
    """Clip small boundary overshoots and reject targets far outside the URDF."""
    joints = np.asarray(joints)
    lower_error = np.maximum(bounds[:, 0] - joints, 0)
    upper_error = np.maximum(joints - bounds[:, 1], 0)
    violation_deg = np.rad2deg(np.maximum(lower_error, upper_error))
    worst = int(np.argmax(violation_deg))
    if violation_deg[worst] > tolerance_deg:
        raise RuntimeError(
            f"J{worst + 1} target exceeds URDF limit by {violation_deg[worst]:.2f}deg"
        )
    return np.clip(joints, bounds[:, 0], bounds[:, 1]), violation_deg


def temporal_ensemble(history, step, decay):
    """Average predictions for one absolute step, favoring newer chunks."""
    predictions = []
    ages = []
    for origin, chunk in history:
        horizon = step - origin
        if 0 <= horizon < len(chunk):
            predictions.append(chunk[horizon])
            ages.append(horizon)
    if not predictions:
        raise RuntimeError("No joint prediction covers the current action step")
    weights = np.exp(-decay * np.asarray(ages, dtype=np.float64))
    weights /= weights.sum()
    return np.average(predictions, axis=0, weights=weights), len(predictions)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / ".local_calibration/hpc/joint/checkpoints/a/best.ckpt",
    )
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--port", default="can_follower")
    parser.add_argument("--wrist-serial", default="427622273943")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--speed-percent", type=int, default=30)
    parser.add_argument(
        "--joint-limit-tolerance-deg",
        type=float,
        default=3.0,
        help="Clip URDF-limit overshoots up to this size; reject larger ones",
    )
    parser.add_argument(
        "--query-every",
        "--chunk-size",
        dest="query_every",
        type=int,
        default=10,
        help="Chunk mode only: execute this prefix before observing again (1..30)",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("ensemble", "chunk"),
        default="ensemble",
        help="Replan every step with temporal ensembling, or execute chunk prefixes",
    )
    parser.add_argument(
        "--ensemble-decay",
        type=float,
        default=0.05,
        help="Exponential age penalty used by temporal ensembling",
    )
    parser.add_argument(
        "--min-start-gripper-mm",
        type=float,
        default=90.0,
        help="Reject execution below this initial opening; use 0 to disable",
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=30.0,
        help="Joint command frequency; training trajectories use 30Hz",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save one uncompressed image/prediction NPZ every N queries; 0 disables",
    )
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / ".local_calibration/rollouts_joint"
        / time.strftime("%Y%m%d_%H%M%S"),
    )
    args = parser.parse_args()
    if (
        not 1 <= args.speed_percent <= 100
        or not 1 <= args.query_every <= 30
        or args.duration <= 0
        or args.joint_limit_tolerance_deg < 0
        or args.ensemble_decay < 0
        or args.min_start_gripper_mm < 0
        or args.control_hz <= 0
        or args.save_every < 0
    ):
        parser.error(
            "speed-percent:1..100, chunk-size:1..30, duration/control-hz must be "
            "positive, save-every must be nonnegative"
        )

    geometry = Geometry(args.calibration)
    policy = JointMLPPolicy(args.checkpoint)
    policy.predict(
        np.zeros(7, np.float32),
        np.zeros((480, 640, 3), np.uint8),
        np.zeros((480, 640, 3), np.uint8),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint.resolve()
    (args.output / "config.json").write_text(
        json.dumps(
            {
                **{
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "checkpoint_step": policy.step,
                "checkpoint_sha256": sha256(checkpoint),
                "source": run_record(ROOT),
                "model_action_hz": 30,
                "control_hz": args.control_hz,
                "action_space": "absolute URDF joints rad + gripper m",
                "replan_hz_nominal": (
                    args.control_hz
                    if args.execution_mode == "ensemble"
                    else args.control_hz / args.query_every
                ),
            },
            indent=2,
        )
    )

    robot = PiperHardware(geometry, args.port, args.wrist_serial)
    started = time.monotonic()
    counter = 0
    action_step = 0
    history = []
    try:
        initial_raw = robot.read_state()
        initial_state = geometry.joint_state(initial_raw)
        initial_gripper_mm = initial_state[6] * 1000
        if (
            args.execute
            and args.min_start_gripper_mm
            and initial_gripper_mm < args.min_start_gripper_mm
        ):
            raise RuntimeError(
                f"Initial gripper opening is {initial_gripper_mm:.1f}mm; "
                f"robot demonstrations start open (required "
                f">={args.min_start_gripper_mm:.1f}mm)"
            )
        if args.execute:
            robot.start_control(args.speed_percent)
        print(
            f"{'EXECUTE' if args.execute else 'READ ONLY'} | checkpoint step "
            f"{policy.step} | logs: {args.output}",
            flush=True,
        )
        with (args.output / "commands.jsonl").open("w") as log:
            while time.monotonic() - started < args.duration:
                front, wrist = robot.images()
                raw = robot.read_state()
                state = geometry.joint_state(raw)
                t0 = time.monotonic()
                chunk, raw_chunk, normalized_overrun = policy.predict(
                    state, front, wrist
                )
                inference_ms = (time.monotonic() - t0) * 1000
                if args.save_every and counter % args.save_every == 0:
                    np.savez(
                        args.output / f"query_{counter:04d}.npz",
                        raw_state=raw,
                        state=state,
                        prediction=chunk,
                        raw_prediction=raw_chunk,
                        normalized_overrun=normalized_overrun,
                        front=front,
                        wrist=wrist,
                    )
                if args.execution_mode == "ensemble":
                    history.append((action_step, chunk))
                    history = [
                        item
                        for item in history
                        if action_step - item[0] < policy.horizon
                    ]
                    ensemble_action, ensemble_sources = temporal_ensemble(
                        history, action_step, args.ensemble_decay
                    )
                    actions = [(-1, ensemble_action)]
                else:
                    ensemble_sources = 1
                    actions = enumerate(chunk[: args.query_every])
                first_command = None
                first_violation = None
                first_action = None
                for horizon, action in actions:
                    if time.monotonic() - started >= args.duration:
                        break
                    tick = time.monotonic()
                    joints, violation_deg = clip_near_joint_limits(
                        action[:6],
                        geometry.model_bounds,
                        args.joint_limit_tolerance_deg,
                    )
                    if horizon == 0:
                        first_command = joints
                        first_violation = violation_deg
                        first_action = action
                    elif horizon == -1:
                        first_command = joints
                        first_violation = violation_deg
                        first_action = action
                    feedback_target = geometry.feedback_from_model(joints)
                    live_raw = robot.read_state()
                    live_state = geometry.joint_state(live_raw)
                    if args.execute:
                        robot.send(feedback_target, action[6])
                    source_horizon = max(horizon, 0)
                    log.write(
                        json.dumps(
                            dict(
                                query=counter,
                                horizon=horizon,
                                action_step=action_step,
                                ensemble_sources=ensemble_sources,
                                elapsed_s=time.monotonic() - started,
                                inference_ms=inference_ms,
                                observed=live_state.tolist(),
                                query_observed=state.tolist(),
                                target=action.tolist(),
                                raw_target=raw_chunk[source_horizon].tolist(),
                                normalized_overrun=normalized_overrun[
                                    source_horizon
                                ].tolist(),
                                commanded_joints_rad=joints.tolist(),
                                joint_limit_violation_deg=violation_deg.tolist(),
                                delta_joint_deg=np.rad2deg(
                                    joints - live_state[:6]
                                ).tolist(),
                                sent=args.execute,
                            )
                        )
                        + "\n"
                    )
                    log.flush()
                    action_step += 1
                    time.sleep(max(0, 1 / args.control_hz - (time.monotonic() - tick)))
                if first_command is None:
                    break
                delta = np.rad2deg(first_command - state[:6])
                print(
                    f"query {counter}: inference {inference_ms:.0f}ms, first joints "
                    f"{np.rad2deg(first_command).round(1)}, max delta "
                    f"{np.abs(delta).max():.1f}deg, limit clip "
                    f"{first_violation.max():.2f}deg, gripper "
                    f"{first_action[6] * 1000:.1f}mm, sources {ensemble_sources}",
                    flush=True,
                )
                counter += 1
    except KeyboardInterrupt:
        print("Interrupted; stopping arm commands.", flush=True)
    finally:
        robot.close()


if __name__ == "__main__":
    main()
