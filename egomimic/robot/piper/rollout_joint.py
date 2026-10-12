"""Deploy a Piper MLP or flow-matching joint-action policy."""

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from egomimic.robot.piper.robot_interface import (
    CALIBRATION,
    ROOT,
    Geometry,
    PiperHardware,
)
from egomimic.scripts.piper_process.evaluate_full import DOMAIN, load_model
from egomimic.scripts.piper_process.evaluate_joint_heads import JOINT_ACTION
from egomimic.utils.provenance import run_record, sha256

INITIAL_POSE = Path(__file__).with_name("initial_pose.json")


class JointPolicy:
    def __init__(self, checkpoint, flow_seed=0):
        self.wrapper, self.step = load_model(checkpoint)
        self.model = self.wrapper.model
        self.policy = self.model.nets["policy"]
        self.head = self.policy.heads[f"{DOMAIN}_{JOINT_ACTION}"]
        self.is_flow = hasattr(self.head, "sample_action")
        self.head_name = getattr(self.head, "contract_name", None) or (
            "FM" if self.is_flow else "MLP"
        )
        self.binary_gripper = self.head_name in {
            "piper_residual_joint_transformer",
            "piper_residual_joint_clean_v2",
        }
        self.flow_seed = flow_seed
        self.model_horizon = self.head.action_horizon
        self.horizon = getattr(self.head, "control_horizon", self.model_horizon)
        self.primary_action_key = self.model.ac_keys[7]
        state_stats = self.model.norm_stats.norm_stats[7]["observations.state.joints"]
        if np.asarray(state_stats["mean"]).size != 7:
            raise ValueError("Checkpoint does not contain seven-channel joint state")

    @torch.inference_mode()
    def predict(self, state, front, wrist):
        raw = {
            "observations.state.joints": torch.as_tensor(state, dtype=torch.float32)[
                None
            ]
        }
        for action_key in {self.primary_action_key, JOINT_ACTION}:
            action_stats = self.model.norm_stats.norm_stats[7][action_key]
            action_dims = int(np.asarray(action_stats["mean"]).shape[-1])
            raw[action_key] = torch.zeros(1, self.model_horizon, action_dims)
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
        if self.is_flow:
            generator = torch.Generator(device=features.device).manual_seed(
                self.flow_seed
            )
            raw_prediction = self.head.sample_action(
                features, DOMAIN, generator=generator
            ).float()
            prediction = raw_prediction
            normalized_overrun = torch.zeros_like(raw_prediction[0]).cpu().numpy()
        else:
            if getattr(self.head, "requires_data_for_inference", False):
                raw_prediction = self.head(features, data).float()
            else:
                raw_prediction = self.head(features).float()
            prediction = raw_prediction.clamp(-1, 1)
            normalized_overrun = (
                (raw_prediction[0].abs() - 1).clamp_min(0).cpu().numpy()
            )
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
        if self.binary_gripper:
            result[:, 6] = (result[:, 6] >= 0.5) * 0.1
            raw_result[:, 6] = (raw_result[:, 6] >= 0.5) * 0.1
        if self.horizon != self.model_horizon:
            control_indices = uniform_horizon_indices(self.model_horizon, self.horizon)
            result = result[control_indices]
            raw_result = raw_result[control_indices]
            normalized_overrun = normalized_overrun[control_indices]
        if not np.isfinite([result, raw_result, normalized_overrun]).all():
            raise RuntimeError("Nonfinite model output")
        return result, raw_result, normalized_overrun


def uniform_horizon_indices(model_horizon, control_horizon):
    """Select control points uniformly across a model trajectory."""
    return np.rint(np.linspace(0, model_horizon - 1, control_horizon)).astype(np.int64)


def apply_joint_limits(joints, bounds, tolerance_deg, allow_small_overshoot=False):
    """Clip or pass small overshoots; reject targets far outside the URDF."""
    joints = np.asarray(joints)
    lower_error = np.maximum(bounds[:, 0] - joints, 0)
    upper_error = np.maximum(joints - bounds[:, 1], 0)
    violation_deg = np.rad2deg(np.maximum(lower_error, upper_error))
    worst = int(np.argmax(violation_deg))
    if violation_deg[worst] > tolerance_deg:
        raise RuntimeError(
            f"J{worst + 1} target exceeds URDF limit by {violation_deg[worst]:.2f}deg"
        )
    if allow_small_overshoot:
        return joints, violation_deg
    return np.clip(joints, bounds[:, 0], bounds[:, 1]), violation_deg


def limit_joint_command_step(target, feedback, max_step_deg):
    """Limit the largest commanded joint delta while preserving its direction."""
    delta = np.asarray(target) - np.asarray(feedback)
    raw_delta_deg = np.rad2deg(delta)
    largest = float(np.abs(raw_delta_deg).max())
    scale = min(1.0, max_step_deg / largest) if largest else 1.0
    command = np.asarray(feedback) + scale * delta
    return command, raw_delta_deg, scale


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


def load_initial_pose(path):
    spec = json.loads(Path(path).read_text())
    state = np.r_[spec["model_joints_rad"], spec["gripper_m"]].astype(float)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError("Initial pose must contain six finite joints and a gripper")
    if not 0 <= state[6] <= 0.1:
        raise ValueError("Initial gripper opening must be between 0 and 0.1m")
    return state


def initialize_robot(
    robot,
    geometry,
    target,
    timeout_s,
    joint_tolerance_deg,
    gripper_tolerance_mm,
    control_hz,
    log=None,
):
    feedback_target = geometry.feedback_from_model(target[:6])
    started = time.monotonic()
    next_report = started
    stable_samples = 0
    while True:
        raw = robot.read_state()
        state = geometry.joint_state(raw)
        joint_error_deg = np.abs(np.rad2deg(target[:6] - state[:6]))
        gripper_error_mm = abs(target[6] - state[6]) * 1000
        if log is not None:
            log.write(
                json.dumps(
                    {
                        "elapsed_s": time.monotonic() - started,
                        "state": state.tolist(),
                        "target": target.tolist(),
                        "joint_error_deg": joint_error_deg.tolist(),
                        "gripper_error_mm": float(gripper_error_mm),
                    }
                )
                + "\n"
            )
            log.flush()
        if (
            joint_error_deg.max() <= joint_tolerance_deg
            and gripper_error_mm <= gripper_tolerance_mm
        ):
            stable_samples += 1
            if stable_samples >= 3:
                return {
                    "duration_s": time.monotonic() - started,
                    "target": target.tolist(),
                    "final_state": state.tolist(),
                    "joint_error_deg": joint_error_deg.tolist(),
                    "gripper_error_mm": float(gripper_error_mm),
                }
        else:
            stable_samples = 0
        now = time.monotonic()
        if now >= next_report:
            print(
                f"INITIALIZING | {now - started:.1f}s | error deg "
                f"{joint_error_deg.round(2)} | gripper {gripper_error_mm:.1f}mm",
                flush=True,
            )
            next_report = now + 1
        if now - started > timeout_s:
            raise RuntimeError(
                "Initialization timed out: max joint error "
                f"{joint_error_deg.max():.2f}deg, gripper error "
                f"{gripper_error_mm:.1f}mm"
            )
        robot.send(feedback_target, target[6])
        time.sleep(1 / control_hz)


def open_start_gripper(robot, geometry, target_mm, tolerance_mm, timeout_s):
    started = time.monotonic()
    while True:
        opening_mm = robot.read_state()[6] * geometry.grip_scale * 1000
        if abs(target_mm - opening_mm) <= tolerance_mm:
            return {"duration_s": time.monotonic() - started, "opening_mm": opening_mm}
        if time.monotonic() - started > timeout_s:
            raise RuntimeError(
                f"Opening gripper timed out: {opening_mm:.1f}mm / {target_mm:.1f}mm"
            )
        robot.send_gripper(target_mm / 1000)
        time.sleep(1 / 30)


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
    parser.add_argument(
        "--flow-seed",
        type=int,
        default=0,
        help="Fixed FM sampling seed; ignored by deterministic MLP checkpoints",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--speed-percent", type=int, default=30)
    parser.add_argument("--initial-pose", type=Path, default=INITIAL_POSE)
    parser.add_argument(
        "--initialize",
        action="store_true",
        help="Move to initial-pose before rollout; disabled by default",
    )
    parser.add_argument(
        "--skip-initialize",
        action="store_false",
        dest="initialize",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(initialize=False)
    parser.add_argument("--init-speed-percent", type=int, default=20)
    parser.add_argument("--init-timeout", type=float, default=30.0)
    parser.add_argument("--init-joint-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--init-gripper-tolerance-mm", type=float, default=5.0)
    parser.add_argument(
        "--max-chunk-displacement-deg",
        type=float,
        default=55.0,
        help="Reject a chunk too far from its observation before sending it",
    )
    parser.add_argument(
        "--max-tracking-error-deg",
        type=float,
        default=30.0,
        help="Stop if a command gets this far ahead of joint feedback",
    )
    parser.add_argument(
        "--max-command-step-deg",
        type=float,
        default=5.0,
        help=(
            "Limit the largest joint delta sent from live feedback; "
            "all six deltas are scaled together"
        ),
    )
    parser.add_argument(
        "--tracking-gate-deg",
        type=float,
        default=0.0,
        help="Hold each action until feedback is within this error; 0 disables",
    )
    parser.add_argument(
        "--tracking-gate-timeout",
        type=float,
        default=1.0,
        help="Stop if a tracking-gated action is not reached within this time",
    )
    parser.add_argument(
        "--joint-limit-tolerance-deg",
        type=float,
        default=3.0,
        help="Maximum URDF-limit overshoot eligible for clipping or pass-through",
    )
    parser.add_argument(
        "--allow-small-joint-limit-overshoot",
        action="store_true",
        help=(
            "Send joint targets outside URDF limits when their violation is within "
            "joint-limit-tolerance-deg; larger violations still stop execution"
        ),
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
        "--start-gripper-mm",
        type=float,
        default=100.0,
        help="Open the gripper to this width before execution; use 0 to disable",
    )
    parser.add_argument("--start-gripper-timeout", type=float, default=5.0)
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
        or not 1 <= args.init_speed_percent <= 100
        or not 1 <= args.query_every <= 30
        or args.duration <= 0
        or args.init_timeout <= 0
        or args.init_joint_tolerance_deg <= 0
        or args.init_gripper_tolerance_mm <= 0
        or args.max_chunk_displacement_deg <= 0
        or args.max_tracking_error_deg <= 0
        or args.max_command_step_deg <= 0
        or args.tracking_gate_deg < 0
        or args.tracking_gate_timeout <= 0
        or args.joint_limit_tolerance_deg < 0
        or args.ensemble_decay < 0
        or args.min_start_gripper_mm < 0
        or not 0 <= args.start_gripper_mm <= 100
        or args.start_gripper_timeout <= 0
        or args.control_hz <= 0
        or args.save_every < 0
    ):
        parser.error(
            "speed percentages:1..100, chunk-size:1..30, durations/tolerances/"
            "control-hz must be positive, save-every must be nonnegative"
        )

    geometry = Geometry(args.calibration)
    initial_target = None
    if args.initialize:
        initial_target = load_initial_pose(args.initial_pose)
        initial_target[:6] = np.clip(
            initial_target[:6],
            geometry.model_bounds[:, 0],
            geometry.model_bounds[:, 1],
        )
    policy = JointPolicy(args.checkpoint, args.flow_seed)
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
                "head": policy.head_name,
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
    counter = 0
    action_step = 0
    history = []
    stage = "hardware setup"
    try:
        if args.execute:
            stage = "initialization" if args.initialize else "control startup"
            robot.start_control(
                args.init_speed_percent if args.initialize else args.speed_percent
            )
            if args.initialize:
                print(
                    f"INITIALIZE | speed {args.init_speed_percent}% | target joints "
                    f"{np.rad2deg(initial_target[:6]).round(2)}deg | gripper "
                    f"{initial_target[6] * 1000:.1f}mm",
                    flush=True,
                )
                with (args.output / "initialization.jsonl").open("w") as init_log:
                    initialization = initialize_robot(
                        robot,
                        geometry,
                        initial_target,
                        args.init_timeout,
                        args.init_joint_tolerance_deg,
                        args.init_gripper_tolerance_mm,
                        args.control_hz,
                        init_log,
                    )
                (args.output / "initialization.json").write_text(
                    json.dumps(initialization, indent=2)
                )
                robot.set_speed_percent(args.speed_percent)
                print(
                    f"INITIALIZED | {initialization['duration_s']:.1f}s | max error "
                    f"{max(initialization['joint_error_deg']):.2f}deg | gripper "
                    f"error {initialization['gripper_error_mm']:.1f}mm",
                    flush=True,
                )
            if args.start_gripper_mm:
                stage = "opening gripper"
                gripper_start = open_start_gripper(
                    robot,
                    geometry,
                    args.start_gripper_mm,
                    args.init_gripper_tolerance_mm,
                    args.start_gripper_timeout,
                )
                (args.output / "gripper_start.json").write_text(
                    json.dumps(gripper_start, indent=2)
                )
                print(
                    f"GRIPPER READY | {gripper_start['opening_mm']:.1f}mm | "
                    f"{gripper_start['duration_s']:.1f}s",
                    flush=True,
                )
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
        started = time.monotonic()
        stage = "policy rollout"
        print(
            f"{'EXECUTE' if args.execute else 'READ ONLY'} | checkpoint step "
            f"{policy.step} | head {policy.head_name} | "
            f"logs: {args.output}",
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
                chunk_displacement_deg = np.abs(
                    np.rad2deg(chunk[:, :6] - state[:6])
                ).max()
                guard_horizon = (
                    args.query_every if args.execution_mode == "chunk" else 1
                )
                guarded_delta_deg = np.abs(
                    np.rad2deg(chunk[:guard_horizon, :6] - state[:6])
                )
                guarded_displacement_deg = guarded_delta_deg.max()
                guard_horizon_index, guard_joint_index = np.unravel_index(
                    np.argmax(guarded_delta_deg), guarded_delta_deg.shape
                )
                guard_failed = (
                    args.execute
                    and guarded_displacement_deg > args.max_chunk_displacement_deg
                )
                if (args.save_every and counter % args.save_every == 0) or guard_failed:
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
                if guard_failed:
                    raise RuntimeError(
                        f"Predicted execution prefix is too far from current feedback: "
                        f"{guarded_displacement_deg:.1f}deg > "
                        f"{args.max_chunk_displacement_deg:.1f}deg at horizon "
                        f"{guard_horizon_index}, J{guard_joint_index + 1}"
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
                first_raw_step_deg = None
                first_applied_step_deg = None
                first_step_scale = None
                for horizon, action in actions:
                    if time.monotonic() - started >= args.duration:
                        break
                    joints, violation_deg = apply_joint_limits(
                        action[:6],
                        geometry.model_bounds,
                        args.joint_limit_tolerance_deg,
                        args.allow_small_joint_limit_overshoot,
                    )
                    if horizon in (0, -1):
                        first_violation = violation_deg
                        first_action = action
                    source_horizon = max(horizon, 0)
                    gate_started = time.monotonic()
                    gate_attempt = 0
                    held_gripper = None
                    while True:
                        tick = time.monotonic()
                        live_raw = robot.read_state()
                        live_state = geometry.joint_state(live_raw)
                        if held_gripper is None:
                            held_gripper = float(live_state[6])
                        commanded_joints, raw_step_deg, step_scale = (
                            limit_joint_command_step(
                                joints,
                                live_state[:6],
                                args.max_command_step_deg,
                            )
                        )
                        applied_step_deg = np.rad2deg(commanded_joints - live_state[:6])
                        if horizon in (0, -1) and first_command is None:
                            first_command = commanded_joints.copy()
                            first_raw_step_deg = raw_step_deg.copy()
                            first_applied_step_deg = applied_step_deg.copy()
                            first_step_scale = step_scale
                        feedback_target = geometry.feedback_from_model(commanded_joints)
                        control_status = robot.control_status()
                        if args.execute and (
                            control_status["ctrl_mode"] != 1
                            or control_status["arm_status"] != 0
                            or any(control_status["joint_limit_errors"])
                            or not all(control_status["enabled"])
                        ):
                            raise RuntimeError(
                                f"Piper control status fault: {control_status}"
                            )
                        tracking_error_deg = np.abs(applied_step_deg)
                        max_tracking_error = float(tracking_error_deg.max())
                        if (
                            args.execute
                            and max_tracking_error > args.max_tracking_error_deg
                        ):
                            raise RuntimeError(
                                "Joint target is outrunning feedback: "
                                f"{max_tracking_error:.1f}deg > "
                                f"{args.max_tracking_error_deg:.1f}deg; "
                                f"status={control_status}"
                            )
                        gate_wait = bool(
                            args.execute
                            and args.tracking_gate_deg
                            and max_tracking_error > args.tracking_gate_deg
                        )
                        gate_elapsed = time.monotonic() - gate_started
                        if gate_wait and gate_elapsed > args.tracking_gate_timeout:
                            raise RuntimeError(
                                f"Tracking gate timed out after {gate_elapsed:.1f}s: "
                                f"{max_tracking_error:.1f}deg > "
                                f"{args.tracking_gate_deg:.1f}deg"
                            )
                        gripper_command = (
                            held_gripper if gate_wait else float(action[6])
                        )
                        if args.execute:
                            robot.send(feedback_target, gripper_command)
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
                                    requested_joints_rad=joints.tolist(),
                                    commanded_joints_rad=commanded_joints.tolist(),
                                    commanded_gripper_m=gripper_command,
                                    joint_limit_violation_deg=violation_deg.tolist(),
                                    raw_command_step_deg=raw_step_deg.tolist(),
                                    delta_joint_deg=applied_step_deg.tolist(),
                                    command_step_scale=float(step_scale),
                                    command_step_limited=bool(step_scale < 1.0),
                                    chunk_displacement_deg=float(
                                        chunk_displacement_deg
                                    ),
                                    guarded_displacement_deg=float(
                                        guarded_displacement_deg
                                    ),
                                    tracking_gate_wait=gate_wait,
                                    tracking_gate_attempt=gate_attempt,
                                    control_status=control_status,
                                    sent=args.execute,
                                )
                            )
                            + "\n"
                        )
                        log.flush()
                        time.sleep(
                            max(0, 1 / args.control_hz - (time.monotonic() - tick))
                        )
                        if not gate_wait:
                            break
                        gate_attempt += 1
                    action_step += 1
                if first_command is None:
                    break
                print(
                    f"query {counter}: inference {inference_ms:.0f}ms, first joints "
                    f"{np.rad2deg(first_command).round(1)}, max delta "
                    f"{np.abs(first_applied_step_deg).max():.1f}deg, step clip "
                    f"{np.abs(first_raw_step_deg).max():.1f}->"
                    f"{np.abs(first_applied_step_deg).max():.1f}deg "
                    f"(scale {first_step_scale:.2f}), limit "
                    f"{'pass' if args.allow_small_joint_limit_overshoot else 'clip'} "
                    f"{first_violation.max():.2f}deg, gripper "
                    f"{first_action[6] * 1000:.1f}mm, sources {ensemble_sources}",
                    flush=True,
                )
                counter += 1
    except KeyboardInterrupt:
        (args.output / "error.json").write_text(
            json.dumps(
                {"stage": stage, "type": "KeyboardInterrupt", "message": ""},
                indent=2,
            )
        )
        print("Interrupted; stopping arm commands.", flush=True)
    except BaseException as error:
        (args.output / "error.json").write_text(
            json.dumps(
                {
                    "stage": stage,
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            )
        )
        raise
    finally:
        robot.close()


if __name__ == "__main__":
    main()
