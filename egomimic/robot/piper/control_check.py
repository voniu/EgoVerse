"""Verify Piper SDK position control with a small reversible joint step."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from egomimic.robot.piper.robot_interface import (
    CALIBRATION,
    ROOT,
    Geometry,
    PiperHardware,
)


def move_joint(robot, geometry, target, opening, joint, timeout, log, phase):
    started = time.monotonic()
    next_report = started
    stable = 0
    while True:
        raw = robot.read_state()
        feedback = np.deg2rad(raw[:6] * geometry.joint_scale)
        error_deg = np.rad2deg(target - feedback)
        status = robot.control_status()
        elapsed = time.monotonic() - started
        row = {
            "phase": phase,
            "elapsed_s": elapsed,
            "raw_state": raw.tolist(),
            "feedback_deg": np.rad2deg(feedback).tolist(),
            "target_deg": np.rad2deg(target).tolist(),
            "error_deg": error_deg.tolist(),
            "control_status": status,
        }
        log.write(json.dumps(row) + "\n")
        log.flush()
        if (
            status["ctrl_mode"] != 1
            or status["arm_status"] != 0
            or not all(status["enabled"])
        ):
            raise RuntimeError(f"Piper left active CAN control: {status}")
        if abs(error_deg[joint]) <= 0.3:
            stable += 1
            if stable >= 3:
                return row
        else:
            stable = 0
        if elapsed >= next_report - started:
            print(
                f"{phase} | {elapsed:.2f}s | J{joint + 1} feedback "
                f"{row['feedback_deg'][joint]:.3f}deg | target "
                f"{row['target_deg'][joint]:.3f}deg | error "
                f"{error_deg[joint]:.3f}deg",
                flush=True,
            )
            next_report += 0.25
        if elapsed > timeout:
            raise RuntimeError(
                f"{phase} timed out: J{joint + 1} error {error_deg[joint]:.2f}deg"
            )
        robot.send(target, opening)
        time.sleep(1 / 30)


def hold_joint(robot, geometry, target, opening, joint, duration, log):
    started = time.monotonic()
    while time.monotonic() - started < duration:
        raw = robot.read_state()
        feedback = np.deg2rad(raw[:6] * geometry.joint_scale)
        error_deg = np.rad2deg(target - feedback)
        status = robot.control_status()
        log.write(
            json.dumps(
                {
                    "phase": "hold",
                    "elapsed_s": time.monotonic() - started,
                    "raw_state": raw.tolist(),
                    "feedback_deg": np.rad2deg(feedback).tolist(),
                    "target_deg": np.rad2deg(target).tolist(),
                    "error_deg": error_deg.tolist(),
                    "control_status": status,
                }
            )
            + "\n"
        )
        log.flush()
        if (
            status["ctrl_mode"] != 1
            or status["arm_status"] != 0
            or not all(status["enabled"])
        ):
            raise RuntimeError(f"Piper left active CAN control: {status}")
        if abs(error_deg[joint]) > 0.5:
            raise RuntimeError(
                f"hold drifted: J{joint + 1} error {error_deg[joint]:.2f}deg"
            )
        robot.send(target, opening)
        time.sleep(1 / 30)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--joint", type=int, default=1, choices=range(1, 7))
    parser.add_argument("--delta-deg", type=float, default=2.0)
    parser.add_argument("--speed-percent", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument("--port", default="can_follower")
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / ".local_calibration/control_checks"
        / time.strftime("%Y%m%d_%H%M%S"),
    )
    args = parser.parse_args()
    if not 1 <= args.speed_percent <= 100 or min(args.timeout, args.hold_seconds) <= 0:
        parser.error("speed-percent must be 1..100 and times must be positive")

    geometry = Geometry(args.calibration)
    robot = PiperHardware(geometry, args.port, use_cameras=False)
    args.output.mkdir(parents=True, exist_ok=False)
    joint = args.joint - 1
    try:
        raw = robot.read_state()
        origin = np.deg2rad(raw[:6] * geometry.joint_scale)
        target = origin.copy()
        target[joint] += np.deg2rad(args.delta_deg)
        feedback_bounds = np.sort(
            (geometry.model_bounds - geometry.offsets[:, None])
            / geometry.signs[:, None],
            axis=1,
        )
        if not feedback_bounds[joint, 0] <= target[joint] <= feedback_bounds[joint, 1]:
            raise ValueError(f"J{args.joint} test target exceeds URDF limits")
        opening = raw[6] * geometry.grip_scale
        robot.start_control(args.speed_percent)
        with (args.output / "feedback.jsonl").open("w") as log:
            outward = move_joint(
                robot,
                geometry,
                target,
                opening,
                joint,
                args.timeout,
                log,
                "outward",
            )
            returned = move_joint(
                robot,
                geometry,
                origin,
                opening,
                joint,
                args.timeout,
                log,
                "return",
            )
            hold_joint(
                robot,
                geometry,
                origin,
                opening,
                joint,
                args.hold_seconds,
                log,
            )
        result = {
            "joint": args.joint,
            "delta_deg": args.delta_deg,
            "speed_percent": args.speed_percent,
            "hold_seconds": args.hold_seconds,
            "origin_deg": np.rad2deg(origin).tolist(),
            "outward": outward,
            "returned": returned,
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        print(f"PASS | J{args.joint} moved and returned | logs: {args.output}")
    except BaseException as error:
        (args.output / "error.json").write_text(
            json.dumps({"type": type(error).__name__, "message": str(error)}, indent=2)
        )
        raise
    finally:
        robot.close()


if __name__ == "__main__":
    main()
