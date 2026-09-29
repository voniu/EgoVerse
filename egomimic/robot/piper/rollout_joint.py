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
        prediction = self.head(features)
        result = (
            self.model.norm_stats.unnormalize({JOINT_ACTION: prediction.float()}, 7)[
                JOINT_ACTION
            ][0]
            .cpu()
            .numpy()
        )
        if not np.isfinite(result).all():
            raise RuntimeError("Nonfinite model output")
        return result


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
    parser.add_argument("--speed-percent", type=int, default=10)
    parser.add_argument(
        "--query-every",
        type=int,
        default=1,
        help="Use this many actions before observing again (1..30)",
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
    ):
        parser.error(
            "speed-percent:1..100, query-every:1..30, duration must be positive"
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
                "action_hz": 30,
                "action_space": "absolute URDF joints rad + gripper m",
            },
            indent=2,
        )
    )

    robot = PiperHardware(geometry, args.port, args.wrist_serial)
    started = time.monotonic()
    counter = 0
    try:
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
                chunk = policy.predict(state, front, wrist)
                inference_ms = (time.monotonic() - t0) * 1000
                np.savez_compressed(
                    args.output / f"query_{counter:04d}.npz",
                    raw_state=raw,
                    state=state,
                    prediction=chunk,
                    front=front,
                    wrist=wrist,
                )
                for horizon, action in enumerate(chunk[: args.query_every]):
                    if time.monotonic() - started >= args.duration:
                        break
                    tick = time.monotonic()
                    joints = action[:6]
                    if (
                        (joints < geometry.model_bounds[:, 0])
                        | (joints > geometry.model_bounds[:, 1])
                    ).any():
                        raise RuntimeError(
                            "Joint-head target exceeds URDF joint limits"
                        )
                    feedback_target = geometry.feedback_from_model(joints)
                    if args.execute:
                        robot.read_state()
                        robot.send(feedback_target, action[6])
                    log.write(
                        json.dumps(
                            dict(
                                query=counter,
                                horizon=horizon,
                                elapsed_s=time.monotonic() - started,
                                inference_ms=inference_ms,
                                observed=state.tolist(),
                                target=action.tolist(),
                                delta_joint_deg=np.rad2deg(joints - state[:6]).tolist(),
                                sent=args.execute,
                            )
                        )
                        + "\n"
                    )
                    log.flush()
                    time.sleep(max(0, 1 / 30 - (time.monotonic() - tick)))
                delta = np.rad2deg(chunk[0, :6] - state[:6])
                print(
                    f"query {counter}: inference {inference_ms:.0f}ms, first joints "
                    f"{np.rad2deg(chunk[0, :6]).round(1)}, max delta "
                    f"{np.abs(delta).max():.1f}deg, gripper {chunk[0, 6] * 1000:.1f}mm",
                    flush=True,
                )
                counter += 1
    except KeyboardInterrupt:
        print("Interrupted; stopping arm commands.", flush=True)
    finally:
        robot.close()


if __name__ == "__main__":
    main()
