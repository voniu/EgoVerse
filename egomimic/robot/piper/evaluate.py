"""Offline deployment parity and IK checks; never connects to the robot."""

import json

import numpy as np
import simplejpeg
import zarr

from egomimic.robot.piper.robot_interface import ROOT, Geometry, PiperHardware
from egomimic.robot.piper.rollout import MLPPolicy


def main():
    geometry = Geometry()
    root = ROOT / ".local_calibration"
    report = {"policy": [], "ik": [], "predicted_ik": []}
    episode = "teddy_0912_1_episode_006"
    z = zarr.open_group(str(root / "piper_all" / f"{episode}.zarr"), mode="r")
    for group in ["a", "b"]:
        policy = MLPPolicy(root / "hpc/mlp/checkpoints" / group / "final.ckpt")
        saved = np.load(
            root / "mlp_final_validation" / f"{group.upper()}_{episode}.npz"
        )
        for frame in [0, 100, 250]:
            raw = z["source.observation_state"][frame]
            state = geometry.state(raw)
            rgb = [
                simplejpeg.decode_jpeg(z[key][frame : frame + 1][0], colorspace="RGB")
                for key in ["images.front_1", "images.right_wrist"]
            ]
            prediction = policy.predict(state, *rgb)
            reference = saved["prediction"][0, frame]
            error = np.max(np.abs(prediction - reference))
            # Batch1 versus saved batch32 can differ slightly in floating-point kernels.
            np.testing.assert_allclose(prediction, reference, atol=3e-4, rtol=1e-4)
            report["policy"].append(
                dict(group=group, frame=frame, max_absolute_difference=float(error))
            )
            seed = np.deg2rad(raw[:6] * geometry.joint_scale)
            for h, action in enumerate(prediction[:10]):
                try:
                    seed, error = geometry.inverse(action, seed)
                    report["predicted_ik"].append(
                        dict(group=group, frame=frame, horizon=h, ok=True, **error)
                    )
                except RuntimeError as exc:
                    report["predicted_ik"].append(
                        dict(
                            group=group,
                            frame=frame,
                            horizon=h,
                            ok=False,
                            error=str(exc),
                        )
                    )
        del policy
    rng = np.random.default_rng(42)
    for i in range(20):
        lo, hi = geometry.bounds.T
        q = rng.uniform(lo + 0.1, hi - 0.1)
        raw = np.r_[np.rad2deg(q) / geometry.joint_scale, 30000]
        action = geometry.state(raw)
        solved, error = geometry.inverse(action, q + rng.normal(0, 0.02, 6))
        assert error["position_mm"] < 1 and error["rotation_deg"] < 0.1
        report["ik"].append(error)
    try:
        geometry.inverse(np.array([10, 10, 10, 0, 0, 0, 0.05]), np.zeros(6))
    except RuntimeError:
        report["unreachable_rejected"] = True
    else:
        raise AssertionError("Unreachable target accepted")

    class Recorder:
        def JointCtrl(self, *args):
            self.joints = args

        def GripperCtrl(self, *args):
            self.gripper = args

    hardware = PiperHardware.__new__(PiperHardware)
    hardware.geometry = geometry
    hardware.controlling = True
    hardware.piper = Recorder()
    hardware.send(np.deg2rad([10, 20, -30, 40, 50, -60]), 0.05)
    assert hardware.piper.joints == (10000, 20000, -30000, 40000, 50000, -60000)
    assert abs(hardware.piper.gripper[0] - 50000) <= 1
    report["sdk_units_verified"] = True
    (root / "deployment_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
