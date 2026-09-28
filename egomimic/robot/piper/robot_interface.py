"""Piper L geometry, passive observation, and SDK joint commands."""

import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from egomimic.robot.piper_kinematics import PiperKinematics

ROOT = Path(__file__).resolve().parents[3]
CALIBRATION = Path(__file__).with_name("calibration.yaml")


def transform(spec):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(spec["quaternion_xyzw"]).as_matrix()
    result[:3, 3] = spec["translation_m"]
    return result


class Geometry:
    def __init__(self, calibration=CALIBRATION):
        self.config = yaml.safe_load(Path(calibration).read_text())
        k = self.config["kinematics"]
        self.signs = np.asarray(k["feedback_to_model"]["signs"])
        self.offsets = np.asarray(k["feedback_to_model"]["offsets_rad"])
        self.joint_scale = k["recorded_sdk_feedback"]["degrees_per_raw_joint_unit"]
        self.grip_scale = k["recorded_sdk_feedback"]["gripper_mm_per_raw_unit"] / 1000
        urdf = ROOT / k["reference_urdf_path"]
        self.fk = PiperKinematics(urdf)
        self.base_camera = transform(self.config["cameras"]["d435if"]["extrinsics"])
        self.link_tcp = transform(self.config["tcp"])
        root = ET.parse(urdf).getroot()
        bounds = np.array(
            [
                [
                    float(root.find(f"joint[@name='{name}']/limit").get(side))
                    for side in ["lower", "upper"]
                ]
                for name in k["joint_names"]
            ]
        )
        self.bounds = np.sort(
            (bounds - self.offsets[:, None]) / self.signs[:, None], axis=1
        )

    def tcp_base(self, q):
        return (
            self.fk.fk_from_feedback_deg(np.rad2deg(q * self.signs + self.offsets))
            @ self.link_tcp
        )

    def state(self, raw):
        q = np.deg2rad(np.asarray(raw[:6]) * self.joint_scale)
        tcp = np.linalg.inv(self.base_camera) @ self.tcp_base(q)
        return np.r_[
            tcp[:3, 3],
            Rotation.from_matrix(tcp[:3, :3]).as_euler("ZYX"),
            raw[6] * self.grip_scale,
        ].astype(np.float32)

    def inverse(self, action, seed):
        """Solve near the previous joint command; reject unreachable pose targets."""
        if np.shape(action) != (7,) or not np.isfinite(action).all():
            raise ValueError("Expected finite [xyz, yaw,pitch,roll, gripper] action")
        camera_tcp = np.eye(4)
        camera_tcp[:3, 3] = action[:3]
        camera_tcp[:3, :3] = Rotation.from_euler("ZYX", action[3:6]).as_matrix()
        target = self.base_camera @ camera_tcp

        def residual(q):
            actual = self.tcp_base(q)
            rot = Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()
            return np.r_[actual[:3, 3] - target[:3, 3], 0.1 * rot]

        lo, hi = self.bounds.T
        result = least_squares(
            residual,
            np.clip(seed, lo + 1e-7, hi - 1e-7),
            bounds=(lo, hi),
            max_nfev=80,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
        error = residual(result.x)
        position_mm = np.linalg.norm(error[:3]) * 1000
        rotation_deg = np.rad2deg(np.linalg.norm(error[3:]) / 0.1)
        if position_mm > 5 or rotation_deg > 3:
            raise RuntimeError(
                f"IK cannot reach target: {position_mm:.1f}mm / {rotation_deg:.1f}deg"
            )
        return result.x, dict(
            position_mm=float(position_mm), rotation_deg=float(rotation_deg)
        )


class PiperHardware:
    def __init__(self, geometry, port="can_follower", wrist_serial="427622273943"):
        import pyrealsense2 as rs
        from piper_sdk import C_PiperInterface_V2

        self.geometry = geometry
        self.piper = C_PiperInterface_V2(port)
        self.pipelines = []
        self.controlling = False
        self.speed_percent = None
        try:
            self.piper.ConnectPort(piper_init=False)
            time.sleep(0.3)
            self.read_state()
            for serial in [
                geometry.config["cameras"]["d435if"]["serial"],
                wrist_serial,
            ]:
                pipeline, config = rs.pipeline(), rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
                pipeline.start(config)
                self.pipelines.append(pipeline)
            # Let automatic exposure settle; consumes frames without controlling the arm.
            until = time.monotonic() + 2
            while time.monotonic() < until:
                self.images()
        except BaseException:
            self.close()
            raise

    def read_state(self):
        joints = self.piper.GetArmJointMsgs()
        grip = self.piper.GetArmGripperMsgs()
        if (
            min(joints.time_stamp, grip.time_stamp) <= 0
            or time.time() - min(joints.time_stamp, grip.time_stamp) > 0.5
        ):
            raise RuntimeError("No fresh follower joint/gripper feedback")
        return np.array(
            [getattr(joints.joint_state, f"joint_{i}") for i in range(1, 7)]
            + [grip.gripper_state.grippers_angle],
            dtype=float,
        )

    def images(self):
        return [
            np.asanyarray(
                p.wait_for_frames(timeout_ms=1000).get_color_frame().get_data()
            ).copy()
            for p in self.pipelines
        ]

    def start_control(self, speed_percent):
        raw = self.read_state()
        self.controlling = True
        self.speed_percent = speed_percent
        # EnablePiper() can return True while status still reports the quick
        # stop left by close(), so reset it explicitly before CAN control.
        self.piper.EmergencyStop(0x02)
        until = time.monotonic() + 5
        while int(self.piper.GetArmStatus().arm_status.arm_status) != 0:
            if time.monotonic() > until:
                raise RuntimeError("Piper emergency-stop reset timed out")
            time.sleep(0.02)
        # Preload the measured position before enabling, avoiding an old target.
        self.piper.JointCtrl(*np.rint(raw[:6]).astype(int).tolist())
        until = time.monotonic() + 5
        while not self.piper.EnablePiper():
            if time.monotonic() > until:
                raise RuntimeError("Piper enable timed out")
            time.sleep(0.02)
        until = time.monotonic() + 5
        while True:
            self.piper.MotionCtrl_2(0x01, 0x01, speed_percent, 0x00)
            status = self.piper.GetArmStatus().arm_status
            if int(status.arm_status) == 0 and int(status.ctrl_mode) == 1:
                break
            if time.monotonic() > until:
                raise RuntimeError(f"Piper did not enter CAN control mode: {status}")
            time.sleep(0.02)
        self.piper.GripperCtrl(int(raw[6]), 1000, 0x01, 0)

    def send(self, q, opening):
        if not self.controlling:
            raise RuntimeError("Control has not been started")
        raw = np.rint(np.rad2deg(q) / self.geometry.joint_scale).astype(int)
        # Piper drops back to standby unless MOVE-J control is refreshed with
        # the streamed position commands.
        self.piper.MotionCtrl_2(0x01, 0x01, self.speed_percent, 0x00)
        self.piper.JointCtrl(*raw.tolist())
        # Piper L nominal physical stroke: 100mm; opening is in meters, not binary.
        self.piper.GripperCtrl(
            int(np.clip(opening, 0, 0.1) / self.geometry.grip_scale), 1000, 0x01, 0
        )

    def close(self):
        try:
            if self.controlling:
                self.piper.EmergencyStop(0x01)
                self.controlling = False
        finally:
            for pipeline in self.pipelines:
                pipeline.stop()
            self.pipelines.clear()
            self.piper.DisconnectPort()
