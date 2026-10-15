"""Piper L geometry, passive observation, and SDK joint commands."""

import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from egomimic.robot.piper.config import check_cameras, load_config
from egomimic.robot.piper_kinematics import PiperKinematics

ROOT = Path(__file__).resolve().parents[3]


def transform(spec):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(spec["quaternion_xyzw"]).as_matrix()
    result[:3, 3] = spec["translation_m"]
    return result


class Geometry:
    def __init__(self, calibration=None, require_cameras=False):
        """calibration: setup file (else $PIPER_SETUP, else robot-model constants)."""
        self.config = load_config(calibration, require_cameras)
        k = self.config["kinematics"]
        self.signs = np.asarray(k["feedback_to_model"]["signs"])
        self.offsets = np.asarray(k["feedback_to_model"]["offsets_rad"])
        self.joint_scale = k["recorded_sdk_feedback"]["degrees_per_raw_joint_unit"]
        self.grip_scale = k["recorded_sdk_feedback"]["gripper_mm_per_raw_unit"] / 1000
        urdf = ROOT / k["reference_urdf_path"]
        self.fk = PiperKinematics(urdf)
        front_extrinsics = ((self.config.get("cameras") or {}).get("d435if") or {}).get(
            "extrinsics"
        ) or {}
        self.base_camera = (
            transform(front_extrinsics)
            if front_extrinsics.get("translation_m") is not None
            else None
        )
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
        self.model_bounds = bounds
        self.bounds = np.sort(
            (bounds - self.offsets[:, None]) / self.signs[:, None], axis=1
        )

    def joint_state(self, raw):
        """Seven-channel model state: URDF joints in radians and gripper in meters."""
        feedback = np.deg2rad(np.asarray(raw[:6]) * self.joint_scale)
        joints = feedback * self.signs + self.offsets
        return np.r_[joints, raw[6] * self.grip_scale].astype(np.float32)

    def feedback_from_model(self, joints):
        """Convert URDF/model joint coordinates to SDK feedback coordinates."""
        return (np.asarray(joints) - self.offsets) / self.signs

    def tcp_base(self, q):
        return (
            self.fk.fk_from_feedback_deg(np.rad2deg(q * self.signs + self.offsets))
            @ self.link_tcp
        )

    def state(self, raw):
        if self.base_camera is None:
            raise RuntimeError("Pose control requires calibrated D435IF extrinsics")
        q = np.deg2rad(np.asarray(raw[:6]) * self.joint_scale)
        tcp = np.linalg.inv(self.base_camera) @ self.tcp_base(q)
        return np.r_[
            tcp[:3, 3],
            Rotation.from_matrix(tcp[:3, :3]).as_euler("ZYX"),
            raw[6] * self.grip_scale,
        ].astype(np.float32)

    def inverse(self, action, seed):
        """Solve near the previous joint command; reject unreachable pose targets."""
        if self.base_camera is None:
            raise RuntimeError("Pose control requires calibrated D435IF extrinsics")
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
    def __init__(
        self,
        geometry,
        port="can_follower",
        wrist_serial=None,
        use_cameras=True,
        gripper_effort=1000,
    ):
        # GripperCtrl effort in SDK units of 0.001 N*m (0..5000). Match the
        # teleoperation value the demonstrations were recorded with.
        if not 0 <= int(gripper_effort) <= 5000:
            raise ValueError("gripper_effort must lie in 0..5000 (0.001 N*m)")
        self.gripper_effort = int(gripper_effort)
        if use_cameras:
            check_cameras(geometry.config)
        from piper_sdk import C_PiperInterface_V2

        self.geometry = geometry
        self.piper = C_PiperInterface_V2(port)
        self.pipelines = []
        self.controlling = False
        self.speed_percent = None
        try:
            self.piper.ConnectPort()
            time.sleep(0.1)
            # The firmware master/slave role is independent of our leader/follower
            # naming. Force this arm into the motion-output role on every connect.
            self.piper.MasterSlaveConfig(0xFC, 0x00, 0x00, 0x00)
            time.sleep(1.0)
            self.read_state()
            if use_cameras:
                import pyrealsense2 as rs

                cameras = geometry.config["cameras"]
                for serial in [
                    cameras["d435if"]["serial"],
                    wrist_serial or cameras["d405"]["serial"],
                ]:
                    pipeline, config = rs.pipeline(), rs.config()
                    config.enable_device(serial)
                    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
                    try:
                        pipeline.start(config)
                    except RuntimeError as error:
                        raise RuntimeError(
                            f"Failed to start RealSense {serial}: {error}"
                        ) from error
                    self.pipelines.append((serial, pipeline))
                # Let automatic exposure settle without controlling the arm.
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
        images = []
        for serial, pipeline in self.pipelines:
            try:
                frame = pipeline.wait_for_frames(timeout_ms=1000).get_color_frame()
                images.append(np.asanyarray(frame.get_data()).copy())
            except RuntimeError as error:
                raise RuntimeError(
                    f"RealSense {serial} frame failed: {error}"
                ) from error
        return images

    def start_control(self, speed_percent):
        raw = self.read_state()
        self.controlling = True
        self.speed_percent = speed_percent
        # EnablePiper() can return True while status still reports the quick
        # stop left by close(), so reset it explicitly before CAN control.
        self.piper.EmergencyStop(0x02)
        until = time.monotonic() + 5
        stable_samples = 0
        while stable_samples < 25:
            status = self.piper.GetArmStatus().arm_status
            if int(status.arm_status) == 0 and int(status.err_code) == 0:
                stable_samples += 1
            else:
                stable_samples = 0
            if time.monotonic() > until:
                raise RuntimeError("Piper emergency-stop reset timed out")
            time.sleep(0.02)
        until = time.monotonic() + 5
        stable_samples = 0
        while stable_samples < 25:
            self.piper.EnablePiper()
            status = self.piper.GetArmStatus().arm_status
            enabled = self.piper.GetArmEnableStatus()
            if (
                int(status.arm_status) == 0
                and int(status.err_code) == 0
                and all(enabled)
            ):
                stable_samples += 1
            else:
                stable_samples = 0
            if time.monotonic() > until:
                raise RuntimeError("Piper enable timed out")
            time.sleep(0.02)
        self.piper.MotionCtrl_2(0x01, 0x01, speed_percent, 0x00)
        # Preload the measured position immediately after entering MOVE-J, so an
        # old controller target cannot move the arm during handover.
        self.piper.JointCtrl(*np.rint(raw[:6]).astype(int).tolist())
        until = time.monotonic() + 5
        stable_samples = 0
        while True:
            status = self.piper.GetArmStatus().arm_status
            enabled = self.piper.GetArmEnableStatus()
            if (
                int(status.arm_status) == 0
                and int(status.ctrl_mode) == 1
                and all(enabled)
            ):
                stable_samples += 1
                if stable_samples >= 25:
                    break
            else:
                stable_samples = 0
            if time.monotonic() > until:
                raise RuntimeError(
                    f"Piper control startup timed out: {status}, enabled={enabled}"
                )
            time.sleep(0.02)
        self.piper.GripperCtrl(int(raw[6]), self.gripper_effort, 0x01, 0)

    def control_status(self):
        status = self.piper.GetArmStatus().arm_status
        errors = status.err_status
        return {
            "ctrl_mode": int(status.ctrl_mode),
            "arm_status": int(status.arm_status),
            "mode_feed": int(status.mode_feed),
            "motion_status": int(status.motion_status),
            "error_code": int(status.err_code),
            "joint_limit_errors": [
                bool(getattr(errors, f"joint_{i}_angle_limit")) for i in range(1, 7)
            ],
            "joint_communication_errors": [
                bool(getattr(errors, f"communication_status_joint_{i}"))
                for i in range(1, 7)
            ],
            "enabled": [bool(value) for value in self.piper.GetArmEnableStatus()],
        }

    def send(self, q, opening):
        if not self.controlling:
            raise RuntimeError("Control has not been started")
        raw = np.rint(np.rad2deg(q) / self.geometry.joint_scale).astype(int)
        self.piper.JointCtrl(*raw.tolist())
        self.send_gripper(opening)

    def send_gripper(self, opening):
        if not self.controlling:
            raise RuntimeError("Control has not been started")
        # Piper L nominal physical stroke: 100mm; opening is in meters, not binary.
        self.piper.GripperCtrl(
            int(np.clip(opening, 0, 0.1) / self.geometry.grip_scale),
            self.gripper_effort,
            0x01,
            0,
        )

    def set_speed_percent(self, speed_percent):
        if not self.controlling:
            raise RuntimeError("Control has not been started")
        self.speed_percent = speed_percent
        self.piper.MotionCtrl_2(0x01, 0x01, speed_percent, 0x00)

    def close(self):
        try:
            if self.controlling:
                self.piper.EmergencyStop(0x01)
                self.controlling = False
        finally:
            for _, pipeline in self.pipelines:
                pipeline.stop()
            self.pipelines.clear()
            self.piper.DisconnectPort()
