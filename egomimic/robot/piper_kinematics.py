"""Offline Piper L FK with explicit encoder-to-model offsets.

This module sends no robot commands. It does not modify raw observations, action
normalization, firmware zeros, or the official URDF. A calibration is specific
to the measured robot and firmware-zero state; consult its validation scope.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


class PiperKinematics:
    """Compute T_base_link6 (meters) from actual feedback joint angles.

    Joints are ordered J1..J6. A calibrated model uses
    q_model = radians(q_feedback_deg + offsets_deg).
    The gripper reading and normalized action values are not joint angles.
    """

    def __init__(
        self, urdf_path: str | Path, calibration_path: str | Path | None = None
    ):
        raw = Path(urdf_path).read_bytes()
        root = ET.fromstring(raw)
        if root.get("name") != "piper_l":
            raise ValueError("This implementation expects the official Piper L model")
        self.profile = None
        self.offsets_deg = np.zeros(6)
        if calibration_path is not None:
            self.profile = json.loads(Path(calibration_path).read_text())
            if self.profile["urdf_sha256"] != hashlib.sha256(raw).hexdigest():
                raise ValueError("URDF hash differs from the calibrated reference")
            if (
                self.profile["angle_convention"]
                != "q_model_deg = q_feedback_deg + offsets_deg"
            ):
                raise ValueError("Unsupported joint-offset convention")
            self.offsets_deg = self._angles(self.profile["offsets_deg"])
            if self.offsets_deg.shape != (6,):
                raise ValueError("Calibration must contain exactly six joint offsets")
        self.origins = []
        self.axes = []
        for index in range(1, 7):
            joint = root.find(f"joint[@name='joint{index}']")
            if joint is None or joint.get("type") != "revolute":
                raise ValueError(f"Missing revolute joint{index}")
            parent = "base_link" if index == 1 else f"link{index - 1}"
            if (
                joint.find("parent").get("link") != parent
                or joint.find("child").get("link") != f"link{index}"
            ):
                raise ValueError("Unexpected Piper joint chain")
            origin = joint.find("origin")
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_euler(
                "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            ).as_matrix()
            transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            if not np.isclose(np.linalg.norm(axis), 1):
                raise ValueError("Expected unit joint axis")
            self.origins.append(transform)
            self.axes.append(axis)

    @staticmethod
    def _angles(values) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim == 0 or array.shape[-1] != 6 or not np.isfinite(array).all():
            raise ValueError(
                "Expected finite joint values with shape (..., 6), excluding gripper"
            )
        return array

    def feedback_deg_to_model_rad(self, feedback_deg) -> np.ndarray:
        return np.deg2rad(self._angles(feedback_deg) + self.offsets_deg)

    def model_rad_to_feedback_deg(self, model_rad) -> np.ndarray:
        """Convert model/IK angles back to hardware coordinates; does not command them."""
        return np.rad2deg(self._angles(model_rad)) - self.offsets_deg

    def fk_from_feedback_raw(self, raw_joint_values) -> np.ndarray:
        """SDK feedback unit is 0.001 degree; pass first six state channels only."""
        return self.fk_from_feedback_deg(self._angles(raw_joint_values) / 1000.0)

    def fk_from_feedback_deg(self, feedback_deg) -> np.ndarray:
        q = self.feedback_deg_to_model_rad(feedback_deg)
        batch_shape = q.shape[:-1]
        flat = q.reshape(-1, 6)
        transforms = np.broadcast_to(np.eye(4), (len(flat), 4, 4)).copy()
        for k, (origin, axis) in enumerate(zip(self.origins, self.axes)):
            rotations = np.broadcast_to(np.eye(4), transforms.shape).copy()
            rotations[:, :3, :3] = Rotation.from_rotvec(
                flat[:, k, None] * axis
            ).as_matrix()
            transforms = transforms @ origin @ rotations
        return transforms.reshape(*batch_shape, 4, 4)
