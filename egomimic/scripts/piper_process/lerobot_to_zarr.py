"""Convert one recorded Piper LeRobot v3 episode to EgoVerse Zarr, offline."""

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow.parquet as pq
import simplejpeg
from scipy.spatial.transform import Rotation

from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.robot.piper.config import load_config
from egomimic.robot.piper_kinematics import PiperKinematics

ROOT = Path(__file__).resolve().parents[3]


def transform(spec):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(spec["quaternion_xyzw"]).as_matrix()
    matrix[:3, 3] = spec["translation_m"]
    return matrix


def poses(matrices):
    """Repository storage convention: xyz + quaternion wxyz, meters."""
    quat = Rotation.from_matrix(matrices[:, :3, :3]).as_quat()
    return np.column_stack((matrices[:, :3, 3], quat[:, [3, 0, 1, 2]])).astype(
        "float32"
    )


def read_episode(dataset, episode_index, info):
    for path in sorted((dataset / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(path).to_pandas()
        rows = table[table.episode_index == episode_index]
        if len(rows):
            meta = rows.iloc[0]
            break
    else:
        raise ValueError(f"Episode {episode_index} not found")
    path = dataset / info["data_path"].format(
        chunk_index=int(meta["data/chunk_index"]),
        file_index=int(meta["data/file_index"]),
    )
    frames = pq.read_table(path).to_pandas()
    frames = frames[frames.episode_index == episode_index].sort_values("frame_index")
    if len(frames) != int(meta["length"]):
        raise ValueError(
            "Episode spans additional data files; this converter expects one data file per episode"
        )
    np.testing.assert_array_equal(frames.frame_index, np.arange(len(frames)))
    np.testing.assert_allclose(
        frames.timestamp, np.arange(len(frames)) / info["fps"], atol=1e-4
    )
    return meta, frames


def read_video(path, timestamps, fps):
    """Decode sequentially using presentation timestamps, including v3 file offsets."""
    images, errors = [], []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            t = float(frame.pts * frame.time_base)
            if t < timestamps[len(images)] - 0.5 / fps:
                continue
            error = t - timestamps[len(images)]
            if abs(error) > 0.5 / fps:
                raise ValueError(f"Video timestamp mismatch: {error:.6f}s in {path}")
            rgb = frame.to_ndarray(format="rgb24")
            images.append(
                simplejpeg.encode_jpeg(
                    rgb, quality=ZarrWriter.JPEG_QUALITY, colorspace="RGB"
                )
            )
            errors.append(error)
            if len(images) == len(timestamps):
                break
    if len(images) != len(timestamps):
        raise ValueError(f"Video ended before episode: {path}")
    return (
        np.array(images, dtype=object),
        list(rgb.shape),
        float(np.max(np.abs(errors))),
    )


def project(points, K):
    uvw = points @ K.T
    return uvw[:, :2] / uvw[:, 2:3]


def write_preview(path, encoded, obs, cmd, K, fps):
    """Green: measured TCP; orange: commanded TCP. Axes: RGB = XYZ."""
    h, w = simplejpeg.decode_jpeg(encoded[0]).shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not open preview video writer")
    panels = []
    selected = set(np.linspace(0, len(encoded) - 1, 6, dtype=int))
    try:
        for i, data in enumerate(encoded):
            image = cv2.cvtColor(simplejpeg.decode_jpeg(data), cv2.COLOR_RGB2BGR)
            for T, color in [(obs[i], (0, 255, 0)), (cmd[i], (0, 165, 255))]:
                if T[2, 3] > 0:
                    uv = project(T[None, :3, 3], K)[0]
                    if 0 <= uv[0] < w and 0 <= uv[1] < h:
                        cv2.circle(image, tuple(np.rint(uv).astype(int)), 6, color, 2)
            T = obs[i]
            points = np.vstack((T[:3, 3], T[:3, 3] + 0.025 * T[:3, :3].T))
            if (points[:, 2] > 0).all():
                uv = np.rint(project(points, K)).astype(int)
                for j, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                    cv2.line(image, tuple(uv[0]), tuple(uv[j + 1]), color, 2)
            cv2.putText(
                image,
                f"{i}/{len(encoded) - 1}  green:state  orange:command",
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            writer.write(image)
            if i in selected:
                panels.append(image)
    finally:
        writer.release()
    cv2.imwrite(
        str(path.with_suffix(".jpg")),
        np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:]))),
    )


def convert(args):
    cfg = load_config(args.calibration)
    limits = json.loads(args.action_calibration.read_text())
    wrist = json.loads(args.wrist_intrinsics.read_text())
    info = json.loads((args.dataset / "meta/info.json").read_text())
    meta, frames = read_episode(args.dataset, args.episode, info)
    task_name = args.task_name or str(meta["tasks"][0])
    state = np.stack(frames["observation.state"]).astype(float)
    action = np.stack(frames.action).astype(float)
    if (
        state.shape[1] != 7
        or action.shape != state.shape
        or not np.isfinite([state, action]).all()
    ):
        raise ValueError("Expected finite seven-channel Piper state and action")
    k = cfg["kinematics"]
    scale = k["recorded_sdk_feedback"]["degrees_per_raw_joint_unit"]
    bounds = np.array(
        [
            [limits[f"joint_{i}"]["range_min"], limits[f"joint_{i}"]["range_max"]]
            for i in range(6)
        ]
    )
    # Match the recorded driver's integer command conversion (truncation toward zero).
    cmd_raw = np.trunc(
        bounds[:, 0] + (action[:, :6] + 100) / 200 * (bounds[:, 1] - bounds[:, 0])
    )
    cmd_gripper_raw = np.trunc(
        np.clip(action[:, 6:7], 0, 100) / 100 * limits["gripper"]["range_max"]
    )
    signs = np.array(k["feedback_to_model"]["signs"])
    offsets = np.array(k["feedback_to_model"]["offsets_rad"])
    q_obs = np.deg2rad(state[:, :6] * scale) * signs + offsets
    q_cmd = np.deg2rad(cmd_raw * scale) * signs + offsets
    fk = PiperKinematics(ROOT / k["reference_urdf_path"])
    T_tcp = transform(cfg["tcp"])
    extrinsics = cfg.get("cameras", {}).get("d435if", {}).get("extrinsics")
    if extrinsics is None:
        raise ValueError(
            f"{args.calibration} lacks cameras.d435if.extrinsics; pass --calibration "
            "with the T_base_camera valid during collection"
        )
    T_camera = transform(extrinsics)
    base_obs = fk.fk_from_feedback_deg(np.rad2deg(q_obs)) @ T_tcp
    base_cmd = fk.fk_from_feedback_deg(np.rad2deg(q_cmd)) @ T_tcp
    cam_obs = np.linalg.inv(T_camera) @ base_obs
    cam_cmd = np.linalg.inv(T_camera) @ base_cmd
    grip_scale = k["recorded_sdk_feedback"]["gripper_mm_per_raw_unit"] / 1000
    numeric = {
        "right.obs_ee_pose": poses(base_obs),
        "right.cmd_ee_pose": poses(base_cmd),
        "right.obs_ee_pose_camframe": poses(cam_obs),
        "right.cmd_ee_pose_camframe": poses(cam_cmd),
        "right.obs_joints": q_obs.astype("float32"),
        "right.cmd_joints": q_cmd.astype("float32"),
        "right.obs_gripper": (state[:, 6:7] * grip_scale).astype("float32"),
        "right.cmd_gripper": (cmd_gripper_raw * grip_scale).astype("float32"),
        "source.observation_state": state.astype("float32"),
        "source.action": action.astype("float32"),
        "timestamp": frames.timestamp.to_numpy(dtype="float32")[:, None],
    }
    images, timing = {}, {}
    for camera, key in [("435if", "front_1"), ("405", "right_wrist")]:
        video_key = f"observation.images.{camera}"
        prefix = f"videos/{video_key}"
        path = args.dataset / info["video_path"].format(
            video_key=video_key,
            chunk_index=int(meta[f"{prefix}/chunk_index"]),
            file_index=int(meta[f"{prefix}/file_index"]),
        )
        timestamps = frames.timestamp.to_numpy(dtype=float) + float(
            meta[f"{prefix}/from_timestamp"]
        )
        encoded, shape, error = read_video(path, timestamps, info["fps"])
        images[f"images.{key}"] = (encoded, shape)
        timing[key] = error
    front = cfg["cameras"]["d435if"]["intrinsics"]
    for key, intr in [("front_1", front), ("right_wrist", wrist["color"])]:
        if images[f"images.{key}"][1][:2] != [intr["height"], intr["width"]]:
            raise ValueError(f"Image dimensions disagree with {key} intrinsics")
    name = f"{args.dataset.name}_episode_{args.episode:03d}"
    out = args.output / f"{name}.zarr"
    if out.exists():
        raise FileExistsError(out)
    args.output.mkdir(parents=True, exist_ok=True)
    K = np.array(front["K"])
    ZarrWriter.create_and_write(
        out,
        numeric_data=numeric,
        pre_encoded_image_data=images,
        embodiment="piper_right_arm",
        fps=info["fps"],
        task_name=task_name,
        annotations=[(task_name, 0, len(frames))],
        intrinsics={
            key: np.column_stack((intr, np.zeros(3)))
            for key, intr in [
                ("front_1", K),
                ("right_wrist", np.array(wrist["color"]["K"])),
            ]
        },
        extrinsics={"right": T_camera},
        metadata_override={
            "pose_format": "xyz_wxyz",
            "ee_pose_frame": "base_link",
            "ee_pose_camframe": "camera_color_optical_frame",
            "ee_reference": "piper_tcp",
            "gripper_unit": "meter",
            "orientation_convention": "native_link6_axes",
            "source_dataset": str(args.dataset.resolve()),
            "source_episode": args.episode,
            "calibration": cfg,
            "action_calibration": limits,
            "wrist_intrinsics": wrist,
            "action_calibration_source": str(args.action_calibration.resolve()),
            "action_label": "recorded command at same row; no temporal shift",
            "human_orientation_alignment_applied": False,
            "geometry_validation": "requires_visual_review",
        },
    )
    write_preview(
        args.output / f"{name}_overlay.mp4",
        images["images.front_1"][0],
        cam_obs,
        cam_cmd,
        K,
        info["fps"],
    )
    uv = project(cam_obs[:, :3, 3], K)
    report = {
        "zarr": str(out),
        "frames": len(frames),
        "fps": info["fps"],
        "max_video_timestamp_error_s": timing,
        "tcp_depth_range_m": [
            float(cam_obs[:, 2, 3].min()),
            float(cam_obs[:, 2, 3].max()),
        ],
        "tcp_in_front_image_fraction": float(
            (
                (cam_obs[:, 2, 3] > 0)
                & (uv[:, 0] >= 0)
                & (uv[:, 0] < front["width"])
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < front["height"])
            ).mean()
        ),
        "median_command_state_distance_m": float(
            np.median(np.linalg.norm(cam_obs[:, :3, 3] - cam_cmd[:, :3, 3], axis=1))
        ),
    }
    (args.output / f"{name}_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="Setup file of the recording setup (front camera intrinsics and "
        "hand-eye extrinsics); see docs/piper_setup.md",
    )
    parser.add_argument("--action-calibration", type=Path, required=True)
    parser.add_argument("--wrist-intrinsics", type=Path, required=True)
    parser.add_argument(
        "--task-name",
        help="Override the LeRobot task string for the name and annotation",
    )
    convert(parser.parse_args())
