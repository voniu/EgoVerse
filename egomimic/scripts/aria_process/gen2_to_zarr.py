"""Convert Gen 2 left-hand tracking, optionally with Piper-compatible tool axes."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import simplejpeg
from projectaria_tools.core import calibration
from projectaria_tools.core.sensor_data import TrackingQuality, VioStatus
from scipy.spatial.transform import Rotation, Slerp

from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.scripts.aria_process.inspect_gen2 import (
    nearest_indices,
    read_tracking,
    valid_runs,
)


def poses(matrices):
    quat = Rotation.from_matrix(matrices[:, :3, :3]).as_quat()
    return np.column_stack((matrices[:, :3, 3], quat[:, [3, 0, 1, 2]])).astype(
        np.float32
    )


def interpolate_poses(matrices, times, good, sessions, queries, max_gap_ns):
    """Interpolate only adjacent valid poses in the same odometry session."""
    right = np.searchsorted(times, queries, side="right").clip(1, len(times) - 1)
    left = right - 1
    valid = (
        good[left]
        & good[right]
        & (sessions[left] == sessions[right])
        & (queries >= times[0])
        & (queries <= times[-1])
        & ((times[right] - times[left]) <= max_gap_ns)
    )
    result = np.repeat(np.eye(4)[None], len(queries), axis=0)
    for i in np.flatnonzero(valid):
        a, b = left[i], right[i]
        alpha = float(queries[i] - times[a]) / float(times[b] - times[a])
        result[i, :3, 3] = (1 - alpha) * matrices[a, :3, 3] + alpha * matrices[b, :3, 3]
        result[i, :3, :3] = Slerp(
            [0, 1], Rotation.from_matrix(matrices[[a, b], :3, :3])
        )([alpha]).as_matrix()[0]
    return result, valid


def interpolate_vio(vio, times, queries):
    """10 Hz on-device VIO; record gaps up to 150 ms are bridged."""
    good = np.array(
        [
            v.status == VioStatus.VALID and v.pose_quality == TrackingQuality.GOOD
            for v in vio
        ]
    )
    sessions = np.array([str(v.frontend_session_uid) for v in vio])
    matrices = np.array(
        [
            (v.transform_odometry_bodyimu @ v.transform_bodyimu_device).to_matrix()
            for v in vio
        ]
    )
    return interpolate_poses(matrices, times, good, sessions, queries, 150_000_000)


def read_vio_high_freq(provider):
    """800 Hz on-device VIO in the same odometry frame as the 10 Hz stream."""
    stream = provider.get_stream_id_from_label("vio_high_frequency")
    samples = [
        provider.get_vio_high_freq_data_by_index(stream, i)
        for i in range(provider.get_num_data(stream))
    ]
    times = np.asarray(
        [round(v.tracking_timestamp.total_seconds() * 1e9) for v in samples],
        dtype=np.int64,
    )
    order = np.argsort(times, kind="stable")
    samples = [samples[i] for i in order]
    return samples, times[order]


def interpolate_vio_high_freq(samples, times, queries):
    good = np.array([v.quality_score >= HIGH_FREQ_MIN_QUALITY for v in samples])
    sessions = np.array([str(v.session_uid) for v in samples])
    matrices = np.array([v.transform_odometry_device.to_matrix() for v in samples])
    return interpolate_poses(matrices, times, good, sessions, queries, 50_000_000)


HIGH_FREQ_MIN_QUALITY = 0.5


def project(points, K):
    pixels = points @ K.T
    return pixels[:, :2] / pixels[:, 2:3]


def convert(
    path,
    output,
    alignment=None,
    task_name="Put bear into basket",
    vio_source="vio",
):
    provider, ids, times, hands, hand_times, vio, vio_times = read_tracking(path)
    nearest = nearest_indices(hand_times, times)
    if vio_source == "high_frequency":
        samples, sample_times = read_vio_high_freq(provider)

        def interpolate(queries):
            return interpolate_vio_high_freq(samples, sample_times, queries)

    else:

        def interpolate(queries):
            return interpolate_vio(vio, vio_times, queries)

    world_device, vio_valid = interpolate(times)
    world_device_hand, hand_vio_valid = interpolate(hand_times[nearest])
    confidence = np.array(
        [
            hands[i].left_hand.confidence if hands[i].left_hand is not None else 0.0
            for i in nearest
        ]
    )
    valid = (
        vio_valid
        & hand_vio_valid
        & (confidence >= 0.5)
        & (abs(hand_times[nearest] - times) <= 25_000_000)
    )
    runs = valid_runs(valid)
    start, end = max(runs, key=lambda run: run[1] - run[0], default=(0, 0))
    assert (
        end - start >= 30
    ), "No continuous segment with a complete 30-frame action chunk"
    assert (
        np.diff(times[start:end]).max() < 50_000_000
    ), "RGB discontinuity: do not compress missing time"

    source_camera = provider.get_device_calibration().get_camera_calib("camera-rgb")
    width, height = 640, 480
    focal = (
        float(source_camera.get_focal_lengths()[0])
        * width
        / source_camera.get_image_size()[0]
    )
    linear_camera = calibration.get_linear_camera_calibration(
        width, height, focal, "camera-rgb", source_camera.get_transform_device_camera()
    )
    T_device_camera = linear_camera.get_transform_device_camera().to_matrix()
    world_camera = world_device @ T_device_camera
    K = np.eye(3)
    K[0, 0], K[1, 1] = linear_camera.get_focal_lengths()
    K[:2, 2] = linear_camera.get_principal_point()

    world_palm, world_wrist, keypoints_world = [], [], []
    for i in range(start, end):
        hand = hands[nearest[i]].left_hand
        wrist = hand.transform_device_wrist.to_matrix()
        palm = wrist.copy()
        palm[:3, 3] = np.asarray(hand.get_palm_position_device()).reshape(3)
        world_palm.append(world_device_hand[i] @ palm)
        world_wrist.append(world_device_hand[i] @ wrist)
        points = np.asarray(hand.landmark_positions_device)
        keypoints_world.append(
            points @ world_device_hand[i, :3, :3].T + world_device_hand[i, :3, 3]
        )
    world_palm, world_wrist, keypoints_world = map(
        np.asarray, (world_palm, world_wrist, keypoints_world)
    )
    if alignment is not None:
        world_palm[:, :3, :3] = world_palm[:, :3, :3] @ alignment
    cameras = world_camera[start:end]
    camera_palm = np.linalg.inv(cameras) @ world_palm
    fps = float(1e9 / np.median(np.diff(times)))
    name = path.stem + f"_left_{start:04d}_{end:04d}"
    if alignment is not None:
        name += "_piper_axes"
    destination = output / (name + ".zarr")
    if destination.exists():
        raise FileExistsError(destination)
    output.mkdir(parents=True, exist_ok=True)
    preview_path = output / (name + "_overlay.mp4")
    video = cv2.VideoWriter(
        str(preview_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    assert video.isOpened(), preview_path
    encoded, snapshots = [], []
    for i in range(len(times)):
        image, _ = provider.get_image_data_by_index(ids["camera-rgb"], i)
        rgb = calibration.distort_by_calibration(
            image.to_numpy_array(), linear_camera, source_camera
        )
        shown = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if start <= i < end:
            j = i - start
            encoded.append(
                simplejpeg.encode_jpeg(
                    np.ascontiguousarray(rgb),
                    quality=ZarrWriter.JPEG_QUALITY,
                    colorspace="RGB",
                )
            )
            T = camera_palm[j]
            axes = np.vstack((T[:3, 3], T[:3, 3] + 0.045 * T[:3, :3].T))
            if np.all(axes[:, 2] > 0):
                pixels = project(axes, K).round().astype(int)
                for endpoint, color in zip(
                    pixels[1:], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
                ):
                    cv2.line(shown, tuple(pixels[0]), tuple(endpoint), color, 2)
                cv2.circle(shown, tuple(pixels[0]), 6, (0, 255, 255), 2)
            cv2.putText(
                shown,
                "KEPT: palm + "
                + ("Piper axes" if alignment is not None else "native wrist axes"),
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
            )
        else:
            cv2.putText(
                shown,
                "EXCLUDED: tracking / boundary",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                1,
            )
        cv2.putText(
            shown,
            f"source frame {i}",
            (10, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        video.write(shown)
        if i in {0, start, (start + end) // 2, end - 1}:
            snapshots.append(shown)
    video.release()
    cv2.imwrite(
        str(output / (name + "_preview.jpg")), np.concatenate(snapshots, axis=1)
    )

    ZarrWriter.create_and_write(
        destination,
        numeric_data={
            "left.obs_ee_pose": poses(world_palm),
            "left.source_wrist_pose": poses(world_wrist),
            "left.source_keypoints": keypoints_world.reshape(-1, 63).astype(np.float32),
            "head_pose": poses(cameras),
            "timestamp": ((times[start:end] - times[0]) / 1e9).reshape(-1, 1),
            "source.rgb_index": np.arange(start, end, dtype=np.int64).reshape(-1, 1),
            "source.device_timestamp_ns": times[start:end].reshape(-1, 1),
            "left.confidence": confidence[start:end].astype(np.float32).reshape(-1, 1),
        },
        pre_encoded_image_data={
            "images.front_1": (np.array(encoded, dtype=object), [height, width, 3])
        },
        embodiment="human_left_arm",
        fps=30,
        task_name=task_name,
        annotations=[(task_name, 0, end - start)],
        intrinsics={"front_1": np.column_stack((K, np.zeros(3)))},
        metadata_override={
            "source_vrs": str(path),
            "source_rgb_frames": len(times),
            "source_frame_range_exclusive": [start, end],
            "source_measured_fps": fps,
            "tracking_source": "gen2_on_device_hand_and_vio_not_cloud_mps",
            "vio_source": vio_source,
            "pose_frame": "odometry",
            "head_pose_frame": "odometry_T_rectified_rgb_optical",
            "pose_origin": "left_palm",
            "pose_axes": (
                "piper_like_left_tool" if alignment is not None else "native_left_wrist"
            ),
            "human_orientation_alignment_applied": alignment is not None,
            "R_wrist_tool": alignment.tolist() if alignment is not None else None,
            "training_status": (
                "requires_clip_review"
                if alignment is not None
                else "pilot_requires_alignment_and_clip_review"
            ),
            "quaternion_order": "wxyz",
            "position_unit": "m",
        },
    )
    report = {
        "source": str(path),
        "output": str(destination),
        "preview": str(preview_path),
        "source_frames": len(times),
        "kept_range": [start, end],
        "kept_frames": end - start,
        "left_hand_only": True,
        "orientation_alignment_applied": alignment is not None,
        "training_ready": False,
    }
    (output / (name + "_report.json")).write_text(json.dumps(report, indent=2))
    print("CONVERTED", json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vrs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task-name",
        default="Put bear into basket",
        help="Task name and whole-episode language annotation",
    )
    parser.add_argument(
        "--vio-source",
        choices=["vio", "high_frequency"],
        default="vio",
        help="On-device pose stream: 10 Hz vio or 800 Hz vio_high_frequency",
    )
    parser.add_argument(
        "--alignment",
        type=Path,
        help="Fixed R_wrist_tool JSON; omit for native wrist axes",
    )
    args = parser.parse_args()
    alignment = None
    if args.alignment:
        alignment = np.asarray(json.loads(args.alignment.read_text())["R_wrist_tool"])
        np.testing.assert_allclose(alignment.T @ alignment, np.eye(3), atol=1e-8)
        assert np.isclose(
            np.linalg.det(alignment), 1
        ), "Alignment must be a proper rotation"
    paths = sorted(args.vrs.glob("*.vrs")) if args.vrs.is_dir() else [args.vrs]
    reports, failures = [], []
    for path in paths:
        try:
            reports.append(
                convert(path, args.output, alignment, args.task_name, args.vio_source)
            )
        except (AssertionError, FileExistsError) as error:
            print("SKIPPED", path, repr(error), flush=True)
            failures.append({"source": str(path), "error": repr(error)})
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "manifest.json").write_text(
            json.dumps(
                {
                    "complete": len(reports) == len(paths),
                    "episodes": reports,
                    "failures": failures,
                    "orientation_alignment_applied": alignment is not None,
                },
                indent=2,
            )
        )
