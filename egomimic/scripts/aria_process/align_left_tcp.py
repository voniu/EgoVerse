"""Estimate and visualize a fixed left-wrist to Piper-like tool-axis convention."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import simplejpeg
import zarr
from scipy.spatial.transform import Rotation


def anatomical_frame(points):
    """Aria indices: wrist=5, index MCP=8, middle MCP=11, pinky MCP=17.

    +Z: wrist toward middle MCP. +Y: pinky toward index, orthogonalized to Z.
    Columns express tool axes in the frame of points. This defines a proper
    rotation for LEFT hands; it is not a mirrored right-hand transform.
    """
    z = points[..., 11, :] - points[..., 5, :]
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    y = points[..., 8, :] - points[..., 17, :]
    y -= np.sum(y * z, axis=-1, keepdims=True) * z
    y /= np.linalg.norm(y, axis=-1, keepdims=True)
    x = np.cross(y, z)
    return np.stack((x, y, z), axis=-1)


def error_summary(rotations):
    degrees = np.rad2deg(Rotation.from_matrix(rotations).magnitude())
    return dict(
        zip(
            ("median_deg", "p95_deg", "max_deg"),
            map(float, np.percentile(degrees, [50, 95, 100])),
        )
    )


def draw_frame(image, origin, rotation, K, title, gripper=False):
    def pixels(points):
        projected = np.asarray(points) @ K.T
        return np.rint(projected[:, :2] / projected[:, 2:]).astype(int)

    axes = np.vstack((origin, origin + 0.06 * rotation.T))
    if np.all(axes[:, 2] > 0):
        uv = pixels(axes)
        for label, end, color in zip(
            "XYZ", uv[1:], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        ):
            cv2.arrowedLine(image, tuple(uv[0]), tuple(end), color, 2, tipLength=0.2)
            cv2.putText(
                image, label, tuple(end), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        cv2.circle(image, tuple(uv[0]), 5, (0, 255, 255), 2)
    if gripper:
        # Schematic only: 60 mm opening, 70 mm fingers; tips flank palm origin.
        local = np.array(
            [[0, -0.03, 0], [0, -0.03, -0.07], [0, 0.03, -0.07], [0, 0.03, 0]]
        )
        points = local @ rotation.T + origin
        if np.all(points[:, 2] > 0):
            cv2.polylines(image, [pixels(points)], False, (255, 255, 0), 2)
    cv2.putText(
        image, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
    )
    return image


def run(episode, output, validation_dir=None):
    output.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(episode), mode="r")
    n = store.attrs["total_frames"]  # Physical arrays can contain allocated padding.
    keypoints = store["left.source_keypoints"][:n].reshape(-1, 21, 3)
    wrist = store["left.source_wrist_pose"][:n]
    wrist_R = Rotation.from_quat(wrist[:, [4, 5, 6, 3]]).as_matrix()
    target_R = anatomical_frame(keypoints)
    local_R = wrist_R.transpose(0, 2, 1) @ target_R
    fixed = Rotation.from_matrix(local_R).mean().as_matrix()
    np.testing.assert_allclose(fixed.T @ fixed, np.eye(3), atol=1e-12)
    assert np.linalg.det(fixed) > 0.999999
    report = {
        "status": "candidate_for_visual_review",
        "fit_episode": str(episode),
        "fit_frames": n,
        "mapping": "R_camera_tool = R_camera_native_left_wrist @ R_wrist_tool",
        "R_wrist_tool": fixed.tolist(),
        "quaternion_xyzw": Rotation.from_matrix(fixed).as_quat().tolist(),
        "tool_origin": "palm center (unchanged)",
        "tool_z": "wrist to middle finger MCP",
        "tool_y": "pinky MCP to index MCP, projected perpendicular to Z",
        "tool_x": "Y cross Z",
        "fit_residual": error_summary(fixed.T @ local_R),
        "interpretation": "SDK skeleton consistency, not physical orientation accuracy",
    }
    if validation_dir:
        from projectaria_tools.core import data_provider

        reports = []
        for path in sorted(validation_dir.glob("*.vrs")):
            provider = data_provider.create_vrs_data_provider(str(path))
            stream = provider.get_stream_id_from_label("handtracking")
            residuals = []
            for i in range(provider.get_num_data(stream)):
                hand = provider.get_hand_pose_data_by_index(stream, i).left_hand
                if hand is None or hand.confidence < 0.5:
                    continue
                frame = anatomical_frame(np.asarray(hand.landmark_positions_device))
                rotation = hand.transform_device_wrist.to_matrix()[:3, :3]
                residuals.append(fixed.T @ rotation.T @ frame)
            reports.append(
                {
                    "file": path.name,
                    "frames": len(residuals),
                    **error_summary(np.asarray(residuals)),
                }
            )
            print(path.name, reports[-1], flush=True)
        report["validation"] = reports

    (output / "alignment.json").write_text(json.dumps(report, indent=2))

    camera = store["head_pose"][:n]
    camera_R = Rotation.from_quat(camera[:, [4, 5, 6, 3]]).as_matrix()
    cam_world_R = camera_R.transpose(0, 2, 1)
    origin = np.einsum(
        "nij,nj->ni", cam_world_R, store["left.obs_ee_pose"][:n, :3] - camera[:, :3]
    )
    native = cam_world_R @ wrist_R
    aligned = native @ fixed
    K = np.asarray(store.attrs["intrinsics"]["front_1"])[:, :3]
    video = cv2.VideoWriter(
        str(output / "comparison.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 480)
    )
    assert video.isOpened()
    snapshots = []
    for i in range(n):
        rgb = simplejpeg.decode_jpeg(
            store["images.front_1"][i : i + 1][0], colorspace="RGB"
        )
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        left = draw_frame(bgr.copy(), origin[i], native[i], K, "Native left wrist axes")
        right = draw_frame(
            bgr.copy(),
            origin[i],
            aligned[i],
            K,
            "Piper-like axes + schematic gripper",
            True,
        )
        joined = np.concatenate((left, right), axis=1)
        cv2.putText(
            joined,
            f'RGB frame {int(store["source.rgb_index"][i, 0])}',
            (10, 465),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        video.write(joined)
        if i in {0, n // 2, n - 1}:
            snapshots.append(joined)
    video.release()
    cv2.imwrite(str(output / "comparison.jpg"), np.concatenate(snapshots, axis=0))
    print(json.dumps({k: v for k, v in report.items() if k != "validation"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path)
    args = parser.parse_args()
    run(args.episode, args.output, args.validation_dir)
