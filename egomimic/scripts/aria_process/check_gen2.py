"""Check aligned episodes against preserved source poses and real training reads."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Subset

from egomimic.rldb.embodiment.aria_gen2_left import AriaGen2Left
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset


def check(folder, alignment_path):
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["complete"]
    alignment = np.asarray(json.loads(alignment_path.read_text())["R_wrist_tool"])
    reports = []
    for episode in manifest["episodes"]:
        path = Path(episode["output"])
        ds = ZarrDataset(
            path,
            AriaGen2Left.get_keymap(norm_mode=True),
            AriaGen2Left.get_transform_list(),
        )
        n = len(ds)
        store = ds.episode_reader._store
        assert ds.metadata["human_orientation_alignment_applied"]
        np.testing.assert_allclose(ds.metadata["R_wrist_tool"], alignment)
        pose, wrist = store["left.obs_ee_pose"][:n], store["left.source_wrist_pose"][:n]
        points = store["left.source_keypoints"][:n].reshape(n, 21, 3)
        np.testing.assert_allclose(pose[:, :3], points[:, 20], atol=1e-6)
        raw_R = Rotation.from_quat(wrist[:, [4, 5, 6, 3]]).as_matrix()
        aligned_R = Rotation.from_quat(pose[:, [4, 5, 6, 3]]).as_matrix()
        np.testing.assert_allclose(aligned_R, raw_R @ alignment, atol=3e-7)
        stamp = store["source.device_timestamp_ns"][:n, 0]
        assert np.all((np.diff(stamp) > 0) & (np.diff(stamp) < 50_000_000))
        indices = [0, n // 2, max(0, n - 5), n - 1]
        for i in indices:
            sample = ds[i]
            camera = store["head_pose"][i]
            camera_R = Rotation.from_quat(camera[[4, 5, 6, 3]]).as_matrix()
            take = np.minimum(np.arange(i, i + 30), n - 1)
            action = sample["actions_cartesian"]
            assert action.shape == (30, 6) and np.isfinite(action).all()
            expected_xyz = (pose[take, :3] - camera[:3]) @ camera_R
            expected_R = camera_R.T @ raw_R[take] @ alignment
            np.testing.assert_allclose(action[:, :3], expected_xyz, atol=2e-6)
            np.testing.assert_allclose(
                Rotation.from_euler("ZYX", action[:, 3:]).as_matrix(),
                expected_R,
                atol=2e-6,
            )
            np.testing.assert_allclose(
                sample["observations.state.ee_pose"], action[0], atol=1e-7
            )
        reports.append(
            {
                "episode": path.name,
                "frames": n,
                "retained_fraction": n / episode["source_frames"],
            }
        )

    image_ds = ZarrDataset(
        Path(manifest["episodes"][0]["output"]),
        AriaGen2Left.get_keymap(),
        AriaGen2Left.get_transform_list(),
    )
    batch = next(
        iter(DataLoader(Subset(image_ds, [0, len(image_ds) - 1]), batch_size=2))
    )
    assert batch["observations.images.front_img_1"].shape == (2, 3, 480, 640)
    assert batch["actions_cartesian"].shape == (2, 30, 6)
    assert (batch["embodiment"] == 2).all()
    report = {
        "passed": True,
        "episodes": reports,
        "total_frames": sum(r["frames"] for r in reports),
        "checks": [
            "source_palm_preserved",
            "fixed_rotation_applied_once",
            "continuous_timestamps",
            "current_camera_future_chunk",
            "tail_padding",
            "real_image_batch",
        ],
        "clip_completeness_manually_reviewed": False,
    }
    (folder / "batch_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--alignment", type=Path, required=True)
    args = parser.parse_args()
    check(args.folder, args.alignment)
