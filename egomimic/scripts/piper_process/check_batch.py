"""Check a real Piper batch, including episode-end padding and rotation semantics."""

import argparse
import json
from pathlib import Path

import numpy as np
import simplejpeg
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Subset

from egomimic.rldb.embodiment.piper import Piper
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset


def check(path, horizon):
    ds = ZarrDataset(
        path, Piper.get_keymap(action_horizon=horizon), Piper.get_transform_list()
    )
    # Start, middle, partially padded chunk, fully padded tail.
    indices = [0, len(ds) // 2, max(0, len(ds) - 5), len(ds) - 1]
    batch = next(iter(DataLoader(Subset(ds, indices), batch_size=4, num_workers=0)))
    state = batch["observations.state.ee_pose"].numpy()
    action = batch["actions_cartesian"].numpy()
    assert state.shape == (4, 7) and action.shape == (4, horizon, 7)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            assert value.dtype == torch.float32 and torch.isfinite(value).all(), key
    assert (batch["embodiment"] == 7).all()
    assert batch["intrinsics"].shape == (4, 3, 4)
    for key in [
        "observations.images.front_img_1",
        "observations.images.right_wrist_img",
    ]:
        assert batch[key].shape == (4, 3, 480, 640)
        assert 0 <= batch[key].min() <= batch[key].max() <= 1

    store = ds.episode_reader
    for b, index in enumerate(indices):
        end = min(index + horizon, len(ds))
        raw = store.read(
            {
                "right.obs_ee_pose_camframe": (index, None),
                "right.obs_gripper": (index, None),
                "right.cmd_ee_pose_camframe": (index, end),
                "right.cmd_gripper": (index, end),
                "timestamp": (index, end),
                "images.front_1": (index, None),
                "images.right_wrist": (index, None),
            }
        )
        for output, pose_key, grip_key in [
            (state[b : b + 1], "right.obs_ee_pose_camframe", "right.obs_gripper"),
            (action[b], "right.cmd_ee_pose_camframe", "right.cmd_gripper"),
        ]:
            pose = np.atleast_2d(raw[pose_key])
            grip = np.asarray(raw[grip_key]).reshape(-1, 1)
            take = np.minimum(np.arange(len(output)), len(pose) - 1)
            # Direct equality of xyz verifies no accidental second camera transform.
            np.testing.assert_allclose(output[:, :3], pose[take, :3], atol=1e-7)
            np.testing.assert_allclose(output[:, 6:], grip[take], atol=1e-7)
            R_out = Rotation.from_euler("ZYX", output[:, 3:6]).as_matrix()
            R_raw = Rotation.from_quat(pose[take][:, [4, 5, 6, 3]]).as_matrix()
            np.testing.assert_allclose(R_out, R_raw, atol=3e-7)
        np.testing.assert_allclose(
            raw["timestamp"].ravel(),
            np.arange(index, end) / ds.metadata["fps"],
            atol=1e-5,
        )
        for image_key, batch_key in [
            ("images.front_1", "observations.images.front_img_1"),
            ("images.right_wrist", "observations.images.right_wrist_img"),
        ]:
            expected = (
                simplejpeg.decode_jpeg(raw[image_key], colorspace="RGB").transpose(
                    2, 0, 1
                )
                / 255.0
            )
            np.testing.assert_allclose(batch[batch_key][b].numpy(), expected, atol=1e-7)

    norm_ds = ZarrDataset(
        path,
        Piper.get_keymap(norm_mode=True, action_horizon=horizon),
        Piper.get_transform_list(),
    )
    norm_sample = norm_ds[0]
    assert not any("images" in k for k in norm_sample)
    np.testing.assert_array_equal(norm_sample["actions_cartesian"], action[0])
    report = {
        "episode": str(path),
        "indices": indices,
        "action_horizon": horizon,
        "fps": ds.metadata["fps"],
        "batch_shapes": {
            k: list(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)
        },
        "checks": [
            "finite_float32",
            "RGB_images_match_current_frame",
            "camera_xyz_unchanged",
            "quaternion_to_intrinsic_ZYX",
            "gripper_meters_unchanged",
            "same_row_command_start",
            "repeat_last_command_at_end",
            "normalization_keymap",
        ],
        "normalized": False,
        "human_orientation_alignment_applied": False,
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = check(args.episode, args.horizon)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
