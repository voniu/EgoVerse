"""Add binary gripper-command intent to converted Piper Zarr episodes."""

import argparse
from pathlib import Path

import numpy as np
import zarr

from egomimic.robot.piper.gripper_utils import binarize_gripper_commands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    episodes = sorted(args.dataset.glob("*.zarr"))
    if not episodes:
        raise FileNotFoundError(f"No Zarr episodes under {args.dataset}")
    counts = {0: 0, 1: 0}
    for episode in episodes:
        group = zarr.open_group(str(episode), mode="a")
        source = np.asarray(group["source.action"][:, 6:7], dtype=np.float32) / 100.0
        binary = binarize_gripper_commands(source)
        key = "right.cmd_gripper_binary"
        if key in group:
            existing = np.asarray(group[key][:], dtype=np.float32)
            if np.array_equal(existing, binary):
                pass
            elif args.overwrite:
                del group[key]
                group[key] = binary
            else:
                raise ValueError(f"{episode.name}: existing {key} differs")
        else:
            group[key] = binary
        counts[0] += int((binary == 0).sum())
        counts[1] += int((binary == 1).sum())
    print(f"episodes={len(episodes)} closed={counts[0]} open={counts[1]}")


if __name__ == "__main__":
    main()
