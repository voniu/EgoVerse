"""Pick front-camera frames for gripper-centre labelling.

Frames are stratified over image position (4 x 3 grid) and depth (two halves) of the
projected TCP so the fit sees the whole workspace. Writes <workdir>/frames/*.jpg and
<workdir>/frames.json (episode, frame, projected TCP pixel, depth, gripper opening).
Frames that already carry labels in <workdir>/labels.json are kept on a re-run.

    python -m egomimic.scripts.piper_process.tcp_calibration.select_frames \
        --dataset <piper_zarr_dir> --workdir <tcp_workdir> --frames 80
"""

import argparse
import json
from pathlib import Path

import numpy as np
import simplejpeg
import zarr


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--stride", type=int, default=15, help="thin near-duplicates")
    parser.add_argument(
        "--margin-px",
        type=int,
        default=60,
        help="projected TCP must lie this far inside the image (the true gripper can "
        "be several centimetres from an uncorrected projection)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    (args.workdir / "frames").mkdir(parents=True, exist_ok=True)

    candidates = []
    for path in sorted(args.dataset.glob("*.zarr")):
        group = zarr.open_group(str(path), mode="r")
        n = int(group.attrs["total_frames"])
        height, width = simplejpeg.decode_jpeg_header(
            bytes(group["images.front_1"][0:1][0])
        )[:2]
        K = np.asarray(group.attrs["intrinsics"]["front_1"])[:, :3]
        pose = np.asarray(group["right.obs_ee_pose_camframe"][:n])
        grip = np.asarray(group["right.obs_gripper"][:n, 0])
        uvw = pose[:, :3] @ K.T
        uv = uvw[:, :2] / uvw[:, 2:3]
        m = args.margin_px
        inside = (
            (uvw[:, 2] > 0)
            & (uv[:, 0] > m)
            & (uv[:, 0] < width - m)
            & (uv[:, 1] > m)
            & (uv[:, 1] < height - m)
        )
        for i in np.flatnonzero(inside)[:: args.stride]:
            candidates.append(
                (
                    path.name,
                    int(i),
                    *uv[i],
                    float(pose[i, 2]),
                    float(grip[i]),
                    width,
                    height,
                )
            )
    if not candidates:
        raise SystemExit("no frame has the projected TCP inside the image")

    values = np.array([c[2:] for c in candidates], dtype=float)
    col = np.clip((values[:, 0] / values[:, 4] * 4).astype(int), 0, 3)
    row = np.clip((values[:, 1] / values[:, 5] * 3).astype(int), 0, 2)
    deep = (values[:, 2] > np.median(values[:, 2])).astype(int)
    cells = {}
    for k, key in enumerate(zip(col, row, deep)):
        cells.setdefault(key, []).append(k)
    per_cell = max(1, args.frames // len(cells))
    chosen = []
    for members in cells.values():
        chosen += list(
            rng.choice(members, size=min(per_cell, len(members)), replace=False)
        )

    frames_path = args.workdir / "frames.json"
    labels_path = args.workdir / "labels.json"
    old = json.loads(frames_path.read_text()) if frames_path.exists() else []
    done = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    frames = [f for f in old if "right" in done.get(f["id"], {})]
    rng.shuffle(chosen)
    for index in chosen:
        name, i, u, v, z, g, _, _ = candidates[index]
        fid = f"{Path(name).stem}_{i:05d}"
        if any(f["id"] == fid for f in frames):
            continue
        group = zarr.open_group(str(args.dataset / name), mode="r")
        jpeg = group["images.front_1"][i : i + 1][0]
        (args.workdir / "frames" / f"{fid}.jpg").write_bytes(bytes(jpeg))
        frames.append(
            {
                "id": fid,
                "episode": name,
                "frame": i,
                "proj_u": u,
                "proj_v": v,
                "depth_m": z,
                "gripper_m": g,
            }
        )
    frames_path.write_text(json.dumps(frames, indent=1))
    print(
        f"{len(frames)} frames from {len({f['episode'] for f in frames})} episodes, "
        f"{len(cells)} occupied cells -> {frames_path}"
    )


if __name__ == "__main__":
    main()
