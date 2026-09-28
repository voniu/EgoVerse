"""Audit Gen 2 on-device hand/VIO coverage at RGB timestamps, without MPS upload."""

import argparse
import json
from pathlib import Path

import numpy as np
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TrackingQuality, VioStatus


def nearest_indices(times, queries):
    right = np.searchsorted(times, queries).clip(0, len(times) - 1)
    left = (right - 1).clip(0)
    return np.where(
        abs(times[left] - queries) <= abs(times[right] - queries), left, right
    )


def valid_runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(
        zip(np.where(edges == 1)[0].tolist(), np.where(edges == -1)[0].tolist())
    )


def read_tracking(path):
    provider = data_provider.create_vrs_data_provider(str(path))
    ids = {
        label: provider.get_stream_id_from_label(label)
        for label in ("camera-rgb", "handtracking", "vio")
    }
    rgb_ns = np.asarray(
        provider.get_timestamps_ns(ids["camera-rgb"], TimeDomain.DEVICE_TIME),
        dtype=np.int64,
    )
    hands = [
        provider.get_hand_pose_data_by_index(ids["handtracking"], i)
        for i in range(provider.get_num_data(ids["handtracking"]))
    ]
    hand_ns = np.asarray(
        [round(h.tracking_timestamp.total_seconds() * 1e9) for h in hands],
        dtype=np.int64,
    )
    vio = [
        provider.get_vio_data_by_index(ids["vio"], i)
        for i in range(provider.get_num_data(ids["vio"]))
    ]
    vio_ns = np.asarray([v.capture_timestamp_ns for v in vio], dtype=np.int64)
    return provider, ids, rgb_ns, hands, hand_ns, vio, vio_ns


def audit(path):
    _, _, times, hands, ht, vio, vt = read_tracking(path)
    hi = nearest_indices(ht, times)
    vi = nearest_indices(vt, times)
    vio_ok = np.asarray(
        [
            v.status == VioStatus.VALID and v.pose_quality == TrackingQuality.GOOD
            for v in vio
        ]
    )
    report = {
        "file": str(path),
        "rgb_frames": len(times),
        "duration_s": float((times[-1] - times[0]) / 1e9),
        "median_fps": float(1e9 / np.median(np.diff(times))),
        "max_rgb_gap_ms": float(np.diff(times).max() / 1e6),
        "hand_nearest_error_ms_p95": float(
            np.percentile(abs(ht[hi] - times) / 1e6, 95)
        ),
        "vio_nearest_error_ms_p95": float(np.percentile(abs(vt[vi] - times) / 1e6, 95)),
        "vio_good_records": int(vio_ok.sum()),
        "vio_records": len(vio),
        "vio_sessions": sorted({v.frontend_session_uid for v in vio}),
    }
    for side in ("left", "right"):
        confidence = np.asarray(
            [
                (
                    getattr(hands[i], side + "_hand").confidence
                    if getattr(hands[i], side + "_hand") is not None
                    else 0.0
                )
                for i in hi
            ]
        )
        mask = (
            (confidence >= 0.5)
            & (abs(ht[hi] - times) <= 25_000_000)
            & vio_ok[vi]
            & (times >= vt[0])
            & (times <= vt[-1])
        )
        report[side] = {
            "hand_present_frames": int((confidence > 0).sum()),
            "confidence_ge_0_5_frames": int((confidence >= 0.5).sum()),
            "usable_frames": int(mask.sum()),
            "contiguous_runs": valid_runs(mask),
            "longest_run_frames": max(
                (end - start for start, end in valid_runs(mask)), default=0
            ),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.input.glob("*.vrs")) if args.input.is_dir() else [args.input]
    reports = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in files:
        reports.append(audit(path))
        print(
            "AUDIT",
            path.name,
            reports[-1]["left"]["usable_frames"],
            reports[-1]["right"]["usable_frames"],
            flush=True,
        )
        args.output.write_text(
            json.dumps({"complete": False, "recordings": reports}, indent=2)
        )
    result = {
        "complete": True,
        "recording_count": len(reports),
        "rgb_frames": sum(r["rgb_frames"] for r in reports),
        "confidence_threshold": 0.5,
        "max_hand_time_error_ms": 25,
        "recordings": reports,
    }
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
