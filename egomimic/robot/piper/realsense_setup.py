"""List connected RealSense cameras and write their serials and intrinsics into a setup.

    python -m egomimic.robot.piper.realsense_setup list
    python -m egomimic.robot.piper.realsense_setup write --setup <setup.yaml> \
        --front <d435if-serial> --wrist <d405-serial>

`write` starts each camera at the streaming profile used for recording and deployment
(640x480 RGB, 30 Hz), stores the factory colour intrinsics in cameras.d435if.intrinsics,
the serials in cameras.*.serial, and the wrist intrinsics as a JSON file next to the
setup (cameras.d405.intrinsics_file; lerobot_to_zarr --wrist-intrinsics). A new setup
file starts from setup.example.yaml. Hand-eye extrinsics are not touched.
"""

import argparse
import datetime
import json
from pathlib import Path

import yaml

TEMPLATE = Path(__file__).with_name("setup.example.yaml")
WIDTH, HEIGHT, FPS = 640, 480, 30


def intrinsics_record(device, intrinsics):
    """JSON record of one colour stream; device: name/serial/firmware strings."""
    K = [
        [intrinsics["fx"], 0.0, intrinsics["ppx"]],
        [0.0, intrinsics["fy"], intrinsics["ppy"]],
        [0.0, 0.0, 1.0],
    ]
    return {
        **device,
        "read_time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "fps": FPS,
        "color": {
            "width": intrinsics["width"],
            "height": intrinsics["height"],
            "fx": intrinsics["fx"],
            "fy": intrinsics["fy"],
            "cx": intrinsics["ppx"],
            "cy": intrinsics["ppy"],
            "distortion_model": intrinsics["model"],
            "coefficients": list(intrinsics["coeffs"]),
            "K": K,
        },
    }


def fill_setup(setup, front, wrist, wrist_file):
    """Write serials and intrinsics into a setup mapping (front/wrist: records)."""
    cameras = setup.setdefault("cameras", {})
    d435if = cameras.setdefault("d435if", {})
    d435if["serial"] = front["serial"]
    color = front["color"]
    d435if["intrinsics"] = {
        "width": color["width"],
        "height": color["height"],
        "K": color["K"],
        "distortion_model": color["distortion_model"].removeprefix("distortion."),
        "distortion_coefficients": color["coefficients"],
        "source": f"realsense factory calibration, {front['read_time_utc'][:10]}",
    }
    d405 = cameras.setdefault("d405", {})
    d405["serial"] = wrist["serial"]
    d405["intrinsics_file"] = str(wrist_file)
    return setup


def read_camera(rs, serial):
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, FPS)
    profile = pipeline.start(config)
    try:
        device = profile.get_device()
        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = stream.get_intrinsics()
        info = {
            "serial": device.get_info(rs.camera_info.serial_number),
            "device_name": device.get_info(rs.camera_info.name),
            "firmware": device.get_info(rs.camera_info.firmware_version),
        }
        return intrinsics_record(
            info,
            {
                "width": intr.width,
                "height": intr.height,
                "fx": intr.fx,
                "fy": intr.fy,
                "ppx": intr.ppx,
                "ppy": intr.ppy,
                "model": str(intr.model),
                "coeffs": intr.coeffs,
            },
        )
    finally:
        pipeline.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="print connected cameras")
    write = sub.add_parser("write", help="fill serials and intrinsics of a setup")
    write.add_argument("--setup", type=Path, required=True)
    write.add_argument("--front", required=True, help="D435IF serial")
    write.add_argument("--wrist", required=True, help="D405 serial")
    args = parser.parse_args()
    import pyrealsense2 as rs

    if args.command == "list":
        for device in rs.context().query_devices():
            print(
                device.get_info(rs.camera_info.name),
                device.get_info(rs.camera_info.serial_number),
                "USB",
                device.get_info(rs.camera_info.usb_type_descriptor),
                "firmware",
                device.get_info(rs.camera_info.firmware_version),
            )
        return
    front, wrist = read_camera(rs, args.front), read_camera(rs, args.wrist)
    source = args.setup if args.setup.exists() else TEMPLATE
    setup = yaml.safe_load(source.read_text()) or {}
    wrist_name = f"d405_{wrist['serial']}_intrinsics_{WIDTH}x{HEIGHT}.json"
    args.setup.parent.mkdir(parents=True, exist_ok=True)
    (args.setup.parent / wrist_name).write_text(json.dumps(wrist, indent=2))
    args.setup.write_text(
        yaml.safe_dump(fill_setup(setup, front, wrist, wrist_name), sort_keys=False)
    )
    print(f"wrote {args.setup} and {args.setup.parent / wrist_name}")


if __name__ == "__main__":
    main()
