import argparse
import json
from typing import Any

from .gemini335l import Gemini335LCamera


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List available Gemini 335L color/depth stream profiles (resolution, format, fps)."
    )
    parser.add_argument("--serial-number-or-name", default=None, help="Exact serial number, device name, or UID.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full profile payload as JSON for scripting.",
    )
    return parser


def _format_profile(profile: dict[str, Any]) -> str:
    return f"{profile['width']}x{profile['height']} @ {profile['fps']} FPS ({profile['format']})"


def _print_resolution_fps(summary: list[dict[str, Any]]) -> None:
    if not summary:
        print("    none")
        return
    for item in summary:
        fps_text = ", ".join(str(fps) for fps in item["fps"])
        print(f"    {item['width']}x{item['height']} [{item['format']}]: {fps_text} FPS")


def _print_camera_info(camera: dict[str, Any], index: int) -> None:
    print(f"[{index}] {camera.get('name', 'Unknown')} ({camera.get('serial_number', 'N/A')})")
    print(f"  uid: {camera.get('uid', 'N/A')}")
    print(f"  connection: {camera.get('connection_type', 'N/A')}")
    print(f"  firmware: {camera.get('firmware_version', 'N/A')}")

    default_color = camera.get("default_color_stream_profile")
    if default_color is not None:
        print(f"  default color: {_format_profile(default_color)}")
    elif "default_color_stream_profile_error" in camera:
        print(f"  default color error: {camera['default_color_stream_profile_error']}")

    default_depth = camera.get("default_depth_stream_profile")
    if default_depth is not None:
        print(f"  default depth: {_format_profile(default_depth)}")
    elif "default_depth_stream_profile_error" in camera:
        print(f"  default depth error: {camera['default_depth_stream_profile_error']}")

    print("  color resolution -> fps:")
    _print_resolution_fps(camera.get("color_resolution_fps", []))

    print("  depth resolution -> fps:")
    _print_resolution_fps(camera.get("depth_resolution_fps", []))

    depth_work_modes = camera.get("depth_work_modes")
    if depth_work_modes:
        print(f"  depth work modes: {', '.join(depth_work_modes)}")
    if camera.get("current_depth_work_mode") is not None:
        print(f"  current depth work mode: {camera['current_depth_work_mode']}")


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    cameras = Gemini335LCamera.list_stream_profiles(serial_number_or_name=args.serial_number_or_name)
    if not cameras:
        print("No Orbbec cameras detected.")
        return

    if args.json:
        print(json.dumps(cameras, indent=2, ensure_ascii=False))
        return

    for index, camera in enumerate(cameras):
        _print_camera_info(camera, index=index)


if __name__ == "__main__":
    main()
