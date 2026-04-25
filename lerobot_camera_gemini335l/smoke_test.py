import argparse
from pathlib import Path

import cv2
import numpy as np

from .config_gemini335l import Gemini335LCameraConfig
from .gemini335l import Gemini335LCamera
from .runtime_env import build_diag_command_hint, find_udev_install_script

UDEV_INSTALL_SCRIPT = find_udev_install_script()


def _save_color_image(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), image_bgr)


def _save_depth_preview(depth_map: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "depth.npy", depth_map)

    if depth_map.dtype != np.uint8:
        normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)
    else:
        normalized = depth_map
    preview = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    cv2.imwrite(str(output_dir / "depth_preview.png"), preview)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test for Gemini 335L LeRobot camera plugin.")
    parser.add_argument("--list", action="store_true", help="List detected Orbbec cameras and exit.")
    parser.add_argument("--list-profiles", action="store_true", help="List available stream profiles and exit.")
    parser.add_argument("--serial-number-or-name", default=None, help="Exact serial number, device name, or UID.")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--color-stream-format", type=str, default=None, help="Example: RGB, BGR, MJPG.")
    parser.add_argument("--depth-stream-format", type=str, default=None, help="Example: Y16, Z16.")
    parser.add_argument(
        "--depth-work-mode",
        type=str,
        default=None,
        help="Depth algorithm mode name, e.g. Default, Hand, High Accuracy.",
    )
    parser.add_argument(
        "--disp-search-range-mode",
        type=int,
        default=None,
        help="Orbbec disparity search range mode integer value. This may affect minimum working distance.",
    )
    parser.add_argument(
        "--disp-search-offset",
        type=int,
        default=None,
        help="Orbbec disparity search offset integer value.",
    )
    parser.add_argument(
        "--profile-selection-strategy",
        choices=("exact", "closest"),
        default="exact",
        help="exact: strict mode matching; closest: fallback to nearest available mode.",
    )
    parser.add_argument("--use-depth", action="store_true", help="Enable depth stream and save depth outputs.")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a live preview window. Press q or Esc to exit.",
    )
    parser.add_argument(
        "--align-depth-to-color",
        action="store_true",
        help="Align depth frames to the color stream. Requires --use-depth.",
    )
    parser.add_argument(
        "--align-mode",
        choices=("hw", "sw"),
        default="hw",
        help="Depth-to-color alignment mode. Use 'sw' when hardware alignment is unsupported.",
    )
    parser.add_argument("--read-count", type=int, default=5, help="Number of frames to read before saving.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/gemini335l_smoke"),
        help="Directory used to store smoke-test outputs.",
    )
    return parser


def _make_depth_preview(depth_map: np.ndarray) -> np.ndarray:
    if depth_map.dtype != np.uint8:
        normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)
    else:
        normalized = depth_map
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def _preview_stream(camera: Gemini335LCamera, use_depth: bool) -> None:
    window_name = "Gemini 335L Preview"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    except cv2.error as exc:
        raise RuntimeError(
            "OpenCV GUI backend is unavailable. Reinstall opencv-python (non-headless), or run without --preview."
        ) from exc

    try:
        while True:
            color_frame = camera.read()
            color_frame_bgr = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)

            if use_depth:
                depth_map = camera.read_depth()
                depth_preview = _make_depth_preview(depth_map)
                preview_frame = np.hstack((color_frame_bgr, depth_preview))
            else:
                preview_frame = color_frame_bgr

            cv2.imshow(window_name, preview_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cv2.destroyAllWindows()


def _print_actionable_error(exc: Exception) -> None:
    message = str(exc)
    print(f"Error: {message}")

    if isinstance(exc, ImportError) and "pyorbbecsdk" in message:
        print("Fix:")
        print("  python -m pip install pyorbbecsdk2")
        print("  # or use vendor package name if provided: pyorbbecsdk")
        return

    if "openUsbDevice failed" in message or "Permission denied" in message:
        print("Fix:")
        if UDEV_INSTALL_SCRIPT is None:
            print("  Cannot auto-locate Orbbec udev installer in current Python env.")
            print("  Reinstall pyorbbecsdk, then run the vendor udev install script.")
        else:
            print(f"  sudo /bin/sh {UDEV_INSTALL_SCRIPT}")
        print("Then unplug and re-plug the Gemini 335L, and run this command again.")
        return

    if "uvc_stream_open_ctrl failed" in message or "uvc_open failed" in message:
        print("Fix:")
        print(f"  {build_diag_command_hint()}")
        print(
            "If RGB-only succeeds but all RGBD probes fail on one camera, "
            "check firmware/cable/USB port or switch to another camera."
        )
        return

    if "disp_search_range_mode" in message or "disp_search_offset" in message:
        print("Fix:")
        print(
            "  Your current firmware/SDK rejected this disparity tuning property. "
            "Try upgrading firmware/pyorbbecsdk, or remove --disp-search-range-mode/--disp-search-offset."
        )
        return

    if "No Orbbec camera matches" in message:
        print("Fix:")
        print("  Run with `--list` first, then use the exact serial number or device name.")

    if "cvNamedWindow" in message or "The function is not implemented" in message:
        print("Fix:")
        print("  Your OpenCV has no GUI backend (often opencv-python-headless).")
        print("  In your active env run:")
        print("    python -m pip uninstall -y opencv-python-headless")
        print("    python -m pip install opencv-python==4.12.0.88")
        print("  Or avoid preview mode by removing --preview.")


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    try:
        if args.list:
            cameras = Gemini335LCamera.find_cameras()
            if not cameras:
                print("No Orbbec cameras detected.")
                return
            for index, camera in enumerate(cameras):
                print(f"[{index}] {camera}")
            return

        if args.list_profiles:
            cameras = Gemini335LCamera.list_stream_profiles(serial_number_or_name=args.serial_number_or_name)
            if not cameras:
                print("No Orbbec cameras detected.")
                return
            for index, camera in enumerate(cameras):
                print(f"[{index}] {camera}")
            return

        config = Gemini335LCameraConfig(
            serial_number_or_name=args.serial_number_or_name,
            fps=args.fps,
            width=args.width,
            height=args.height,
            color_stream_format=args.color_stream_format,
            depth_stream_format=args.depth_stream_format,
            depth_work_mode=args.depth_work_mode,
            disp_search_range_mode=args.disp_search_range_mode,
            disp_search_offset=args.disp_search_offset,
            profile_selection_strategy=args.profile_selection_strategy,
            use_depth=args.use_depth,
            align_depth_to_color=args.align_depth_to_color,
            align_mode=args.align_mode,
        )
        camera = Gemini335LCamera(config)
        camera.connect()

        if args.preview:
            try:
                _preview_stream(camera, use_depth=args.use_depth)
                return
            finally:
                camera.disconnect()

        color_frame = None
        depth_map = None
        for _ in range(args.read_count):
            color_frame = camera.read()
            if args.use_depth:
                depth_map = camera.read_depth()

        if color_frame is None:
            raise RuntimeError("Smoke test did not receive a color frame.")

        _save_color_image(color_frame, args.output_dir / "color.png")
        print(f"Saved color frame to {args.output_dir / 'color.png'}")
        print(f"Color frame shape: {tuple(color_frame.shape)}")

        if depth_map is not None:
            _save_depth_preview(depth_map, args.output_dir)
            print(f"Saved depth outputs to {args.output_dir}")
            print(f"Depth map shape: {tuple(depth_map.shape)} dtype={depth_map.dtype}")
        camera.disconnect()
    except Exception as exc:
        _print_actionable_error(exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
