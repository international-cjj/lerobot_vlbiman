import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config_gemini335l import Gemini335LCameraConfig
from .gemini335l import Gemini335LCamera
from .runtime_env import build_diag_command_hint, find_udev_install_script

UDEV_INSTALL_SCRIPT = find_udev_install_script()

_COLORMAPS = {
    "bone": cv2.COLORMAP_BONE,
    "jet": cv2.COLORMAP_JET,
    "turbo": cv2.COLORMAP_TURBO,
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified real-time viewer for Gemini 335L. Supports rgb/depth/rgbd preview."
    )
    parser.add_argument("--list", action="store_true", help="List detected Orbbec cameras and exit.")
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List all available color/depth stream profiles and exit.",
    )
    parser.add_argument("--serial-number-or-name", default=None, help="Exact serial number, device name, or UID.")
    parser.add_argument("--stream", choices=("rgb", "depth", "rgbd"), default="rgbd")
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
    parser.add_argument(
        "--align-depth-to-color",
        action="store_true",
        help="Align depth frames to color stream. Valid for depth/rgbd modes.",
    )
    parser.add_argument(
        "--align-mode",
        choices=("hw", "sw"),
        default="hw",
        help="Depth-to-color alignment mode. Use 'sw' when hardware alignment is unsupported.",
    )
    parser.add_argument("--timeout-ms", type=int, default=500)
    parser.add_argument("--min-depth-mm", type=int, default=200)
    parser.add_argument("--max-depth-mm", type=int, default=1500)
    parser.add_argument("--colormap", choices=tuple(sorted(_COLORMAPS)), default="turbo")
    parser.add_argument("--frame-limit", type=int, default=0, help="0 means run until q/Esc.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gemini335l_stream_viewer"))
    return parser


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
        print("  If you only need algorithm calls without display, add: --headless")


def _normalize_depth_map(depth_map: np.ndarray, min_depth_mm: int, max_depth_mm: int) -> np.ndarray:
    if max_depth_mm <= min_depth_mm:
        raise ValueError("`max_depth_mm` must be greater than `min_depth_mm`.")

    normalized = np.zeros(depth_map.shape, dtype=np.uint8)
    valid_mask = depth_map > 0
    if not np.any(valid_mask):
        return normalized

    clipped = np.clip(depth_map[valid_mask].astype(np.float32), min_depth_mm, max_depth_mm)
    scaled = (clipped - min_depth_mm) / float(max_depth_mm - min_depth_mm)
    normalized[valid_mask] = np.clip((1.0 - scaled) * 255.0, 0, 255).astype(np.uint8)
    return normalized


def _render_depth(
    depth_map: np.ndarray,
    *,
    min_depth_mm: int,
    max_depth_mm: int,
    colormap: str,
) -> np.ndarray:
    normalized = _normalize_depth_map(depth_map, min_depth_mm=min_depth_mm, max_depth_mm=max_depth_mm)
    preview = cv2.applyColorMap(normalized, _COLORMAPS[colormap])
    preview[depth_map <= 0] = 0
    return preview


def _put_overlay_text(image: np.ndarray, text: str, line_index: int) -> None:
    origin = (16, 28 + line_index * 28)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


def _save_snapshot(
    *,
    output_dir: Path,
    stream: str,
    color_frame: np.ndarray | None,
    depth_map: np.ndarray | None,
    preview_frame: np.ndarray,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_paths: list[Path] = []

    preview_path = output_dir / f"{stream}_preview_{timestamp}.png"
    cv2.imwrite(str(preview_path), preview_frame)
    saved_paths.append(preview_path)

    if color_frame is not None:
        color_path = output_dir / f"rgb_{timestamp}.png"
        cv2.imwrite(str(color_path), cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR))
        saved_paths.append(color_path)

    if depth_map is not None:
        depth_npy_path = output_dir / f"depth_{timestamp}.npy"
        np.save(depth_npy_path, depth_map)
        saved_paths.append(depth_npy_path)

    return saved_paths


def _show_profiles(serial_number_or_name: str | None) -> None:
    profiles = Gemini335LCamera.list_stream_profiles(serial_number_or_name=serial_number_or_name)
    print(json.dumps(profiles, indent=2, ensure_ascii=False))


def _run_viewer(camera: Gemini335LCamera, args: argparse.Namespace) -> None:
    window_name = "Gemini 335L Stream Viewer"
    if not args.headless:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            raise RuntimeError(
                "OpenCV GUI backend is unavailable. Reinstall opencv-python (non-headless), or run with --headless."
            ) from exc

    active_profiles = camera.get_active_stream_profiles()
    color_profile_text = (
        f"{active_profiles['color']['width']}x{active_profiles['color']['height']}@{active_profiles['color']['fps']}"
        if active_profiles["color"] is not None
        else "disabled"
    )
    depth_profile_text = (
        f"{active_profiles['depth']['width']}x{active_profiles['depth']['height']}@{active_profiles['depth']['fps']}"
        if active_profiles["depth"] is not None
        else "disabled"
    )

    frame_count = 0
    fps_window_start = time.perf_counter()
    fps_window_count = 0
    display_fps = 0.0

    try:
        while True:
            if args.stream == "rgb":
                color_frame = camera.read(timeout_ms=args.timeout_ms)
                depth_map = None
                preview_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
            elif args.stream == "depth":
                color_frame = None
                depth_map = camera.read_depth(timeout_ms=args.timeout_ms)
                preview_frame = _render_depth(
                    depth_map,
                    min_depth_mm=args.min_depth_mm,
                    max_depth_mm=args.max_depth_mm,
                    colormap=args.colormap,
                )
            else:
                color_frame, depth_map = camera.read_rgbd(timeout_ms=args.timeout_ms)
                color_bgr = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
                depth_preview = _render_depth(
                    depth_map,
                    min_depth_mm=args.min_depth_mm,
                    max_depth_mm=args.max_depth_mm,
                    colormap=args.colormap,
                )
                if depth_preview.shape[:2] != color_bgr.shape[:2]:
                    depth_preview = cv2.resize(
                        depth_preview,
                        (color_bgr.shape[1], color_bgr.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                preview_frame = np.hstack((color_bgr, depth_preview))

            frame_count += 1
            fps_window_count += 1
            now = time.perf_counter()
            elapsed = now - fps_window_start
            if elapsed >= 0.5:
                display_fps = fps_window_count / elapsed
                fps_window_start = now
                fps_window_count = 0

            _put_overlay_text(preview_frame, f"mode: {args.stream}", 0)
            _put_overlay_text(preview_frame, f"color: {color_profile_text}", 1)
            _put_overlay_text(preview_frame, f"depth: {depth_profile_text}", 2)
            _put_overlay_text(preview_frame, f"display fps: {display_fps:.1f}", 3)

            if not args.headless:
                cv2.imshow(window_name, preview_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s"):
                    saved_paths = _save_snapshot(
                        output_dir=args.output_dir,
                        stream=args.stream,
                        color_frame=color_frame,
                        depth_map=depth_map,
                        preview_frame=preview_frame,
                    )
                    print("Saved snapshot:")
                    for path in saved_paths:
                        print(f"  {path}")

            if args.frame_limit > 0 and frame_count >= args.frame_limit:
                print(f"Processed {frame_count} frames successfully.")
                break
    finally:
        if not args.headless:
            cv2.destroyAllWindows()


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
            _show_profiles(serial_number_or_name=args.serial_number_or_name)
            return

        use_depth = args.stream in {"depth", "rgbd"}
        config = Gemini335LCameraConfig(
            serial_number_or_name=args.serial_number_or_name,
            fps=args.fps,
            width=args.width,
            height=args.height,
            use_depth=use_depth,
            align_depth_to_color=args.align_depth_to_color if use_depth else False,
            align_mode=args.align_mode,
            color_stream_format=args.color_stream_format,
            depth_stream_format=args.depth_stream_format if use_depth else None,
            depth_work_mode=args.depth_work_mode if use_depth else None,
            disp_search_range_mode=args.disp_search_range_mode if use_depth else None,
            disp_search_offset=args.disp_search_offset if use_depth else None,
            profile_selection_strategy=args.profile_selection_strategy,
        )
        camera = Gemini335LCamera(config)
        camera.connect()
        try:
            _run_viewer(camera, args)
        finally:
            camera.disconnect()
    except Exception as exc:
        _print_actionable_error(exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
