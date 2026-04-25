from __future__ import annotations

import json
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config_gemini335l import Gemini335LCameraConfig
from .gemini335l import Gemini335LCamera
from .runtime_env import get_python_executable

_SAFE_RGBD_FALLBACKS: list[tuple[int, int, int]] = [
    (640, 400, 30),
    (640, 400, 15),
    (640, 400, 5),
    (640, 480, 15),
    (640, 480, 5),
    (848, 480, 10),
    (480, 270, 15),
    (480, 270, 5),
]


def _normalize_identifier(camera: dict[str, Any]) -> set[str]:
    keys = {camera.get("serial_number"), camera.get("name"), camera.get("uid")}
    return {key for key in keys if isinstance(key, str) and key}


def _ordered_cameras(
    cameras: list[dict[str, Any]],
    preferred_identifier: str | None,
    *,
    probe_all_cameras: bool,
) -> list[dict[str, Any]]:
    if not cameras:
        return []
    if preferred_identifier is None:
        return cameras

    preferred: list[dict[str, Any]] = []
    others: list[dict[str, Any]] = []
    for camera in cameras:
        if preferred_identifier in _normalize_identifier(camera):
            preferred.append(camera)
        else:
            others.append(camera)

    if not preferred:
        return cameras if probe_all_cameras else []
    if not probe_all_cameras:
        return preferred
    return preferred + others


def _make_rgb_probe_config(
    *,
    serial_number_or_name: str,
    requested: dict[str, Any],
) -> Gemini335LCameraConfig:
    return Gemini335LCameraConfig(
        serial_number_or_name=serial_number_or_name,
        width=requested.get("width"),
        height=requested.get("height"),
        fps=requested.get("fps"),
        color_stream_format=requested.get("color_stream_format"),
        use_depth=False,
    )


def _build_rgbd_probe_candidates(requested: dict[str, Any]) -> list[dict[str, Any]]:
    width = requested.get("width")
    height = requested.get("height")
    fps = requested.get("fps")
    depth_work_mode = requested.get("depth_work_mode")
    disp_search_range_mode = requested.get("disp_search_range_mode")
    disp_search_offset = requested.get("disp_search_offset")
    align_depth_to_color = bool(requested.get("align_depth_to_color", False))
    align_mode = requested.get("align_mode") or "sw"
    profile_selection_strategy = requested.get("profile_selection_strategy") or "exact"
    color_stream_format = requested.get("color_stream_format")
    depth_stream_format = requested.get("depth_stream_format")

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def add_candidate(
        *,
        candidate_width: int | None,
        candidate_height: int | None,
        candidate_fps: int | None,
        candidate_depth_work_mode: str | None,
        candidate_disp_search_range_mode: int | None,
        candidate_disp_search_offset: int | None,
        candidate_align_depth_to_color: bool,
    ) -> None:
        values = (candidate_width, candidate_height, candidate_fps)
        if any(value is None for value in values) and any(value is not None for value in values):
            return

        key = (
            candidate_width,
            candidate_height,
            candidate_fps,
            color_stream_format,
            depth_stream_format,
            candidate_depth_work_mode,
            candidate_disp_search_range_mode,
            candidate_disp_search_offset,
            candidate_align_depth_to_color,
            align_mode,
            profile_selection_strategy,
        )
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            {
                "width": candidate_width,
                "height": candidate_height,
                "fps": candidate_fps,
                "color_stream_format": color_stream_format,
                "depth_stream_format": depth_stream_format,
                "depth_work_mode": candidate_depth_work_mode,
                "disp_search_range_mode": candidate_disp_search_range_mode,
                "disp_search_offset": candidate_disp_search_offset,
                "align_depth_to_color": candidate_align_depth_to_color,
                "align_mode": align_mode,
                "profile_selection_strategy": profile_selection_strategy,
            }
        )

    # 1) Try the exact user request first.
    add_candidate(
        candidate_width=width,
        candidate_height=height,
        candidate_fps=fps,
        candidate_depth_work_mode=depth_work_mode,
        candidate_disp_search_range_mode=disp_search_range_mode,
        candidate_disp_search_offset=disp_search_offset,
        candidate_align_depth_to_color=align_depth_to_color,
    )
    # 2) Same mode but without forcing depth work mode.
    if depth_work_mode is not None:
        add_candidate(
            candidate_width=width,
            candidate_height=height,
            candidate_fps=fps,
            candidate_depth_work_mode=None,
            candidate_disp_search_range_mode=disp_search_range_mode,
            candidate_disp_search_offset=disp_search_offset,
            candidate_align_depth_to_color=align_depth_to_color,
        )
    # 2.5) Same profile but without forcing disparity tuning properties.
    if disp_search_range_mode is not None or disp_search_offset is not None:
        add_candidate(
            candidate_width=width,
            candidate_height=height,
            candidate_fps=fps,
            candidate_depth_work_mode=depth_work_mode,
            candidate_disp_search_range_mode=None,
            candidate_disp_search_offset=None,
            candidate_align_depth_to_color=align_depth_to_color,
        )
    # 3) Disable alignment as a safer startup path.
    if align_depth_to_color:
        add_candidate(
            candidate_width=width,
            candidate_height=height,
            candidate_fps=fps,
            candidate_depth_work_mode=depth_work_mode,
            candidate_disp_search_range_mode=disp_search_range_mode,
            candidate_disp_search_offset=disp_search_offset,
            candidate_align_depth_to_color=False,
        )

    # 4) Probe conservative profiles commonly stable on USB2.x.
    for fallback_width, fallback_height, fallback_fps in _SAFE_RGBD_FALLBACKS:
        add_candidate(
            candidate_width=fallback_width,
            candidate_height=fallback_height,
            candidate_fps=fallback_fps,
            candidate_depth_work_mode=None,
            candidate_disp_search_range_mode=None,
            candidate_disp_search_offset=None,
            candidate_align_depth_to_color=False,
        )

    # 5) Retry selected safe profiles with software alignment enabled.
    for fallback_width, fallback_height, fallback_fps in ((640, 400, 15), (640, 400, 5), (640, 480, 15)):
        add_candidate(
            candidate_width=fallback_width,
            candidate_height=fallback_height,
            candidate_fps=fallback_fps,
            candidate_depth_work_mode=None,
            candidate_disp_search_range_mode=None,
            candidate_disp_search_offset=None,
            candidate_align_depth_to_color=True,
        )

    return candidates


def _profile_exists(
    profiles: list[dict[str, Any]],
    *,
    width: int | None,
    height: int | None,
    fps: int | None,
    format_name: str | None,
) -> bool:
    if width is None or height is None or fps is None:
        return True

    target_format = format_name.upper() if isinstance(format_name, str) else None
    for profile in profiles:
        if profile.get("width") != width:
            continue
        if profile.get("height") != height:
            continue
        if profile.get("fps") != fps:
            continue
        profile_format = profile.get("format")
        if target_format is not None and str(profile_format).upper() != target_format:
            continue
        return True
    return False


def _candidate_supported_by_camera(camera: dict[str, Any], candidate: dict[str, Any]) -> bool:
    color_profiles = camera.get("color_stream_profiles", [])
    depth_profiles = camera.get("depth_stream_profiles", [])
    width = candidate.get("width")
    height = candidate.get("height")
    fps = candidate.get("fps")

    if not _profile_exists(
        color_profiles,
        width=width,
        height=height,
        fps=fps,
        format_name=candidate.get("color_stream_format"),
    ):
        return False
    if not _profile_exists(
        depth_profiles,
        width=width,
        height=height,
        fps=fps,
        format_name=candidate.get("depth_stream_format"),
    ):
        return False
    return True


def _probe_camera(
    config: Gemini335LCameraConfig,
    *,
    stream: str,
    read_count: int,
    timeout_ms: int,
) -> tuple[bool, str | None, int]:
    start_ts = time.perf_counter()
    camera = Gemini335LCamera(config)
    try:
        camera.connect()
        for _ in range(max(1, read_count)):
            if stream == "rgb":
                camera.read(timeout_ms=timeout_ms)
            else:
                camera.read_rgbd(timeout_ms=timeout_ms)
        return True, None, int((time.perf_counter() - start_ts) * 1000)
    except Exception as exc:
        return False, str(exc), int((time.perf_counter() - start_ts) * 1000)
    finally:
        if camera.is_connected:
            try:
                camera.disconnect()
            except Exception:
                pass


def diagnose_depth_startup(
    *,
    preferred_identifier: str | None = None,
    requested: dict[str, Any] | None = None,
    read_count: int = 15,
    timeout_ms: int = 700,
    probe_all_cameras: bool = True,
    stop_on_first_success: bool = True,
) -> dict[str, Any]:
    requested = dict(requested or {})
    cameras = Gemini335LCamera.list_stream_profiles()
    ordered = _ordered_cameras(
        cameras,
        preferred_identifier=preferred_identifier,
        probe_all_cameras=probe_all_cameras,
    )

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "preferred_identifier": preferred_identifier,
        "requested": requested,
        "read_count": read_count,
        "timeout_ms": timeout_ms,
        "probe_all_cameras": probe_all_cameras,
        "attempts": [],
        "recommended": None,
        "available_cameras": [
            {
                "serial_number": camera.get("serial_number"),
                "name": camera.get("name"),
                "uid": camera.get("uid"),
                "connection_type": camera.get("connection_type"),
                "firmware_version": camera.get("firmware_version"),
            }
            for camera in cameras
        ],
    }

    if preferred_identifier is not None and not ordered:
        report["error"] = f"No camera matches preferred identifier '{preferred_identifier}'."
        return report

    for camera in ordered:
        serial_number = camera.get("serial_number")
        if not serial_number:
            continue

        identity = {
            "serial_number": serial_number,
            "name": camera.get("name"),
            "uid": camera.get("uid"),
            "connection_type": camera.get("connection_type"),
            "firmware_version": camera.get("firmware_version"),
        }

        rgb_config = _make_rgb_probe_config(serial_number_or_name=serial_number, requested=requested)
        rgb_success, rgb_error, rgb_elapsed_ms = _probe_camera(
            rgb_config,
            stream="rgb",
            read_count=read_count,
            timeout_ms=timeout_ms,
        )
        report["attempts"].append(
            {
                "camera": identity,
                "stream": "rgb",
                "config": {
                    "width": rgb_config.width,
                    "height": rgb_config.height,
                    "fps": rgb_config.fps,
                    "color_stream_format": rgb_config.color_stream_format,
                    "use_depth": False,
                },
                "success": rgb_success,
                "elapsed_ms": rgb_elapsed_ms,
                "error": rgb_error,
            }
        )

        for candidate in _build_rgbd_probe_candidates(requested):
            if not _candidate_supported_by_camera(camera, candidate):
                continue
            try:
                rgbd_config = Gemini335LCameraConfig(
                    serial_number_or_name=serial_number,
                    width=candidate["width"],
                    height=candidate["height"],
                    fps=candidate["fps"],
                    color_stream_format=candidate["color_stream_format"],
                    depth_stream_format=candidate["depth_stream_format"],
                    depth_work_mode=candidate["depth_work_mode"],
                    disp_search_range_mode=candidate["disp_search_range_mode"],
                    disp_search_offset=candidate["disp_search_offset"],
                    profile_selection_strategy=candidate["profile_selection_strategy"],
                    use_depth=True,
                    align_depth_to_color=candidate["align_depth_to_color"],
                    align_mode=candidate["align_mode"],
                )
            except Exception as config_exc:
                report["attempts"].append(
                    {
                        "camera": identity,
                        "stream": "rgbd",
                        "config": candidate,
                        "success": False,
                        "elapsed_ms": 0,
                        "error": f"Invalid config: {config_exc}",
                    }
                )
                continue

            success, error, elapsed_ms = _probe_camera(
                rgbd_config,
                stream="rgbd",
                read_count=read_count,
                timeout_ms=timeout_ms,
            )
            attempt = {
                "camera": identity,
                "stream": "rgbd",
                "config": candidate,
                "success": success,
                "elapsed_ms": elapsed_ms,
                "error": error,
            }
            report["attempts"].append(attempt)

            if success and report["recommended"] is None:
                report["recommended"] = attempt
                if stop_on_first_success:
                    return report

    return report


def build_recommended_depth_viewer_command(
    report: dict[str, Any],
    *,
    python_bin: str | None = None,
) -> list[str] | None:
    recommended = report.get("recommended")
    if not isinstance(recommended, dict):
        return None

    camera = recommended.get("camera", {})
    config = recommended.get("config", {})
    serial_number = camera.get("serial_number")
    if not isinstance(serial_number, str) or not serial_number:
        return None

    command = [
        python_bin or get_python_executable(),
        "-m",
        "lerobot_camera_gemini335l.depth_viewer",
        "--serial-number-or-name",
        serial_number,
    ]

    if config.get("width") is not None:
        command += ["--width", str(config["width"])]
    if config.get("height") is not None:
        command += ["--height", str(config["height"])]
    if config.get("fps") is not None:
        command += ["--fps", str(config["fps"])]
    if config.get("color_stream_format") is not None:
        command += ["--color-stream-format", str(config["color_stream_format"])]
    if config.get("depth_stream_format") is not None:
        command += ["--depth-stream-format", str(config["depth_stream_format"])]
    if config.get("depth_work_mode") is not None:
        command += ["--depth-work-mode", str(config["depth_work_mode"])]
    if config.get("disp_search_range_mode") is not None:
        command += ["--disp-search-range-mode", str(config["disp_search_range_mode"])]
    if config.get("disp_search_offset") is not None:
        command += ["--disp-search-offset", str(config["disp_search_offset"])]

    align_mode = config.get("align_mode")
    if align_mode is not None:
        command += ["--align-mode", str(align_mode)]
    if config.get("align_depth_to_color"):
        command.append("--align-depth-to-color")

    command.append("--show-color")
    return command


def format_diagnosis_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"Generated at (UTC): {report.get('generated_at_utc', 'unknown')}")
    preferred = report.get("preferred_identifier")
    if preferred is not None:
        lines.append(f"Preferred camera: {preferred}")
    lines.append(f"Probe all cameras: {bool(report.get('probe_all_cameras', True))}")
    lines.append(f"Read count per attempt: {report.get('read_count')}  timeout_ms: {report.get('timeout_ms')}")

    available_cameras = report.get("available_cameras", [])
    if available_cameras:
        lines.append("Available cameras:")
        for camera in available_cameras:
            lines.append(
                "  - {serial} | {name} | uid={uid} | {conn} | fw={fw}".format(
                    serial=camera.get("serial_number"),
                    name=camera.get("name"),
                    uid=camera.get("uid"),
                    conn=camera.get("connection_type"),
                    fw=camera.get("firmware_version"),
                )
            )

    error = report.get("error")
    if error:
        lines.append(f"Error: {error}")
        return "\n".join(lines)

    lines.append("Attempts:")
    for index, attempt in enumerate(report.get("attempts", []), start=1):
        camera = attempt.get("camera", {})
        config = attempt.get("config", {})
        status = "PASS" if attempt.get("success") else "FAIL"
        lines.append(
            f"  [{index:02d}] {status} {attempt.get('stream')} serial={camera.get('serial_number')} "
            f"({camera.get('connection_type')}, fw={camera.get('firmware_version')}) "
            f"config={json.dumps(config, ensure_ascii=False)} elapsed={attempt.get('elapsed_ms')}ms"
        )
        if attempt.get("error"):
            lines.append(f"       error={attempt['error']}")

    recommended = report.get("recommended")
    if recommended:
        camera = recommended.get("camera", {})
        lines.append(
            f"Recommended camera: {camera.get('serial_number')} "
            f"({camera.get('connection_type')}, fw={camera.get('firmware_version')})"
        )
        command = build_recommended_depth_viewer_command(report)
        if command:
            lines.append(f"Recommended depth_viewer command: {shlex.join(command)}")
    else:
        lines.append("Recommended camera: none (no RGBD startup probe succeeded).")

    return "\n".join(lines)


def save_report_json(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
