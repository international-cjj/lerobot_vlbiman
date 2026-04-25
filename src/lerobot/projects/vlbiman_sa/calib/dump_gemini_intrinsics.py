#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def _bootstrap_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[5]
    extra_paths = [
        repo_root / "src",
        repo_root,
        repo_root / "lerobot_camera_gemini335l",
    ]
    for path in extra_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo_root


def _camera_matrix_from_intrinsics(intrinsic: Any) -> list[list[float]]:
    return [
        [float(intrinsic.fx), 0.0, float(intrinsic.cx)],
        [0.0, float(intrinsic.fy), float(intrinsic.cy)],
        [0.0, 0.0, 1.0],
    ]


def _dist_coeffs_from_distortion(distortion: Any) -> list[float]:
    return [
        float(distortion.k1),
        float(distortion.k2),
        float(distortion.p1),
        float(distortion.p2),
        float(distortion.k3),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read Gemini335L intrinsics from SDK and write JSON.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/lerobot/projects/vlbiman_sa/configs/handeye_auto.yaml"),
        help="Hand-eye auto config path used to resolve camera options.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/lerobot/projects/vlbiman_sa/configs/camera_intrinsics.json"),
        help="Output intrinsics JSON path.",
    )
    parser.add_argument(
        "--stream",
        choices=("rgb", "depth"),
        default="rgb",
        help="Which intrinsic/distortion block to export.",
    )
    parser.add_argument(
        "--serial-number-or-name",
        default=None,
        help="Override camera serial/name from config.",
    )
    return parser.parse_args()


def main() -> int:
    _bootstrap_paths()
    args = _parse_args()

    config_path = args.config
    if config_path.is_dir():
        config_path = config_path / "handeye_auto.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg_payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    camera_cfg = dict(cfg_payload.get("camera", {}))

    serial_number_or_name = args.serial_number_or_name or camera_cfg.get("serial_number_or_name")
    width = int(camera_cfg.get("width", 640))
    height = int(camera_cfg.get("height", 400))
    fps = int(camera_cfg.get("fps", 15))

    from lerobot.cameras.configs import ColorMode
    from lerobot_camera_gemini335l.config_gemini335l import Gemini335LCameraConfig
    from lerobot_camera_gemini335l.gemini335l import Gemini335LCamera

    color_mode = str(camera_cfg.get("color_mode", "rgb")).lower()
    gemini_cfg = Gemini335LCameraConfig(
        serial_number_or_name=serial_number_or_name,
        width=width,
        height=height,
        fps=fps,
        color_mode=ColorMode.RGB if color_mode == "rgb" else ColorMode.BGR,
        use_depth=True,
        align_depth_to_color=bool(camera_cfg.get("align_depth_to_color", False)),
        align_mode=str(camera_cfg.get("align_mode", "sw")),
        profile_selection_strategy=str(camera_cfg.get("profile_selection_strategy", "closest")),
        depth_work_mode=(
            str(camera_cfg["depth_work_mode"])
            if camera_cfg.get("depth_work_mode") is not None
            else None
        ),
        disp_search_range_mode=(
            int(camera_cfg["disp_search_range_mode"])
            if camera_cfg.get("disp_search_range_mode") is not None
            else None
        ),
        disp_search_offset=(
            int(camera_cfg["disp_search_offset"])
            if camera_cfg.get("disp_search_offset") is not None
            else None
        ),
    )
    camera = Gemini335LCamera(gemini_cfg)
    camera.connect()
    try:
        camera_param = camera.pipeline.get_camera_param()
        if args.stream == "rgb":
            intrinsic = camera_param.rgb_intrinsic
            distortion = camera_param.rgb_distortion
        else:
            intrinsic = camera_param.depth_intrinsic
            distortion = camera_param.depth_distortion

        payload = {
            "camera_matrix": _camera_matrix_from_intrinsics(intrinsic),
            "dist_coeffs": _dist_coeffs_from_distortion(distortion),
            "metadata": {
                "stream": args.stream,
                "serial_number_or_name": serial_number_or_name,
                "width": int(intrinsic.width),
                "height": int(intrinsic.height),
                "fx": float(intrinsic.fx),
                "fy": float(intrinsic.fy),
                "cx": float(intrinsic.cx),
                "cy": float(intrinsic.cy),
            },
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote intrinsics to {args.output}")
    finally:
        camera.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
