#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

def _bootstrap_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[5]
    extra_paths = [
        repo_root / "src",
        repo_root,
        repo_root / "lerobot_robot_cjjarm",
        repo_root / "lerobot_camera_gemini335l",
        repo_root / "lerobot_teleoperator_zhonglin",
    ]
    for path in extra_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo_root


_bootstrap_paths()

from lerobot.projects.vlbiman_sa.demo.rgbd_recorder import RGBDRecorder
from lerobot.projects.vlbiman_sa.demo.schema import RecorderConfig


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "one_shot_record.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record one-shot RGBD demo synchronized with robot state.")
    parser.add_argument("--config", type=Path, default=_default_config_path(), help="Path to one_shot_record.yaml.")
    parser.add_argument("--fps", type=int, default=None, help="Override recording.fps.")
    parser.add_argument("--control-rate-hz", type=float, default=None, help="Override recording.control_rate_hz.")
    parser.add_argument("--duration-s", type=float, default=None, help="Override recording.duration_s.")
    parser.add_argument("--max-frames", type=int, default=None, help="Override recording.max_frames.")
    parser.add_argument("--output-root", type=Path, default=None, help="Override recording.output_root.")
    parser.add_argument("--run-name", type=str, default=None, help="Override recording.run_name.")
    parser.add_argument("--camera-serial-number", type=str, default=None, help="Override camera serial number.")
    parser.add_argument("--robot-serial-port", type=str, default=None, help="Override robot serial port.")
    parser.add_argument("--teleop-port", type=str, default=None, help="Override Zhonglin teleop serial port.")
    parser.add_argument(
        "--no-teleop-calibrate",
        action="store_true",
        help="Skip Zhonglin zero-position calibration on connect.",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping, got {type(payload).__name__}.")
    return payload


def _build_recorder_config(payload: dict[str, Any], args: argparse.Namespace) -> RecorderConfig:
    recording_payload = dict(payload.get("recording", {}))
    if args.fps is not None:
        recording_payload["fps"] = args.fps
    if args.control_rate_hz is not None:
        recording_payload["control_rate_hz"] = args.control_rate_hz
    if args.duration_s is not None:
        recording_payload["duration_s"] = args.duration_s
    if args.max_frames is not None:
        recording_payload["max_frames"] = args.max_frames
    if args.output_root is not None:
        recording_payload["output_root"] = args.output_root
    if args.run_name is not None:
        recording_payload["run_name"] = args.run_name
    if "output_root" in recording_payload:
        recording_payload["output_root"] = Path(recording_payload["output_root"])
    return RecorderConfig(**recording_payload)


def _build_camera(camera_cfg: dict[str, Any]) -> Any:
    from lerobot.cameras.configs import ColorMode
    from lerobot_camera_gemini335l.config_gemini335l import Gemini335LCameraConfig
    from lerobot_camera_gemini335l.gemini335l import Gemini335LCamera

    backend = str(camera_cfg.get("backend", "gemini335l")).lower()
    if backend != "gemini335l":
        raise ValueError(f"Unsupported camera backend for T2 recording: {backend}")

    color_mode = str(camera_cfg.get("color_mode", "rgb")).lower()
    gemini_cfg = Gemini335LCameraConfig(
        serial_number_or_name=camera_cfg.get("serial_number_or_name"),
        width=int(camera_cfg.get("width", 1280)),
        height=int(camera_cfg.get("height", 800)),
        fps=int(camera_cfg.get("fps", 30)),
        color_mode=ColorMode.RGB if color_mode == "rgb" else ColorMode.BGR,
        use_depth=True,
        align_depth_to_color=bool(camera_cfg.get("align_depth_to_color", True)),
        align_mode=str(camera_cfg.get("align_mode", "sw")),
        profile_selection_strategy=str(camera_cfg.get("profile_selection_strategy", "closest")),
        depth_work_mode=(
            str(camera_cfg["depth_work_mode"]) if camera_cfg.get("depth_work_mode") is not None else None
        ),
        disp_search_range_mode=(
            int(camera_cfg["disp_search_range_mode"])
            if camera_cfg.get("disp_search_range_mode") is not None
            else None
        ),
        disp_search_offset=(
            int(camera_cfg["disp_search_offset"]) if camera_cfg.get("disp_search_offset") is not None else None
        ),
    )
    return Gemini335LCamera(gemini_cfg)


def _build_teleop(teleop_cfg: dict[str, Any]) -> Any | None:
    if not teleop_cfg:
        return None

    from lerobot_teleoperator_zhonglin.config_zhonglin import ZhonglinTeleopConfig
    from lerobot_teleoperator_zhonglin.zhonglin_teleop import ZhonglinTeleop

    teleop_type = str(teleop_cfg.get("type", "zhonglin_teleop")).lower()
    if teleop_type != "zhonglin_teleop":
        raise ValueError(f"Unsupported teleop type for T2 recording: {teleop_type}")

    cfg = ZhonglinTeleopConfig()
    if teleop_cfg.get("port"):
        cfg.port = str(teleop_cfg["port"])
    if teleop_cfg.get("baudrate") is not None:
        cfg.baudrate = int(teleop_cfg["baudrate"])
    if teleop_cfg.get("timeout") is not None:
        cfg.timeout = float(teleop_cfg["timeout"])
    if teleop_cfg.get("command_delay_s") is not None:
        cfg.command_delay_s = float(teleop_cfg["command_delay_s"])
    if teleop_cfg.get("output_signs"):
        cfg.output_signs = {str(k): float(v) for k, v in dict(teleop_cfg["output_signs"]).items()}
    teleop = ZhonglinTeleop(cfg)
    teleop._vlbiman_calibrate_on_connect = bool(teleop_cfg.get("calibrate_on_connect", True))
    return teleop


def _build_robot(robot_cfg: dict[str, Any]) -> Any:
    from lerobot_robot_cjjarm.config_cjjarm import CjjArmConfig
    from lerobot_robot_cjjarm.cjjarm_robot import CjjArm

    robot_type = str(robot_cfg.get("type", "cjjarm")).lower()
    if robot_type != "cjjarm":
        raise ValueError(f"Unsupported robot type for T2 recording: {robot_type}")

    config = CjjArmConfig()
    if robot_cfg.get("serial_port"):
        config.serial_port = str(robot_cfg["serial_port"])
    if bool(robot_cfg.get("disable_robot_cameras", True)):
        config.cameras = {}
    return CjjArm(config)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)

    payload = _load_yaml(args.config)
    camera_cfg = dict(payload.get("camera", {}))
    robot_cfg = dict(payload.get("robot", {}))
    teleop_cfg = dict(payload.get("teleop", {}))
    if args.camera_serial_number is not None:
        camera_cfg["serial_number_or_name"] = args.camera_serial_number
    if args.robot_serial_port is not None:
        robot_cfg["serial_port"] = args.robot_serial_port
    if args.teleop_port is not None:
        teleop_cfg["port"] = args.teleop_port
    if args.no_teleop_calibrate:
        teleop_cfg["calibrate_on_connect"] = False

    recorder_cfg = _build_recorder_config(payload, args)
    camera = _build_camera(camera_cfg)
    robot = _build_robot(robot_cfg)
    teleop = _build_teleop(teleop_cfg)

    if teleop is not None:
        original_connect = teleop.connect

        def _connect_with_cfg() -> None:
            original_connect(calibrate=getattr(teleop, "_vlbiman_calibrate_on_connect", True))

        teleop.connect = _connect_with_cfg  # type: ignore[method-assign]

    recorder = RGBDRecorder(
        recorder_cfg,
        robot=robot,
        camera=camera,
        teleop=teleop,
        manifest_extra={"camera": camera_cfg, "robot": robot_cfg, "teleop": teleop_cfg},
    )
    summary = recorder.record()

    logging.info("Recording status: %s", summary.status)
    logging.info("Recorded frames: %s/%s", summary.recorded_frames, summary.target_frame_slots)
    logging.info("Dropped frames: %s, failed frames: %s", summary.dropped_frames, summary.failed_frames)
    logging.info("Achieved FPS: %.2f", summary.achieved_fps)
    logging.info("Average skew: %.2f ms, max skew: %.2f ms", summary.average_time_skew_ms, summary.max_time_skew_ms)
    logging.info("Session dir: %s", summary.session_dir)
    return 0 if summary.recorded_frames > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
