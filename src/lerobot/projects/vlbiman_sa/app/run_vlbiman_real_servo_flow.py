#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def _bootstrap_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[5]
    for path in (repo_root / "src", repo_root / "lerobot_robot_cjjarm", repo_root):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo_root


REPO_ROOT = _bootstrap_paths()

from lerobot.projects.vlbiman_sa.core.contracts import TaskGraspConfig  # noqa: E402
from lerobot.projects.vlbiman_sa.demo.io import load_frame_records  # noqa: E402
from lerobot.projects.vlbiman_sa.inv_servo.config import (  # noqa: E402
    InvRGBServoConfig,
    default_inv_rgb_servo_config_path,
    load_inv_rgb_servo_config,
)
from lerobot.projects.vlbiman_sa.inv_servo.detector import GroundingDINODetector  # noqa: E402
from lerobot.projects.vlbiman_sa.inv_servo.metrics import mask_iou, mask_state_from_mask  # noqa: E402
from lerobot.projects.vlbiman_sa.inv_servo.rgb_servo_controller import RGBServoController  # noqa: E402
from lerobot.projects.vlbiman_sa.inv_servo.robot_backend import RobotBackendConfig, RobotExecutionBackend  # noqa: E402
from lerobot.projects.vlbiman_sa.inv_servo.sam2_live_tracker import SAM2LiveTracker  # noqa: E402
from lerobot.projects.vlbiman_sa.inv_servo.servo_safety import ServoSafetyConfig, ServoSafetyFilter  # noqa: E402
from lerobot.projects.vlbiman_sa.inv_servo.target_state import MaskState, ServoCommand, ServoTarget  # noqa: E402
from lerobot.projects.vlbiman_sa.inv_servo.trace_logger import TraceLogger, TraceLoggerConfig  # noqa: E402
from lerobot.projects.vlbiman_sa.skills import InvarianceClassifierConfig, SegmenterConfig, SkillBank, SkillSegment, build_skill_bank  # noqa: E402


class FlowFailure(RuntimeError):
    def __init__(self, reason: str, payload: dict[str, Any] | None = None):
        super().__init__(reason)
        self.reason = reason
        self.payload = payload or {}


@dataclass(slots=True)
class TargetArtifacts:
    target: ServoTarget
    target_mask: np.ndarray
    target_frame: np.ndarray | None
    target_mask_path: Path
    target_frame_path: Path | None
    target_state_path: Path
    detection_path: Path | None


@dataclass(slots=True)
class ConnectedRobot:
    robot: Any
    backend: RobotExecutionBackend


def _default_task_config_path() -> Path:
    return REPO_ROOT / "src" / "lerobot" / "projects" / "vlbiman_sa" / "configs" / "task_grasp.yaml"


def _default_output_root() -> Path:
    return REPO_ROOT / "outputs" / "vlbiman_sa" / "real_servo_flow"


def _default_capture_config_path() -> Path:
    return REPO_ROOT / "src" / "lerobot" / "projects" / "vlbiman_sa" / "configs" / "one_shot_record.yaml"


def _default_vision_config_path() -> Path:
    return REPO_ROOT / "src" / "lerobot" / "projects" / "vlbiman_sa" / "configs" / "vision_analysis.yaml"


def _default_live_output_root() -> Path:
    return REPO_ROOT / "outputs" / "vlbiman_sa" / "live_orange_pose"


def _timestamp_run_id() -> str:
    return f"real_servo_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VLBiMan on a real CJJArm with one online RGB visual-servo segment.")
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument(
        "--live-result-path",
        type=Path,
        default=None,
        help="Optional precomputed live scene target localization result. Use --capture-live-result to generate one.",
    )
    parser.add_argument(
        "--capture-live-result",
        action="store_true",
        help="Capture the current scene before VLBiMan planning and generate the live target localization result.",
    )
    parser.add_argument("--task-config", type=Path, default=_default_task_config_path())
    parser.add_argument("--servo-config", type=Path, default=default_inv_rgb_servo_config_path())
    parser.add_argument("--capture-config", type=Path, default=_default_capture_config_path())
    parser.add_argument("--vision-config", type=Path, default=_default_vision_config_path())
    parser.add_argument("--output-root", type=Path, default=_default_output_root())
    parser.add_argument("--live-output-root", type=Path, default=_default_live_output_root())
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--target-phrase", type=str, default="redcan")
    parser.add_argument("--aux-target-phrase", action="append", default=[])
    parser.add_argument("--servo-segment", type=str, default=None, help="T3 segment id reserved for online visual servo.")
    parser.add_argument(
        "--servo-frame-range",
        type=str,
        default=None,
        help="1-based inclusive frame range used to locate the servo segment, for example 51-55.",
    )
    parser.add_argument(
        "--servo-target-frame",
        type=int,
        default=None,
        help="Recorded frame_index used as the servo target. Defaults to metrics.target_frame or segment end_frame.",
    )
    parser.add_argument("--target-mask-path", type=Path, default=None, help="Optional precomputed target mask to bypass DINO/SAM2.")
    parser.add_argument("--camera", type=str, default="wrist")
    parser.add_argument("--robot-serial-port", type=str, default=None)
    parser.add_argument("--camera-serial-number", type=str, default=None)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--camera-timeout-ms", type=int, default=2500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-t5", action="store_true", help="Accepted for compatibility; T5 is generated per run.")
    parser.add_argument("--force-t6", action="store_true", help="Regenerate T6 outputs even if run artifacts exist.")
    parser.add_argument("--step-duration-s", type=float, default=0.05)
    parser.add_argument(
        "--servo-step-duration-s",
        type=float,
        default=None,
        help="Live visual-servo command execution duration. Defaults to --step-duration-s.",
    )
    parser.add_argument("--bridge-max-joint-step-rad", type=float, default=0.08)
    parser.add_argument("--max-servo-steps", type=int, default=None)
    parser.add_argument("--stable-servo-frames", type=int, default=None)
    parser.add_argument("--servo-k-u", type=float, default=None, help="Override visual-servo proportional gain from horizontal image error to x motion.")
    parser.add_argument("--servo-k-v", type=float, default=None, help="Override visual-servo proportional gain from vertical image error to y motion.")
    parser.add_argument("--servo-k-a", type=float, default=None, help="Override visual-servo proportional gain from area error to z approach motion.")
    parser.add_argument("--servo-axis-sign-x", type=float, default=None, help="Override visual-servo command x sign/scale.")
    parser.add_argument("--servo-axis-sign-y", type=float, default=None, help="Override visual-servo command y sign/scale.")
    parser.add_argument("--servo-axis-sign-z", type=float, default=None, help="Override visual-servo command z sign/scale.")
    parser.add_argument("--servo-max-step-xy-m", type=float, default=None, help="Override max visual-servo x/y translation per step in meters.")
    parser.add_argument("--servo-max-step-z-m", type=float, default=None, help="Override max visual-servo z translation per step in meters.")
    parser.add_argument(
        "--servo-arm-velocity",
        type=float,
        default=2.0,
        help="CjjArm position/velocity motor speed for live servo commands. Use a lower value for compliant servo motion.",
    )
    parser.add_argument(
        "--servo-arm-smooth-factor",
        type=float,
        default=0.35,
        help="CjjArm per-action joint target smoothing for live servo commands. Use 1.0 for direct targets.",
    )
    parser.add_argument(
        "--servo-interpolation-hz",
        type=float,
        default=20.0,
        help="Substep frequency for smooth execution of each visual-servo command. Set <=0 to disable interpolation.",
    )
    parser.add_argument(
        "--servo-interpolation-profile",
        choices=("linear", "smootherstep"),
        default="linear",
        help="Substep spacing profile. linear gives constant small increments; smootherstep accelerates/decelerates within one command.",
    )
    parser.add_argument(
        "--servo-command-filter-alpha",
        type=float,
        default=0.35,
        help="EMA factor for live servo command smoothing. 1.0 disables temporal filtering.",
    )
    parser.add_argument("--servo-command-deadband-xy-m", type=float, default=0.00008)
    parser.add_argument("--servo-command-deadband-z-m", type=float, default=0.00015)
    parser.add_argument("--servo-command-max-change-xy-m", type=float, default=0.00035)
    parser.add_argument("--servo-command-max-change-z-m", type=float, default=0.0015)
    parser.add_argument("--save-overlay-every", type=int, default=5)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _load_task_config(path: Path) -> TaskGraspConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Task config must be a mapping: {path}")
    for key in (
        "data_root",
        "transforms_path",
        "handeye_result_path",
        "recording_session_dir",
        "skill_output_dir",
        "skill_bank_path",
        "vision_output_dir",
        "pose_output_dir",
        "trajectory_output_dir",
        "live_result_path",
        "intrinsics_path",
    ):
        if key in payload and payload[key] is not None:
            payload[key] = Path(payload[key])
    return TaskGraspConfig(**payload)


def _normalize_semantic(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _is_visual_servo_segment(segment: SkillSegment) -> bool:
    return (
        _normalize_semantic(segment.metrics.get("semantic_state")) == "visual_servo"
        or _normalize_semantic(segment.label) == "visual_servo"
    )


def _parse_frame_range_1based(raw: str | None) -> tuple[int, int] | None:
    if not raw:
        return None
    try:
        start_text, end_text = raw.split("-", 1)
        start_frame = int(start_text) - 1
        end_frame = int(end_text) - 1
    except Exception as exc:
        raise ValueError(f"Invalid --servo-frame-range {raw!r}; expected start-end.") from exc
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError(f"Invalid --servo-frame-range {raw!r}; frames are 1-based and inclusive.")
    return start_frame, end_frame


def _find_servo_segment(
    skill_bank: SkillBank,
    *,
    servo_segment_id: str | None,
    servo_frame_range: tuple[int, int] | None,
) -> SkillSegment:
    if servo_segment_id and servo_frame_range is not None:
        raise ValueError("Specify either --servo-segment or --servo-frame-range, not both.")
    segments = list(skill_bank.segments)
    if servo_segment_id:
        for segment in segments:
            if str(segment.segment_id) == servo_segment_id:
                return segment
        raise ValueError(f"Servo segment {servo_segment_id!r} not found.")
    if servo_frame_range is not None:
        start_frame, end_frame = servo_frame_range
        matches = [
            segment
            for segment in segments
            if int(segment.start_frame) <= start_frame and end_frame <= int(segment.end_frame)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one T3 segment containing frames {start_frame}-{end_frame}, found {len(matches)}."
            )
        return matches[0]
    visual_segments = [segment for segment in segments if _is_visual_servo_segment(segment)]
    if len(visual_segments) != 1:
        raise ValueError(f"Expected exactly one visual_servo segment, found {len(visual_segments)}.")
    return visual_segments[0]


def _metric_int(segment: SkillSegment, key: str) -> int | None:
    value = segment.metrics.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _metric_str(segment: SkillSegment, key: str) -> str | None:
    value = segment.metrics.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_servo_target_frame(segment: SkillSegment, explicit_frame: int | None) -> int:
    if explicit_frame is not None:
        return int(explicit_frame)
    metric_frame = _metric_int(segment, "target_frame")
    if metric_frame is not None:
        return metric_frame
    return int(segment.end_frame)


def _resolve_servo_target_phrase(segment: SkillSegment, cli_phrase: str) -> str:
    return _metric_str(segment, "target_phrase") or str(cli_phrase)


def _camera_asset_candidates(camera: str) -> list[str]:
    candidates = [camera]
    for value in list(candidates):
        if value.startswith("observation.images."):
            candidates.append(value.removeprefix("observation.images."))
        else:
            candidates.append(f"observation.images.{value}")
        if value.endswith("_camera"):
            candidates.append(value.removesuffix("_camera"))
        else:
            candidates.append(f"{value}_camera")
    candidates.extend(["wrist", "wrist_rgb", "dabaidcw_rgb", "hand", "hand_camera", "wrist_camera"])
    return list(dict.fromkeys(item for item in candidates if item))


def _load_recorded_rgb_frame(session_dir: Path, records_by_index: dict[int, Any], frame_index: int, camera: str) -> np.ndarray:
    record = records_by_index.get(int(frame_index))
    if record is None:
        raise FlowFailure("recorded_target_frame_not_found", {"frame_index": frame_index})
    assets = dict(getattr(record, "camera_assets", {}))
    for candidate in _camera_asset_candidates(camera):
        if candidate not in assets:
            continue
        color_path = assets[candidate].get("color_path")
        if not color_path:
            continue
        image_bgr = cv2.imread(str(session_dir / color_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FlowFailure("recorded_camera_frame_unreadable", {"path": str(session_dir / color_path)})
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if getattr(record, "color_path", ""):
        image_bgr = cv2.imread(str(session_dir / record.color_path), cv2.IMREAD_COLOR)
        if image_bgr is not None:
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    raise FlowFailure(
        "recorded_camera_frame_not_found",
        {"frame_index": frame_index, "camera": camera, "available_camera_assets": sorted(assets.keys())},
    )


def _write_rgb(path: Path, frame_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_bgr = cv2.cvtColor(np.asarray(frame_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), frame_bgr):
        raise RuntimeError(f"Failed to write image: {path}")


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if not cv2.imwrite(str(path), binary):
        raise RuntimeError(f"Failed to write mask: {path}")


def _save_overlay(path: Path, frame_rgb: np.ndarray, mask: np.ndarray, label: str) -> None:
    frame_bgr = cv2.cvtColor(np.asarray(frame_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame_bgr, contours, -1, (0, 255, 255), 2)
    cv2.putText(frame_bgr, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame_bgr):
        raise RuntimeError(f"Failed to write overlay: {path}")


def _mask_state_from_payload(payload: dict[str, Any]) -> MaskState:
    state = dict(payload.get("mask_state") or {})
    if not state:
        state = {
            "frame_index": payload.get("frame_index", 0),
            "image_size_hw": payload.get("image_size_hw"),
            "mask_area_px": payload.get("mask_area_px", 0),
            "centroid_uv": payload.get("center_uv"),
            "bbox_xyxy": payload.get("bbox_xyxy"),
            "source": "sam2",
            "debug": {"payload_without_mask_state": True},
        }
    return MaskState(
        frame_index=int(state["frame_index"]),
        image_size_hw=tuple(state["image_size_hw"]),
        mask_area_px=int(state["mask_area_px"]),
        centroid_uv=None if state.get("centroid_uv") is None else tuple(state["centroid_uv"]),
        bbox_xyxy=None if state.get("bbox_xyxy") is None else tuple(state["bbox_xyxy"]),
        source=str(state.get("source", "sam2")),
        mask_path=Path(state["mask_path"]) if state.get("mask_path") else None,
        debug=dict(state.get("debug", {})),
    )


def _servo_command_from_dict(payload: dict[str, Any]) -> ServoCommand:
    return ServoCommand(
        delta_xyz_m=tuple(payload["delta_xyz_m"]),
        delta_rpy_rad=tuple(payload["delta_rpy_rad"]),
        gripper_position=payload.get("gripper_position"),
        stop=bool(payload.get("stop", False)),
        reason=payload.get("reason"),
        debug=dict(payload.get("debug", {})),
    )


def _elapsed_ms(start_s: float) -> float:
    return float((time.perf_counter() - start_s) * 1000.0)


def _smootherstep(alpha: float) -> float:
    value = max(0.0, min(1.0, float(alpha)))
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


def _clamp_delta(value: float, limit: float) -> float:
    bound = abs(float(limit))
    return max(-bound, min(bound, float(value)))


def _servo_command_is_zero(command: ServoCommand) -> bool:
    values = [*command.delta_xyz_m, *command.delta_rpy_rad]
    return all(abs(float(value)) <= 1e-12 for value in values)


def _scale_servo_command(command: ServoCommand, scale: float, *, final_substep: bool, substep_index: int, substep_count: int) -> ServoCommand:
    return ServoCommand(
        delta_xyz_m=tuple(float(value) * float(scale) for value in command.delta_xyz_m),
        delta_rpy_rad=tuple(float(value) * float(scale) for value in command.delta_rpy_rad),
        gripper_position=command.gripper_position if final_substep else None,
        stop=bool(command.stop and final_substep),
        reason=command.reason,
        debug={
            **dict(command.debug),
            "interpolation": {
                "substep_index": int(substep_index),
                "substep_count": int(substep_count),
                "scale": float(scale),
            },
        },
    )


def _replace_servo_command_delta(
    command: ServoCommand,
    *,
    delta_xyz_m: tuple[float, float, float],
    delta_rpy_rad: tuple[float, float, float] | None = None,
    reason: str | None = None,
    debug_update: dict[str, Any] | None = None,
) -> ServoCommand:
    debug = dict(command.debug)
    if debug_update:
        debug.update(debug_update)
    return ServoCommand(
        delta_xyz_m=tuple(float(value) for value in delta_xyz_m),
        delta_rpy_rad=tuple(float(value) for value in (command.delta_rpy_rad if delta_rpy_rad is None else delta_rpy_rad)),
        gripper_position=command.gripper_position,
        stop=command.stop,
        reason=command.reason if reason is None else reason,
        debug=debug,
    )


def _apply_component_deadband(values: np.ndarray, *, xy_m: float, z_m: float) -> np.ndarray:
    thresholds = np.asarray([max(float(xy_m), 0.0), max(float(xy_m), 0.0), max(float(z_m), 0.0)], dtype=float)
    filtered = values.astype(float, copy=True)
    filtered[np.abs(filtered) < thresholds] = 0.0
    return filtered


def _limit_component_change(values: np.ndarray, previous: np.ndarray, *, xy_m: float, z_m: float) -> np.ndarray:
    limits = np.asarray([max(float(xy_m), 0.0), max(float(xy_m), 0.0), max(float(z_m), 0.0)], dtype=float)
    if np.all(limits <= 0.0):
        return values
    delta = values - previous
    limited_delta = np.asarray(
        [
            _clamp_delta(float(delta[axis]), float(limits[axis])) if limits[axis] > 0.0 else float(delta[axis])
            for axis in range(3)
        ],
        dtype=float,
    )
    return previous + limited_delta


def _shape_servo_command(
    command: ServoCommand,
    *,
    previous_command: ServoCommand | None,
    filter_alpha: float,
    deadband_xy_m: float,
    deadband_z_m: float,
    max_change_xy_m: float,
    max_change_z_m: float,
) -> tuple[ServoCommand, dict[str, Any]]:
    raw_xyz = np.asarray(command.delta_xyz_m, dtype=float)
    alpha = max(0.0, min(1.0, float(filter_alpha)))
    previous_xyz = None if previous_command is None else np.asarray(previous_command.delta_xyz_m, dtype=float)

    shaped_xyz = raw_xyz.copy()
    if previous_xyz is not None and alpha < 1.0 and not _servo_command_is_zero(command):
        shaped_xyz = previous_xyz + alpha * (shaped_xyz - previous_xyz)
    after_filter_xyz = shaped_xyz.copy()

    if previous_xyz is not None and not _servo_command_is_zero(command):
        shaped_xyz = _limit_component_change(
            shaped_xyz,
            previous_xyz,
            xy_m=max_change_xy_m,
            z_m=max_change_z_m,
        )
    after_rate_limit_xyz = shaped_xyz.copy()

    shaped_xyz = _apply_component_deadband(shaped_xyz, xy_m=deadband_xy_m, z_m=deadband_z_m)
    if _servo_command_is_zero(command):
        shaped_xyz = np.zeros(3, dtype=float)

    shaping = {
        "filter_alpha": alpha,
        "deadband_xy_m": float(deadband_xy_m),
        "deadband_z_m": float(deadband_z_m),
        "max_change_xy_m": float(max_change_xy_m),
        "max_change_z_m": float(max_change_z_m),
        "raw_delta_xyz_m": [float(value) for value in raw_xyz],
        "after_filter_delta_xyz_m": [float(value) for value in after_filter_xyz],
        "after_rate_limit_delta_xyz_m": [float(value) for value in after_rate_limit_xyz],
        "shaped_delta_xyz_m": [float(value) for value in shaped_xyz],
        "previous_delta_xyz_m": None if previous_xyz is None else [float(value) for value in previous_xyz],
    }
    shaped_command = _replace_servo_command_delta(
        command,
        delta_xyz_m=tuple(float(value) for value in shaped_xyz),
        reason="command_deadband" if np.allclose(shaped_xyz, 0.0) and not _servo_command_is_zero(command) else None,
        debug_update={"command_shaping": shaping},
    )
    return shaped_command, shaping


def _interpolated_servo_commands(
    command: ServoCommand,
    *,
    duration_s: float,
    interpolation_hz: float,
    profile: str = "linear",
) -> list[ServoCommand]:
    if interpolation_hz <= 0.0 or duration_s <= 0.0 or _servo_command_is_zero(command):
        return [command]
    substep_count = max(1, int(np.ceil(float(duration_s) * float(interpolation_hz))))
    if substep_count <= 1:
        return [command]
    commands: list[ServoCommand] = []
    previous_fraction = 0.0
    for substep_index in range(substep_count):
        linear_fraction = float(substep_index + 1) / float(substep_count)
        fraction = _smootherstep(linear_fraction) if str(profile) == "smootherstep" else linear_fraction
        scale = fraction - previous_fraction
        previous_fraction = fraction
        commands.append(
            _scale_servo_command(
                command,
                scale,
                final_substep=substep_index == substep_count - 1,
                substep_index=substep_index,
                substep_count=substep_count,
            )
        )
    return commands


def _execute_servo_command_smooth(
    *,
    backend: RobotExecutionBackend,
    command: ServoCommand,
    step_duration_s: float,
    interpolation_hz: float,
    interpolation_profile: str,
) -> tuple[Any, dict[str, Any]]:
    duration_s = max(float(step_duration_s), 1e-3)
    subcommands = _interpolated_servo_commands(
        command,
        duration_s=duration_s,
        interpolation_hz=float(interpolation_hz),
        profile=interpolation_profile,
    )
    substep_count = max(len(subcommands), 1)
    substep_period_s = duration_s / float(substep_count)
    execution_start = time.perf_counter()
    send_action_ms_total = 0.0
    sleep_ms_total = 0.0
    substep_payloads: list[dict[str, Any]] = []
    last_result: Any = None

    for substep_index, subcommand in enumerate(subcommands):
        send_start = time.perf_counter()
        result = backend.execute_servo_command(subcommand)
        send_action_ms = _elapsed_ms(send_start)
        send_action_ms_total += send_action_ms
        last_result = result
        result_payload = result.to_dict()
        action = dict((result_payload.get("state") or {}).get("action") or {})
        substep_payloads.append(
            {
                "substep_index": substep_index,
                "ok": bool(result_payload.get("ok")),
                "failure_reason": result_payload.get("failure_reason"),
                "send_action_ms": send_action_ms,
                "action": action,
            }
        )
        if not result.ok:
            break

        target_time_s = execution_start + (float(substep_index + 1) * substep_period_s)
        sleep_s = max(0.0, target_time_s - time.perf_counter())
        if sleep_s > 0.0:
            time.sleep(sleep_s)
            sleep_ms_total += sleep_s * 1000.0

    timing = {
        "interpolation_enabled": bool(float(interpolation_hz) > 0.0 and substep_count > 1),
        "interpolation_hz": float(interpolation_hz),
        "interpolation_profile": str(interpolation_profile),
        "substep_count": substep_count,
        "substep_period_ms": substep_period_s * 1000.0,
        "send_action_ms": send_action_ms_total,
        "sleep_ms": sleep_ms_total,
        "execution_total_ms": _elapsed_ms(execution_start),
        "substeps": substep_payloads,
    }
    return last_result, timing


def _prepare_target_artifacts(
    *,
    target_dir: Path,
    target_phrase: str,
    target_frame_index: int,
    target_frame_rgb: np.ndarray,
    target_mask_path: Path | None,
    servo_config: InvRGBServoConfig,
) -> TargetArtifacts:
    target_dir.mkdir(parents=True, exist_ok=True)
    frame_out = target_dir / f"target_frame_{target_frame_index:06d}.png"
    mask_out = target_dir / f"target_mask_{target_frame_index:06d}.png"
    target_state_path = target_dir / "target_state.json"
    detection_path: Path | None = None
    _write_rgb(frame_out, target_frame_rgb)

    if target_mask_path is not None:
        loaded_mask = cv2.imread(str(target_mask_path), cv2.IMREAD_GRAYSCALE)
        if loaded_mask is None:
            raise FlowFailure("target_mask_unreadable", {"target_mask_path": str(target_mask_path)})
        target_mask = loaded_mask
        detection_payload = {"source": "provided_target_mask", "target_mask_path": str(target_mask_path)}
    else:
        detector = GroundingDINODetector(servo_config.detector.to_detector_config())
        detection_result = detector.detect(target_frame_rgb, phrase=target_phrase, frame_index=target_frame_index)
        if not detection_result.ok or detection_result.state is None:
            raise FlowFailure(
                detection_result.failure_reason or "target_detection_failed",
                {"detector": detection_result.to_dict(), "target_frame": target_frame_index},
            )
        detection_payload = detection_result.state
        bbox_xyxy = detection_payload["bbox_xyxy"]
        tracker = SAM2LiveTracker(servo_config.sam2.to_tracker_config())
        tracker_payload = tracker.initialize(target_frame_rgb, bbox_xyxy)
        if not tracker_payload.get("ok") or tracker_payload.get("mask") is None:
            raise FlowFailure(
                "target_sam2_mask_failed",
                {"sam2": {key: value for key, value in tracker_payload.items() if key != "mask"}},
            )
        target_mask = np.asarray(tracker_payload["mask"], dtype=np.uint8)
        detection_path = target_dir / "target_detection.json"
        _save_json(detection_path, detection_payload)

    target_mask_state = mask_state_from_mask(
        target_mask,
        frame_index=target_frame_index,
        source="recorded_target_mask",
        mask_path=mask_out,
        debug={"target_phrase": target_phrase},
    )
    if not target_mask_state.visible:
        raise FlowFailure("target_mask_empty", {"target_frame": target_frame_index})
    target = ServoTarget(phrase=target_phrase, mask=target_mask_state)
    _write_mask(mask_out, target_mask)
    _save_overlay(target_dir / "target_overlay.png", target_frame_rgb, target_mask, target_phrase)
    _save_json(
        target_state_path,
        {
            "target": target.to_dict(),
            "target_frame_index": target_frame_index,
            "target_frame_path": str(frame_out),
            "target_mask_path": str(mask_out),
            "detection": detection_payload,
        },
    )
    return TargetArtifacts(
        target=target,
        target_mask=(np.asarray(target_mask) > 0).astype(np.uint8) * 255,
        target_frame=target_frame_rgb,
        target_mask_path=mask_out,
        target_frame_path=frame_out,
        target_state_path=target_state_path,
        detection_path=detection_path,
    )


def _joint_action_dict(joint_positions: np.ndarray, gripper_raw: float) -> dict[str, float]:
    q = np.asarray(joint_positions, dtype=float).reshape(6)
    return {
        "joint_1.pos": float(q[0]),
        "joint_2.pos": float(q[1]),
        "joint_3.pos": float(q[2]),
        "joint_4.pos": float(q[3]),
        "joint_5.pos": float(q[4]),
        "joint_6.pos": float(q[5]),
        "gripper.pos": float(gripper_raw),
    }


def _segment_gripper_command(segment_label: str | None, previous_raw: float) -> float:
    label = str(segment_label or "")
    if label == "gripper_close":
        return -1.0
    if label == "gripper_open":
        return 1.0
    return previous_raw


def _bridge_joint_positions(current_q: np.ndarray, target_q: np.ndarray, max_step_rad: float) -> list[np.ndarray]:
    current_q = np.asarray(current_q, dtype=float)
    target_q = np.asarray(target_q, dtype=float)
    step_limit = max(float(max_step_rad), 1e-6)
    max_delta = float(np.max(np.abs(target_q - current_q)))
    if max_delta <= step_limit:
        return []
    step_count = max(1, int(np.ceil(max_delta / step_limit)))
    return [
        (1.0 - alpha) * current_q + alpha * target_q
        for alpha in np.linspace(1.0 / (step_count + 1), 1.0 - (1.0 / (step_count + 1)), num=step_count)
    ]


def _joints_from_observation(observation: dict[str, Any]) -> list[float]:
    return [float(observation[f"joint_{index}.pos"]) for index in range(1, 7)]


def _execute_trajectory_points(
    *,
    robot: Any | None,
    points: list[dict[str, Any]],
    step_duration_s: float,
    bridge_max_joint_step_rad: float,
    dry_run: bool,
    label: str,
) -> dict[str, Any]:
    if not points:
        return {"label": label, "status": "skipped", "point_count": 0, "dry_run": dry_run}
    if dry_run:
        return {
            "label": label,
            "status": "dry_run",
            "point_count": len(points),
            "first_segment_id": points[0].get("segment_id"),
            "last_segment_id": points[-1].get("segment_id"),
        }
    if robot is None:
        raise FlowFailure("robot_not_connected_for_trajectory", {"label": label})

    observation = robot.get_observation()
    current_q = np.asarray(_joints_from_observation(observation), dtype=float)
    first_target_q = np.asarray(points[0]["joint_positions"], dtype=float)
    bridge_points = _bridge_joint_positions(current_q, first_target_q, bridge_max_joint_step_rad)
    current_gripper_raw = 1.0
    for bridge_q in bridge_points:
        robot.send_action(_joint_action_dict(bridge_q, current_gripper_raw))
        time.sleep(max(float(step_duration_s), 1e-3))
    for point in points:
        current_gripper_raw = _segment_gripper_command(point.get("segment_label"), current_gripper_raw)
        robot.send_action(_joint_action_dict(np.asarray(point["joint_positions"], dtype=float), current_gripper_raw))
        time.sleep(max(float(step_duration_s), 1e-3))
    return {
        "label": label,
        "status": "executed",
        "point_count": len(points),
        "bridge_point_count": len(bridge_points),
        "first_segment_id": points[0].get("segment_id"),
        "last_segment_id": points[-1].get("segment_id"),
    }


def _segment_order_map(skill_bank: SkillBank) -> dict[str, int]:
    return {str(segment.segment_id): index for index, segment in enumerate(skill_bank.segments)}


def _split_pre_servo_points(points: list[dict[str, Any]], skill_bank: SkillBank, servo_segment_id: str) -> list[dict[str, Any]]:
    order = _segment_order_map(skill_bank)
    if servo_segment_id not in order:
        raise ValueError(f"Servo segment {servo_segment_id!r} not present in skill bank.")
    servo_index = order[servo_segment_id]
    pre_points: list[dict[str, Any]] = []
    current_segment_index: int | None = None
    for point in points:
        segment_id = point.get("segment_id")
        if segment_id in order:
            current_segment_index = order[str(segment_id)]
        if current_segment_index is not None and current_segment_index < servo_index:
            pre_points.append(point)
    return pre_points


def _write_trajectory_subset(path: Path, source_payload: dict[str, Any], points: list[dict[str, Any]]) -> None:
    reindexed = []
    for index, point in enumerate(points):
        item = dict(point)
        item["trajectory_index"] = index
        reindexed.append(item)
    payload = {
        "joint_keys": list(source_payload.get("joint_keys", [])),
        "points": reindexed,
        "summary": {
            **dict(source_payload.get("summary", {})),
            "point_count": len(reindexed),
            "subset": True,
        },
    }
    _save_json(path, payload)


def _synthetic_current_joints_for_dry_run(
    *,
    pre_points: list[dict[str, Any]],
    records_by_index: dict[int, Any],
    joint_keys: list[str],
    servo_segment: SkillSegment,
) -> list[float]:
    if pre_points:
        return [float(value) for value in pre_points[-1]["joint_positions"]]
    record = records_by_index.get(int(servo_segment.end_frame))
    if record is None:
        record = next(iter(records_by_index.values()))
    return [float(record.joint_positions[key]) for key in joint_keys]


def _connect_robot(
    serial_port: str | None,
    *,
    camera: str,
    servo_arm_velocity: float | None,
    servo_arm_smooth_factor: float | None,
) -> ConnectedRobot:
    from lerobot_robot_cjjarm.cjjarm_robot import CjjArm
    from lerobot_robot_cjjarm.config_cjjarm import CjjArmConfig

    default_config = CjjArmConfig()
    requested_camera = str(camera).strip()
    camera_candidates = [
        requested_camera,
        requested_camera.removeprefix("observation.images."),
        requested_camera.removesuffix("_camera"),
    ]
    camera_candidates = [item for item in dict.fromkeys(camera_candidates) if item]
    selected_cameras = {
        name: default_config.cameras[name]
        for name in camera_candidates
        if name in default_config.cameras
    }
    if not selected_cameras:
        raise FlowFailure(
            "robot_camera_config_missing",
            {
                "requested_camera": requested_camera,
                "available_cameras": sorted(default_config.cameras.keys()),
            },
        )

    robot_config = CjjArmConfig(
        serial_port=serial_port or default_config.serial_port,
        cameras=selected_cameras,
    )
    if not robot_config.serial_port:
        raise FlowFailure("robot_serial_port_missing")
    robot = CjjArm(robot_config)
    robot.connect()
    backend = RobotExecutionBackend(
        robot,
        RobotBackendConfig(
            enabled=True,
            require_explicit_enable=False,
            camera=camera,
            arm_velocity=servo_arm_velocity,
            arm_smooth_factor=servo_arm_smooth_factor,
        ),
    )
    return ConnectedRobot(robot=robot, backend=backend)


def _mask_safety_check(
    *,
    mask_state: MaskState,
    previous_mask_state: MaskState | None,
    safety_config: InvRGBServoConfig,
) -> tuple[bool, str | None, dict[str, Any]]:
    height, width = mask_state.image_size_hw
    area_ratio = float(mask_state.mask_area_px) / max(float(height * width), 1.0)
    safety = safety_config.safety
    if area_ratio < float(safety.min_area_ratio):
        return False, "mask_area_too_small", {"area_ratio": area_ratio}
    if area_ratio > float(safety.max_area_ratio):
        return False, "mask_area_too_large", {"area_ratio": area_ratio}
    if previous_mask_state is None:
        return True, None, {"area_ratio": area_ratio}
    prev_height, prev_width = previous_mask_state.image_size_hw
    prev_area_ratio = float(previous_mask_state.mask_area_px) / max(float(prev_height * prev_width), 1.0)
    if prev_area_ratio > 0.0 and abs(area_ratio - prev_area_ratio) / prev_area_ratio > float(safety.max_area_jump_ratio):
        return False, "mask_area_jump_too_large", {"area_ratio": area_ratio, "previous_area_ratio": prev_area_ratio}
    if mask_state.centroid_uv is not None and previous_mask_state.centroid_uv is not None:
        current = np.asarray(mask_state.centroid_uv, dtype=float)
        previous = np.asarray(previous_mask_state.centroid_uv, dtype=float)
        jump_ratio = float(np.linalg.norm(current - previous) / max(np.hypot(height, width), 1.0))
        if jump_ratio > float(safety.max_bbox_jump_ratio):
            return False, "mask_center_jump_too_large", {"jump_ratio": jump_ratio}
    return True, None, {"area_ratio": area_ratio, "previous_area_ratio": prev_area_ratio}


def _run_live_visual_servo(
    *,
    backend: RobotExecutionBackend,
    target_artifacts: TargetArtifacts,
    target_phrase: str,
    servo_config: InvRGBServoConfig,
    trace_logger: TraceLogger,
    output_dir: Path,
    max_steps: int,
    stable_frames: int,
    save_overlay_every: int,
    step_duration_s: float,
    servo_interpolation_hz: float,
    servo_interpolation_profile: str,
    servo_command_filter_alpha: float,
    servo_command_deadband_xy_m: float,
    servo_command_deadband_z_m: float,
    servo_command_max_change_xy_m: float,
    servo_command_max_change_z_m: float,
) -> dict[str, Any]:
    detector = GroundingDINODetector(servo_config.detector.to_detector_config())
    tracker = SAM2LiveTracker(servo_config.sam2.to_tracker_config())
    controller = RGBServoController(servo_config.servo.to_controller_config())
    safety_filter = ServoSafetyFilter(
        ServoSafetyConfig(
            max_step_xy_m=servo_config.servo.max_step_xy_m,
            max_step_z_m=servo_config.servo.max_step_z_m,
            max_rotation_rad=servo_config.safety.max_joint_step_rad,
        )
    )

    init_start = time.perf_counter()
    first_obs_start = time.perf_counter()
    first_obs = backend.get_rgb_frame()
    first_observation_ms = _elapsed_ms(first_obs_start)
    if not first_obs.ok or first_obs.state is None:
        raise FlowFailure(first_obs.failure_reason or "robot_rgb_frame_unavailable", {"observation": first_obs.to_dict()})
    first_frame = np.asarray(first_obs.state["image_rgb"], dtype=np.uint8)
    first_frame_index = int(first_obs.state["frame_index"])
    dino_start = time.perf_counter()
    detection_result = detector.detect(first_frame, phrase=target_phrase, frame_index=first_frame_index)
    dino_ms = _elapsed_ms(dino_start)
    if not detection_result.ok or detection_result.state is None:
        raise FlowFailure(detection_result.failure_reason or "live_detection_failed", {"detector": detection_result.to_dict()})
    sam2_init_start = time.perf_counter()
    tracker_payload = tracker.initialize(first_frame, detection_result.state["bbox_xyxy"])
    sam2_init_ms = _elapsed_ms(sam2_init_start)
    trace_logger.write_event(
        "servo_init_timing",
        {
            "wall_time_s": time.time(),
            "frame_index": first_frame_index,
            "first_observation_ms": first_observation_ms,
            "grounding_dino_ms": dino_ms,
            "sam2_initialize_ms": sam2_init_ms,
            "init_total_ms": _elapsed_ms(init_start),
        },
    )

    stable_count = 0
    lost_count = 0
    previous_mask_state: MaskState | None = None
    previous_servo_command: ServoCommand | None = None
    last_error: dict[str, Any] | None = None
    overlay_dir = output_dir / "overlays"

    for step_index in range(max_steps):
        cycle_start = time.perf_counter()
        timing_payload: dict[str, Any] = {
            "step_index": step_index,
            "wall_time_s": time.time(),
            "observation_ms": 0.0,
            "sam2_update_ms": 0.0,
            "control_ms": 0.0,
            "overlay_ms": 0.0,
            "send_action_ms": 0.0,
            "sleep_ms": 0.0,
            "execution_total_ms": 0.0,
            "cycle_total_ms": None,
            "interpolation": None,
        }
        if step_index > 0:
            observation_start = time.perf_counter()
            observation = backend.get_rgb_frame()
            timing_payload["observation_ms"] = _elapsed_ms(observation_start)
            if not observation.ok or observation.state is None:
                lost_count += 1
                trace_logger.write_event(
                    "servo_observation_lost",
                    {"step_index": step_index, "lost_count": lost_count, "observation": observation.to_dict()},
                )
                if lost_count > int(servo_config.safety.max_lost_frames):
                    raise FlowFailure("servo_lost_too_many_frames", {"lost_count": lost_count})
                continue
            frame_rgb = np.asarray(observation.state["image_rgb"], dtype=np.uint8)
            frame_index = int(observation.state["frame_index"])
            sam2_update_start = time.perf_counter()
            tracker_payload = tracker.update(frame_rgb, frame_index=frame_index)
            timing_payload["sam2_update_ms"] = _elapsed_ms(sam2_update_start)
        else:
            frame_rgb = first_frame
            frame_index = first_frame_index

        if not tracker_payload.get("ok") or tracker_payload.get("mask") is None:
            lost_count += 1
            trace_logger.write_event(
                "servo_mask_lost",
                {
                    "step_index": step_index,
                    "frame_index": frame_index,
                    "lost_count": lost_count,
                    "tracker": {key: value for key, value in tracker_payload.items() if key != "mask"},
                },
            )
            if lost_count > int(servo_config.safety.max_lost_frames):
                raise FlowFailure("servo_mask_lost_too_many_frames", {"lost_count": lost_count})
            continue
        lost_count = 0

        control_start = time.perf_counter()
        current_mask = (np.asarray(tracker_payload["mask"]) > 0).astype(np.uint8) * 255
        current_mask_state = _mask_state_from_payload(tracker_payload)
        safe, safety_reason, safety_payload = _mask_safety_check(
            mask_state=current_mask_state,
            previous_mask_state=previous_mask_state,
            safety_config=servo_config,
        )
        if not safe:
            raise FlowFailure(safety_reason or "servo_mask_safety_failed", safety_payload)
        current_iou = None
        if tuple(current_mask.shape) == tuple(target_artifacts.target_mask.shape):
            current_iou = mask_iou(current_mask, target_artifacts.target_mask)
        command_result = controller.compute_command(current_mask_state, target_artifacts.target, mask_iou=current_iou)
        if not command_result.ok or command_result.state is None:
            raise FlowFailure(command_result.failure_reason or "servo_command_failed", command_result.to_dict())
        command = _servo_command_from_dict(command_result.state["command"])
        safe_result = safety_filter.filter_command(command)
        if not safe_result.ok or safe_result.state is None:
            raise FlowFailure(safe_result.failure_reason or "servo_safety_failed", safe_result.to_dict())
        raw_safe_command = _servo_command_from_dict(safe_result.state["safe_command"])
        safe_command, command_shaping = _shape_servo_command(
            raw_safe_command,
            previous_command=previous_servo_command,
            filter_alpha=servo_command_filter_alpha,
            deadband_xy_m=servo_command_deadband_xy_m,
            deadband_z_m=servo_command_deadband_z_m,
            max_change_xy_m=servo_command_max_change_xy_m,
            max_change_z_m=servo_command_max_change_z_m,
        )
        timing_payload["control_ms"] = _elapsed_ms(control_start)

        error = dict(command_result.state["error"])
        last_error = error
        if error.get("converged"):
            stable_count += 1
        else:
            stable_count = 0

        safety_state = dict(safe_result.state)
        safety_state["raw_safe_command"] = raw_safe_command.to_dict()
        safety_state["shaped_safe_command"] = safe_command.to_dict()
        event_payload = {
            "step_index": step_index,
            "frame_index": frame_index,
            "mask_state": current_mask_state.to_dict(),
            "error": error,
            "safety": safety_state,
            "command_shaping": command_shaping,
            "stable_count": stable_count,
            "safety_mask": safety_payload,
        }
        trace_logger.write_event("servo_step", event_payload)
        if save_overlay_every > 0 and step_index % save_overlay_every == 0:
            overlay_start = time.perf_counter()
            _save_overlay(overlay_dir / f"servo_overlay_{step_index:04d}.png", frame_rgb, current_mask, target_phrase)
            timing_payload["overlay_ms"] = _elapsed_ms(overlay_start)

        if stable_count >= stable_frames:
            timing_payload["cycle_total_ms"] = _elapsed_ms(cycle_start)
            timing_payload["terminated"] = "converged"
            trace_logger.write_event("servo_timing", timing_payload)
            trace_logger.write_event("servo_converged", {"step_index": step_index, "stable_count": stable_count, "error": error})
            return {
                "status": "converged",
                "step_count": step_index + 1,
                "stable_count": stable_count,
                "trace_path": str(trace_logger.path),
                "last_error": last_error,
            }

        backend_result, execution_timing = _execute_servo_command_smooth(
            backend=backend,
            command=safe_command,
            step_duration_s=step_duration_s,
            interpolation_hz=servo_interpolation_hz,
            interpolation_profile=servo_interpolation_profile,
        )
        timing_payload["send_action_ms"] = execution_timing["send_action_ms"]
        timing_payload["sleep_ms"] = execution_timing["sleep_ms"]
        timing_payload["execution_total_ms"] = execution_timing["execution_total_ms"]
        timing_payload["interpolation"] = {
            key: value for key, value in execution_timing.items() if key not in {"send_action_ms", "sleep_ms", "execution_total_ms"}
        }
        trace_logger.write_event(
            "servo_command",
            {"step_index": step_index, "backend": backend_result.to_dict(), "execution": execution_timing},
        )
        if not backend_result.ok:
            raise FlowFailure(backend_result.failure_reason or "servo_backend_failed", backend_result.to_dict())
        previous_servo_command = safe_command
        previous_mask_state = current_mask_state
        timing_payload["cycle_total_ms"] = _elapsed_ms(cycle_start)
        trace_logger.write_event("servo_timing", timing_payload)

    raise FlowFailure(
        "servo_max_steps_exceeded",
        {"max_steps": max_steps, "stable_count": stable_count, "last_error": last_error, "trace_path": str(trace_logger.path)},
    )


def _ensure_skill_bank(session_dir: Path) -> Path:
    skill_bank_path = session_dir / "analysis" / "t3_skill_bank" / "skill_bank.json"
    if skill_bank_path.exists():
        return skill_bank_path
    result = build_skill_bank(
        session_dir=session_dir,
        output_dir=session_dir / "analysis" / "t3_skill_bank",
        segmenter_config=SegmenterConfig(),
        classifier_config=InvarianceClassifierConfig(),
    )
    return result.skill_bank_path


def _resolve_live_result_path(
    *,
    args: argparse.Namespace,
    task_config: TaskGraspConfig,
    target_phrase: str,
    run_dir: Path,
) -> Path:
    if args.capture_live_result and args.live_result_path is not None:
        raise ValueError("Specify either --capture-live-result or --live-result-path, not both.")
    if args.capture_live_result:
        from lerobot.projects.vlbiman_sa.app.run_live_grasp_preview import _capture_live_result

        vision_payload = yaml.safe_load(args.vision_config.read_text(encoding="utf-8")) or {}
        if not isinstance(vision_payload, dict):
            raise ValueError(f"Vision config must be a mapping: {args.vision_config}")
        vision_payload["intrinsics_path"] = str(task_config.intrinsics_path)
        vision_payload["target_phrase"] = target_phrase
        vision_payload["task_prompt"] = target_phrase
        effective_vision_config = run_dir / "capture_vision_config.yaml"
        effective_vision_config.parent.mkdir(parents=True, exist_ok=True)
        effective_vision_config.write_text(yaml.safe_dump(vision_payload, sort_keys=False), encoding="utf-8")

        logging.info("Capturing live target localization for phrase %r.", target_phrase)
        live_result_path = _capture_live_result(
            capture_config_path=args.capture_config,
            vision_config_path=effective_vision_config,
            handeye_result_path=task_config.handeye_result_path,
            output_root=args.live_output_root,
            camera_serial_number=args.camera_serial_number or task_config.camera_serial_number,
            target_phrase=target_phrase,
            warmup_frames=args.warmup_frames,
            camera_timeout_ms=args.camera_timeout_ms,
        )
    elif args.live_result_path is not None:
        live_result_path = args.live_result_path
    else:
        raise ValueError("Provide --capture-live-result or an explicit --live-result-path for the current scene.")

    if not live_result_path.exists():
        raise FileNotFoundError(f"Live result not found: {live_result_path}")
    return live_result_path


def _live_result_diagnostics(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return {"path": str(path), "valid": False, "reason": "not_a_mapping"}
    objects = payload.get("objects") if isinstance(payload.get("objects"), dict) else {}
    return {
        "path": str(path),
        "valid": True,
        "status": payload.get("status"),
        "mode": payload.get("mode"),
        "target_phrase": payload.get("target_phrase"),
        "base_xyz_m": payload.get("base_xyz_m"),
        "object_count": len(objects),
        "object_keys": sorted(str(key) for key in objects.keys()),
    }


def _ensure_pose(
    *,
    args: argparse.Namespace,
    task_config: TaskGraspConfig,
    session_dir: Path,
    analysis_dir: Path,
    output_dir: Path,
    live_result_path: Path,
) -> Path:
    from lerobot.projects.vlbiman_sa.app.run_pose_adaptation import (
        build_pose_pipeline_config,
        run_pose_adaptation_pipeline,
    )

    adapted_pose_path = output_dir / "adapted_pose.json"

    task_config.recording_session_dir = session_dir
    task_config.skill_bank_path = analysis_dir / "t3_skill_bank" / "skill_bank.json"
    task_config.skill_output_dir = analysis_dir / "t3_skill_bank"
    task_config.vision_output_dir = analysis_dir / "t4_vision"
    task_config.pose_output_dir = output_dir
    task_config.live_result_path = live_result_path
    task_config.target_phrase = args.target_phrase
    task_config.task_prompt = args.target_phrase
    task_config.primary_reference_phrase = args.target_phrase
    if args.aux_target_phrase:
        task_config.secondary_target_phrase = str(args.aux_target_phrase[0])
        task_config.secondary_reference_phrase = str(args.aux_target_phrase[0])
        task_config.secondary_vision_dir_name = f"t4_vision_{_safe_name(args.aux_target_phrase[0])}"

    pose_summary = run_pose_adaptation_pipeline(
        build_pose_pipeline_config(
            task_config=task_config,
            session_dir=session_dir,
            analysis_dir=analysis_dir,
            output_dir=output_dir,
            live_result_path=live_result_path,
            auxiliary_target_phrases=list(args.aux_target_phrase or []),
            allow_configured_secondary_fallback=bool(args.aux_target_phrase),
        )
    )
    if pose_summary.get("status") != "ok":
        raise FlowFailure("t5_pose_adaptation_failed", {"pose_summary": pose_summary})
    return adapted_pose_path


def _safe_name(value: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "_" for char in str(value).strip())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "object"


def _run_t6(
    *,
    session_dir: Path,
    analysis_dir: Path,
    output_dir: Path,
    skill_bank_path: Path,
    adapted_pose_path: Path,
    current_joint_positions: list[float] | None = None,
    current_joint_positions_source: str | None = None,
    start_after_segment_id: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    from lerobot.projects.vlbiman_sa.app.run_trajectory_generation import (
        TrajectoryPipelineConfig,
        run_trajectory_generation_pipeline,
    )

    summary_path = output_dir / "summary.json"
    points_path = output_dir / "trajectory_points.json"
    if points_path.exists() and summary_path.exists() and not force:
        return _load_json(summary_path)
    return run_trajectory_generation_pipeline(
        TrajectoryPipelineConfig(
            session_dir=session_dir,
            analysis_dir=analysis_dir,
            output_dir=output_dir,
            skill_bank_path=skill_bank_path,
            adapted_pose_path=adapted_pose_path,
            current_joint_positions=current_joint_positions,
            current_joint_positions_source=current_joint_positions_source,
            start_after_segment_id=start_after_segment_id,
        )
    )


def _apply_servo_cli_overrides(servo_config: InvRGBServoConfig, args: argparse.Namespace) -> None:
    if args.servo_k_u is not None:
        servo_config.servo.k_u = float(args.servo_k_u)
    if args.servo_k_v is not None:
        servo_config.servo.k_v = float(args.servo_k_v)
    if args.servo_k_a is not None:
        servo_config.servo.k_a = float(args.servo_k_a)
    if args.servo_axis_sign_x is not None:
        servo_config.servo.axis_sign_x = float(args.servo_axis_sign_x)
    if args.servo_axis_sign_y is not None:
        servo_config.servo.axis_sign_y = float(args.servo_axis_sign_y)
    if args.servo_axis_sign_z is not None:
        servo_config.servo.axis_sign_z = float(args.servo_axis_sign_z)
    if args.servo_max_step_xy_m is not None:
        servo_config.servo.max_step_xy_m = float(args.servo_max_step_xy_m)
    if args.servo_max_step_z_m is not None:
        servo_config.servo.max_step_z_m = float(args.servo_max_step_z_m)


def _annotate_summary(path: Path, updates: dict[str, Any]) -> dict[str, Any]:
    payload = _load_json(path) if path.exists() else {}
    payload.update(updates)
    _save_json(path, payload)
    return payload


def run_flow(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.run_id or _timestamp_run_id()
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    session_dir = args.session_dir
    analysis_dir = session_dir / "analysis"
    task_config = _load_task_config(args.task_config)
    servo_config = load_inv_rgb_servo_config(args.servo_config)
    servo_config.target.phrase = str(args.target_phrase)
    servo_config.backend.camera = str(args.camera)
    _apply_servo_cli_overrides(servo_config, args)
    servo_step_duration_s = float(args.servo_step_duration_s) if args.servo_step_duration_s is not None else float(args.step_duration_s)
    trace_logger = TraceLogger(TraceLoggerConfig(run_dir / "servo", filename="servo_trace.jsonl"))
    connected: ConnectedRobot | None = None

    skill_bank_path = _ensure_skill_bank(session_dir)
    skill_bank = SkillBank.load(skill_bank_path)
    servo_segment = _find_servo_segment(
        skill_bank,
        servo_segment_id=args.servo_segment,
        servo_frame_range=_parse_frame_range_1based(args.servo_frame_range),
    )
    if not _is_visual_servo_segment(servo_segment):
        logging.warning("Servo segment %s is not annotated as visual_servo; it will still be treated as online servo.", servo_segment.segment_id)

    target_phrase = _resolve_servo_target_phrase(servo_segment, args.target_phrase)
    servo_config.target.phrase = target_phrase
    target_frame_index = _resolve_servo_target_frame(servo_segment, args.servo_target_frame)
    live_result_path = _resolve_live_result_path(
        args=args,
        task_config=task_config,
        target_phrase=target_phrase,
        run_dir=run_dir,
    )
    live_result_diagnostics = _live_result_diagnostics(live_result_path)
    adapted_pose_path = _ensure_pose(
        args=args,
        task_config=task_config,
        session_dir=session_dir,
        analysis_dir=analysis_dir,
        output_dir=run_dir / "analysis" / "t5_pose",
        live_result_path=live_result_path,
    )
    records = load_frame_records(session_dir)
    records_by_index = {int(record.frame_index): record for record in records}
    try:
        target_frame_rgb = _load_recorded_rgb_frame(session_dir, records_by_index, target_frame_index, args.camera)
        target_artifacts = _prepare_target_artifacts(
            target_dir=run_dir / "target",
            target_phrase=target_phrase,
            target_frame_index=target_frame_index,
            target_frame_rgb=target_frame_rgb,
            target_mask_path=args.target_mask_path,
            servo_config=servo_config,
        )
    except FlowFailure as exc:
        failure = {
            "status": "failed",
            "failure_reason": exc.reason,
            "failure": exc.payload,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "dry_run": bool(args.dry_run),
            "session_dir": str(session_dir),
            "live_result_path": str(live_result_path),
            "servo_trace_path": str(trace_logger.path),
        }
        _save_json(run_dir / "failure_report.json", failure)
        _save_json(run_dir / "flow_summary.json", failure)
        return failure

    full_t6_summary = _run_t6(
        session_dir=session_dir,
        analysis_dir=analysis_dir,
        output_dir=run_dir / "t6_full",
        skill_bank_path=skill_bank_path,
        adapted_pose_path=adapted_pose_path,
        force=args.force_t6,
    )
    full_points_path = Path(full_t6_summary["trajectory_points_path"])
    full_payload = _load_json(full_points_path)
    pre_points = _split_pre_servo_points(list(full_payload.get("points", [])), skill_bank, str(servo_segment.segment_id))
    pre_points_path = run_dir / "pre_servo_trajectory_points.json"
    _write_trajectory_subset(pre_points_path, full_payload, pre_points)

    pre_execution: dict[str, Any]
    servo_summary: dict[str, Any]
    post_execution: dict[str, Any]
    current_joint_positions: list[float]
    try:
        if args.dry_run:
            pre_execution = _execute_trajectory_points(
                robot=None,
                points=pre_points,
                step_duration_s=args.step_duration_s,
                bridge_max_joint_step_rad=args.bridge_max_joint_step_rad,
                dry_run=True,
                label="pre_servo",
            )
            current_joint_positions = _synthetic_current_joints_for_dry_run(
                pre_points=pre_points,
                records_by_index=records_by_index,
                joint_keys=skill_bank.joint_keys,
                servo_segment=servo_segment,
            )
            servo_summary = {
                "status": "dry_run_skipped_live_servo",
                "trace_path": str(trace_logger.path),
                "synthetic_current_joint_positions": current_joint_positions,
            }
            trace_logger.write_event("servo_dry_run", servo_summary)
        else:
            connected = _connect_robot(
                args.robot_serial_port,
                camera=args.camera,
                servo_arm_velocity=args.servo_arm_velocity,
                servo_arm_smooth_factor=args.servo_arm_smooth_factor,
            )
            pre_execution = _execute_trajectory_points(
                robot=connected.robot,
                points=pre_points,
                step_duration_s=args.step_duration_s,
                bridge_max_joint_step_rad=args.bridge_max_joint_step_rad,
                dry_run=False,
                label="pre_servo",
            )
            servo_summary = _run_live_visual_servo(
                backend=connected.backend,
                target_artifacts=target_artifacts,
                target_phrase=target_phrase,
                servo_config=servo_config,
                trace_logger=trace_logger,
                output_dir=run_dir / "servo",
                max_steps=int(args.max_servo_steps or servo_config.servo.max_steps),
                stable_frames=int(args.stable_servo_frames or servo_config.servo.stable_frames),
                save_overlay_every=max(0, int(args.save_overlay_every)),
                step_duration_s=servo_step_duration_s,
                servo_interpolation_hz=float(args.servo_interpolation_hz),
                servo_interpolation_profile=str(args.servo_interpolation_profile),
                servo_command_filter_alpha=float(args.servo_command_filter_alpha),
                servo_command_deadband_xy_m=float(args.servo_command_deadband_xy_m),
                servo_command_deadband_z_m=float(args.servo_command_deadband_z_m),
                servo_command_max_change_xy_m=float(args.servo_command_max_change_xy_m),
                servo_command_max_change_z_m=float(args.servo_command_max_change_z_m),
            )
            current_joint_positions = _joints_from_observation(connected.robot.get_observation())

        post_t6_summary = _run_t6(
            session_dir=session_dir,
            analysis_dir=analysis_dir,
            output_dir=run_dir / "t6_post_servo",
            skill_bank_path=skill_bank_path,
            adapted_pose_path=adapted_pose_path,
            current_joint_positions=current_joint_positions,
            current_joint_positions_source="real_robot_after_visual_servo" if not args.dry_run else "dry_run_pre_servo_endpoint",
            start_after_segment_id=str(servo_segment.segment_id),
            force=True,
        )
        post_summary_path = Path(post_t6_summary["summary_path"])
        post_t6_summary = _annotate_summary(
            post_summary_path,
            {
                "resume_after_segment_id": str(servo_segment.segment_id),
                "servo_trace_path": str(trace_logger.path),
                "continuity_input_joint_positions": current_joint_positions,
            },
        )
        post_points_path = Path(post_t6_summary["trajectory_points_path"])
        post_points = list(_load_json(post_points_path).get("points", []))
        post_execution = _execute_trajectory_points(
            robot=None if connected is None else connected.robot,
            points=post_points,
            step_duration_s=args.step_duration_s,
            bridge_max_joint_step_rad=args.bridge_max_joint_step_rad,
            dry_run=args.dry_run,
            label="post_servo",
        )

        summary = {
            "status": "ok",
            "run_id": run_id,
            "run_dir": str(run_dir),
            "dry_run": bool(args.dry_run),
            "session_dir": str(session_dir),
            "live_result_path": str(live_result_path),
            "live_result": live_result_diagnostics,
            "target_phrase": target_phrase,
            "servo_gains": {
                "k_u": float(servo_config.servo.k_u),
                "k_v": float(servo_config.servo.k_v),
                "k_a": float(servo_config.servo.k_a),
            },
            "servo_interpolation_hz": float(args.servo_interpolation_hz),
            "servo_interpolation_profile": str(args.servo_interpolation_profile),
            "servo_step_duration_s": servo_step_duration_s,
            "servo_arm_velocity": None if args.servo_arm_velocity is None else float(args.servo_arm_velocity),
            "servo_arm_smooth_factor": None if args.servo_arm_smooth_factor is None else float(args.servo_arm_smooth_factor),
            "servo_command_filter": {
                "alpha": float(args.servo_command_filter_alpha),
                "deadband_xy_m": float(args.servo_command_deadband_xy_m),
                "deadband_z_m": float(args.servo_command_deadband_z_m),
                "max_change_xy_m": float(args.servo_command_max_change_xy_m),
                "max_change_z_m": float(args.servo_command_max_change_z_m),
            },
            "servo_segment": {
                "segment_id": str(servo_segment.segment_id),
                "label": servo_segment.label,
                "semantic_state": servo_segment.metrics.get("semantic_state"),
                "start_frame": int(servo_segment.start_frame),
                "end_frame": int(servo_segment.end_frame),
                "target_frame": target_frame_index,
            },
            "skill_bank_path": str(skill_bank_path),
            "adapted_pose_path": str(adapted_pose_path),
            "full_t6_summary_path": str(full_t6_summary["summary_path"]),
            "pre_servo_trajectory_points_path": str(pre_points_path),
            "post_servo_trajectory_points_path": str(post_points_path),
            "post_servo_summary_path": str(post_summary_path),
            "target_artifacts": {
                "target_mask_path": str(target_artifacts.target_mask_path),
                "target_frame_path": None if target_artifacts.target_frame_path is None else str(target_artifacts.target_frame_path),
                "target_state_path": str(target_artifacts.target_state_path),
                "detection_path": None if target_artifacts.detection_path is None else str(target_artifacts.detection_path),
            },
            "pre_execution": pre_execution,
            "servo_summary": servo_summary,
            "post_execution": post_execution,
        }
        _save_json(run_dir / "flow_summary.json", summary)
        return summary
    except FlowFailure as exc:
        failure = {
            "status": "failed",
            "failure_reason": exc.reason,
            "failure": exc.payload,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "dry_run": bool(args.dry_run),
            "session_dir": str(session_dir),
            "live_result_path": str(live_result_path),
            "servo_trace_path": str(trace_logger.path),
        }
        _save_json(run_dir / "failure_report.json", failure)
        _save_json(run_dir / "flow_summary.json", failure)
        return failure
    finally:
        if connected is not None and getattr(connected.robot, "is_connected", False):
            connected.robot.disconnect()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)
    summary = run_flow(args)
    logging.info("Real servo flow summary: %s", json.dumps(_to_jsonable(summary), ensure_ascii=False))
    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
