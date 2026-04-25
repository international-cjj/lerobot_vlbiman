from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

from ..contracts import FRRGGraspConfig
from .default_provider import default_theta_sources_from_config
from .theta_schema import DemoTheta, THETA_PARAMETER_NAMES, ThetaParameterSource


@dataclass
class ThetaExtractionResult:
    theta: DemoTheta
    parameter_sources: dict[str, ThetaParameterSource]
    session_dir: Path
    trajectory_points_path: Path
    selected_paths: dict[str, str] = field(default_factory=dict)
    available_segment_labels: list[str] = field(default_factory=list)

    def theta_samples_dict(self) -> dict[str, float]:
        return self.theta.to_dict()

    def report_dict(self) -> dict[str, Any]:
        return {
            "session_dir": str(self.session_dir),
            "trajectory_points_path": str(self.trajectory_points_path),
            "selected_paths": dict(self.selected_paths),
            "available_segment_labels": list(self.available_segment_labels),
            "theta": self.theta.to_dict(),
            "parameters": {
                name: self.parameter_sources[name].to_dict()
                for name in THETA_PARAMETER_NAMES
            },
            "default_parameter_count": sum(1 for source in self.parameter_sources.values() if source.default_used),
            "extracted_parameter_count": sum(1 for source in self.parameter_sources.values() if not source.default_used),
        }


def _load_json(path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}.")
    return payload


def _select_analysis_file(session_dir: Path, preferred_relative_paths: list[str], glob_patterns: list[str]) -> tuple[Path, str]:
    session_dir = Path(session_dir).resolve()
    for relative_path in preferred_relative_paths:
        candidate = session_dir / relative_path
        if candidate.exists():
            return candidate, f"preferred:{relative_path}"
    analysis_dir = session_dir / "analysis"
    for pattern in glob_patterns:
        matches = sorted(analysis_dir.glob(pattern))
        if matches:
            relative = matches[0].relative_to(session_dir)
            return matches[0], f"glob:{relative}"
    raise FileNotFoundError(f"No matching analysis file found under {session_dir}.")


def _segment_points(points: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    return [point for point in points if point.get("segment_label") == label]


def _pose_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [point for point in points if isinstance(point.get("target_pose_6d"), list) and len(point["target_pose_6d"]) >= 3]


def _path_length_m(pose_points: list[dict[str, Any]]) -> float:
    xyz_points = [tuple(float(value) for value in point["target_pose_6d"][:3]) for point in pose_points]
    return sum(math.dist(start, end) for start, end in zip(xyz_points, xyz_points[1:]))


def _segment_duration_s(points: list[dict[str, Any]]) -> float:
    if len(points) < 2:
        return 0.0
    return float(points[-1]["relative_time_s"]) - float(points[0]["relative_time_s"])


def _default_parameter(
    name: str,
    defaults: dict[str, ThetaParameterSource],
    *,
    reason: str,
    diagnostics: dict[str, Any] | None = None,
) -> ThetaParameterSource:
    default_source = defaults[name]
    return ThetaParameterSource(
        name=name,
        value=default_source.value,
        source=default_source.source,
        default_used=True,
        reason=reason,
        diagnostics=diagnostics or dict(default_source.diagnostics),
    )


def _extract_advance_speed(
    points: list[dict[str, Any]],
    defaults: dict[str, ThetaParameterSource],
) -> ThetaParameterSource:
    segment = _pose_points(_segment_points(points, "approach"))
    if len(segment) < 2:
        return _default_parameter("advance_speed_mps", defaults, reason="approach_segment_missing_or_too_short")
    duration_s = _segment_duration_s(segment)
    path_length_m = _path_length_m(segment)
    if duration_s <= 1e-9 or path_length_m <= 1e-6:
        return _default_parameter(
            "advance_speed_mps",
            defaults,
            reason="approach_segment_has_no_measurable_motion",
            diagnostics={"duration_s": duration_s, "path_length_m": path_length_m},
        )
    return ThetaParameterSource(
        name="advance_speed_mps",
        value=path_length_m / duration_s,
        source="demo_segment:approach",
        diagnostics={"duration_s": duration_s, "path_length_m": path_length_m, "point_count": len(segment)},
    )


def _extract_preclose_distance(
    points: list[dict[str, Any]],
    config: FRRGGraspConfig,
    defaults: dict[str, ThetaParameterSource],
) -> ThetaParameterSource:
    segment = _pose_points(_segment_points(points, "gripper_close"))
    if len(segment) < 2:
        return _default_parameter("preclose_distance_m", defaults, reason="gripper_close_pose_points_missing")
    start_z = float(segment[0]["target_pose_6d"][2])
    end_z = float(segment[-1]["target_pose_6d"][2])
    candidate = abs(end_z - start_z)
    max_plausible = max(float(config.capture_build.target_depth_max_m), float(config.close_hold.preclose_distance_m) * 3.0)
    if candidate <= 1e-4:
        return _default_parameter(
            "preclose_distance_m",
            defaults,
            reason="gripper_close_pose_motion_too_small",
            diagnostics={"candidate_distance_m": candidate},
        )
    if candidate > max_plausible:
        return _default_parameter(
            "preclose_distance_m",
            defaults,
            reason="gripper_close_pose_motion_out_of_frrg_range",
            diagnostics={"candidate_distance_m": candidate, "max_plausible_m": max_plausible},
        )
    return ThetaParameterSource(
        name="preclose_distance_m",
        value=candidate,
        source="demo_segment:gripper_close_pose",
        diagnostics={"start_z_m": start_z, "end_z_m": end_z, "point_count": len(segment)},
    )


def _extract_close_speed(
    points: list[dict[str, Any]],
    defaults: dict[str, ThetaParameterSource],
) -> ThetaParameterSource:
    segment = [point for point in _segment_points(points, "gripper_close") if point.get("gripper_position") is not None]
    if len(segment) < 2:
        return _default_parameter("close_speed_raw_per_s", defaults, reason="gripper_close_gripper_signal_missing")
    duration_s = _segment_duration_s(segment)
    gripper_delta = abs(float(segment[-1]["gripper_position"]) - float(segment[0]["gripper_position"]))
    if duration_s <= 1e-9 or gripper_delta <= 1e-4:
        return _default_parameter(
            "close_speed_raw_per_s",
            defaults,
            reason="gripper_close_delta_too_small",
            diagnostics={"duration_s": duration_s, "gripper_delta": gripper_delta},
        )
    return ThetaParameterSource(
        name="close_speed_raw_per_s",
        value=gripper_delta / duration_s,
        source="demo_segment:gripper_close_gripper",
        diagnostics={"duration_s": duration_s, "gripper_delta": gripper_delta, "point_count": len(segment)},
    )


def _extract_settle_time(
    points: list[dict[str, Any]],
    config: FRRGGraspConfig,
    defaults: dict[str, ThetaParameterSource],
) -> ThetaParameterSource:
    segment = [point for point in _segment_points(points, "gripper_close") if point.get("gripper_position") is not None]
    if len(segment) < 3:
        return _default_parameter("settle_time_s", defaults, reason="gripper_close_segment_too_short_for_settle_detection")
    rate_samples: list[tuple[float, float]] = []
    for start, end in zip(segment, segment[1:]):
        dt = float(end["relative_time_s"]) - float(start["relative_time_s"])
        if dt <= 1e-9:
            continue
        rate_samples.append((float(end["relative_time_s"]), abs(float(end["gripper_position"]) - float(start["gripper_position"])) / dt))
    if not rate_samples:
        return _default_parameter("settle_time_s", defaults, reason="gripper_close_rate_samples_missing")
    mean_rate = sum(rate for _, rate in rate_samples) / len(rate_samples)
    threshold = max(mean_rate * 0.1, 1e-4)
    last_active_t = float(segment[0]["relative_time_s"])
    for timestamp_s, rate in rate_samples:
        if rate > threshold:
            last_active_t = timestamp_s
    settle_time_s = float(segment[-1]["relative_time_s"]) - last_active_t
    min_settle_time = 1.0 / max(float(config.runtime.control_hz), 1e-9)
    if settle_time_s <= min_settle_time:
        return _default_parameter(
            "settle_time_s",
            defaults,
            reason="no_observable_settle_tail",
            diagnostics={
                "settle_time_s": settle_time_s,
                "min_settle_time_s": min_settle_time,
                "mean_rate": mean_rate,
                "threshold": threshold,
            },
        )
    return ThetaParameterSource(
        name="settle_time_s",
        value=settle_time_s,
        source="demo_segment:gripper_close_tail",
        diagnostics={
            "mean_rate": mean_rate,
            "threshold": threshold,
            "last_active_t": last_active_t,
            "segment_end_t": float(segment[-1]["relative_time_s"]),
        },
    )


def _extract_lift_height(
    points: list[dict[str, Any]],
    defaults: dict[str, ThetaParameterSource],
) -> ThetaParameterSource:
    segment = _pose_points(_segment_points(points, "transfer"))
    if len(segment) < 2:
        return _default_parameter("lift_height_m", defaults, reason="transfer_pose_points_missing")
    window_end_t = float(segment[0]["relative_time_s"]) + 0.5
    early_segment = [point for point in segment if float(point["relative_time_s"]) <= window_end_t]
    if len(early_segment) < 2:
        early_segment = segment[: min(len(segment), 10)]
    start_z = float(early_segment[0]["target_pose_6d"][2])
    lift_height_m = max(max(float(point["target_pose_6d"][2]) - start_z, 0.0) for point in early_segment)
    if lift_height_m <= 1e-4:
        return _default_parameter(
            "lift_height_m",
            defaults,
            reason="transfer_segment_has_no_positive_lift_window",
            diagnostics={"window_point_count": len(early_segment), "start_z_m": start_z},
        )
    return ThetaParameterSource(
        name="lift_height_m",
        value=lift_height_m,
        source="demo_segment:transfer_early_window",
        diagnostics={
            "start_z_m": start_z,
            "window_point_count": len(early_segment),
            "window_duration_s": float(early_segment[-1]["relative_time_s"]) - float(early_segment[0]["relative_time_s"]),
        },
    )


def extract_theta_from_session(session_dir: str | Path, config: FRRGGraspConfig) -> ThetaExtractionResult:
    session_path = Path(session_dir).resolve()
    trajectory_points_path, trajectory_select_reason = _select_analysis_file(
        session_path,
        preferred_relative_paths=[
            "analysis/t6_trajectory/trajectory_points.json",
            "analysis/ros2_smoke_t6/trajectory_points.json",
        ],
        glob_patterns=[
            "t6_baseline*/trajectory_points.json",
            "t6_*/trajectory_points.json",
            "*/trajectory_points.json",
        ],
    )
    selected_paths = {"trajectory_points": trajectory_select_reason}
    try:
        adapted_pose_path, adapted_pose_reason = _select_analysis_file(
            session_path,
            preferred_relative_paths=[
                "analysis/t5_pose/adapted_pose.json",
                "analysis/ros2_smoke_t5/adapted_pose.json",
            ],
            glob_patterns=["t5*/adapted_pose.json", "*/adapted_pose.json"],
        )
        selected_paths["adapted_pose"] = adapted_pose_reason
        selected_paths["adapted_pose_path"] = str(adapted_pose_path)
    except FileNotFoundError:
        pass
    payload = _load_json(trajectory_points_path)
    points = payload.get("points")
    if not isinstance(points, list):
        raise ValueError(f"{trajectory_points_path} does not contain a 'points' list.")

    defaults = default_theta_sources_from_config(config)
    parameter_sources = {
        "advance_speed_mps": _extract_advance_speed(points, defaults),
        "preclose_distance_m": _extract_preclose_distance(points, config, defaults),
        "close_speed_raw_per_s": _extract_close_speed(points, defaults),
        "settle_time_s": _extract_settle_time(points, config, defaults),
        "lift_height_m": _extract_lift_height(points, defaults),
    }
    theta = DemoTheta.from_mapping({name: parameter_sources[name].value for name in THETA_PARAMETER_NAMES})
    return ThetaExtractionResult(
        theta=theta,
        parameter_sources=parameter_sources,
        session_dir=session_path,
        trajectory_points_path=trajectory_points_path,
        selected_paths=selected_paths,
        available_segment_labels=list(dict.fromkeys(str(point.get("segment_label")) for point in points if point.get("segment_label") is not None)),
    )


__all__ = [
    "ThetaExtractionResult",
    "extract_theta_from_session",
]
