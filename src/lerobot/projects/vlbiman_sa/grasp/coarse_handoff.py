from __future__ import annotations

from typing import Any

from .state_machine import PHASE_HANDOFF


COARSE_HANDOFF_SOURCE = "coarse_python"
REQUIRED_COARSE_FIELDS = (
    "target_pose_base",
    "pregrasp_pose_base",
    "gripper_initial_width",
    "vision_summary.target_visible",
    "vision_summary.vision_conf",
    "vision_summary.corridor_center_px",
    "vision_summary.object_center_px",
    "vision_summary.object_axis_angle",
    "vision_summary.object_proj_width_px",
    "vision_summary.object_proj_height_px",
)


class CoarseHandoffError(ValueError):
    def __init__(self, reason: str, message: str, *, missing_fields: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.reason = reason
        self.missing_fields = missing_fields


def _require_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CoarseHandoffError(
            "invalid_coarse_field",
            f"{field_name} must be a mapping, got {type(value).__name__}.",
        )
    return value


def _require_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CoarseHandoffError(
            "invalid_coarse_field",
            f"{field_name} must be numeric, got {type(value).__name__}.",
        )
    return float(value)


def _require_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise CoarseHandoffError(
            "invalid_coarse_field",
            f"{field_name} must be bool, got {type(value).__name__}.",
        )
    return value


def _require_vector(value: Any, *, field_name: str, size: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise CoarseHandoffError(
            "invalid_coarse_field",
            f"{field_name} must be a list of {size} numeric values.",
        )
    return [_require_number(item, field_name=f"{field_name}[{idx}]") for idx, item in enumerate(value)]


def _require_field(payload: dict[str, Any], field_name: str, *, key: str | None = None) -> Any:
    resolved_key = key or field_name
    if resolved_key not in payload or payload[resolved_key] is None:
        raise CoarseHandoffError(
            "missing_coarse_field",
            f"Missing required coarse field: {field_name}",
            missing_fields=(field_name,),
        )
    return payload[resolved_key]


def _require_pose(payload: dict[str, Any], field_name: str) -> dict[str, list[float]]:
    pose = _require_mapping(_require_field(payload, field_name), field_name=field_name)
    xyz = _require_vector(_require_field(pose, f"{field_name}.xyz", key="xyz"), field_name=f"{field_name}.xyz", size=3)
    rpy = _require_vector(_require_field(pose, f"{field_name}.rpy", key="rpy"), field_name=f"{field_name}.rpy", size=3)
    return {"xyz": xyz, "rpy": rpy}


def build_coarse_input_summary(coarse_summary: dict[str, Any]) -> dict[str, Any]:
    vision_summary = coarse_summary.get("vision_summary")
    return {
        "keys": sorted(coarse_summary.keys()),
        "vision_keys": sorted(vision_summary.keys()) if isinstance(vision_summary, dict) else [],
        "timestamp": coarse_summary.get("timestamp"),
        "gripper_initial_width": coarse_summary.get("gripper_initial_width"),
    }


def build_frrg_input_from_coarse_summary(coarse_summary: dict[str, Any]) -> dict[str, Any]:
    coarse_summary = _require_mapping(coarse_summary, field_name="coarse_summary")
    target_pose_base = _require_pose(coarse_summary, "target_pose_base")
    pregrasp_pose_base = _require_pose(coarse_summary, "pregrasp_pose_base")
    gripper_initial_width = _require_number(
        _require_field(coarse_summary, "gripper_initial_width"),
        field_name="gripper_initial_width",
    )
    vision_summary = _require_mapping(_require_field(coarse_summary, "vision_summary"), field_name="vision_summary")

    target_visible = _require_bool(
        _require_field(vision_summary, "vision_summary.target_visible", key="target_visible"),
        field_name="vision_summary.target_visible",
    )
    vision_conf = _require_number(
        _require_field(vision_summary, "vision_summary.vision_conf", key="vision_conf"),
        field_name="vision_summary.vision_conf",
    )
    corridor_center_px = _require_vector(
        _require_field(vision_summary, "vision_summary.corridor_center_px", key="corridor_center_px"),
        field_name="vision_summary.corridor_center_px",
        size=2,
    )
    object_center_px = _require_vector(
        _require_field(vision_summary, "vision_summary.object_center_px", key="object_center_px"),
        field_name="vision_summary.object_center_px",
        size=2,
    )
    object_axis_angle = _require_number(
        _require_field(vision_summary, "vision_summary.object_axis_angle", key="object_axis_angle"),
        field_name="vision_summary.object_axis_angle",
    )
    object_proj_width_px = _require_number(
        _require_field(vision_summary, "vision_summary.object_proj_width_px", key="object_proj_width_px"),
        field_name="vision_summary.object_proj_width_px",
    )
    object_proj_height_px = _require_number(
        _require_field(vision_summary, "vision_summary.object_proj_height_px", key="object_proj_height_px"),
        field_name="vision_summary.object_proj_height_px",
    )

    timestamp = _require_number(coarse_summary.get("timestamp", 0.0), field_name="timestamp")
    gripper_current_proxy = _require_number(
        coarse_summary.get("gripper_current_proxy", 0.0),
        field_name="gripper_current_proxy",
    )

    return {
        "timestamp": timestamp,
        "phase": PHASE_HANDOFF,
        "mode": COARSE_HANDOFF_SOURCE,
        "retry_count": 0,
        "stable_count": 0,
        "phase_elapsed_s": 0.0,
        "ee_pose_base": pregrasp_pose_base,
        "object_pose_base": target_pose_base,
        "gripper_width": gripper_initial_width,
        "gripper_cmd": gripper_initial_width,
        "gripper_current_proxy": gripper_current_proxy,
        "vision_conf": vision_conf,
        "target_visible": target_visible,
        "corridor_center_px": corridor_center_px,
        "object_center_px": object_center_px,
        "object_axis_angle": object_axis_angle,
        "object_proj_width_px": object_proj_width_px,
        "object_proj_height_px": object_proj_height_px,
        "e_dep": 0.0,
        "e_lat": 0.0,
        "e_vert": 0.0,
        "e_ang": 0.0,
        "e_sym": 0.0,
        "occ_corridor": 0.0,
        "drift_obj": 0.0,
        "capture_score": 0.0,
        "hold_score": 0.0,
        "lift_score": 0.0,
    }


__all__ = [
    "COARSE_HANDOFF_SOURCE",
    "CoarseHandoffError",
    "REQUIRED_COARSE_FIELDS",
    "build_coarse_input_summary",
    "build_frrg_input_from_coarse_summary",
]
