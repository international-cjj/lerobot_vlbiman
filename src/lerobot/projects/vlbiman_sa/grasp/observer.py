from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

from .contracts import GraspState, Pose6D
from .frame_math import compose_transform, invert_transform, matrix_to_pose6d, pose6d_to_matrix


@dataclass
class ObserverResult:
    state: GraspState
    missing_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": asdict(self.state),
            "missing_fields": list(self.missing_fields),
        }


def _missing(payload: dict[str, Any], field_name: str, default: Any, missing_fields: list[str]) -> Any:
    if field_name not in payload or payload[field_name] is None:
        missing_fields.append(field_name)
        return default
    return payload[field_name]


def _coerce_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric, got {type(value).__name__}.")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite, got {value!r}.")
    return float(value)


def _coerce_int(value: Any, *, field_name: str) -> int:
    numeric = _coerce_number(value, field_name=field_name)
    if not float(numeric).is_integer():
        raise ValueError(f"{field_name} must be an integer-like value, got {numeric}.")
    return int(numeric)


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be bool, got {type(value).__name__}.")
    return value


def _coerce_str(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be str, got {type(value).__name__}.")
    return value


def _coerce_vector(
    value: Any,
    *,
    field_name: str,
    size: int,
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{field_name} must be a list of {size} numeric values.")
    return tuple(_coerce_number(item, field_name=f"{field_name}[{idx}]") for idx, item in enumerate(value))


def _read_pose(payload: dict[str, Any], field_name: str, missing_fields: list[str]) -> Pose6D:
    raw_pose = payload.get(field_name)
    if raw_pose is None:
        missing_fields.append(field_name)
        return Pose6D.zeros()
    if not isinstance(raw_pose, dict):
        raise ValueError(f"{field_name} must be a mapping, got {type(raw_pose).__name__}.")

    xyz = raw_pose.get("xyz")
    if xyz is None:
        missing_fields.append(f"{field_name}.xyz")
        xyz = (0.0, 0.0, 0.0)
    xyz = _coerce_vector(xyz, field_name=f"{field_name}.xyz", size=3)

    rpy = raw_pose.get("rpy")
    if rpy is None:
        missing_fields.append(f"{field_name}.rpy")
        rpy = (0.0, 0.0, 0.0)
    rpy = _coerce_vector(rpy, field_name=f"{field_name}.rpy", size=3)

    return Pose6D(xyz=xyz, rpy=rpy)


def _compute_ee_pose_object(ee_pose_base: Pose6D, object_pose_base: Pose6D) -> Pose6D:
    object_t_base = pose6d_to_matrix(object_pose_base)
    ee_t_base = pose6d_to_matrix(ee_pose_base)
    ee_t_object = compose_transform(invert_transform(object_t_base), ee_t_base)
    return matrix_to_pose6d(ee_t_object)


def build_grasp_state(payload: dict[str, Any], *, default_mode: str = "mock") -> ObserverResult:
    if not isinstance(payload, dict):
        raise ValueError(f"Observer payload must be a mapping, got {type(payload).__name__}.")

    missing_fields: list[str] = []
    ee_pose_base = _read_pose(payload, "ee_pose_base", missing_fields)
    object_pose_base = _read_pose(payload, "object_pose_base", missing_fields)
    ee_pose_object = _compute_ee_pose_object(ee_pose_base, object_pose_base)

    state = GraspState(
        timestamp=_coerce_number(_missing(payload, "timestamp", 0.0, missing_fields), field_name="timestamp"),
        phase=_coerce_str(_missing(payload, "phase", "HANDOFF", missing_fields), field_name="phase"),
        mode=_coerce_str(_missing(payload, "mode", default_mode, missing_fields), field_name="mode"),
        retry_count=_coerce_int(_missing(payload, "retry_count", 0, missing_fields), field_name="retry_count"),
        stable_count=_coerce_int(_missing(payload, "stable_count", 0, missing_fields), field_name="stable_count"),
        phase_elapsed_s=_coerce_number(
            _missing(payload, "phase_elapsed_s", 0.0, missing_fields),
            field_name="phase_elapsed_s",
        ),
        ee_pose_base=ee_pose_base,
        object_pose_base=object_pose_base,
        ee_pose_object=ee_pose_object,
        gripper_width=_coerce_number(
            _missing(payload, "gripper_width", 0.0, missing_fields),
            field_name="gripper_width",
        ),
        gripper_cmd=_coerce_number(_missing(payload, "gripper_cmd", 0.0, missing_fields), field_name="gripper_cmd"),
        gripper_current_proxy=_coerce_number(
            _missing(payload, "gripper_current_proxy", 0.0, missing_fields),
            field_name="gripper_current_proxy",
        ),
        vision_conf=_coerce_number(_missing(payload, "vision_conf", 0.0, missing_fields), field_name="vision_conf"),
        target_visible=_coerce_bool(
            _missing(payload, "target_visible", False, missing_fields),
            field_name="target_visible",
        ),
        corridor_center_px=_coerce_vector(
            _missing(payload, "corridor_center_px", (0.0, 0.0), missing_fields),
            field_name="corridor_center_px",
            size=2,
        ),
        object_center_px=_coerce_vector(
            _missing(payload, "object_center_px", (0.0, 0.0), missing_fields),
            field_name="object_center_px",
            size=2,
        ),
        object_axis_angle=_coerce_number(
            _missing(payload, "object_axis_angle", 0.0, missing_fields),
            field_name="object_axis_angle",
        ),
        object_proj_width_px=_coerce_number(
            _missing(payload, "object_proj_width_px", 0.0, missing_fields),
            field_name="object_proj_width_px",
        ),
        object_proj_height_px=_coerce_number(
            _missing(payload, "object_proj_height_px", 0.0, missing_fields),
            field_name="object_proj_height_px",
        ),
        e_dep=_coerce_number(_missing(payload, "e_dep", 0.0, missing_fields), field_name="e_dep"),
        e_lat=_coerce_number(_missing(payload, "e_lat", 0.0, missing_fields), field_name="e_lat"),
        e_vert=_coerce_number(_missing(payload, "e_vert", 0.0, missing_fields), field_name="e_vert"),
        e_ang=_coerce_number(_missing(payload, "e_ang", 0.0, missing_fields), field_name="e_ang"),
        e_sym=_coerce_number(_missing(payload, "e_sym", 0.0, missing_fields), field_name="e_sym"),
        occ_corridor=_coerce_number(
            _missing(payload, "occ_corridor", 0.0, missing_fields),
            field_name="occ_corridor",
        ),
        drift_obj=_coerce_number(_missing(payload, "drift_obj", 0.0, missing_fields), field_name="drift_obj"),
        object_lift_m=_coerce_number(
            _missing(payload, "object_lift_m", 0.0, missing_fields),
            field_name="object_lift_m",
        ),
        capture_score=_coerce_number(
            _missing(payload, "capture_score", 0.0, missing_fields),
            field_name="capture_score",
        ),
        hold_score=_coerce_number(_missing(payload, "hold_score", 0.0, missing_fields), field_name="hold_score"),
        lift_score=_coerce_number(_missing(payload, "lift_score", 0.0, missing_fields), field_name="lift_score"),
    )
    return ObserverResult(state=state, missing_fields=sorted(set(missing_fields)))


__all__ = ["ObserverResult", "build_grasp_state"]
