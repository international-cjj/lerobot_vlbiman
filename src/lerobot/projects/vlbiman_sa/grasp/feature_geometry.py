from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any

from .contracts import FRRGGraspConfig, GraspState
from .frame_math import wrap_to_pi


@dataclass
class FeatureGeometryResult:
    state: GraspState
    debug_terms: dict[str, Any] = field(default_factory=dict)


def compute_lateral_error(
    corridor_center_px: tuple[float, float],
    object_center_px: tuple[float, float],
    *,
    object_depth_m: float | None = None,
    fx_px: float | None = None,
) -> tuple[float, str]:
    lateral_error_px = float(object_center_px[0] - corridor_center_px[0])
    if object_depth_m is None or fx_px is None:
        return lateral_error_px, "px"
    if not math.isfinite(object_depth_m) or not math.isfinite(fx_px) or fx_px <= 0.0:
        raise ValueError("object_depth_m and fx_px must be finite, with fx_px > 0.")
    return (float(object_depth_m) / float(fx_px)) * lateral_error_px, "m"


def compute_depth_error(ee_pose_object_xyz: tuple[float, float, float], *, target_depth_goal_m: float) -> float:
    return float(target_depth_goal_m) - float(ee_pose_object_xyz[2])


def compute_vertical_error(
    ee_pose_object_xyz: tuple[float, float, float],
    *,
    target_vertical_goal_m: float = 0.0,
    enabled: bool = False,
) -> float:
    if not enabled:
        return 0.0
    return float(target_vertical_goal_m) - float(ee_pose_object_xyz[1])


def compute_symmetry_error(distance_left: float, distance_right: float, *, epsilon: float = 1e-6) -> float:
    numerator = float(distance_left) - float(distance_right)
    denominator = float(distance_left) + float(distance_right) + float(epsilon)
    return numerator / denominator


def compute_angle_error(object_axis_angle: float, gripper_axis_angle: float) -> float:
    return wrap_to_pi(float(object_axis_angle) - float(gripper_axis_angle))


def compute_corridor_occupancy(
    object_center_px: tuple[float, float],
    corridor_center_px: tuple[float, float],
    *,
    object_width_px: float,
    corridor_width_px: float | None = None,
    epsilon: float = 1e-6,
) -> float:
    object_width = max(0.0, float(object_width_px))
    corridor_width = object_width if corridor_width_px is None else max(0.0, float(corridor_width_px))
    if object_width <= 0.0:
        return 0.0

    object_half = object_width / 2.0
    corridor_half = corridor_width / 2.0
    object_left = float(object_center_px[0]) - object_half
    object_right = float(object_center_px[0]) + object_half
    corridor_left = float(corridor_center_px[0]) - corridor_half
    corridor_right = float(corridor_center_px[0]) + corridor_half
    overlap = max(0.0, min(object_right, corridor_right) - max(object_left, corridor_left))
    occupancy = overlap / (object_width + float(epsilon))
    return max(0.0, min(1.0, occupancy))


def compute_object_drift(
    object_center_px: tuple[float, float],
    corridor_center_px: tuple[float, float],
    *,
    previous_object_center_px: tuple[float, float] | None = None,
    previous_corridor_center_px: tuple[float, float] | None = None,
) -> float:
    if previous_object_center_px is None or previous_corridor_center_px is None:
        return 0.0
    current_dx = float(object_center_px[0]) - float(corridor_center_px[0])
    current_dy = float(object_center_px[1]) - float(corridor_center_px[1])
    previous_dx = float(previous_object_center_px[0]) - float(previous_corridor_center_px[0])
    previous_dy = float(previous_object_center_px[1]) - float(previous_corridor_center_px[1])
    return math.hypot(current_dx - previous_dx, current_dy - previous_dy)


def _edge_distances_from_projection(
    object_center_px: tuple[float, float],
    corridor_center_px: tuple[float, float],
    object_width_px: float,
) -> tuple[float, float]:
    half_width = max(0.0, float(object_width_px)) / 2.0
    object_left = float(object_center_px[0]) - half_width
    object_right = float(object_center_px[0]) + half_width
    corridor_center_x = float(corridor_center_px[0])
    distance_left = abs(corridor_center_x - object_left)
    distance_right = abs(object_right - corridor_center_x)
    return distance_left, distance_right


def apply_feature_geometry(
    state: GraspState,
    config: FRRGGraspConfig,
    *,
    previous_state: GraspState | None = None,
    object_depth_m: float | None = None,
    fx_px: float | None = None,
    target_vertical_goal_m: float = 0.0,
    vertical_enabled: bool = False,
    corridor_width_px: float | None = None,
) -> FeatureGeometryResult:
    lateral_error, lateral_unit = compute_lateral_error(
        state.corridor_center_px,
        state.object_center_px,
        object_depth_m=object_depth_m,
        fx_px=fx_px,
    )
    depth_error = compute_depth_error(
        state.ee_pose_object.xyz,
        target_depth_goal_m=config.capture_build.target_depth_goal_m,
    )
    vertical_error = compute_vertical_error(
        state.ee_pose_object.xyz,
        target_vertical_goal_m=target_vertical_goal_m,
        enabled=vertical_enabled,
    )
    distance_left, distance_right = _edge_distances_from_projection(
        state.object_center_px,
        state.corridor_center_px,
        state.object_proj_width_px,
    )
    symmetry_error = compute_symmetry_error(distance_left, distance_right)
    angle_error = compute_angle_error(state.object_axis_angle, state.ee_pose_object.rpy[2])
    corridor_occupancy = compute_corridor_occupancy(
        state.object_center_px,
        state.corridor_center_px,
        object_width_px=state.object_proj_width_px,
        corridor_width_px=corridor_width_px,
    )
    drift = compute_object_drift(
        state.object_center_px,
        state.corridor_center_px,
        previous_object_center_px=previous_state.object_center_px if previous_state else None,
        previous_corridor_center_px=previous_state.corridor_center_px if previous_state else None,
    )

    updated_state = replace(
        state,
        e_dep=depth_error,
        e_lat=lateral_error,
        e_vert=vertical_error,
        e_ang=angle_error,
        e_sym=symmetry_error,
        occ_corridor=corridor_occupancy,
        drift_obj=drift,
    )
    debug_terms = {
        "lateral_error_unit": lateral_unit,
        "lateral_error_px_raw": float(state.object_center_px[0] - state.corridor_center_px[0]),
        "depth_target_m": config.capture_build.target_depth_goal_m,
        "vertical_enabled": bool(vertical_enabled),
        "e_vert_disabled": 0 if vertical_enabled else 1,
        "symmetry_distance_left": distance_left,
        "symmetry_distance_right": distance_right,
        "corridor_width_px": float(state.object_proj_width_px if corridor_width_px is None else corridor_width_px),
        "object_width_px": float(state.object_proj_width_px),
        "object_height_px": float(state.object_proj_height_px),
        "drift_unit": "px",
    }
    if lateral_unit == "m":
        debug_terms["lateral_scale_object_depth_m"] = float(object_depth_m)
        debug_terms["lateral_scale_fx_px"] = float(fx_px)
    return FeatureGeometryResult(state=updated_state, debug_terms=debug_terms)


__all__ = [
    "FeatureGeometryResult",
    "apply_feature_geometry",
    "compute_angle_error",
    "compute_corridor_occupancy",
    "compute_depth_error",
    "compute_lateral_error",
    "compute_object_drift",
    "compute_symmetry_error",
    "compute_vertical_error",
]
