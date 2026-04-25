from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..contracts import FRRGGraspConfig, GraspAction, GraspState
from ..scores import capture_lateral_context


def compute_forward_gate(
    state: GraspState,
    config: FRRGGraspConfig,
    *,
    feature_debug_terms: Mapping[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    lateral_context = capture_lateral_context(state, config, feature_debug_terms)
    visible_ready = bool(state.target_visible) and float(state.vision_conf) >= float(config.handoff.handoff_vis_min)
    lateral_ready = abs(float(state.e_lat)) <= float(lateral_context["forward_lateral_tol"])
    angle_ready = abs(float(state.e_ang)) <= float(config.capture_build.forward_enable_ang_tol_rad)
    occupancy_ready = float(state.occ_corridor) >= float(config.capture_build.forward_enable_occ_min)
    forward_gate = visible_ready and lateral_ready and angle_ready and occupancy_ready
    return forward_gate, {
        **lateral_context,
        "visible_ready": visible_ready,
        "lateral_ready": lateral_ready,
        "angle_ready": angle_ready,
        "occupancy_ready": occupancy_ready,
        "forward_gate": forward_gate,
    }


def nominal_capture_action(
    state: GraspState,
    config: FRRGGraspConfig,
    *,
    feature_debug_terms: Mapping[str, Any] | None = None,
) -> GraspAction:
    forward_gate, forward_debug_terms = compute_forward_gate(
        state,
        config,
        feature_debug_terms=feature_debug_terms,
    )
    dx = float(config.capture_build.lat_gain) * float(state.e_lat) + float(config.capture_build.sym_gain) * float(state.e_sym)
    dy = float(config.capture_build.vert_gain) * float(state.e_vert)
    dz = float(config.capture_build.dep_gain) * float(state.e_dep) if forward_gate else 0.0
    dyaw = float(config.capture_build.yaw_gain) * float(state.e_ang)
    return GraspAction(
        delta_pose_object=(dx, dy, dz, 0.0, 0.0, dyaw),
        delta_gripper=0.0,
        debug_terms={
            **forward_debug_terms,
            "primitive": "capture",
            "is_raw_action": True,
            "raw_dx": dx,
            "raw_dy": dy,
            "raw_dz": dz,
            "raw_dyaw": dyaw,
        },
    )


__all__ = [
    "compute_forward_gate",
    "nominal_capture_action",
]
