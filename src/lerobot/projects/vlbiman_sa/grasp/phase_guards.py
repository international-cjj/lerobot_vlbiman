from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .contracts import FRRGGraspConfig, GraspState, GuardResult
from .scores import compute_capture_score, compute_hold_score, compute_lift_score


def handoff_guard(
    state: GraspState,
    config: FRRGGraspConfig,
    *,
    feature_debug_terms: Mapping[str, Any] | None = None,
) -> GuardResult:
    capture_score, capture_debug = compute_capture_score(
        state,
        config,
        feature_debug_terms=feature_debug_terms,
    )
    visible_ready = bool(state.target_visible) and float(state.vision_conf) >= float(config.handoff.handoff_vis_min)
    gripper_ready = float(state.gripper_width) >= float(config.handoff.handoff_open_width_m)
    yaw_ready = abs(float(state.e_ang)) <= float(config.handoff.handoff_yaw_tol_rad)
    lateral_ready = abs(float(state.e_lat)) <= float(capture_debug["handoff_lateral_tol"])
    passed = visible_ready and gripper_ready and yaw_ready and lateral_ready

    reason = None
    if not visible_ready:
        reason = "vision_lost"
    elif not passed:
        reason = "corridor_not_formed"

    score = 0.25 * float(visible_ready) + 0.25 * float(gripper_ready) + 0.25 * float(yaw_ready) + 0.25 * float(
        lateral_ready
    )
    return GuardResult(
        passed=passed,
        score=score,
        reason=reason,
        debug_terms={
            "capture_score_snapshot": capture_score,
            "visible_ready": visible_ready,
            "gripper_ready": gripper_ready,
            "yaw_ready": yaw_ready,
            "lateral_ready": lateral_ready,
            **capture_debug,
        },
    )


def capture_to_close_guard(
    state: GraspState,
    config: FRRGGraspConfig,
    *,
    feature_debug_terms: Mapping[str, Any] | None = None,
) -> GuardResult:
    debug_terms = dict(feature_debug_terms or {})
    capture_score, capture_debug = compute_capture_score(
        state,
        config,
        feature_debug_terms=feature_debug_terms,
    )
    visible_ready = bool(state.target_visible) and float(state.vision_conf) >= float(config.handoff.handoff_vis_min)
    lateral_ready = abs(float(state.e_lat)) <= float(capture_debug["forward_lateral_tol"])
    angle_ready = abs(float(state.e_ang)) <= float(config.capture_build.forward_enable_ang_tol_rad)
    occupancy_ready = float(state.occ_corridor) >= float(config.capture_build.forward_enable_occ_min)
    depth_ready = float(state.ee_pose_object.xyz[2]) <= float(config.capture_build.target_depth_max_m)
    vertical_enabled = bool(debug_terms.get("vertical_enabled", False))
    vertical_tol_m = float(debug_terms.get("vertical_tol_m", config.handoff.handoff_pos_tol_m))
    vertical_ready = (not vertical_enabled) or abs(float(state.e_vert)) <= float(vertical_tol_m)
    forward_gate = visible_ready and lateral_ready and angle_ready and occupancy_ready
    stable_ready = int(state.stable_count) >= int(config.runtime.stable_window_frames)
    score_ready = float(capture_score) >= float(config.capture_build.close_score_threshold)
    timeout_reached = float(state.phase_elapsed_s) >= float(config.capture_build.capture_timeout_s)
    passed = forward_gate and depth_ready and vertical_ready and stable_ready and score_ready and not timeout_reached

    reason = None
    if not visible_ready:
        reason = "vision_lost"
    elif timeout_reached:
        reason = "capture_timeout"
    elif not depth_ready:
        reason = "depth_not_ready"
    elif not vertical_ready:
        reason = "vertical_not_ready"
    elif not passed:
        reason = "corridor_not_formed"

    return GuardResult(
        passed=passed,
        score=float(capture_score),
        reason=reason,
        debug_terms={
            **capture_debug,
            "forward_gate": forward_gate,
            "visible_ready": visible_ready,
            "lateral_ready": lateral_ready,
            "angle_ready": angle_ready,
            "occupancy_ready": occupancy_ready,
            "depth_ready": depth_ready,
            "ee_depth_m": float(state.ee_pose_object.xyz[2]),
            "target_depth_max_m": float(config.capture_build.target_depth_max_m),
            "vertical_enabled": vertical_enabled,
            "vertical_ready": vertical_ready,
            "vertical_tol_m": vertical_tol_m,
            "e_vert_m": float(state.e_vert),
            "stable_ready": stable_ready,
            "score_ready": score_ready,
            "timeout_reached": timeout_reached,
            "stable_count": int(state.stable_count),
            "required_stable_count": int(config.runtime.stable_window_frames),
        },
    )


def close_to_lift_guard(
    state: GraspState,
    config: FRRGGraspConfig,
    *,
    feature_debug_terms: Mapping[str, Any] | None = None,
    contact_current_available: bool = True,
) -> GuardResult:
    hold_score, hold_debug = compute_hold_score(
        state,
        config,
        feature_debug_terms=feature_debug_terms,
        contact_current_available=contact_current_available,
    )
    visible_ready = bool(state.target_visible) and float(state.vision_conf) >= float(config.handoff.handoff_vis_min)
    contact_ready = bool(hold_debug["contact_detected"])
    drift_ready = abs(float(state.drift_obj)) <= float(hold_debug["hold_drift_tol"])
    stable_ready = int(state.stable_count) >= int(config.runtime.stable_window_frames)
    score_ready = float(hold_score) >= float(config.close_hold.hold_score_threshold)
    passed = visible_ready and contact_ready and drift_ready and stable_ready and score_ready

    reason = None
    if not visible_ready:
        reason = "vision_lost"
    elif not drift_ready:
        reason = "large_drift"
    elif not contact_ready:
        reason = "contact_not_detected"
    elif not passed:
        reason = "contact_not_detected"

    return GuardResult(
        passed=passed,
        score=float(hold_score),
        reason=reason,
        debug_terms={
            **hold_debug,
            "visible_ready": visible_ready,
            "contact_ready": contact_ready,
            "drift_ready": drift_ready,
            "stable_ready": stable_ready,
            "score_ready": score_ready,
            "stable_count": int(state.stable_count),
            "required_stable_count": int(config.runtime.stable_window_frames),
        },
    )


def lift_to_success_guard(
    state: GraspState,
    config: FRRGGraspConfig,
    *,
    feature_debug_terms: Mapping[str, Any] | None = None,
    contact_current_available: bool = True,
) -> GuardResult:
    lift_score, lift_debug = compute_lift_score(
        state,
        config,
        feature_debug_terms=feature_debug_terms,
        contact_current_available=contact_current_available,
    )
    visible_ready = bool(state.target_visible) and float(state.vision_conf) >= float(config.handoff.handoff_vis_min)
    contact_ready = bool(lift_debug["contact_detected"])
    slip_free = abs(float(state.drift_obj)) <= float(lift_debug["slip_drift_tol"])
    required_lift_height = max(float(config.lift_test.lift_height_m) * 0.5, 0.005)
    lift_height_ready = float(state.object_lift_m) >= required_lift_height
    stable_ready = int(state.stable_count) >= int(config.runtime.stable_window_frames)
    score_ready = float(lift_score) >= float(config.lift_test.lift_score_threshold)
    passed = visible_ready and contact_ready and slip_free and lift_height_ready and stable_ready and score_ready

    reason = None
    if not visible_ready:
        reason = "vision_lost"
    elif not slip_free:
        reason = "slip_detected"
    elif not contact_ready:
        reason = "contact_not_detected"
    elif not lift_height_ready:
        reason = "lift_height_not_reached"
    elif not passed:
        reason = "slip_detected"

    return GuardResult(
        passed=passed,
        score=float(lift_score),
        reason=reason,
        debug_terms={
            **lift_debug,
            "visible_ready": visible_ready,
            "contact_ready": contact_ready,
            "slip_free": slip_free,
            "lift_height_ready": lift_height_ready,
            "object_lift_m": float(state.object_lift_m),
            "required_lift_height_m": float(required_lift_height),
            "stable_ready": stable_ready,
            "score_ready": score_ready,
            "stable_count": int(state.stable_count),
            "required_stable_count": int(config.runtime.stable_window_frames),
        },
    )


def vision_lost_guard(state: GraspState, config: FRRGGraspConfig) -> GuardResult:
    visible = bool(state.target_visible)
    confidence_ok = float(state.vision_conf) >= float(config.safety.vision_hardstop_min)
    triggered = (not visible) or (not confidence_ok)
    score = min(float(state.vision_conf) / max(float(config.safety.vision_hardstop_min), 1e-9), 1.0)
    return GuardResult(
        passed=triggered,
        score=score,
        reason="vision_lost" if triggered else None,
        debug_terms={
            "target_visible": visible,
            "vision_conf": float(state.vision_conf),
            "vision_hardstop_min": float(config.safety.vision_hardstop_min),
            "confidence_ok": confidence_ok,
        },
    )


def capture_timeout_guard(state: GraspState, config: FRRGGraspConfig) -> GuardResult:
    timeout_reached = float(state.phase_elapsed_s) >= float(config.capture_build.capture_timeout_s)
    return GuardResult(
        passed=timeout_reached,
        score=float(state.phase_elapsed_s) / float(config.capture_build.capture_timeout_s),
        reason="capture_timeout" if timeout_reached else None,
        debug_terms={
            "phase": state.phase,
            "phase_elapsed_s": float(state.phase_elapsed_s),
            "capture_timeout_s": float(config.capture_build.capture_timeout_s),
        },
    )


def slip_detected_guard(
    state: GraspState,
    config: FRRGGraspConfig,
    *,
    feature_debug_terms: Mapping[str, Any] | None = None,
) -> GuardResult:
    lift_score, lift_debug = compute_lift_score(
        state,
        config,
        feature_debug_terms=feature_debug_terms,
        contact_current_available=True,
    )
    slip_detected = abs(float(state.drift_obj)) > float(lift_debug["slip_drift_tol"])
    return GuardResult(
        passed=slip_detected,
        score=float(lift_score),
        reason="slip_detected" if slip_detected else None,
        debug_terms={
            **lift_debug,
            "slip_detected": slip_detected,
        },
    )


__all__ = [
    "capture_timeout_guard",
    "capture_to_close_guard",
    "close_to_lift_guard",
    "handoff_guard",
    "lift_to_success_guard",
    "slip_detected_guard",
    "vision_lost_guard",
]
