from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import math
from typing import Any

from .contracts import FRRGGraspConfig, GraspState


@dataclass
class ScoreResult:
    state: GraspState
    debug_terms: dict[str, Any] = field(default_factory=dict)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _gaussian_term(error: float, sigma: float) -> float:
    sigma_value = max(float(sigma), 1e-9)
    normalized = float(error) / sigma_value
    return math.exp(-(normalized * normalized))


def _feature_terms(feature_debug_terms: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(feature_debug_terms or {})


def capture_lateral_context(
    state: GraspState,
    config: FRRGGraspConfig,
    feature_debug_terms: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    debug_terms = _feature_terms(feature_debug_terms)
    lateral_unit = str(debug_terms.get("lateral_error_unit", "m"))
    object_width_px = max(0.0, float(debug_terms.get("object_width_px", state.object_proj_width_px)))
    if lateral_unit == "px":
        handoff_tol = max(1.0, object_width_px * 0.10)
        forward_tol = max(1.0, object_width_px * 0.05)
        score_sigma = max(handoff_tol, 1.0)
    else:
        handoff_tol = float(config.handoff.handoff_pos_tol_m)
        forward_tol = float(config.capture_build.forward_enable_lat_tol_m)
        score_sigma = max(handoff_tol, forward_tol, 1e-9)

    return {
        "lateral_error_unit": lateral_unit,
        "object_width_px": object_width_px,
        "handoff_lateral_tol": handoff_tol,
        "forward_lateral_tol": forward_tol,
        "capture_lateral_sigma": score_sigma,
    }


def _drift_context(
    state: GraspState,
    config: FRRGGraspConfig,
    feature_debug_terms: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    debug_terms = _feature_terms(feature_debug_terms)
    lateral_unit = str(debug_terms.get("lateral_error_unit", "m"))
    drift_unit = str(debug_terms.get("drift_unit", "px" if lateral_unit == "px" else "m"))
    object_width_px = max(0.0, float(debug_terms.get("object_width_px", state.object_proj_width_px)))
    if drift_unit == "px":
        hold_tol = max(1.0, object_width_px * 0.03)
        slip_tol = max(hold_tol, object_width_px * 0.04)
        hold_sigma = max(hold_tol, 1.0)
        slip_sigma = max(slip_tol, 1.0)
    else:
        hold_tol = float(config.close_hold.hold_drift_max_m)
        slip_tol = float(config.lift_test.slip_threshold_m)
        hold_sigma = max(hold_tol, 1e-9)
        slip_sigma = max(slip_tol, hold_sigma, 1e-9)

    return {
        "drift_unit": drift_unit,
        "object_width_px": object_width_px,
        "hold_drift_tol": hold_tol,
        "slip_drift_tol": slip_tol,
        "hold_drift_sigma": hold_sigma,
        "slip_drift_sigma": slip_sigma,
    }


def infer_contact_proxy(
    state: GraspState,
    config: FRRGGraspConfig,
    *,
    contact_current_available: bool = True,
) -> tuple[bool, dict[str, Any]]:
    close_width_trigger = max(
        float(config.close_hold.close_width_target_m),
        float(config.handoff.handoff_open_width_m) - float(config.close_hold.preclose_distance_m),
    )
    width_closed_enough = float(state.gripper_width) <= close_width_trigger
    current_in_range = (
        contact_current_available
        and float(config.close_hold.contact_current_min)
        <= float(state.gripper_current_proxy)
        <= float(config.close_hold.contact_current_max)
    )
    contact_detected = width_closed_enough and current_in_range
    debug_terms = {
        "contact_detected": contact_detected,
        "contact_current_unavailable": not contact_current_available,
        "contact_width_trigger_m": close_width_trigger,
        "width_closed_enough": width_closed_enough,
        "gripper_current_proxy": float(state.gripper_current_proxy),
        "contact_current_min": float(config.close_hold.contact_current_min),
        "contact_current_max": float(config.close_hold.contact_current_max),
        "current_in_range": current_in_range,
    }
    return contact_detected, debug_terms


def compute_capture_score(
    state: GraspState,
    config: FRRGGraspConfig,
    *,
    feature_debug_terms: Mapping[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    lateral_context = capture_lateral_context(state, config, feature_debug_terms)
    angle_sigma = max(
        float(config.handoff.handoff_yaw_tol_rad),
        float(config.capture_build.forward_enable_ang_tol_rad),
        1e-9,
    )
    symmetry_sigma = max(0.35 / max(config.feature_geometry.angle_symmetry_weight_default, 1e-6), 1e-6)
    visibility_term = _clamp01(float(state.vision_conf)) if state.target_visible else 0.0
    lateral_term = _gaussian_term(abs(float(state.e_lat)), float(lateral_context["capture_lateral_sigma"]))
    angle_term = _gaussian_term(abs(float(state.e_ang)), angle_sigma)
    symmetry_term = _gaussian_term(
        abs(float(state.e_sym) * float(config.feature_geometry.angle_symmetry_weight_default)),
        symmetry_sigma,
    )
    occupancy_term = _clamp01(float(state.occ_corridor))

    score = (
        0.25 * lateral_term
        + 0.20 * angle_term
        + 0.25 * occupancy_term
        + 0.15 * symmetry_term
        + 0.15 * visibility_term
    )
    debug_terms = {
        **lateral_context,
        "lateral_term": lateral_term,
        "angle_term": angle_term,
        "occupancy_term": occupancy_term,
        "symmetry_term": symmetry_term,
        "visibility_term": visibility_term,
        "angle_sigma": angle_sigma,
        "symmetry_sigma": symmetry_sigma,
        "score_weights": {
            "lateral": 0.25,
            "angle": 0.20,
            "occupancy": 0.25,
            "symmetry": 0.15,
            "visibility": 0.15,
        },
    }
    return score, debug_terms


def compute_hold_score(
    state: GraspState,
    config: FRRGGraspConfig,
    *,
    feature_debug_terms: Mapping[str, Any] | None = None,
    contact_current_available: bool = True,
) -> tuple[float, dict[str, Any]]:
    drift_context = _drift_context(state, config, feature_debug_terms)
    contact_detected, contact_debug = infer_contact_proxy(
        state,
        config,
        contact_current_available=contact_current_available,
    )
    hold_width_min = float(config.close_hold.close_width_target_m)
    hold_width_max = max(
        hold_width_min,
        float(config.handoff.handoff_open_width_m) - float(config.close_hold.preclose_distance_m),
    )
    width_in_hold_band = hold_width_min <= float(state.gripper_width) <= hold_width_max
    drift_term = _gaussian_term(abs(float(state.drift_obj)), float(drift_context["hold_drift_sigma"]))
    visibility_term = _clamp01(float(state.vision_conf)) if state.target_visible else 0.0

    score = (
        0.40 * (1.0 if contact_detected else 0.0)
        + 0.25 * drift_term
        + 0.20 * (1.0 if width_in_hold_band else 0.0)
        + 0.15 * visibility_term
    )
    debug_terms = {
        **drift_context,
        **contact_debug,
        "drift_term": drift_term,
        "visibility_term": visibility_term,
        "hold_width_min_m": hold_width_min,
        "hold_width_max_m": hold_width_max,
        "width_in_hold_band": width_in_hold_band,
        "score_weights": {
            "contact": 0.40,
            "drift": 0.25,
            "width_band": 0.20,
            "visibility": 0.15,
        },
    }
    return score, debug_terms


def compute_lift_score(
    state: GraspState,
    config: FRRGGraspConfig,
    *,
    feature_debug_terms: Mapping[str, Any] | None = None,
    contact_current_available: bool = True,
) -> tuple[float, dict[str, Any]]:
    drift_context = _drift_context(state, config, feature_debug_terms)
    contact_detected, contact_debug = infer_contact_proxy(
        state,
        config,
        contact_current_available=contact_current_available,
    )
    slip_term = _gaussian_term(abs(float(state.drift_obj)), float(drift_context["slip_drift_sigma"]))
    visibility_term = _clamp01(float(state.vision_conf)) if state.target_visible else 0.0

    score = 0.50 * slip_term + 0.20 * visibility_term + 0.30 * (1.0 if contact_detected else 0.0)
    debug_terms = {
        **drift_context,
        **contact_debug,
        "slip_term": slip_term,
        "visibility_term": visibility_term,
        "score_weights": {
            "slip": 0.50,
            "visibility": 0.20,
            "contact": 0.30,
        },
    }
    return score, debug_terms


def apply_scores(
    state: GraspState,
    config: FRRGGraspConfig,
    *,
    feature_debug_terms: Mapping[str, Any] | None = None,
    contact_current_available: bool = True,
) -> ScoreResult:
    capture_score, capture_debug_terms = compute_capture_score(
        state,
        config,
        feature_debug_terms=feature_debug_terms,
    )
    hold_score, hold_debug_terms = compute_hold_score(
        state,
        config,
        feature_debug_terms=feature_debug_terms,
        contact_current_available=contact_current_available,
    )
    lift_score, lift_debug_terms = compute_lift_score(
        state,
        config,
        feature_debug_terms=feature_debug_terms,
        contact_current_available=contact_current_available,
    )
    updated_state = replace(
        state,
        capture_score=capture_score,
        hold_score=hold_score,
        lift_score=lift_score,
    )
    return ScoreResult(
        state=updated_state,
        debug_terms={
            "capture": capture_debug_terms,
            "hold": hold_debug_terms,
            "lift": lift_debug_terms,
        },
    )


__all__ = [
    "ScoreResult",
    "apply_scores",
    "capture_lateral_context",
    "compute_capture_score",
    "compute_hold_score",
    "compute_lift_score",
    "infer_contact_proxy",
]
