from __future__ import annotations

import math
from pathlib import Path

from lerobot.projects.vlbiman_sa.grasp.contracts import load_frrg_config
from lerobot.projects.vlbiman_sa.grasp.feature_geometry import (
    apply_feature_geometry,
    compute_angle_error,
    compute_corridor_occupancy,
    compute_lateral_error,
    compute_object_drift,
    compute_symmetry_error,
)
from lerobot.projects.vlbiman_sa.grasp.observer import build_grasp_state


def test_compute_lateral_error_preserves_px_when_scale_is_missing():
    error, unit = compute_lateral_error((320.0, 240.0), (326.0, 238.0))

    assert error == 6.0
    assert unit == "px"


def test_compute_lateral_error_uses_metric_scale_when_available():
    error, unit = compute_lateral_error(
        (320.0, 240.0),
        (330.0, 240.0),
        object_depth_m=0.5,
        fx_px=500.0,
    )

    assert math.isclose(error, 0.01, abs_tol=1e-9)
    assert unit == "m"


def test_compute_symmetry_error_sign_matches_left_right_bias():
    assert compute_symmetry_error(7.0, 3.0) > 0.0
    assert compute_symmetry_error(3.0, 7.0) < 0.0


def test_compute_angle_error_wraps_across_pi_boundary():
    error = compute_angle_error(-math.pi + 0.1, math.pi - 0.1)

    assert math.isclose(error, 0.2, abs_tol=1e-9)


def test_apply_feature_geometry_updates_state_with_finite_values():
    config = load_frrg_config(Path("src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml"))
    current = build_grasp_state(
        {
            "timestamp": 1.0,
            "phase": "HANDOFF",
            "mode": "mock",
            "ee_pose_base": {"xyz": [0.42, 0.01, 0.18], "rpy": [0.0, 0.0, 0.04]},
            "object_pose_base": {"xyz": [0.44, 0.0, 0.12], "rpy": [0.0, 0.0, 0.0]},
            "target_visible": True,
            "vision_conf": 0.88,
            "corridor_center_px": [320.0, 240.0],
            "object_center_px": [326.0, 238.0],
            "object_axis_angle": -math.pi + 0.02,
            "object_proj_width_px": 88.0,
            "object_proj_height_px": 126.0,
            "gripper_width": 0.06,
        }
    ).state
    previous = build_grasp_state(
        {
            "timestamp": 0.5,
            "phase": "HANDOFF",
            "mode": "mock",
            "ee_pose_base": {"xyz": [0.42, 0.01, 0.18], "rpy": [0.0, 0.0, 0.04]},
            "object_pose_base": {"xyz": [0.44, 0.0, 0.12], "rpy": [0.0, 0.0, 0.0]},
            "corridor_center_px": [320.0, 240.0],
            "object_center_px": [324.0, 239.0],
            "object_axis_angle": 0.0,
            "object_proj_width_px": 88.0,
            "object_proj_height_px": 126.0,
        }
    ).state

    result = apply_feature_geometry(current, config, previous_state=previous)

    assert all(
        math.isfinite(value)
        for value in (
            result.state.e_lat,
            result.state.e_dep,
            result.state.e_vert,
            result.state.e_ang,
            result.state.e_sym,
            result.state.occ_corridor,
            result.state.drift_obj,
        )
    )
    assert result.debug_terms["lateral_error_unit"] == "px"
    assert result.debug_terms["e_vert_disabled"] == 1
    assert result.state.e_lat == 6.0
    assert math.isclose(result.state.e_dep, 0.012 - 0.06, abs_tol=1e-9)
    assert math.isclose(result.state.e_ang, wrap_like(-math.pi + 0.02 - 0.04), abs_tol=1e-9)
    assert result.state.e_sym < 0.0
    assert 0.0 <= result.state.occ_corridor <= 1.0
    assert math.isclose(result.state.drift_obj, math.hypot(2.0, -1.0), abs_tol=1e-9)


def test_compute_corridor_occupancy_and_drift_are_deterministic():
    occupancy = compute_corridor_occupancy(
        (330.0, 240.0),
        (320.0, 240.0),
        object_width_px=20.0,
        corridor_width_px=20.0,
    )
    drift = compute_object_drift(
        (330.0, 241.0),
        (320.0, 240.0),
        previous_object_center_px=(327.0, 240.0),
        previous_corridor_center_px=(320.0, 240.0),
    )

    assert math.isclose(occupancy, 0.5, abs_tol=1e-6)
    assert math.isclose(drift, math.hypot(3.0, 1.0), abs_tol=1e-9)


def wrap_like(angle: float) -> float:
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    if math.isclose(wrapped, -math.pi, abs_tol=1e-12) and angle > 0.0:
        return math.pi
    return wrapped
