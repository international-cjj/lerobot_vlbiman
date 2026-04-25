from __future__ import annotations

import math

import numpy as np

from lerobot.projects.vlbiman_sa.grasp.observer import build_grasp_state


def test_build_grasp_state_from_nominal_mock_reads_all_fields():
    payload = {
        "timestamp": 0.0,
        "phase": "HANDOFF",
        "mode": "mock",
        "retry_count": 0,
        "stable_count": 0,
        "phase_elapsed_s": 0.0,
        "ee_pose_base": {"xyz": [0.42, 0.01, 0.18], "rpy": [0.0, 0.0, 0.04]},
        "object_pose_base": {"xyz": [0.44, 0.0, 0.12], "rpy": [0.0, 0.0, 0.0]},
        "gripper_width": 0.06,
        "gripper_cmd": 0.06,
        "gripper_current_proxy": 0.0,
        "vision_conf": 0.88,
        "target_visible": True,
        "corridor_center_px": [320.0, 240.0],
        "object_center_px": [326.0, 238.0],
        "object_axis_angle": 0.03,
        "object_proj_width_px": 88.0,
        "object_proj_height_px": 126.0,
        "e_dep": 0.014,
        "e_lat": 0.002,
        "e_vert": 0.001,
        "e_ang": 0.03,
        "e_sym": 0.0,
        "occ_corridor": 0.61,
        "drift_obj": 0.0,
        "capture_score": 0.0,
        "hold_score": 0.0,
        "lift_score": 0.0,
    }

    result = build_grasp_state(payload)

    assert result.missing_fields == []
    assert result.state.phase == "HANDOFF"
    assert result.state.mode == "mock"
    assert np.allclose(result.state.ee_pose_object.xyz, (-0.020000000000000018, 0.01, 0.06), atol=1e-9)
    assert np.allclose(result.state.ee_pose_object.rpy, (0.0, 0.0, 0.04), atol=1e-9)


def test_build_grasp_state_defaults_missing_fields_and_tracks_them():
    payload = {
        "ee_pose_base": {"xyz": [0.5, -0.1, 0.2], "rpy": [0.0, 0.0, 0.0]},
        "object_pose_base": {"xyz": [0.4, -0.2, 0.05], "rpy": [0.0, 0.0, 0.1]},
    }

    result = build_grasp_state(payload, default_mode="mock")

    assert "timestamp" in result.missing_fields
    assert "gripper_width" in result.missing_fields
    assert "target_visible" in result.missing_fields
    assert result.state.phase == "HANDOFF"
    assert result.state.mode == "mock"
    assert result.state.gripper_width == 0.0
    assert result.state.target_visible is False
    assert all(math.isfinite(value) for value in result.state.ee_pose_object.xyz)
    assert all(math.isfinite(value) for value in result.state.ee_pose_object.rpy)


def test_build_grasp_state_tracks_partial_pose_fields():
    payload = {
        "ee_pose_base": {"xyz": [0.1, 0.2, 0.3]},
        "object_pose_base": {"rpy": [0.0, 0.0, 0.0]},
    }

    result = build_grasp_state(payload)

    assert "ee_pose_base.rpy" in result.missing_fields
    assert "object_pose_base.xyz" in result.missing_fields
    assert result.state.ee_pose_base.rpy == (0.0, 0.0, 0.0)
    assert result.state.object_pose_base.xyz == (0.0, 0.0, 0.0)
