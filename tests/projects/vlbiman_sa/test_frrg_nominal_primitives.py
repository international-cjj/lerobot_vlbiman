from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from lerobot.projects.vlbiman_sa.grasp.contracts import load_frrg_config
from lerobot.projects.vlbiman_sa.grasp.observer import build_grasp_state
from lerobot.projects.vlbiman_sa.grasp.primitives.minimum_jerk import minimum_jerk_profile
from lerobot.projects.vlbiman_sa.grasp.primitives.nominal_capture import nominal_capture_action
from lerobot.projects.vlbiman_sa.grasp.primitives.nominal_close import nominal_close_action
from lerobot.projects.vlbiman_sa.grasp.primitives.nominal_lift import nominal_lift_action


FIXTURE_DIR = Path("tests/fixtures/vlbiman_sa")
CONFIG_PATH = Path("src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml")


def _load_fixture(name: str) -> tuple[object, dict[str, object]]:
    payload = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    feature_debug_terms = dict(payload.get("feature_debug_terms", {}))
    state = build_grasp_state(payload).state
    return state, feature_debug_terms


def test_minimum_jerk_profile_hits_expected_endpoints():
    assert minimum_jerk_profile(0.0) == 0.0
    assert minimum_jerk_profile(1.0) == 1.0
    assert 0.0 < minimum_jerk_profile(0.5) < 1.0


def test_nominal_capture_keeps_gripper_open_and_disables_forward_motion_when_gate_is_closed():
    config = load_frrg_config(CONFIG_PATH)
    state, feature_debug_terms = _load_fixture("frrg_capture_timeout.json")

    action = nominal_capture_action(state, config, feature_debug_terms=feature_debug_terms)

    assert action.delta_pose_object[2] == 0.0
    assert action.delta_gripper == 0.0
    assert action.debug_terms["forward_gate"] is False
    assert action.debug_terms["is_raw_action"] is True


def test_nominal_capture_outputs_raw_action_without_safety_clipping():
    config = load_frrg_config(CONFIG_PATH)
    state, feature_debug_terms = _load_fixture("frrg_capture_ready.json")

    action = nominal_capture_action(state, config, feature_debug_terms=feature_debug_terms)

    assert action.delta_pose_object[0] > 0.0
    assert action.delta_pose_object[2] > 0.0
    assert action.delta_pose_object[5] > 0.0
    assert action.delta_gripper == 0.0
    assert action.debug_terms["forward_gate"] is True
    assert action.debug_terms["is_raw_action"] is True
    assert abs(action.delta_pose_object[0]) > config.safety.max_step_xyz_m[0]


def test_nominal_close_uses_preclose_motion_before_closing_gripper():
    config = load_frrg_config(CONFIG_PATH)
    state, _ = _load_fixture("frrg_capture_timeout.json")
    state = replace(state, phase_elapsed_s=0.0, e_dep=0.01)

    action = nominal_close_action(state, config)

    assert action.delta_pose_object[2] > 0.0
    assert action.delta_gripper == 0.0
    assert action.debug_terms["close_mode"] == "preclose"
    assert action.debug_terms["is_raw_action"] is True


def test_nominal_close_then_closes_gripper_with_raw_delta():
    config = load_frrg_config(CONFIG_PATH)
    state, _ = _load_fixture("frrg_nominal_success.json")

    action = nominal_close_action(state, config)

    assert action.delta_pose_object == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert action.delta_gripper < 0.0
    assert action.debug_terms["close_mode"] == "gripper_close"
    assert action.debug_terms["is_raw_action"] is True
    assert abs(action.delta_gripper) > config.safety.max_gripper_delta_m


def test_nominal_lift_only_generates_small_positive_z_motion():
    config = load_frrg_config(CONFIG_PATH)
    state, _ = _load_fixture("frrg_nominal_success.json")

    action = nominal_lift_action(state, config)

    assert action.delta_pose_object[0] == 0.0
    assert action.delta_pose_object[1] == 0.0
    assert action.delta_pose_object[2] > 0.0
    assert action.delta_pose_object[3:] == (0.0, 0.0, 0.0)
    assert action.delta_gripper == 0.0
    assert action.debug_terms["is_raw_action"] is True
    assert action.delta_pose_object[2] <= config.lift_test.lift_speed_mps / config.runtime.control_hz
