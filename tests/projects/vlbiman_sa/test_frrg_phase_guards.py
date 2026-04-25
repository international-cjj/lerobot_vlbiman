from __future__ import annotations

import json
from pathlib import Path

from lerobot.projects.vlbiman_sa.grasp.contracts import load_frrg_config
from lerobot.projects.vlbiman_sa.grasp.observer import build_grasp_state
from lerobot.projects.vlbiman_sa.grasp.phase_guards import (
    capture_timeout_guard,
    capture_to_close_guard,
    close_to_lift_guard,
    handoff_guard,
    lift_to_success_guard,
    slip_detected_guard,
    vision_lost_guard,
)


FIXTURE_DIR = Path("tests/fixtures/vlbiman_sa")
CONFIG_PATH = Path("src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml")


def _load_fixture(name: str) -> tuple[object, dict[str, object]]:
    payload = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    feature_debug_terms = dict(payload.get("feature_debug_terms", {}))
    state = build_grasp_state(payload).state
    return state, feature_debug_terms


def _assert_guard_shape(result) -> None:
    assert isinstance(result.passed, bool)
    assert isinstance(result.score, float)
    assert isinstance(result.debug_terms, dict)


def test_handoff_guard_passes_for_ready_fixture():
    config = load_frrg_config(CONFIG_PATH)
    state, feature_debug_terms = _load_fixture("frrg_handoff_ready.json")

    result = handoff_guard(state, config, feature_debug_terms=feature_debug_terms)

    _assert_guard_shape(result)
    assert result.passed is True
    assert result.reason is None


def test_capture_to_close_guard_passes_for_capture_ready_fixture():
    config = load_frrg_config(CONFIG_PATH)
    state, feature_debug_terms = _load_fixture("frrg_capture_ready.json")

    result = capture_to_close_guard(state, config, feature_debug_terms=feature_debug_terms)

    _assert_guard_shape(result)
    assert result.passed is True
    assert result.reason is None
    assert result.debug_terms["forward_gate"] is True
    assert result.debug_terms["depth_ready"] is True


def test_close_to_lift_and_lift_to_success_pass_for_nominal_success_fixture():
    config = load_frrg_config(CONFIG_PATH)
    state, feature_debug_terms = _load_fixture("frrg_nominal_success.json")

    close_guard = close_to_lift_guard(
        state,
        config,
        feature_debug_terms=feature_debug_terms,
        contact_current_available=True,
    )
    lift_guard = lift_to_success_guard(
        state,
        config,
        feature_debug_terms=feature_debug_terms,
        contact_current_available=True,
    )

    _assert_guard_shape(close_guard)
    _assert_guard_shape(lift_guard)
    assert close_guard.passed is True
    assert lift_guard.passed is True


def test_failure_guards_identify_standard_failure_reasons():
    config = load_frrg_config(CONFIG_PATH)

    vision_state, vision_debug = _load_fixture("frrg_vision_lost.json")
    timeout_state, timeout_debug = _load_fixture("frrg_capture_timeout.json")
    slip_state, slip_debug = _load_fixture("frrg_slip_detected.json")

    vision_result = vision_lost_guard(vision_state, config)
    timeout_result = capture_timeout_guard(timeout_state, config)
    slip_result = slip_detected_guard(slip_state, config, feature_debug_terms=slip_debug)
    capture_transition = capture_to_close_guard(timeout_state, config, feature_debug_terms=timeout_debug)
    lift_transition = lift_to_success_guard(slip_state, config, feature_debug_terms=slip_debug)
    handoff_transition = handoff_guard(vision_state, config, feature_debug_terms=vision_debug)

    for result in (
        vision_result,
        timeout_result,
        slip_result,
        capture_transition,
        lift_transition,
        handoff_transition,
    ):
        _assert_guard_shape(result)

    assert vision_result.passed is True
    assert vision_result.reason == "vision_lost"
    assert timeout_result.passed is True
    assert timeout_result.reason == "capture_timeout"
    assert slip_result.passed is True
    assert slip_result.reason == "slip_detected"
    assert capture_transition.passed is False
    assert capture_transition.reason == "capture_timeout"
    assert lift_transition.passed is False
    assert lift_transition.reason == "slip_detected"
    assert handoff_transition.passed is False
    assert handoff_transition.reason == "vision_lost"


def test_capture_to_close_guard_blocks_when_depth_is_still_far():
    config = load_frrg_config(CONFIG_PATH)
    state, feature_debug_terms = _load_fixture("frrg_capture_ready.json")
    state = type(state)(
        **{
            **state.__dict__,
            "ee_pose_object": type(state.ee_pose_object)(
                xyz=(state.ee_pose_object.xyz[0], state.ee_pose_object.xyz[1], 0.12),
                rpy=state.ee_pose_object.rpy,
            ),
            "e_dep": float(config.capture_build.target_depth_goal_m) - 0.12,
        }
    )

    result = capture_to_close_guard(state, config, feature_debug_terms=feature_debug_terms)

    _assert_guard_shape(result)
    assert result.passed is False
    assert result.reason == "depth_not_ready"
    assert result.debug_terms["depth_ready"] is False


def test_capture_to_close_guard_blocks_when_vertical_alignment_is_not_ready():
    config = load_frrg_config(CONFIG_PATH)
    state, feature_debug_terms = _load_fixture("frrg_capture_ready.json")
    state = type(state)(**{**state.__dict__, "e_vert": 0.03})
    feature_debug_terms = {
        **feature_debug_terms,
        "vertical_enabled": True,
        "vertical_tol_m": 0.015,
    }

    result = capture_to_close_guard(state, config, feature_debug_terms=feature_debug_terms)

    _assert_guard_shape(result)
    assert result.passed is False
    assert result.reason == "vertical_not_ready"
    assert result.debug_terms["vertical_ready"] is False


def test_lift_to_success_guard_requires_real_object_lift():
    config = load_frrg_config(CONFIG_PATH)
    state, feature_debug_terms = _load_fixture("frrg_nominal_success.json")
    state = type(state)(**{**state.__dict__, "object_lift_m": 0.001})

    result = lift_to_success_guard(
        state,
        config,
        feature_debug_terms=feature_debug_terms,
        contact_current_available=True,
    )

    _assert_guard_shape(result)
    assert result.passed is False
    assert result.reason == "lift_height_not_reached"
    assert result.debug_terms["lift_height_ready"] is False
