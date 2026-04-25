from __future__ import annotations

import json
from pathlib import Path

from lerobot.projects.vlbiman_sa.grasp.contracts import GuardResult, load_frrg_config
from lerobot.projects.vlbiman_sa.grasp.observer import build_grasp_state
from lerobot.projects.vlbiman_sa.grasp.phase_guards import (
    capture_to_close_guard,
    close_to_lift_guard,
    handoff_guard,
    lift_to_success_guard,
)
from lerobot.projects.vlbiman_sa.grasp.state_machine import (
    PHASE_CAPTURE_BUILD,
    PHASE_CLOSE_HOLD,
    PHASE_FAILURE,
    PHASE_HANDOFF,
    PHASE_LIFT_TEST,
    PHASE_RECOVERY,
    PHASE_SUCCESS,
    next_phase,
)


FIXTURE_DIR = Path("tests/fixtures/vlbiman_sa")
CONFIG_PATH = Path("src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml")


def _load_fixture(name: str) -> tuple[object, dict[str, object]]:
    payload = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    feature_debug_terms = dict(payload.get("feature_debug_terms", {}))
    state = build_grasp_state(payload).state
    return state, feature_debug_terms


def test_next_phase_walks_full_success_chain():
    config = load_frrg_config(CONFIG_PATH)
    handoff_state, handoff_debug = _load_fixture("frrg_handoff_ready.json")
    capture_state, capture_debug = _load_fixture("frrg_capture_ready.json")
    success_state, success_debug = _load_fixture("frrg_nominal_success.json")

    handoff_transition = next_phase(
        PHASE_HANDOFF,
        {"handoff": handoff_guard(handoff_state, config, feature_debug_terms=handoff_debug)},
    )
    capture_transition = next_phase(
        PHASE_CAPTURE_BUILD,
        {"capture_to_close": capture_to_close_guard(capture_state, config, feature_debug_terms=capture_debug)},
    )
    close_transition = next_phase(
        PHASE_CLOSE_HOLD,
        {
            "close_to_lift": close_to_lift_guard(
                success_state,
                config,
                feature_debug_terms=success_debug,
                contact_current_available=True,
            )
        },
    )
    lift_transition = next_phase(
        PHASE_LIFT_TEST,
        {
            "lift_to_success": lift_to_success_guard(
                success_state,
                config,
                feature_debug_terms=success_debug,
                contact_current_available=True,
            )
        },
    )

    assert handoff_transition.next_phase == PHASE_CAPTURE_BUILD
    assert capture_transition.next_phase == PHASE_CLOSE_HOLD
    assert close_transition.next_phase == PHASE_LIFT_TEST
    assert lift_transition.next_phase == PHASE_SUCCESS
    assert lift_transition.transition_kind == "advance"
    assert lift_transition.to_dict()["next_phase"] == PHASE_SUCCESS


def test_failure_guard_can_force_failure_from_any_execution_phase():
    phases = (PHASE_HANDOFF, PHASE_CAPTURE_BUILD, PHASE_CLOSE_HOLD, PHASE_LIFT_TEST)
    failure_guard = GuardResult(passed=True, score=0.0, reason="vision_lost", debug_terms={"target_visible": False})

    for phase in phases:
        transition = next_phase(phase, {"vision_lost": failure_guard})

        assert transition.next_phase == PHASE_FAILURE
        assert transition.reason == "vision_lost"
        assert transition.transition_kind == "fail"


def test_capture_timeout_enters_recovery_then_exceeds_retry_limit():
    recoverable_guard = GuardResult(
        passed=True,
        score=1.0,
        reason="capture_timeout",
        debug_terms={"phase_elapsed_s": 8.6},
    )

    recovery_transition = next_phase(
        PHASE_CAPTURE_BUILD,
        {"capture_timeout": recoverable_guard},
        retry_count=0,
        max_retry_count=1,
    )
    recovery_exit = next_phase(
        PHASE_RECOVERY,
        {},
        retry_count=1,
        max_retry_count=1,
        recovery_target=PHASE_HANDOFF,
    )
    exceeded_transition = next_phase(
        PHASE_CAPTURE_BUILD,
        {"capture_timeout": recoverable_guard},
        retry_count=1,
        max_retry_count=1,
    )

    assert recovery_transition.next_phase == PHASE_RECOVERY
    assert recovery_transition.reason == "capture_timeout"
    assert recovery_transition.transition_kind == "recover"
    assert recovery_exit.next_phase == PHASE_HANDOFF
    assert recovery_exit.transition_kind == "recover_exit"
    assert exceeded_transition.next_phase == PHASE_FAILURE
    assert exceeded_transition.reason == "max_retry_exceeded"
    assert exceeded_transition.debug_terms["triggered_failure_reason"] == "capture_timeout"


def test_illegal_jump_is_rejected_and_marked_invalid_phase():
    transition = next_phase(
        PHASE_HANDOFF,
        {"lift_to_success": GuardResult(passed=True, score=0.95, reason=None, debug_terms={"score_ready": True})},
    )

    assert transition.next_phase == PHASE_HANDOFF
    assert transition.allowed is False
    assert transition.invalid_phase is True
    assert transition.reason == "invalid_phase"
    assert transition.debug_terms["expected_transition_guard"] == "handoff"
    assert transition.debug_terms["illegal_transition_guards"] == ["lift_to_success"]


def test_unknown_state_falls_back_to_failure():
    transition = next_phase("NOT_A_REAL_PHASE", {})

    assert transition.next_phase == PHASE_FAILURE
    assert transition.reason == "unknown_state"
    assert transition.invalid_phase is True
