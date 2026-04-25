from __future__ import annotations

import json
from pathlib import Path

from lerobot.projects.vlbiman_sa.grasp.contracts import load_frrg_config
from lerobot.projects.vlbiman_sa.grasp.observer import build_grasp_state
from lerobot.projects.vlbiman_sa.grasp.scores import (
    apply_scores,
    compute_capture_score,
    compute_hold_score,
    compute_lift_score,
)


FIXTURE_DIR = Path("tests/fixtures/vlbiman_sa")
CONFIG_PATH = Path("src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml")


def _load_fixture(name: str) -> tuple[object, dict[str, object]]:
    payload = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    feature_debug_terms = dict(payload.get("feature_debug_terms", {}))
    state = build_grasp_state(payload).state
    return state, feature_debug_terms


def test_compute_capture_score_prefers_capture_ready_fixture():
    config = load_frrg_config(CONFIG_PATH)
    ready_state, ready_debug = _load_fixture("frrg_capture_ready.json")
    timeout_state, timeout_debug = _load_fixture("frrg_capture_timeout.json")

    ready_score, ready_terms = compute_capture_score(ready_state, config, feature_debug_terms=ready_debug)
    timeout_score, _ = compute_capture_score(timeout_state, config, feature_debug_terms=timeout_debug)

    assert ready_score > config.capture_build.close_score_threshold
    assert timeout_score < ready_score
    assert ready_terms["lateral_error_unit"] == "m"
    assert ready_terms["capture_lateral_sigma"] == config.handoff.handoff_pos_tol_m


def test_compute_hold_score_requires_contact_and_flags_missing_current_proxy():
    config = load_frrg_config(CONFIG_PATH)
    success_state, success_debug = _load_fixture("frrg_nominal_success.json")

    hold_score, hold_terms = compute_hold_score(
        success_state,
        config,
        feature_debug_terms=success_debug,
        contact_current_available=True,
    )
    no_current_score, no_current_terms = compute_hold_score(
        success_state,
        config,
        feature_debug_terms=success_debug,
        contact_current_available=False,
    )

    assert hold_score > config.close_hold.hold_score_threshold
    assert hold_terms["contact_detected"] is True
    assert no_current_terms["contact_current_unavailable"] is True
    assert no_current_terms["contact_detected"] is False
    assert no_current_score < hold_score


def test_compute_lift_score_penalizes_slip_fixture():
    config = load_frrg_config(CONFIG_PATH)
    success_state, success_debug = _load_fixture("frrg_nominal_success.json")
    slip_state, slip_debug = _load_fixture("frrg_slip_detected.json")

    success_score, success_terms = compute_lift_score(success_state, config, feature_debug_terms=success_debug)
    slip_score, slip_terms = compute_lift_score(slip_state, config, feature_debug_terms=slip_debug)

    assert success_score > config.lift_test.lift_score_threshold
    assert slip_score < config.lift_test.lift_score_threshold
    assert success_terms["drift_unit"] == "m"
    assert slip_terms["slip_drift_tol"] == config.lift_test.slip_threshold_m


def test_apply_scores_updates_state_fields():
    config = load_frrg_config(CONFIG_PATH)
    state, feature_debug_terms = _load_fixture("frrg_nominal_success.json")

    result = apply_scores(state, config, feature_debug_terms=feature_debug_terms)

    assert result.state.capture_score > 0.0
    assert result.state.hold_score > 0.0
    assert result.state.lift_score > 0.0
    assert set(result.debug_terms.keys()) == {"capture", "hold", "lift"}
