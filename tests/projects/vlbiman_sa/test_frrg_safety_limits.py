from __future__ import annotations

import json
from pathlib import Path

from lerobot.projects.vlbiman_sa.grasp.contracts import GraspAction, load_frrg_config
from lerobot.projects.vlbiman_sa.grasp.observer import build_grasp_state
from lerobot.projects.vlbiman_sa.grasp.primitives.nominal_capture import nominal_capture_action
from lerobot.projects.vlbiman_sa.grasp.primitives.nominal_close import nominal_close_action
from lerobot.projects.vlbiman_sa.grasp.safety_limits import apply_safety_limits


FIXTURE_DIR = Path("tests/fixtures/vlbiman_sa")
CONFIG_PATH = Path("src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml")


def _load_fixture(name: str) -> tuple[object, dict[str, object]]:
    payload = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    feature_debug_terms = dict(payload.get("feature_debug_terms", {}))
    state = build_grasp_state(payload).state
    return state, feature_debug_terms


def test_apply_safety_limits_clips_raw_capture_action():
    config = load_frrg_config(CONFIG_PATH)
    state, feature_debug_terms = _load_fixture("frrg_capture_ready.json")
    raw_action = nominal_capture_action(state, config, feature_debug_terms=feature_debug_terms)

    result = apply_safety_limits(state, config, raw_action)

    assert result.limited is True
    assert result.stop is False
    assert abs(result.safe_action.delta_pose_object[0]) <= config.safety.max_step_xyz_m[0]
    assert result.raw_action_norm >= result.safe_action_norm
    assert "raw_action" in result.to_dict()
    assert "safe_action" in result.to_dict()


def test_apply_safety_limits_clips_gripper_delta_from_close_action():
    config = load_frrg_config(CONFIG_PATH)
    state, _ = _load_fixture("frrg_nominal_success.json")
    raw_action = nominal_close_action(state, config)

    result = apply_safety_limits(state, config, raw_action)

    assert result.limited is True
    assert abs(result.safe_action.delta_gripper) <= config.safety.max_gripper_delta_m
    assert result.safe_action.delta_pose_object == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_apply_safety_limits_hardstops_on_vision_loss():
    config = load_frrg_config(CONFIG_PATH)
    state, _ = _load_fixture("frrg_vision_lost.json")
    raw_action = GraspAction(delta_pose_object=(0.01, 0.0, 0.0, 0.0, 0.0, 0.0), delta_gripper=-0.01)

    result = apply_safety_limits(state, config, raw_action)

    assert result.stop is True
    assert result.reason == "vision_lost"
    assert result.safe_action.delta_pose_object == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert result.safe_action.delta_gripper == 0.0


def test_apply_safety_limits_hardstops_on_object_jump_invalid_phase_and_non_finite_action():
    config = load_frrg_config(CONFIG_PATH)
    state, _ = _load_fixture("frrg_capture_ready.json")

    object_jump_result = apply_safety_limits(
        state,
        config,
        GraspAction(delta_pose_object=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), delta_gripper=0.0),
        object_jump_m=config.safety.obj_jump_stop_m + 0.001,
    )
    invalid_phase_result = apply_safety_limits(
        state,
        config,
        GraspAction(delta_pose_object=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), delta_gripper=0.0),
        invalid_phase=True,
    )
    non_finite_action = GraspAction()
    non_finite_action.delta_pose_object = (float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0)
    non_finite_result = apply_safety_limits(state, config, non_finite_action)

    assert object_jump_result.stop is True
    assert object_jump_result.reason == "object_jump"
    assert invalid_phase_result.stop is True
    assert invalid_phase_result.reason == "invalid_phase"
    assert non_finite_result.stop is True
    assert non_finite_result.reason == "non_finite_action"
