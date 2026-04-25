from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from lerobot.projects.vlbiman_sa.grasp.closed_loop_controller import FRRGClosedLoopController
from lerobot.projects.vlbiman_sa.grasp.contracts import load_frrg_config
from lerobot.projects.vlbiman_sa.grasp.recovery_policy import RecoveryPolicy
from lerobot.projects.vlbiman_sa.grasp.state_machine import PHASE_FAILURE, PHASE_RECOVERY


FIXTURE_DIR = Path("tests/fixtures/vlbiman_sa")
CONFIG_PATH = Path("src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml")


def _load_fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_recovery_policy_supports_backoff_half_open_recenter_and_abort():
    config = load_frrg_config(CONFIG_PATH)
    payload = _load_fixture("frrg_capture_ready.json")
    state = FRRGClosedLoopController(config, payload).step(0, payload).scored.state
    state = replace(state, gripper_width=0.02)
    policy = RecoveryPolicy()

    backoff = policy.propose(state, config, "capture_timeout")
    half_open = policy.propose(state, config, "contact_not_detected")
    recenter = policy.propose(state, config, "large_drift")
    abort = policy.propose(state, config, "vision_lost")

    assert backoff.proposal_type == "backoff"
    assert backoff.action.delta_pose_object[2] < 0.0
    assert half_open.proposal_type == "half_open"
    assert half_open.action.delta_gripper > 0.0
    assert recenter.proposal_type == "recenter"
    assert recenter.action.delta_pose_object[0] != 0.0 or recenter.action.delta_pose_object[5] != 0.0
    assert abort.proposal_type == "abort"
    assert abort.action.stop is True
    assert abort.action.reason == "vision_lost"


def test_controller_uses_recovery_action_for_capture_timeout_before_retry_exhaustion():
    config = load_frrg_config(CONFIG_PATH)
    payload = _load_fixture("frrg_capture_timeout.json")
    payload["retry_count"] = 0
    controller = FRRGClosedLoopController(config, payload)

    step_result = controller.step(0)

    assert step_result.phase_transition.next_phase == PHASE_RECOVERY
    assert step_result.phase_transition.reason == "capture_timeout"
    assert step_result.recovery_result is not None
    assert step_result.recovery_result.proposal_type == "backoff"
    assert step_result.raw_action.delta_pose_object[2] < 0.0
    assert step_result.next_payload["retry_count"] == 1


def test_controller_emits_failure_report_for_vision_lost():
    config = load_frrg_config(CONFIG_PATH)
    payload = _load_fixture("frrg_vision_lost.json")

    result = FRRGClosedLoopController(config, payload).run(max_steps=10)

    assert result.final_phase == PHASE_FAILURE
    assert result.failure_reason == "vision_lost"
    assert result.failure_report is not None
    assert result.failure_report["failure_reason"] == "vision_lost"
    assert "last_state" in result.failure_report
    assert "safe_action" in result.failure_report
    assert isinstance(result.failure_report["phase_trace"], list)


def test_controller_emits_failure_report_for_object_jump():
    config = load_frrg_config(CONFIG_PATH)
    payload = _load_fixture("frrg_capture_ready.json")
    payload["object_jump_m"] = config.safety.obj_jump_stop_m + 0.001

    result = FRRGClosedLoopController(config, payload).run(max_steps=5)

    assert result.final_phase == PHASE_FAILURE
    assert result.failure_reason == "object_jump"
    assert result.failure_report is not None
    assert result.failure_report["failure_reason"] == "object_jump"
    assert result.failure_report["safe_action"]["stop"] is True
    assert result.failure_report["recovery_proposal"]["proposal_type"] == "abort"
