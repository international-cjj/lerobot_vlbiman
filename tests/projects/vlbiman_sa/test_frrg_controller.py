from __future__ import annotations

import json
from pathlib import Path

from lerobot.projects.vlbiman_sa.grasp.closed_loop_controller import FRRGClosedLoopController
from lerobot.projects.vlbiman_sa.grasp.contracts import load_frrg_config
from lerobot.projects.vlbiman_sa.grasp.state_machine import PHASE_FAILURE, PHASE_SUCCESS


FIXTURE_DIR = Path("tests/fixtures/vlbiman_sa")
CONFIG_PATH = Path("src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml")


def _load_fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_controller_run_reaches_success_for_nominal_success_fixture():
    config = load_frrg_config(CONFIG_PATH)
    payload = _load_fixture("frrg_nominal_success.json")

    result = FRRGClosedLoopController(config, payload).run(max_steps=80)

    assert result.status == "success"
    assert result.final_phase == PHASE_SUCCESS
    assert result.failure_reason is None
    assert result.steps_run >= 1
    assert len(result.phase_trace) == result.steps_run
    assert len(result.actions) == result.steps_run
    assert result.max_residual_norm == 0.0
    assert all("safe_action" in action for action in result.actions)


def test_controller_run_reports_failure_for_vision_lost_fixture():
    config = load_frrg_config(CONFIG_PATH)
    payload = _load_fixture("frrg_vision_lost.json")

    result = FRRGClosedLoopController(config, payload).run(max_steps=10)

    assert result.status == "failure"
    assert result.final_phase == PHASE_FAILURE
    assert result.failure_reason == "vision_lost"
    assert result.actions[0]["stop"] is True
    assert result.actions[0]["reason"] == "vision_lost"


def test_controller_run_reports_failure_for_capture_timeout_fixture():
    config = load_frrg_config(CONFIG_PATH)
    payload = _load_fixture("frrg_capture_timeout.json")

    result = FRRGClosedLoopController(config, payload).run(max_steps=10)

    assert result.status == "failure"
    assert result.final_phase == PHASE_FAILURE
    assert result.failure_reason == "max_retry_exceeded"
    assert result.phase_trace[0]["reason"] == "max_retry_exceeded"
    assert result.max_residual_norm == 0.0


def test_controller_summary_has_required_fields():
    config = load_frrg_config(CONFIG_PATH)
    payload = _load_fixture("frrg_nominal_success.json")
    result = FRRGClosedLoopController(config, payload).run(max_steps=80)

    summary = result.summary_dict(
        config_path=str(CONFIG_PATH.resolve()),
        mock_state_path=str((FIXTURE_DIR / "frrg_nominal_success.json").resolve()),
        output_dir="/tmp/frrg",
        max_steps=80,
        input_mode=config.runtime.default_input_mode,
        input_summary={"phase": payload["phase"]},
    )

    assert summary["status"] == "success"
    assert summary["final_phase"] == PHASE_SUCCESS
    assert summary["max_residual_norm"] == 0.0
    assert summary["all_actions_limited"] is True
    assert isinstance(summary["phase_trace"], list)
