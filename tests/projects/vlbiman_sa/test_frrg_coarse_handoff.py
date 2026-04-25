from __future__ import annotations

import json
from pathlib import Path

from lerobot.projects.vlbiman_sa.app.run_frrg_from_coarse import run_from_coarse_summary
from lerobot.projects.vlbiman_sa.grasp.coarse_handoff import (
    COARSE_HANDOFF_SOURCE,
    CoarseHandoffError,
    build_frrg_input_from_coarse_summary,
)
from lerobot.projects.vlbiman_sa.grasp.state_machine import PHASE_CAPTURE_BUILD, PHASE_HANDOFF


FIXTURE_DIR = Path("tests/fixtures/vlbiman_sa")
CONFIG_PATH = Path("src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml")


def _load_fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_build_frrg_input_from_coarse_summary_maps_required_fields():
    coarse_summary = _load_fixture("coarse_handoff_summary.json")

    payload = build_frrg_input_from_coarse_summary(coarse_summary)

    assert payload["phase"] == PHASE_HANDOFF
    assert payload["mode"] == COARSE_HANDOFF_SOURCE
    assert payload["ee_pose_base"] == coarse_summary["pregrasp_pose_base"]
    assert payload["object_pose_base"] == coarse_summary["target_pose_base"]
    assert payload["gripper_width"] == coarse_summary["gripper_initial_width"]
    assert payload["gripper_cmd"] == coarse_summary["gripper_initial_width"]
    assert payload["vision_conf"] == coarse_summary["vision_summary"]["vision_conf"]
    assert payload["corridor_center_px"] == coarse_summary["vision_summary"]["corridor_center_px"]


def test_missing_required_coarse_field_reports_explicit_error():
    coarse_summary = _load_fixture("coarse_handoff_summary.json")
    coarse_summary["vision_summary"].pop("corridor_center_px")

    try:
        build_frrg_input_from_coarse_summary(coarse_summary)
    except CoarseHandoffError as error:
        assert error.reason == "missing_coarse_field"
        assert error.missing_fields == ("vision_summary.corridor_center_px",)
    else:
        raise AssertionError("Expected missing coarse field error.")


def test_run_from_coarse_summary_starts_frrg_and_records_handoff_source(tmp_path):
    exit_code, summary_path, summary = run_from_coarse_summary(
        config_path=CONFIG_PATH,
        coarse_summary_path=FIXTURE_DIR / "coarse_handoff_summary.json",
        max_steps=80,
        output_root=tmp_path,
        run_id="coarse_success",
    )

    assert exit_code == 0
    assert summary_path.exists()
    assert summary["handoff_source"] == COARSE_HANDOFF_SOURCE
    assert summary["input_mode"] == COARSE_HANDOFF_SOURCE
    assert summary["steps_run"] > 0
    assert summary["phase_trace"][0]["current_phase"] == PHASE_HANDOFF
    assert summary["phase_trace"][0]["next_phase"] == PHASE_CAPTURE_BUILD


def test_run_from_coarse_summary_reports_missing_field_failure(tmp_path):
    coarse_summary = _load_fixture("coarse_handoff_summary.json")
    coarse_summary.pop("target_pose_base")
    coarse_path = tmp_path / "coarse_missing.json"
    coarse_path.write_text(json.dumps(coarse_summary, ensure_ascii=False) + "\n", encoding="utf-8")

    exit_code, summary_path, summary = run_from_coarse_summary(
        config_path=CONFIG_PATH,
        coarse_summary_path=coarse_path,
        max_steps=80,
        output_root=tmp_path,
        run_id="coarse_missing",
    )

    assert exit_code == 1
    assert summary_path.exists()
    assert summary["handoff_source"] == COARSE_HANDOFF_SOURCE
    assert summary["failure_reason"] == "missing_coarse_field"
    assert summary["handoff_error"]["missing_fields"] == ["target_pose_base"]
