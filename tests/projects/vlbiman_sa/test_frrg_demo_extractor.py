from __future__ import annotations

import json
from pathlib import Path

from lerobot.projects.vlbiman_sa.app.run_frrg_extract_demo import run_extract_demo
from lerobot.projects.vlbiman_sa.app.run_frrg_grasp_dryrun import run_dryrun
from lerobot.projects.vlbiman_sa.grasp.parameters import THETA_PARAMETER_NAMES, load_theta_samples


CONFIG_PATH = Path("src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml")
SESSION_DIR = Path("outputs/vlbiman_sa/recordings/one_shot_full_20260327T231705")
MOCK_STATE_PATH = Path("tests/fixtures/vlbiman_sa/frrg_nominal_success.json")


def test_run_extract_demo_writes_theta_samples_and_report(tmp_path):
    output_dir = tmp_path / "demo_params"

    exit_code, theta_path, report_path, report = run_extract_demo(
        session_dir=SESSION_DIR,
        output_dir=output_dir,
        config_path=CONFIG_PATH,
    )

    assert exit_code == 0
    assert theta_path.exists()
    assert report_path.exists()
    theta_payload = json.loads(theta_path.read_text(encoding="utf-8"))
    for name in THETA_PARAMETER_NAMES:
        assert name in theta_payload
        assert theta_payload[name] >= 0.0
        assert name in report["parameters"]
        assert report["parameters"][name]["source"]
    assert report["dryrun_loadable"] is True
    assert "theta_overrides_applied" in report


def test_dryrun_can_load_extracted_theta_samples(tmp_path):
    theta_dir = tmp_path / "theta"
    _, theta_path, _, _ = run_extract_demo(
        session_dir=SESSION_DIR,
        output_dir=theta_dir,
        config_path=CONFIG_PATH,
    )
    output_root = tmp_path / "dryrun"

    exit_code, summary_path, summary = run_dryrun(
        config_path=CONFIG_PATH,
        mock_state_path=MOCK_STATE_PATH,
        max_steps=5,
        output_root=output_root,
        run_id="theta_load",
        theta_path=theta_path,
    )

    assert exit_code == 0
    assert summary_path.exists()
    assert summary["theta_path"] == str(theta_path.resolve())
    assert "theta_samples" in summary
    assert "theta_overrides_applied" in summary
    loaded_theta = load_theta_samples(theta_path)
    assert summary["theta_samples"]["preclose_distance_m"] == loaded_theta.preclose_distance_m
