from __future__ import annotations

import json
from pathlib import Path

from lerobot.projects.vlbiman_sa.app.run_frrg_benchmark import run_benchmark


CONFIG_PATH = Path("src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml")
FIXTURE_DIR = Path("tests/fixtures/vlbiman_sa")


def test_run_benchmark_writes_metrics_episodes_and_report(tmp_path):
    output_dir = tmp_path / "benchmark"

    exit_code, metrics_path, metrics = run_benchmark(
        config_path=CONFIG_PATH,
        fixtures=FIXTURE_DIR,
        output_dir=output_dir,
        max_steps=20,
    )

    assert exit_code == 0
    assert metrics_path.exists()
    assert (output_dir / "episodes.jsonl").exists()
    assert (output_dir / "report.md").exists()
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["episode_count"] >= 1
    assert "success_rate" in metrics_payload
    assert "failure_reason_top_list" in metrics_payload
    assert metrics_payload["hardware_called"] is False
    assert metrics_payload["mujoco_available"] is False
