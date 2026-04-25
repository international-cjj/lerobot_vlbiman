#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _bootstrap_pythonpath() -> None:
    repo_root = _repo_root()
    for candidate in (repo_root / "src", repo_root):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


_bootstrap_pythonpath()

from lerobot.projects.vlbiman_sa.grasp.closed_loop_controller import FRRGClosedLoopController
from lerobot.projects.vlbiman_sa.grasp.contracts import load_frrg_config
from lerobot.projects.vlbiman_sa.grasp.parameters import apply_theta_overrides, load_theta_samples


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "frrg_grasp.yaml"


def _default_output_root() -> Path:
    return _repo_root() / "outputs" / "vlbiman_sa" / "frrg" / "dryrun"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FRRG grasp dry-run entrypoint without hardware.")
    parser.add_argument("--config", type=Path, default=_default_config_path(), help="Path to FRRG YAML config.")
    parser.add_argument("--mock-state", type=Path, required=True, help="Path to the mock grasp state JSON file.")
    parser.add_argument("--max-steps", type=int, default=3, help="Maximum dry-run steps to reserve in the summary.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_default_output_root(),
        help="Directory that stores dry-run outputs.",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional explicit run id.")
    parser.add_argument("--theta-path", type=Path, default=None, help="Optional theta_samples.json file.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Python logging level.")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Mock state file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Mock state must be a JSON object, got {type(payload).__name__}.")
    return payload


def _make_run_id(explicit: str | None) -> str:
    if explicit:
        return explicit
    return datetime.now().strftime("%Y%m%dT%H%M%S%f")


def _build_input_summary(mock_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "keys": sorted(mock_state.keys()),
        "phase": mock_state.get("phase"),
        "mode": mock_state.get("mode"),
        "target_visible": mock_state.get("target_visible"),
        "vision_conf": mock_state.get("vision_conf"),
        "gripper_width": mock_state.get("gripper_width"),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def run_dryrun(
    *,
    config_path: Path,
    mock_state_path: Path,
    max_steps: int,
    output_root: Path,
    run_id: str | None = None,
    theta_path: Path | None = None,
) -> tuple[int, Path, dict[str, Any]]:
    config = load_frrg_config(config_path)
    theta_applied = None
    theta_samples = None
    if theta_path is not None:
        theta_samples = load_theta_samples(theta_path)
        theta_applied = apply_theta_overrides(config, theta_samples)
        config = theta_applied.config
    mock_state = _load_json(mock_state_path.resolve())

    resolved_output_root = output_root.resolve()
    run_id = _make_run_id(run_id)
    run_dir = resolved_output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    controller = FRRGClosedLoopController(
        config,
        mock_state,
        input_mode=config.runtime.default_input_mode,
    )
    run_result = controller.run(max_steps=int(max_steps))
    summary = run_result.summary_dict(
        config_path=str(Path(config_path).resolve()),
        mock_state_path=str(Path(mock_state_path).resolve()),
        output_dir=str(run_dir),
        max_steps=int(max_steps),
        input_mode=config.runtime.default_input_mode,
        input_summary=_build_input_summary(mock_state),
    )
    summary["run_id"] = run_id
    summary["runtime"] = {
        "control_hz": config.runtime.control_hz,
        "default_input_mode": config.runtime.default_input_mode,
    }
    if theta_path is not None and theta_samples is not None and theta_applied is not None:
        summary["theta_path"] = str(Path(theta_path).resolve())
        summary["theta_samples"] = theta_samples.to_dict()
        summary["theta_overrides_applied"] = dict(theta_applied.applied_overrides)
        summary["unused_theta_parameters"] = list(theta_applied.unused_parameters)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_jsonl(run_dir / "phase_trace.jsonl", run_result.phase_trace)
    _write_jsonl(run_dir / "actions.jsonl", run_result.actions)
    _write_jsonl(run_dir / "states.jsonl", run_result.states)
    _write_jsonl(run_dir / "guards.jsonl", run_result.guards)
    if run_result.failure_report is not None:
        (run_dir / "failure_report.json").write_text(
            json.dumps(run_result.failure_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    latest_summary_path = resolved_output_root / "latest_summary.json"
    latest_summary_path.parent.mkdir(parents=True, exist_ok=True)
    latest_summary_path.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    return 0, summary_path, summary

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)
    exit_code, summary_path, _ = run_dryrun(
        config_path=args.config,
        mock_state_path=args.mock_state,
        max_steps=int(args.max_steps),
        output_root=args.output_root,
        run_id=args.run_id,
        theta_path=args.theta_path,
    )
    logging.info("Dry-run summary written to %s", summary_path)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
