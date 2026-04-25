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
from lerobot.projects.vlbiman_sa.grasp.coarse_handoff import (
    COARSE_HANDOFF_SOURCE,
    CoarseHandoffError,
    REQUIRED_COARSE_FIELDS,
    build_coarse_input_summary,
    build_frrg_input_from_coarse_summary,
)
from lerobot.projects.vlbiman_sa.grasp.contracts import load_frrg_config
from lerobot.projects.vlbiman_sa.grasp.state_machine import PHASE_FAILURE


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "frrg_grasp.yaml"


def _default_output_root() -> Path:
    return _repo_root() / "outputs" / "vlbiman_sa" / "frrg" / "coarse_handoff"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FRRG from a coarse handoff summary.")
    parser.add_argument("--config", type=Path, default=_default_config_path(), help="Path to FRRG YAML config.")
    parser.add_argument("--coarse-summary", type=Path, required=True, help="Path to the coarse handoff JSON file.")
    parser.add_argument("--max-steps", type=int, default=80, help="Maximum controller steps to execute.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_default_output_root(),
        help="Directory that stores coarse handoff outputs.",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional explicit run id.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Python logging level.")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Coarse summary file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Coarse summary must be a JSON object, got {type(payload).__name__}.")
    return payload


def _make_run_id(explicit: str | None) -> str:
    if explicit:
        return explicit
    return datetime.now().strftime("%Y%m%dT%H%M%S%f")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _failure_summary(
    *,
    config_path: Path,
    coarse_summary_path: Path,
    output_dir: Path,
    max_steps: int,
    coarse_summary: dict[str, Any],
    error: CoarseHandoffError,
) -> dict[str, Any]:
    failure_report = {
        "status": "failure",
        "failure_reason": error.reason,
        "final_phase": PHASE_FAILURE,
        "message": str(error),
        "missing_fields": list(error.missing_fields),
    }
    return {
        "status": "failure",
        "final_phase": PHASE_FAILURE,
        "failure_reason": error.reason,
        "config_path": str(config_path.resolve()),
        "coarse_summary_path": str(coarse_summary_path.resolve()),
        "output_dir": str(output_dir),
        "max_steps": int(max_steps),
        "steps_run": 0,
        "phase_trace": [],
        "max_raw_action_norm": 0.0,
        "max_safe_action_norm": 0.0,
        "max_residual_norm": 0.0,
        "all_actions_limited": True,
        "input_mode": COARSE_HANDOFF_SOURCE,
        "input_summary": build_coarse_input_summary(coarse_summary),
        "last_state_summary": None,
        "hardware_called": False,
        "camera_opened": False,
        "mujoco_available": False,
        "mujoco_validation_status": "not_run",
        "handoff_source": COARSE_HANDOFF_SOURCE,
        "coarse_required_fields": list(REQUIRED_COARSE_FIELDS),
        "handoff_error": failure_report,
        "failure_report": failure_report,
    }


def run_from_coarse_summary(
    *,
    config_path: Path,
    coarse_summary_path: Path,
    max_steps: int,
    output_root: Path,
    run_id: str | None = None,
) -> tuple[int, Path, dict[str, Any]]:
    config = load_frrg_config(config_path)
    coarse_summary = _load_json(coarse_summary_path.resolve())

    resolved_output_root = output_root.resolve()
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    resolved_run_id = _make_run_id(run_id)
    run_dir = resolved_output_root / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    try:
        handoff_payload = build_frrg_input_from_coarse_summary(coarse_summary)
        controller = FRRGClosedLoopController(
            config,
            handoff_payload,
            input_mode=COARSE_HANDOFF_SOURCE,
        )
        run_result = controller.run(max_steps=int(max_steps))
        summary = run_result.summary_dict(
            config_path=str(config_path.resolve()),
            mock_state_path=str(coarse_summary_path.resolve()),
            output_dir=str(run_dir),
            max_steps=int(max_steps),
            input_mode=COARSE_HANDOFF_SOURCE,
            input_summary=build_coarse_input_summary(coarse_summary),
        )
        summary["coarse_summary_path"] = summary.pop("mock_state_path")
        summary["handoff_source"] = COARSE_HANDOFF_SOURCE
        summary["coarse_required_fields"] = list(REQUIRED_COARSE_FIELDS)
        exit_code = 0

        _write_jsonl(run_dir / "phase_trace.jsonl", run_result.phase_trace)
        _write_jsonl(run_dir / "actions.jsonl", run_result.actions)
        _write_jsonl(run_dir / "states.jsonl", run_result.states)
        _write_jsonl(run_dir / "guards.jsonl", run_result.guards)
        if run_result.failure_report is not None:
            (run_dir / "failure_report.json").write_text(
                json.dumps(run_result.failure_report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    except CoarseHandoffError as error:
        summary = _failure_summary(
            config_path=config_path,
            coarse_summary_path=coarse_summary_path,
            output_dir=run_dir,
            max_steps=max_steps,
            coarse_summary=coarse_summary,
            error=error,
        )
        (run_dir / "failure_report.json").write_text(
            json.dumps(summary["failure_report"], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        exit_code = 1

    summary["run_id"] = resolved_run_id
    summary["runtime"] = {
        "control_hz": config.runtime.control_hz,
        "default_input_mode": config.runtime.default_input_mode,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    latest_summary_path = resolved_output_root / "latest_summary.json"
    latest_summary_path.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    return exit_code, summary_path, summary


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)
    exit_code, summary_path, _ = run_from_coarse_summary(
        config_path=args.config,
        coarse_summary_path=args.coarse_summary,
        max_steps=int(args.max_steps),
        output_root=args.output_root,
        run_id=args.run_id,
    )
    logging.info("Coarse handoff summary written to %s", summary_path)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
