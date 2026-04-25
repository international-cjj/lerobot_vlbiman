#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _bootstrap_pythonpath() -> None:
    repo_root = _repo_root()
    for candidate in (repo_root / "src", repo_root):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


_bootstrap_pythonpath()

from lerobot.projects.vlbiman_sa.grasp.contracts import load_frrg_config
from lerobot.projects.vlbiman_sa.grasp.parameters import apply_theta_overrides, extract_theta_from_session


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "frrg_grasp.yaml"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract FRRG theta parameters from a one-shot demo session.")
    parser.add_argument("--session-dir", type=Path, required=True, help="Path to the one-shot session directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for theta extraction outputs.")
    parser.add_argument("--config", type=Path, default=_default_config_path(), help="Path to FRRG YAML config.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Python logging level.")
    return parser.parse_args(argv)


def run_extract_demo(
    *,
    session_dir: Path,
    output_dir: Path,
    config_path: Path,
) -> tuple[int, Path, Path, dict[str, object]]:
    config = load_frrg_config(config_path)
    extraction_result = extract_theta_from_session(session_dir, config)
    apply_result = apply_theta_overrides(config, extraction_result.theta)

    resolved_output_dir = output_dir.resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    theta_path = resolved_output_dir / "theta_samples.json"
    report_path = resolved_output_dir / "extraction_report.json"

    theta_payload = extraction_result.theta_samples_dict()
    theta_path.write_text(json.dumps(theta_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = extraction_result.report_dict()
    report.update(
        {
            "config_path": str(config_path.resolve()),
            "theta_samples_path": str(theta_path),
            "dryrun_loadable": True,
            "theta_overrides_applied": dict(apply_result.applied_overrides),
            "unused_theta_parameters": list(apply_result.unused_parameters),
        }
    )
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0, theta_path, report_path, report


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)
    exit_code, theta_path, report_path, _ = run_extract_demo(
        session_dir=args.session_dir,
        output_dir=args.output_dir,
        config_path=args.config,
    )
    logging.info("Theta samples written to %s", theta_path)
    logging.info("Extraction report written to %s", report_path)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
