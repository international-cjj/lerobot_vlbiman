#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import statistics
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
from lerobot.projects.vlbiman_sa.grasp.coarse_handoff import COARSE_HANDOFF_SOURCE, CoarseHandoffError, build_frrg_input_from_coarse_summary
from lerobot.projects.vlbiman_sa.grasp.contracts import load_frrg_config
from lerobot.projects.vlbiman_sa.grasp.parameters import apply_theta_overrides, load_theta_samples
from lerobot.projects.vlbiman_sa.grasp.state_machine import PHASE_FAILURE, PHASE_SUCCESS


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "frrg_grasp.yaml"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FRRG benchmark on mock/coarse fixtures.")
    parser.add_argument("--config", type=Path, default=_default_config_path(), help="Path to FRRG YAML config.")
    parser.add_argument("--fixtures", type=Path, required=True, help="Fixture file or directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for benchmark outputs.")
    parser.add_argument("--max-steps", type=int, default=80, help="Maximum steps per episode.")
    parser.add_argument("--theta-path", type=Path, default=None, help="Optional theta_samples.json file.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Python logging level.")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}.")
    return payload


def _discover_fixture_paths(fixtures: Path) -> list[Path]:
    resolved = fixtures.resolve()
    if resolved.is_file():
        return [resolved]
    if not resolved.is_dir():
        raise FileNotFoundError(f"fixtures path not found: {resolved}")
    return [
        path
        for path in sorted(resolved.glob("*.json"))
        if path.name.startswith("frrg_") or path.name.startswith("coarse_")
    ]


def _mujoco_available() -> bool:
    try:
        import mujoco  # noqa: F401
    except Exception:
        return False
    return True


def _episode_from_run_result(
    *,
    fixture_path: Path,
    input_kind: str,
    run_result: Any,
) -> dict[str, Any]:
    terminal_success = run_result.final_phase == PHASE_SUCCESS
    failure_reason = run_result.failure_reason
    if not terminal_success and failure_reason is None:
        failure_reason = "max_steps_exhausted"
    return {
        "episode_id": fixture_path.stem,
        "fixture_path": str(fixture_path),
        "input_kind": input_kind,
        "status": "success" if terminal_success else "failure",
        "final_phase": run_result.final_phase,
        "failure_reason": failure_reason,
        "steps_run": int(run_result.steps_run),
        "phase_trace": list(run_result.phase_trace),
        "max_raw_action_norm": float(run_result.max_raw_action_norm),
        "max_safe_action_norm": float(run_result.max_safe_action_norm),
        "max_residual_norm": float(run_result.max_residual_norm),
        "hardware_called": False,
        "camera_opened": False,
    }


def _run_fixture_episode(config, fixture_path: Path, max_steps: int) -> dict[str, Any]:
    payload = _load_json(fixture_path)
    input_kind = "coarse" if fixture_path.name.startswith("coarse_") else "mock"
    if input_kind == "coarse":
        try:
            handoff_payload = build_frrg_input_from_coarse_summary(payload)
        except CoarseHandoffError as error:
            return {
                "episode_id": fixture_path.stem,
                "fixture_path": str(fixture_path),
                "input_kind": input_kind,
                "status": "failure",
                "final_phase": PHASE_FAILURE,
                "failure_reason": error.reason,
                "steps_run": 0,
                "phase_trace": [],
                "max_raw_action_norm": 0.0,
                "max_safe_action_norm": 0.0,
                "max_residual_norm": 0.0,
                "hardware_called": False,
                "camera_opened": False,
                "failure_report": {"message": str(error), "missing_fields": list(error.missing_fields)},
            }
        controller = FRRGClosedLoopController(config, handoff_payload, input_mode=COARSE_HANDOFF_SOURCE)
        return _episode_from_run_result(fixture_path=fixture_path, input_kind=input_kind, run_result=controller.run(max_steps=max_steps))

    controller = FRRGClosedLoopController(config, payload, input_mode=config.runtime.default_input_mode)
    return _episode_from_run_result(fixture_path=fixture_path, input_kind=input_kind, run_result=controller.run(max_steps=max_steps))


def _render_report(metrics: dict[str, Any], episodes: list[dict[str, Any]]) -> str:
    lines = [
        "# FRRG Benchmark Report",
        "",
        f"- episode_count: {metrics['episode_count']}",
        f"- success_count: {metrics['success_count']}",
        f"- failure_count: {metrics['failure_count']}",
        f"- success_rate: {metrics['success_rate']:.3f}",
        f"- average_steps: {metrics['average_steps']:.3f}",
        f"- max_action_step: {metrics['max_action_step']:.6f}",
        f"- hardware_called: {str(metrics['hardware_called']).lower()}",
        f"- mujoco_available: {str(metrics['mujoco_available']).lower()}",
        "",
        "## Failure Reasons",
    ]
    if metrics["failure_reason_top_list"]:
        for item in metrics["failure_reason_top_list"]:
            lines.append(f"- {item['reason']}: {item['count']}")
    else:
        lines.append("- none")
    lines.extend(["", "## Episodes"])
    for episode in episodes:
        lines.append(
            f"- {episode['episode_id']}: status={episode['status']} final_phase={episode['final_phase']} "
            f"failure_reason={episode['failure_reason']} steps={episode['steps_run']} "
            f"max_safe_action_norm={episode['max_safe_action_norm']:.6f}"
        )
    return "\n".join(lines) + "\n"


def run_benchmark(
    *,
    config_path: Path,
    fixtures: Path,
    output_dir: Path,
    max_steps: int,
    theta_path: Path | None = None,
) -> tuple[int, Path, dict[str, Any]]:
    config = load_frrg_config(config_path)
    theta_applied = None
    theta_samples = None
    if theta_path is not None:
        theta_samples = load_theta_samples(theta_path)
        theta_applied = apply_theta_overrides(config, theta_samples)
        config = theta_applied.config

    fixture_paths = _discover_fixture_paths(fixtures)
    episodes = [_run_fixture_episode(config, fixture_path, int(max_steps)) for fixture_path in fixture_paths]
    failure_reasons = Counter(
        episode["failure_reason"]
        for episode in episodes
        if episode["status"] != "success" and episode["failure_reason"] is not None
    )
    success_count = sum(1 for episode in episodes if episode["status"] == "success")
    episode_count = len(episodes)
    metrics = {
        "config_path": str(config_path.resolve()),
        "fixtures_path": str(fixtures.resolve()),
        "output_dir": str(output_dir.resolve()),
        "episode_count": episode_count,
        "success_count": success_count,
        "failure_count": episode_count - success_count,
        "success_rate": (success_count / episode_count) if episode_count else 0.0,
        "average_steps": statistics.fmean(episode["steps_run"] for episode in episodes) if episodes else 0.0,
        "max_action_step": max((episode["max_safe_action_norm"] for episode in episodes), default=0.0),
        "max_raw_action_norm": max((episode["max_raw_action_norm"] for episode in episodes), default=0.0),
        "max_safe_action_norm": max((episode["max_safe_action_norm"] for episode in episodes), default=0.0),
        "max_residual_norm": max((episode["max_residual_norm"] for episode in episodes), default=0.0),
        "failure_reason_top_list": [{"reason": reason, "count": count} for reason, count in failure_reasons.most_common()],
        "hardware_called": False,
        "camera_opened": False,
        "mujoco_available": _mujoco_available(),
        "mujoco_validation_status": "not_run",
    }
    if theta_path is not None and theta_samples is not None and theta_applied is not None:
        metrics["theta_path"] = str(theta_path.resolve())
        metrics["theta_samples"] = theta_samples.to_dict()
        metrics["theta_overrides_applied"] = dict(theta_applied.applied_overrides)
        metrics["unused_theta_parameters"] = list(theta_applied.unused_parameters)

    resolved_output_dir = output_dir.resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = resolved_output_dir / "metrics.json"
    episodes_path = resolved_output_dir / "episodes.jsonl"
    report_path = resolved_output_dir / "report.md"

    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    episodes_path.write_text("".join(json.dumps(episode, ensure_ascii=False) + "\n" for episode in episodes), encoding="utf-8")
    report_path.write_text(_render_report(metrics, episodes), encoding="utf-8")
    return 0, metrics_path, metrics


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)
    exit_code, metrics_path, _ = run_benchmark(
        config_path=args.config,
        fixtures=args.fixtures,
        output_dir=args.output_dir,
        max_steps=int(args.max_steps),
        theta_path=args.theta_path,
    )
    logging.info("Benchmark metrics written to %s", metrics_path)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
