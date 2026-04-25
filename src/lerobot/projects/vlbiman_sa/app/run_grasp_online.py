#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml


def _maybe_reexec_in_repo_venv() -> None:
    if os.environ.get("PYTHON_BIN") or os.environ.get("CONDA_PREFIX") or os.environ.get("VIRTUAL_ENV"):
        return
    repo_root = Path(__file__).resolve().parents[5]
    default_conda_python = Path.home() / "miniconda3" / "envs" / "lerobot" / "bin" / "python"
    repo_python = default_conda_python if default_conda_python.exists() else Path(sys.executable)
    if not repo_python.exists():
        return
    if Path(sys.executable).resolve() == repo_python.resolve():
        return
    if os.environ.get("VLBIMAN_REEXEC") == "1":
        return
    env = os.environ.copy()
    env["VLBIMAN_REEXEC"] = "1"
    os.execve(str(repo_python), [str(repo_python), __file__, *sys.argv[1:]], env)


_maybe_reexec_in_repo_venv()

try:
    from lerobot.projects.vlbiman_sa.core.contracts import TaskGraspConfig
    from lerobot.projects.vlbiman_sa.core.grasp_orchestrator import GraspOrchestrator
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[5]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from lerobot.projects.vlbiman_sa.core.contracts import TaskGraspConfig
    from lerobot.projects.vlbiman_sa.core.grasp_orchestrator import GraspOrchestrator


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "task_grasp.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VLBiMan single-arm online grasp pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="Path to task_grasp.yaml.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Override config and run without hardware.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def _load_config(config_path: Path) -> TaskGraspConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a dictionary, got {type(payload).__name__}.")

    if "data_root" in payload:
        payload["data_root"] = Path(payload["data_root"])
    if "transforms_path" in payload:
        payload["transforms_path"] = Path(payload["transforms_path"])
    for key in (
        "handeye_result_path",
        "recording_session_dir",
        "skill_output_dir",
        "skill_bank_path",
        "vision_output_dir",
        "pose_output_dir",
        "trajectory_output_dir",
        "live_result_path",
        "intrinsics_path",
    ):
        if key in payload and payload[key] is not None:
            payload[key] = Path(payload[key])
    return TaskGraspConfig(**payload)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)

    cfg = _load_config(args.config)
    if args.dry_run:
        cfg.dry_run = True

    orchestrator = GraspOrchestrator(cfg)
    result = orchestrator.run()

    logging.info("Execution status: %s", result["status"])
    logging.info("Plan summary: %s", result["plan"])
    logging.info("Effective config: %s", orchestrator.dump_effective_config())
    return 0


if __name__ == "__main__":
    sys.exit(main())
