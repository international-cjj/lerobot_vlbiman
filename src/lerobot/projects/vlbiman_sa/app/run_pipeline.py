#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

STAGE_ORDER = ("handeye", "record", "skill_build", "vision", "grasp")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "pipeline.yaml"


def _runner_python() -> str:
    override = os.environ.get("PYTHON_BIN")
    if override:
        return override
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_python = Path(conda_prefix) / "bin" / "python"
        if conda_python.exists():
            return str(conda_python)
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        venv_python = Path(virtual_env) / "bin" / "python"
        if venv_python.exists():
            return str(venv_python)
    default_conda_python = Path.home() / "miniconda3" / "envs" / "lerobot" / "bin" / "python"
    if default_conda_python.exists():
        return str(default_conda_python)
    return sys.executable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the VLBiMan single-arm reproduction pipeline.")
    parser.add_argument("--config", type=Path, default=_default_config_path(), help="Path to pipeline.yaml.")
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Comma-separated stage list. Choices: handeye,record,grasp.",
    )
    parser.add_argument("--camera-serial-number", type=str, default=None, help="Override camera serial number.")
    parser.add_argument("--robot-serial-port", type=str, default=None, help="Override robot serial port.")
    parser.add_argument("--teleop-port", type=str, default=None, help="Override Zhonglin teleop serial port.")
    parser.add_argument("--control-rate-hz", type=float, default=None, help="Override record stage control rate.")
    parser.add_argument("--duration-s", type=float, default=None, help="Override record stage duration.")
    parser.add_argument("--max-frames", type=int, default=None, help="Override record stage max frames.")
    parser.add_argument("--no-teleop-calibrate", action="store_true", help="Skip teleop calibration in record stage.")
    parser.add_argument("--dry-run", action="store_true", help="Force grasp stage into dry-run mode.")
    parser.add_argument("--plan-only", action="store_true", help="Print the planned commands without executing.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping, got {type(payload).__name__}.")
    return payload


def _normalize_stages(raw_stages: str | None, payload: dict[str, Any]) -> list[str]:
    if raw_stages:
        stages = [item.strip().lower() for item in raw_stages.split(",") if item.strip()]
    else:
        stage_cfg = dict(payload.get("stages", {}))
        stages = [name for name in STAGE_ORDER if bool(stage_cfg.get(name, {}).get("enabled", False))]

    if not stages:
        raise ValueError("No stages selected. Use --stages or enable stages in pipeline.yaml.")

    invalid = [name for name in stages if name not in STAGE_ORDER]
    if invalid:
        raise ValueError(f"Unsupported stages: {invalid}. Supported stages: {list(STAGE_ORDER)}.")

    seen: set[str] = set()
    normalized: list[str] = []
    for name in stages:
        if name in seen:
            continue
        seen.add(name)
        normalized.append(name)
    return normalized


def _script_path(stage_name: str) -> Path:
    repo_root = _repo_root()
    mapping = {
        "handeye": repo_root / "src/lerobot/projects/vlbiman_sa/calib/run_handeye_auto.py",
        "record": repo_root / "src/lerobot/projects/vlbiman_sa/app/run_one_shot_record.py",
        "skill_build": repo_root / "src/lerobot/projects/vlbiman_sa/app/run_skill_build.py",
        "vision": repo_root / "src/lerobot/projects/vlbiman_sa/app/run_visual_analysis.py",
        "grasp": repo_root / "src/lerobot/projects/vlbiman_sa/app/run_grasp_online.py",
    }
    return mapping[stage_name]


def _stringify_path(value: Any) -> str:
    return str(value) if isinstance(value, Path) else str(value)


def _append_optional_arg(cmd: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    cmd.extend([flag, _stringify_path(value)])


def build_stage_command(stage_name: str, payload: dict[str, Any], args: argparse.Namespace) -> list[str]:
    stage_cfg = dict(payload.get("stages", {}).get(stage_name, {}))
    hardware_cfg = dict(payload.get("hardware", {}))
    cmd = [_runner_python(), str(_script_path(stage_name))]

    config_path = stage_cfg.get("config")
    if config_path:
        _append_optional_arg(cmd, "--config", config_path)

    if stage_name == "handeye":
        _append_optional_arg(cmd, "--log-level", args.log_level)
        return cmd

    if stage_name == "record":
        duration_s = args.duration_s if args.duration_s is not None else stage_cfg.get("duration_s")
        control_rate_hz = args.control_rate_hz if args.control_rate_hz is not None else stage_cfg.get("control_rate_hz")
        max_frames = args.max_frames if args.max_frames is not None else stage_cfg.get("max_frames")
        camera_serial_number = (
            args.camera_serial_number
            if args.camera_serial_number is not None
            else stage_cfg.get("camera_serial_number", hardware_cfg.get("camera_serial_number"))
        )
        robot_serial_port = (
            args.robot_serial_port
            if args.robot_serial_port is not None
            else stage_cfg.get("robot_serial_port", hardware_cfg.get("robot_serial_port"))
        )
        teleop_port = (
            args.teleop_port
            if args.teleop_port is not None
            else stage_cfg.get("teleop_port", hardware_cfg.get("teleop_port"))
        )
        no_teleop_calibrate = bool(
            args.no_teleop_calibrate or stage_cfg.get("no_teleop_calibrate", False)
        )

        _append_optional_arg(cmd, "--duration-s", duration_s)
        _append_optional_arg(cmd, "--control-rate-hz", control_rate_hz)
        _append_optional_arg(cmd, "--max-frames", max_frames)
        _append_optional_arg(cmd, "--camera-serial-number", camera_serial_number)
        _append_optional_arg(cmd, "--robot-serial-port", robot_serial_port)
        _append_optional_arg(cmd, "--teleop-port", teleop_port)
        if no_teleop_calibrate:
            cmd.append("--no-teleop-calibrate")
        _append_optional_arg(cmd, "--log-level", args.log_level)
        return cmd

    if stage_name in {"skill_build", "vision"}:
        _append_optional_arg(cmd, "--log-level", args.log_level)
        return cmd

    if stage_name == "grasp":
        dry_run = bool(args.dry_run or stage_cfg.get("dry_run", False))
        if dry_run:
            cmd.append("--dry-run")
        _append_optional_arg(cmd, "--log-level", args.log_level)
        return cmd

    raise ValueError(f"Unsupported stage: {stage_name}")


def _run_stage(stage_name: str, cmd: list[str], repo_root: Path) -> int:
    logging.info("Running stage %s", stage_name)
    logging.info("Command: %s", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=repo_root, check=False)
    return int(completed.returncode)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)

    repo_root = _repo_root()
    payload = _load_yaml(args.config)
    stages = _normalize_stages(args.stages, payload)

    commands = [(stage_name, build_stage_command(stage_name, payload, args)) for stage_name in stages]
    for stage_name, cmd in commands:
        logging.info("Planned stage %s: %s", stage_name, " ".join(cmd))

    if args.plan_only:
        return 0

    for stage_name, cmd in commands:
        return_code = _run_stage(stage_name, cmd, repo_root)
        if return_code != 0:
            logging.error("Stage %s failed with exit code %s.", stage_name, return_code)
            return return_code

    logging.info("Pipeline completed successfully: %s", ",".join(stages))
    return 0


if __name__ == "__main__":
    sys.exit(main())
