#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


CONDA_LEROBOT_ROOT = Path("/home/cjj/miniconda3/envs/lerobot")


def _maybe_reexec_in_conda_lerobot() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    conda_root = Path(os.environ.get("VLBIMAN_CONDA_LEROBOT_PREFIX", CONDA_LEROBOT_ROOT))
    conda_python = conda_root / "bin" / "python"
    already_conda = Path(sys.prefix).resolve() == conda_root.resolve() or Path(sys.executable).resolve() == conda_python.resolve()
    if already_conda and os.environ.get("PYTHONNOUSERSITE") == "1":
        return
    if not conda_python.exists():
        return
    env = os.environ.copy()
    env["VLBIMAN_CONDA_LEROBOT_REEXEC"] = "1"
    env["CONDA_PREFIX"] = str(conda_root)
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("VIRTUAL_ENV", None)
    env["PATH"] = os.pathsep.join([str(conda_root / "bin"), env.get("PATH", "")])
    pythonpath = [str(repo_root / "src"), str(repo_root)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    os.execve(str(conda_python), [str(conda_python), __file__, *sys.argv[1:]], env)


_maybe_reexec_in_conda_lerobot()


def _bootstrap_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[5]
    for path in (repo_root / "src", repo_root):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)
    return repo_root


REPO_ROOT = _bootstrap_paths()

from lerobot.projects.vlbiman_sa.inv_servo.config import (
    InvServoConfigError,
    default_inv_rgb_servo_config_path,
    load_inv_rgb_servo_config,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run the inv RGB visual-servo grasp pipeline.")
    parser.add_argument("--config", type=Path, default=default_inv_rgb_servo_config_path())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-json", action="store_true", help="Print the loaded config summary as JSON.")
    return parser.parse_args()


def _json_dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def main() -> None:
    args = _parse_args()
    try:
        config = load_inv_rgb_servo_config(args.config)
    except (FileNotFoundError, InvServoConfigError, ValueError) as exc:
        print("CONFIG_LOAD_FAILED")
        print(f"failure_reason={exc}")
        raise SystemExit(1) from exc

    effective_dry_run = bool(args.dry_run or config.backend.dry_run)
    print("CONFIG_LOAD_OK")
    print(f"DRY_RUN={'true' if effective_dry_run else 'false'}")

    summary = config.dry_run_summary(cli_dry_run=args.dry_run)
    print(f"TARGET_PHRASE={summary['target_phrase']}")
    print(f"ORIGINAL_FLOW_DIR={summary['original_flow_dir']}")
    print(f"GROUNDINGDINO_REPO_PATH={summary['groundingdino_repo_path']}")
    print(f"GROUNDINGDINO_CHECK_FRAME={summary['groundingdino_check_frame']}")
    print(f"SAM2_VALIDATION_RANGE={summary['sam2_validation_range'][0]}-{summary['sam2_validation_range'][1]}")
    print(f"SERVO_VALIDATION_RANGE={summary['servo_validation_range'][0]}-{summary['servo_validation_range'][1]}")
    if args.print_json:
        print(_json_dump(summary))

    if effective_dry_run:
        return

    print("RUN_NOT_IMPLEMENTED")
    print("failure_reason=inv_rgb_servo_execution_will_be_enabled_after_detector_sam2_and_backend_stages")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
