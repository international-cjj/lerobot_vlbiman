#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def _bootstrap_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[5]
    for path in (repo_root / "src", repo_root, repo_root / "lerobot_robot_cjjarm"):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo_root


_maybe_reexec_in_repo_venv()
REPO_ROOT = _bootstrap_paths()

from lerobot.projects.vlbiman_sa.app.export_t6_visuals import ExportT6Config, export_t6_visuals
from lerobot.projects.vlbiman_sa.app.run_pose_adaptation import build_pose_pipeline_config, run_pose_adaptation_pipeline
from lerobot.projects.vlbiman_sa.app.run_trajectory_generation import (
    TrajectoryPipelineConfig,
    run_trajectory_generation_pipeline,
)
from lerobot.projects.vlbiman_sa.core.contracts import TaskGraspConfig
from lerobot.projects.vlbiman_sa.skills import InvarianceClassifierConfig, SegmenterConfig, build_skill_bank


@dataclass(slots=True)
class ClosedLoopArtifacts:
    run_dir: Path
    analysis_dir: Path
    skill_bank_path: Path
    pose_summary_path: Path
    trajectory_summary_path: Path
    trajectory_points_path: Path
    summary_path: Path


def _default_task_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "task_grasp.yaml"


def _default_skill_build_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "skill_build.yaml"


def _default_output_root() -> Path:
    return REPO_ROOT / "outputs" / "vlbiman_sa" / "t3_closed_loop_preview"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild T3, rerun T5/T6, and preview the new trajectory in MuJoCo.")
    parser.add_argument("--task-config", type=Path, default=_default_task_config_path())
    parser.add_argument("--skill-build-config", type=Path, default=_default_skill_build_config_path())
    parser.add_argument("--session-dir", type=Path, default=None)
    parser.add_argument("--live-result-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=_default_output_root())
    parser.add_argument("--display", type=str, default=":1")
    parser.add_argument("--no-launch-mujoco", action="store_true")
    parser.add_argument("--kill-existing-viewers", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _timestamp_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(payload).__name__}.")
    return payload


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_repo_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (REPO_ROOT / path)


def _load_task_config(path: Path) -> TaskGraspConfig:
    payload = _load_yaml(path)
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


def _link_or_copy_tree(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    source = source.resolve()
    try:
        destination.symlink_to(source, target_is_directory=True)
    except OSError:
        shutil.copytree(source, destination)


def _segments_payload(skill_bank_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(skill_bank_path.read_text(encoding="utf-8"))
    return [
        {
            "segment_id": item["segment_id"],
            "start_frame": int(item["start_frame"]),
            "end_frame": int(item["end_frame"]),
            "label": str(item["label"]),
            "invariance": str(item.get("invariance", "unknown")),
            "frame_count": int(item["frame_count"]),
        }
        for item in payload.get("segments", [])
    ]


def _kill_existing_viewers() -> None:
    subprocess.run(
        ["pkill", "-f", "src/lerobot/projects/vlbiman_sa/app/play_mujoco_trajectory.py"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _launch_mujoco(trajectory_points_path: Path, display: str) -> int | None:
    env = os.environ.copy()
    env["DISPLAY"] = display
    env["PYTHONPATH"] = "src:lerobot_robot_cjjarm"
    process = subprocess.Popen(
        [
            sys.executable,
            str(REPO_ROOT / "src" / "lerobot" / "projects" / "vlbiman_sa" / "app" / "play_mujoco_trajectory.py"),
            "--trajectory-points",
            str(trajectory_points_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return int(process.pid)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)

    task_config = _load_task_config(args.task_config)
    session_dir = _resolve_repo_path(args.session_dir or task_config.recording_session_dir)
    if session_dir is None:
        raise ValueError("task_config.recording_session_dir is required.")
    live_result_path = _resolve_repo_path(args.live_result_path or task_config.live_result_path)
    if live_result_path is None:
        raise ValueError("Provide --live-result-path or set live_result_path in task_config.")
    if not live_result_path.exists():
        raise FileNotFoundError(f"Live result not found: {live_result_path}")

    run_dir = args.output_root / _timestamp_name("t3_loop")
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    skill_payload = _load_yaml(args.skill_build_config)
    segmenter_config = SegmenterConfig(**dict(skill_payload.get("segmenter", {})))
    classifier_config = InvarianceClassifierConfig(**dict(skill_payload.get("classifier", {})))
    skill_result = build_skill_bank(
        session_dir=session_dir,
        output_dir=analysis_dir / "t3_skill_bank",
        segmenter_config=segmenter_config,
        classifier_config=classifier_config,
    )

    source_t4_dir = session_dir / "analysis" / "t4_vision"
    if not source_t4_dir.exists():
        raise FileNotFoundError(f"T4 analysis not found: {source_t4_dir}")
    _link_or_copy_tree(source_t4_dir, analysis_dir / "t4_vision")

    pose_summary = run_pose_adaptation_pipeline(
        build_pose_pipeline_config(
            task_config=task_config,
            session_dir=session_dir,
            analysis_dir=analysis_dir,
            output_dir=analysis_dir / "t5_pose",
            live_result_path=live_result_path,
        )
    )
    trajectory_summary = run_trajectory_generation_pipeline(
        TrajectoryPipelineConfig(
            session_dir=session_dir,
            analysis_dir=analysis_dir,
            output_dir=analysis_dir / "t6_trajectory",
            skill_bank_path=analysis_dir / "t3_skill_bank" / "skill_bank.json",
            adapted_pose_path=analysis_dir / "t5_pose" / "adapted_pose.json",
        )
    )
    visuals = export_t6_visuals(
        ExportT6Config(
            session_dir=session_dir,
            analysis_dir=analysis_dir,
            t6_dir=analysis_dir / "t6_trajectory",
            t5_dir=analysis_dir / "t5_pose",
            output_dir=analysis_dir / "t6_trajectory" / "visuals",
        )
    )

    if args.kill_existing_viewers:
        _kill_existing_viewers()
    mujoco_pid = None
    if not args.no_launch_mujoco:
        mujoco_pid = _launch_mujoco(analysis_dir / "t6_trajectory" / "trajectory_points.json", args.display)

    artifacts = ClosedLoopArtifacts(
        run_dir=run_dir,
        analysis_dir=analysis_dir,
        skill_bank_path=analysis_dir / "t3_skill_bank" / "skill_bank.json",
        pose_summary_path=analysis_dir / "t5_pose" / "summary.json",
        trajectory_summary_path=analysis_dir / "t6_trajectory" / "summary.json",
        trajectory_points_path=analysis_dir / "t6_trajectory" / "trajectory_points.json",
        summary_path=run_dir / "summary.json",
    )
    payload = {
        "status": "ok",
        "run_dir": str(artifacts.run_dir),
        "analysis_dir": str(artifacts.analysis_dir),
        "session_dir": str(session_dir),
        "live_result_path": str(live_result_path),
        "skill_build_config": str(args.skill_build_config),
        "skill_bank_path": str(artifacts.skill_bank_path),
        "pose_summary_path": str(artifacts.pose_summary_path),
        "trajectory_summary_path": str(artifacts.trajectory_summary_path),
        "trajectory_points_path": str(artifacts.trajectory_points_path),
        "t6_visuals_index_path": str(analysis_dir / "t6_trajectory" / "visuals" / "index.json"),
        "mujoco_pid": mujoco_pid,
        "pose_summary": pose_summary,
        "trajectory_summary": trajectory_summary,
        "t6_visuals": visuals,
        "segments": _segments_payload(artifacts.skill_bank_path),
    }
    _save_json(artifacts.summary_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
