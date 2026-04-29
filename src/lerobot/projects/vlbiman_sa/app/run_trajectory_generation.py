#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


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
    extra_paths = [
        repo_root / "src",
        repo_root,
        repo_root / "lerobot_robot_cjjarm",
    ]
    for path in extra_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo_root


_maybe_reexec_in_repo_venv()
REPO_ROOT = _bootstrap_paths()

from lerobot.projects.vlbiman_sa.demo.io import load_frame_records
from lerobot.projects.vlbiman_sa.skills import SkillBank
from lerobot.projects.vlbiman_sa.trajectory import (
    ComposedTrajectory,
    TrajectoryComposer,
    TrajectoryComposerConfig,
    build_ikpy_state,
    forward_kinematics_tool,
    full_q_from_arm_q,
)


@dataclass(slots=True)
class TrajectoryPipelineConfig:
    session_dir: Path
    analysis_dir: Path
    output_dir: Path
    skill_bank_path: Path
    adapted_pose_path: Path
    current_joint_positions: list[float] | None = None
    current_joint_positions_source: str | None = None
    start_segment_id: str | None = None
    start_after_segment_id: str | None = None


def _default_session_dir() -> Path:
    return Path("outputs/vlbiman_sa/recordings/one_shot_20260323T185847Z")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose T6 trajectory and solve it with progressive IK.")
    parser.add_argument("--session-dir", type=Path, default=_default_session_dir())
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skill-bank-path", type=Path, default=None)
    parser.add_argument("--adapted-pose-path", type=Path, default=None)
    parser.add_argument(
        "--current-joint-positions",
        type=str,
        default=None,
        help="Comma/space separated 6-joint vector in joint_1..joint_6 order for continuity-aware T6 start.",
    )
    parser.add_argument(
        "--current-joint-positions-path",
        type=Path,
        default=None,
        help="JSON path for continuity-aware start joints (list[6] or {'joint_positions': list[6]}).",
    )
    parser.add_argument(
        "--start-segment-id",
        type=str,
        default=None,
        help="Only generate a suffix beginning with this segment id.",
    )
    parser.add_argument(
        "--start-after-segment-id",
        type=str,
        default=None,
        help="Only generate a suffix beginning after this segment id.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> TrajectoryPipelineConfig:
    analysis_dir = args.analysis_dir or (args.session_dir / "analysis")
    output_dir = args.output_dir or (analysis_dir / "t6_trajectory")
    current_joint_positions: list[float] | None = None
    current_joint_positions_source: str | None = None
    if args.current_joint_positions and args.current_joint_positions_path is not None:
        raise ValueError("Specify either --current-joint-positions or --current-joint-positions-path, not both.")
    if args.current_joint_positions:
        current_joint_positions = _parse_joint_positions_text(args.current_joint_positions)
        current_joint_positions_source = "cli_inline"
    elif args.current_joint_positions_path is not None:
        current_joint_positions = _load_joint_positions_payload(args.current_joint_positions_path)
        current_joint_positions_source = str(args.current_joint_positions_path)
    return TrajectoryPipelineConfig(
        session_dir=args.session_dir,
        analysis_dir=analysis_dir,
        output_dir=output_dir,
        skill_bank_path=args.skill_bank_path or (analysis_dir / "t3_skill_bank" / "skill_bank.json"),
        adapted_pose_path=args.adapted_pose_path or (analysis_dir / "t5_pose" / "adapted_pose.json"),
        current_joint_positions=current_joint_positions,
        current_joint_positions_source=current_joint_positions_source,
        start_segment_id=args.start_segment_id,
        start_after_segment_id=args.start_after_segment_id,
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_joint_positions_text(raw: str) -> list[float]:
    tokens = [token for token in re.split(r"[,\s]+", raw.strip()) if token]
    values = [float(token) for token in tokens]
    if len(values) != 6:
        raise ValueError(f"Expected 6 joint values, got {len(values)} from: {raw!r}")
    return values


def _load_joint_positions_payload(path: Path) -> list[float]:
    payload = _load_json(path)
    if isinstance(payload, dict) and "joint_positions" in payload:
        payload = payload["joint_positions"]
    if not isinstance(payload, list):
        raise ValueError(f"Invalid joint payload at {path}: expected list[6] or dict with joint_positions.")
    values = [float(item) for item in payload]
    if len(values) != 6:
        raise ValueError(f"Invalid joint payload at {path}: expected 6 values, got {len(values)}.")
    return values


def _demo_pose_matrices(records: list[Any], ik_state: Any, joint_keys: list[str]) -> dict[int, np.ndarray]:
    matrices: dict[int, np.ndarray] = {}
    for record in records:
        arm_q = np.asarray([float(record.joint_positions[key]) for key in joint_keys], dtype=float)
        q = full_q_from_arm_q(ik_state, arm_q)
        matrices[int(record.frame_index)] = forward_kinematics_tool(ik_state, q)
    return matrices


def _filter_skill_bank_segments(
    skill_bank: SkillBank,
    *,
    start_segment_id: str | None,
    start_after_segment_id: str | None,
) -> tuple[SkillBank, dict[str, Any]]:
    if start_segment_id and start_after_segment_id:
        raise ValueError("Specify either start_segment_id or start_after_segment_id, not both.")

    segments = list(skill_bank.segments)
    segment_ids = [str(segment.segment_id) for segment in segments]
    filter_summary: dict[str, Any] = {
        "mode": "all",
        "start_segment_id": start_segment_id,
        "start_after_segment_id": start_after_segment_id,
        "original_segment_count": len(segments),
        "selected_segment_count": len(segments),
        "first_selected_segment_id": segment_ids[0] if segment_ids else None,
    }
    if not start_segment_id and not start_after_segment_id:
        return skill_bank, filter_summary

    requested_id = start_segment_id or start_after_segment_id
    assert requested_id is not None
    if requested_id not in segment_ids:
        raise ValueError(f"Segment id {requested_id!r} not found in skill bank. Available: {segment_ids}")

    requested_index = segment_ids.index(requested_id)
    start_index = requested_index + (1 if start_after_segment_id else 0)
    selected = segments[start_index:]
    filter_summary.update(
        {
            "mode": "suffix_after" if start_after_segment_id else "suffix_from",
            "requested_segment_id": requested_id,
            "requested_segment_index": requested_index,
            "selected_start_index": start_index,
            "selected_segment_count": len(selected),
            "first_selected_segment_id": str(selected[0].segment_id) if selected else None,
        }
    )

    return (
        SkillBank(
            session_dir=skill_bank.session_dir,
            output_dir=skill_bank.output_dir,
            frame_count=skill_bank.frame_count,
            joint_keys=list(skill_bank.joint_keys),
            segments=selected,
            summary={**skill_bank.summary, "segment_filter": filter_summary},
        ),
        filter_summary,
    )


def run_trajectory_generation_pipeline(config: TrajectoryPipelineConfig) -> dict[str, Any]:
    records = load_frame_records(config.session_dir)
    skill_bank = SkillBank.load(config.skill_bank_path)
    skill_bank, segment_filter_summary = _filter_skill_bank_segments(
        skill_bank,
        start_segment_id=config.start_segment_id,
        start_after_segment_id=config.start_after_segment_id,
    )
    adapted_pose = _load_json(config.adapted_pose_path)
    ik_state = build_ikpy_state()
    composer = TrajectoryComposer(ik_state, config=TrajectoryComposerConfig())
    demo_pose_matrices = _demo_pose_matrices(records, ik_state, skill_bank.joint_keys)
    trajectory = composer.compose(
        records=records,
        skill_bank=skill_bank,
        adapted_pose=adapted_pose,
        demo_pose_matrices=demo_pose_matrices,
        start_joint_positions=config.current_joint_positions,
    )
    trajectory.summary["segment_filter"] = segment_filter_summary

    config.output_dir.mkdir(parents=True, exist_ok=True)
    points_path = config.output_dir / "trajectory_points.json"
    summary_path = config.output_dir / "summary.json"
    npz_path = config.output_dir / "trajectory_joint_path.npz"
    _save_json(points_path, trajectory.to_dict())
    _save_json(summary_path, trajectory.summary)
    np.savez_compressed(
        npz_path,
        joint_keys=np.asarray(trajectory.joint_keys),
        joint_positions=np.asarray([point.joint_positions for point in trajectory.points], dtype=np.float32),
        relative_time_s=np.asarray([point.relative_time_s for point in trajectory.points], dtype=np.float32),
        gripper_positions=np.asarray(
            [np.nan if point.gripper_position is None else point.gripper_position for point in trajectory.points],
            dtype=np.float32,
        ),
    )

    summary = {
        **trajectory.summary,
        "output_dir": str(config.output_dir),
        "trajectory_points_path": str(points_path),
        "trajectory_npz_path": str(npz_path),
        "summary_path": str(summary_path),
        "adapted_pose_path": str(config.adapted_pose_path),
        "skill_bank_path": str(config.skill_bank_path),
    }
    if config.current_joint_positions is not None:
        summary["continuity_input_joint_positions"] = [float(value) for value in config.current_joint_positions]
        summary["continuity_input_source"] = config.current_joint_positions_source
    if config.start_segment_id is not None:
        summary["start_segment_id"] = config.start_segment_id
    if config.start_after_segment_id is not None:
        summary["start_after_segment_id"] = config.start_after_segment_id
    _save_json(summary_path, summary)
    return summary


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)
    config = _build_config(args)
    summary = run_trajectory_generation_pipeline(config)
    logging.info("T6 trajectory output: %s", summary["output_dir"])
    logging.info("T6 summary: %s", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
