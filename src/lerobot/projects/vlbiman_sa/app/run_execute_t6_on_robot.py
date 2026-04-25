#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
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
        repo_root / "lerobot_robot_cjjarm",
        repo_root,
    ]
    for path in reversed(extra_paths):
        if not path.exists():
            continue
        path_str = str(path)
        while path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)
    return repo_root


_maybe_reexec_in_repo_venv()
REPO_ROOT = _bootstrap_paths()

from lerobot_robot_cjjarm.cjjarm_robot import CjjArm
from lerobot_robot_cjjarm.config_cjjarm import CjjArmConfig


def _default_trajectory_points() -> Path:
    return REPO_ROOT / "outputs" / "vlbiman_sa" / "recordings" / "one_shot_full_20260329T014609" / "analysis" / "t6_trajectory" / "trajectory_points.json"


def _default_trajectory_summary(trajectory_points: Path) -> Path:
    return trajectory_points.parent / "summary.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a T6 joint trajectory on the real CJJArm robot.")
    parser.add_argument("--trajectory-points", type=Path, default=_default_trajectory_points())
    parser.add_argument("--trajectory-summary", type=Path, default=None, help="Optional T6 summary.json for continuity metadata.")
    parser.add_argument("--target-phrase", type=str, default="orange", help="Primary object phrase for this execution.")
    parser.add_argument(
        "--aux-target-phrase",
        action="append",
        default=[],
        help="Auxiliary object phrase for this execution; can be repeated.",
    )
    parser.add_argument("--robot-serial-port", type=str, default=None)
    parser.add_argument("--step-duration-s", type=float, default=0.05)
    parser.add_argument("--bridge-max-joint-step-rad", type=float, default=0.08)
    parser.add_argument("--initial-gripper-state", choices=("open", "closed"), default="open")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _joint_action_dict(joint_positions: np.ndarray, gripper_raw: float) -> dict[str, float]:
    joint_positions = np.asarray(joint_positions, dtype=float).reshape(-1)
    return {
        "joint_1.pos": float(joint_positions[0]),
        "joint_2.pos": float(joint_positions[1]),
        "joint_3.pos": float(joint_positions[2]),
        "joint_4.pos": float(joint_positions[3]),
        "joint_5.pos": float(joint_positions[4]),
        "joint_6.pos": float(joint_positions[5]),
        "gripper.pos": float(gripper_raw),
    }


def _bridge_joint_positions(current_q: np.ndarray, target_q: np.ndarray, max_step_rad: float) -> list[np.ndarray]:
    current_q = np.asarray(current_q, dtype=float)
    target_q = np.asarray(target_q, dtype=float)
    step_limit = max(float(max_step_rad), 1e-6)
    max_delta = float(np.max(np.abs(target_q - current_q)))
    if max_delta <= step_limit:
        return []
    step_count = max(1, int(np.ceil(max_delta / step_limit)))
    return [(1.0 - alpha) * current_q + alpha * target_q for alpha in np.linspace(1.0 / (step_count + 1), 1.0 - (1.0 / (step_count + 1)), num=step_count, dtype=float)]


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    return None


def _continuity_signal(
    trajectory_summary: dict[str, Any] | None,
    points: list[dict[str, Any]],
) -> tuple[bool, str]:
    summary = trajectory_summary if isinstance(trajectory_summary, dict) else {}
    for key in (
        "continuity_encoded",
        "continuity_from_current",
        "current_state_continuity",
        "start_from_current_state",
        "trajectory_starts_from_current_state",
    ):
        coerced = _coerce_bool(summary.get(key))
        if coerced is True:
            return True, f"summary:{key}"
    continuity_payload = summary.get("continuity")
    if isinstance(continuity_payload, dict):
        for key in (
            "enabled",
            "from_current_state",
            "encoded",
            "trajectory_starts_from_current_state",
        ):
            coerced = _coerce_bool(continuity_payload.get(key))
            if coerced is True:
                return True, f"summary:continuity.{key}"
    if points:
        first_point = points[0]
        source = str(first_point.get("source", "")).strip().lower()
        segment_id = str(first_point.get("segment_id", "")).strip().lower()
        for token in ("continuity", "current_state", "start_state", "state_seed"):
            if token in source:
                return True, f"first_point_source:{source}"
            if token in segment_id:
                return True, f"first_point_segment:{segment_id}"
    return False, "none"


def _plan_start_bridge(
    *,
    current_q: np.ndarray,
    first_target_q: np.ndarray,
    bridge_max_joint_step_rad: float,
    continuity_declared: bool,
    continuity_signal: str,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    current_q = np.asarray(current_q, dtype=float)
    first_target_q = np.asarray(first_target_q, dtype=float)
    step_limit = float(max(float(bridge_max_joint_step_rad), 1e-3))
    max_delta = float(np.max(np.abs(first_target_q - current_q)))

    if max_delta <= 1e-6:
        return [], {
            "bridge_injected": False,
            "bridge_point_count": 0,
            "bridge_reason": "already_aligned",
            "bridge_fallback_triggered": False,
            "continuity_declared_by_t6": continuity_declared,
            "continuity_signal": continuity_signal,
            "initial_joint_gap_rad_inf": max_delta,
            "bridge_step_limit_rad": step_limit,
        }

    if continuity_declared and max_delta <= step_limit:
        return [], {
            "bridge_injected": False,
            "bridge_point_count": 0,
            "bridge_reason": "continuity_declared_safe_gap",
            "bridge_fallback_triggered": False,
            "continuity_declared_by_t6": True,
            "continuity_signal": continuity_signal,
            "initial_joint_gap_rad_inf": max_delta,
            "bridge_step_limit_rad": step_limit,
        }

    bridge_points = _bridge_joint_positions(current_q, first_target_q, step_limit)
    fallback_triggered = continuity_declared and max_delta > step_limit
    bridge_reason = (
        "continuity_declared_gap_too_large_fallback_bridge"
        if fallback_triggered
        else "continuity_not_declared_bridge"
    )
    return bridge_points, {
        "bridge_injected": bool(bridge_points),
        "bridge_point_count": len(bridge_points),
        "bridge_reason": bridge_reason,
        "bridge_fallback_triggered": fallback_triggered,
        "continuity_declared_by_t6": continuity_declared,
        "continuity_signal": continuity_signal,
        "initial_joint_gap_rad_inf": max_delta,
        "bridge_step_limit_rad": step_limit,
    }


def _segment_gripper_command(segment_label: str | None, previous_raw: float) -> float:
    label = str(segment_label or "")
    if label == "gripper_close":
        return -1.0
    if label == "gripper_open":
        return 1.0
    return previous_raw


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)

    trajectory_points_path = args.trajectory_points
    trajectory_payload = _load_json(trajectory_points_path)
    points = list(trajectory_payload.get("points", []))
    if not points:
        raise ValueError(f"No points found in {trajectory_points_path}")

    trajectory_summary_path = args.trajectory_summary or _default_trajectory_summary(trajectory_points_path)
    trajectory_summary: dict[str, Any] | None = None
    if trajectory_summary_path.exists():
        payload = _load_json(trajectory_summary_path)
        if isinstance(payload, dict):
            trajectory_summary = payload
    continuity_declared, continuity_signal = _continuity_signal(trajectory_summary, points)

    robot_config = CjjArmConfig(
        serial_port=args.robot_serial_port or CjjArmConfig().serial_port,
        cameras={},
    )
    if not robot_config.serial_port and args.dry_run:
        robot_config.serial_port = "<dry-run>"
    if not robot_config.serial_port:
        raise RuntimeError("No robot serial port detected. Pass --robot-serial-port explicitly.")

    open_raw = 1.0
    closed_raw = -1.0
    current_gripper_raw = open_raw if args.initial_gripper_state == "open" else closed_raw

    robot = CjjArm(robot_config)
    try:
        if args.dry_run:
            logging.info("Dry-run enabled. Robot will not connect.")
            initial_q = np.asarray(points[0]["joint_positions"], dtype=float)
            bridge_points, bridge_decision = _plan_start_bridge(
                current_q=initial_q,
                first_target_q=initial_q,
                bridge_max_joint_step_rad=float(args.bridge_max_joint_step_rad),
                continuity_declared=continuity_declared,
                continuity_signal=continuity_signal,
            )
            logging.info(
                "Dry-run summary: target=%s aux_targets=%s point_count=%s robot_port=%s first_segment=%s bridge_count=%s bridge_reason=%s continuity_declared=%s continuity_signal=%s summary_path=%s",
                args.target_phrase,
                args.aux_target_phrase,
                len(points),
                robot_config.serial_port,
                points[0].get("segment_id"),
                len(bridge_points),
                bridge_decision["bridge_reason"],
                bridge_decision["continuity_declared_by_t6"],
                bridge_decision["continuity_signal"],
                trajectory_summary_path,
            )
            return 0

        robot.connect()
        observation = robot.get_observation()
        current_q = np.asarray(
            [observation[f"joint_{index}.pos"] for index in range(1, 7)],
            dtype=float,
        )
        first_target_q = np.asarray(points[0]["joint_positions"], dtype=float)
        bridge_points, bridge_decision = _plan_start_bridge(
            current_q=current_q,
            first_target_q=first_target_q,
            bridge_max_joint_step_rad=float(args.bridge_max_joint_step_rad),
            continuity_declared=continuity_declared,
            continuity_signal=continuity_signal,
        )
        logging.info(
            "Executing trajectory: target=%s aux_targets=%s port=%s bridge_points=%s trajectory_points=%s bridge_reason=%s continuity_declared=%s continuity_signal=%s initial_joint_gap_rad_inf=%.6f",
            args.target_phrase,
            args.aux_target_phrase,
            robot_config.serial_port,
            len(bridge_points),
            len(points),
            bridge_decision["bridge_reason"],
            bridge_decision["continuity_declared_by_t6"],
            bridge_decision["continuity_signal"],
            float(bridge_decision["initial_joint_gap_rad_inf"]),
        )

        for bridge_q in bridge_points:
            robot.send_action(_joint_action_dict(bridge_q, current_gripper_raw))
            time.sleep(max(float(args.step_duration_s), 1e-3))

        for point in points:
            current_gripper_raw = _segment_gripper_command(point.get("segment_label"), current_gripper_raw)
            robot.send_action(_joint_action_dict(np.asarray(point["joint_positions"], dtype=float), current_gripper_raw))
            time.sleep(max(float(args.step_duration_s), 1e-3))

        logging.info("Trajectory execution completed.")
        return 0
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
