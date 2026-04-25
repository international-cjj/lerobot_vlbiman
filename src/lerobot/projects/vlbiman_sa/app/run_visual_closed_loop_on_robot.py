#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
import glob
import json
import logging
import multiprocessing as mp
import os
import queue
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
    for path in (
        repo_root / "src",
        repo_root,
        repo_root / "lerobot_robot_cjjarm",
        repo_root / "lerobot_camera_gemini335l",
    ):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo_root


_maybe_reexec_in_repo_venv()
REPO_ROOT = _bootstrap_paths()

from lerobot.projects.vlbiman_sa.runtime_env import strip_user_site_packages

strip_user_site_packages()

from lerobot.projects.vlbiman_sa.app.run_live_grasp_preview import (
    _load_json,
    _load_task_config,
    _load_yaml,
    _run_t5_t6,
)
from lerobot.projects.vlbiman_sa.app.run_pose_adaptation import resolve_pose_target_phrases
from lerobot.projects.vlbiman_sa.app.run_visual_closed_loop_validation import (
    _await_single_capture_sample,
    _build_continuous_points,
    _build_execution_points,
    _current_segment_state,
    _default_capture_config_path,
    _default_live_output_root,
    _default_task_config_path,
    _default_vision_config_path,
    _drain_latest_sample_from_queue,
    _load_stability_config,
    _phrase_key,
    _pose_stream_worker,
    _resolve_repo_path,
    _save_json,
    _single_capture_stability,
    _stability_metrics,
    _stop_capture_process,
    _timestamp_name,
    _write_summary,
)
from lerobot_robot_cjjarm.config_cjjarm import CjjArmConfig
from lerobot_robot_cjjarm.cjjarm_robot import CjjArm


def _default_output_root() -> Path:
    return REPO_ROOT / "outputs" / "vlbiman_sa" / "robot_closed_loop"


def _auto_detect_robot_serial_port() -> str:
    ports = sorted(glob.glob("/dev/ttyACM*"))
    return ports[0] if ports else ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously localize the objects, replan T5/T6, and execute the trajectory on the real robot."
    )
    parser.add_argument("--task-config", type=Path, default=_default_task_config_path())
    parser.add_argument("--capture-config", type=Path, default=_default_capture_config_path())
    parser.add_argument("--vision-config", type=Path, default=_default_vision_config_path())
    parser.add_argument("--output-root", type=Path, default=_default_output_root())
    parser.add_argument("--live-output-root", type=Path, default=_default_live_output_root())
    parser.add_argument("--display", type=str, default=":1", help="Accepted for CLI compatibility; unused on robot execution.")
    parser.add_argument("--target-phrase", type=str, default=None)
    parser.add_argument("--container-target-phrase", type=str, default=None, help="Legacy alias for a single auxiliary target phrase.")
    parser.add_argument("--aux-target-phrase", action="append", default=None, help="Additional live target phrase to localize alongside the primary target. Repeat for multiple targets.")
    parser.add_argument("--camera-serial-number", type=str, default=None)
    parser.add_argument("--capture-mode", choices=("single_shot", "stream"), default="single_shot")
    parser.add_argument("--single-capture-timeout-s", type=float, default=20.0)
    parser.add_argument("--robot-serial-port", type=str, default=None)
    parser.add_argument("--stream-interval-s", type=float, default=1.0)
    parser.add_argument("--replan-distance-threshold-m", type=float, default=0.015)
    parser.add_argument("--step-duration-s", type=float, default=0.04)
    parser.add_argument("--stability-window-size", type=int, default=None)
    parser.add_argument("--position-variance-threshold-mm2", type=float, default=None)
    parser.add_argument("--orientation-variance-threshold-deg2", type=float, default=None)
    parser.add_argument("--bridge-max-joint-step-rad", type=float, default=0.08)
    parser.add_argument("--final-hold-s", type=float, default=1.0)
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--max-runtime-s", type=float, default=0.0)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--camera-timeout-ms", type=int, default=2500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _robot_joint_vector(observation: dict[str, Any]) -> np.ndarray:
    return np.asarray([observation[f"joint_{idx}.pos"] for idx in range(1, 7)], dtype=float)


def _segment_gripper_command(
    segment_label: str | None,
    previous_raw: float,
    *,
    open_raw: float,
    close_raw: float,
) -> float:
    label = str(segment_label or "")
    if label == "gripper_close":
        return float(close_raw)
    if label == "gripper_open":
        return float(open_raw)
    return float(previous_raw)


def _robot_action_dict(joint_positions: np.ndarray, gripper_raw: float) -> dict[str, float]:
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


def _load_session_home_state(session_dir: Path) -> tuple[np.ndarray, float]:
    metadata_path = session_dir / "metadata.jsonl"
    with metadata_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    if not first_line:
        raise RuntimeError(f"Recording metadata is empty: {metadata_path}")
    payload = json.loads(first_line)
    joint_payload = payload.get("joint_positions")
    if not isinstance(joint_payload, dict):
        raise RuntimeError(f"Recording metadata missing joint_positions in first frame: {metadata_path}")
    home_q = np.asarray(
        [joint_payload[f"joint_{idx}.pos"] for idx in range(1, 7)],
        dtype=float,
    )
    gripper_payload = payload.get("gripper_state")
    gripper_raw = 1.0
    if isinstance(gripper_payload, dict) and gripper_payload.get("gripper.pos") is not None:
        gripper_raw = float(gripper_payload["gripper.pos"])
    return home_q, gripper_raw


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)

    task_config = _load_task_config(args.task_config)
    task_config.handeye_result_path = _resolve_repo_path(task_config.handeye_result_path)
    task_config.recording_session_dir = _resolve_repo_path(task_config.recording_session_dir)
    task_config.intrinsics_path = _resolve_repo_path(task_config.intrinsics_path)
    task_config.transforms_path = _resolve_repo_path(task_config.transforms_path)
    task_config.skill_bank_path = _resolve_repo_path(task_config.skill_bank_path)
    task_config.target_phrase = args.target_phrase or task_config.target_phrase
    capture_config = _load_yaml(args.capture_config)
    stability_config = _load_stability_config(args, args.vision_config)
    home_joint_positions, recorded_home_gripper_raw = _load_session_home_state(task_config.recording_session_dir)

    aux_target_phrases: list[str] = []
    for phrase in list(args.aux_target_phrase or []):
        phrase = str(phrase).strip()
        if phrase:
            aux_target_phrases.append(phrase)
    legacy_container_phrase = str(args.container_target_phrase).strip() if args.container_target_phrase else ""
    if legacy_container_phrase:
        aux_target_phrases.insert(0, legacy_container_phrase)
    configured_aux_phrase = str(getattr(task_config, "secondary_target_phrase", "") or "").strip()
    if not aux_target_phrases and configured_aux_phrase:
        aux_target_phrases.append(configured_aux_phrase)
    deduped_aux_target_phrases: list[str] = []
    seen_aux_keys: set[str] = set()
    for phrase in aux_target_phrases:
        phrase_key = _phrase_key(phrase)
        if not phrase_key or phrase_key == _phrase_key(task_config.target_phrase) or phrase_key in seen_aux_keys:
            continue
        deduped_aux_target_phrases.append(phrase)
        seen_aux_keys.add(phrase_key)
    aux_target_phrases = deduped_aux_target_phrases
    planning_primary_target, planning_secondary_target = resolve_pose_target_phrases(
        task_config,
        auxiliary_target_phrases=aux_target_phrases,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    cycle_root = args.output_root / _timestamp_name("robot_loop")
    cycle_root.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Closed-loop planning targets: primary=%s secondary=%s aux=%s",
        planning_primary_target,
        planning_secondary_target or "<none>",
        aux_target_phrases,
    )

    sample_queue: mp.Queue = mp.Queue(maxsize=1)
    stop_event = mp.Event()
    capture_process = mp.Process(
        target=_pose_stream_worker,
        kwargs={
            "output_queue": sample_queue,
            "stop_event": stop_event,
            "capture_config_path": str(args.capture_config),
            "vision_config_path": str(args.vision_config),
            "handeye_result_path": str(task_config.handeye_result_path),
            "output_root": str(args.live_output_root),
            "camera_serial_number": args.camera_serial_number or capture_config.get("camera", {}).get("serial_number_or_name"),
            "target_phrase": task_config.target_phrase,
            "aux_target_phrases": list(aux_target_phrases),
            "warmup_frames": int(args.warmup_frames),
            "camera_timeout_ms": int(args.camera_timeout_ms),
            "stream_interval_s": float(args.stream_interval_s),
            "log_level": str(args.log_level),
        },
        daemon=True,
    )
    capture_process.start()

    robot_serial_port = args.robot_serial_port or _auto_detect_robot_serial_port()
    robot_config = CjjArmConfig(
        serial_port=robot_serial_port,
        zhongling_serial_port="",
        cameras={},
    )
    if bool(task_config.dry_run) and not args.dry_run:
        logging.info(
            "task_config.dry_run=%s is ignored by run_visual_closed_loop_on_robot.py; hardware motion is controlled by --dry-run.",
            task_config.dry_run,
        )
    if not args.dry_run and not robot_config.serial_port:
        raise RuntimeError("No robot serial port detected. Pass --robot-serial-port explicitly.")

    robot = CjjArm(robot_config)
    current_points: list[dict[str, Any]] = []
    current_index = 0
    current_joint_positions: np.ndarray | None = None
    current_target_base_xyz: np.ndarray | None = None
    current_aux_object_positions: dict[str, np.ndarray] = {}
    stable_target_base_xyz: np.ndarray | None = None
    last_replan_base_xyz: np.ndarray | None = None
    last_step_time = 0.0
    latest_cycle_summary: dict[str, Any] | None = None
    capture_cycle_count = 0
    replan_count = 0
    latest_live_result_path: Path | None = None
    pose_window: deque[dict[str, Any]] = deque(maxlen=stability_config.window_size)
    stability = _stability_metrics(pose_window, stability_config)
    stream_status = "starting"
    loop_started_at = time.monotonic()
    last_segment_key: tuple[str, str, str] | None = None
    startup_gripper_raw = float(robot_config.gripper_open_pos)
    if abs(float(recorded_home_gripper_raw) - startup_gripper_raw) > 1e-6:
        logging.info(
            "Ignoring recorded startup gripper raw %.4f and seeding open gripper raw %.4f for robot execution.",
            float(recorded_home_gripper_raw),
            startup_gripper_raw,
        )
    current_gripper_raw = startup_gripper_raw
    trajectory_completed_at: float | None = None
    stop_after_completion = False
    single_capture_mode = str(args.capture_mode) == "single_shot"

    try:
        if args.dry_run:
            logging.info("Dry-run enabled. Robot motion will be skipped.")
            current_joint_positions = np.asarray(home_joint_positions, dtype=float).copy()
            logging.info(
                "Dry-run seeded from recording home pose %s.",
                np.array2string(current_joint_positions, precision=4),
            )
        else:
            robot.connect()
            observed_joint_positions = _robot_joint_vector(robot.get_observation())
            logging.info("Real robot connected on %s", robot_config.serial_port)
            home_seed_point = {
                "joint_positions": np.asarray(home_joint_positions, dtype=float).astype(float).tolist(),
                "segment_id": "startup_home",
                "segment_label": "startup_home",
                "invariance": "bridge",
                "source": "startup_home",
                "translation_error_mm": 0.0,
                "rotation_error_deg": 0.0,
                "max_joint_step_rad": 0.0,
            }
            home_points = _build_continuous_points(
                current_joint_positions=observed_joint_positions,
                new_points=[home_seed_point],
                bridge_max_joint_step_rad=float(args.bridge_max_joint_step_rad),
            )
            for home_point in home_points:
                joint_positions = np.asarray(home_point["joint_positions"], dtype=float)
                robot.send_action(_robot_action_dict(joint_positions, current_gripper_raw))
                current_joint_positions = joint_positions
                time.sleep(max(float(args.step_duration_s), 1e-3))
            current_joint_positions = np.asarray(home_joint_positions, dtype=float).copy()
            logging.info(
                "Robot moved to recording home pose %s before closed-loop execution.",
                np.array2string(current_joint_positions, precision=4),
            )

        if single_capture_mode:
            _drain_latest_sample_from_queue(sample_queue)
            initial_sample = _await_single_capture_sample(
                sample_queue,
                timeout_s=float(args.single_capture_timeout_s),
            )
            stream_status = str(initial_sample.get("status", "ok"))
            current_target_base_xyz = np.asarray(initial_sample["base_xyz_m"], dtype=float).reshape(3)
            object_positions = initial_sample.get("object_positions_m") if isinstance(initial_sample.get("object_positions_m"), dict) else {}
            current_aux_object_positions = {
                str(name): np.asarray(position, dtype=float).reshape(3)
                for name, position in object_positions.items()
                if position is not None and str(name) != _phrase_key(task_config.target_phrase)
            }
            latest_live_result_path = Path(str(initial_sample["live_result_path"]))
            pose_window.clear()
            pose_window.append(initial_sample)
            stability = _single_capture_stability(initial_sample)
            stable_target_base_xyz = np.asarray(initial_sample["base_xyz_m"], dtype=float).reshape(3)
            capture_cycle_count = 1
            _stop_capture_process(stop_event, capture_process)
            logging.info(
                "Captured a single live scene for %s and stopped the camera stream.",
                task_config.target_phrase,
            )

        while True:
            now = time.monotonic()
            if float(args.max_runtime_s) > 0.0 and now - loop_started_at >= float(args.max_runtime_s):
                logging.info("Stopping robot closed loop after %.2f seconds.", now - loop_started_at)
                break

            drained_samples = 0
            latest_sample: dict[str, Any] | None = None
            if not single_capture_mode:
                drained_samples, latest_sample = _drain_latest_sample_from_queue(sample_queue)
            if latest_sample is not None:
                stream_status = str(latest_sample.get("status", "warn"))
                if latest_sample.get("status") == "ok" and latest_sample.get("base_xyz_m") is not None:
                    current_target_base_xyz = np.asarray(latest_sample["base_xyz_m"], dtype=float).reshape(3)
                    object_positions = latest_sample.get("object_positions_m") if isinstance(latest_sample.get("object_positions_m"), dict) else {}
                    current_aux_object_positions = {
                        str(name): np.asarray(position, dtype=float).reshape(3)
                        for name, position in object_positions.items()
                        if position is not None and str(name) != _phrase_key(task_config.target_phrase)
                    }
                    latest_live_result_path = Path(str(latest_sample["live_result_path"]))
                    pose_window.append(latest_sample)
                    stability = _stability_metrics(pose_window, stability_config)
                    stable_target = stability.get("mean_position_m")
                    stable_target_base_xyz = np.asarray(stable_target, dtype=float).reshape(3) if stable_target is not None else None
                    capture_cycle_count += drained_samples
                elif latest_sample.get("status") == "error":
                    logging.warning("Pose stream sample failed: %s", latest_sample.get("error"))

            if (
                latest_live_result_path is not None
                and stability.get("stable")
                and stable_target_base_xyz is not None
                and trajectory_completed_at is None
            ):
                should_replan = (
                    last_replan_base_xyz is None
                    or np.linalg.norm(stable_target_base_xyz - last_replan_base_xyz) >= float(args.replan_distance_threshold_m)
                )
                if should_replan:
                    run_dir = cycle_root / _timestamp_name("cycle")
                    run_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        pose_summary, trajectory_summary = _run_t5_t6(
                            task_config=task_config,
                            live_result_path=latest_live_result_path,
                            run_dir=run_dir,
                            aux_target_phrases=list(aux_target_phrases),
                        )
                        trajectory_payload = _load_json(run_dir / "analysis" / "t6_trajectory" / "trajectory_points.json")
                        planned_points = list(trajectory_payload.get("points", []))
                        if planned_points:
                            current_points, bridge_decision = _build_execution_points(
                                current_joint_positions=current_joint_positions,
                                planned_points=planned_points,
                                bridge_max_joint_step_rad=float(args.bridge_max_joint_step_rad),
                                trajectory_summary=trajectory_summary,
                            )
                            current_index = 0
                            last_replan_base_xyz = stable_target_base_xyz.copy()
                            replan_count += 1
                            segment_planning_modes = trajectory_summary.get("segment_planning_modes", {})
                            stop_after_completion = any(
                                str(mode) == "return_to_home_blend"
                                for mode in segment_planning_modes.values()
                            )
                            trajectory_completed_at = None
                            latest_cycle_summary = {
                                "status": "ok",
                                "run_dir": str(run_dir),
                                "live_result_path": str(latest_live_result_path),
                                "target_base_xyz_m": stable_target_base_xyz.astype(float).tolist(),
                                "stability": stability,
                                "bridge_point_count": int(bridge_decision["bridge_point_count"]),
                                "bridge_injected": bool(bridge_decision["bridge_injected"]),
                                "bridge_reason": str(bridge_decision["bridge_reason"]),
                                "bridge_fallback_triggered": bool(bridge_decision["bridge_fallback_triggered"]),
                                "continuity_declared_by_t6": bool(bridge_decision["continuity_declared_by_t6"]),
                                "continuity_signal": str(bridge_decision["continuity_signal"]),
                                "initial_joint_gap_rad_inf": bridge_decision["initial_joint_gap_rad_inf"],
                                "bridge_step_limit_rad": bridge_decision["bridge_step_limit_rad"],
                                "pose_summary": pose_summary,
                                "trajectory_summary": trajectory_summary,
                                "aux_target_phrases": list(aux_target_phrases),
                            }
                            _save_json(cycle_root / "latest_cycle.json", latest_cycle_summary)
                            logging.info(
                                "Replanned robot trajectory for %s at %s with %s points (%s bridge, reason=%s).",
                                task_config.target_phrase,
                                np.array2string(stable_target_base_xyz, precision=4),
                                len(current_points),
                                int(bridge_decision["bridge_point_count"]),
                                str(bridge_decision["bridge_reason"]),
                            )
                        else:
                            latest_cycle_summary = {
                                "status": "warn",
                                "reason": "empty_trajectory",
                                "run_dir": str(run_dir),
                                "live_result_path": str(latest_live_result_path),
                                "target_base_xyz_m": stable_target_base_xyz.astype(float).tolist(),
                                "aux_target_phrases": list(aux_target_phrases),
                            }
                            _save_json(cycle_root / "latest_cycle.json", latest_cycle_summary)
                            logging.warning("Robot closed-loop replan produced no trajectory points.")
                    except Exception as exc:
                        latest_cycle_summary = {
                            "status": "error",
                            "reason": "replan_failed",
                            "error": repr(exc),
                            "run_dir": str(run_dir),
                            "live_result_path": str(latest_live_result_path),
                            "target_base_xyz_m": stable_target_base_xyz.astype(float).tolist(),
                            "aux_target_phrases": list(aux_target_phrases),
                        }
                        _save_json(cycle_root / "latest_cycle.json", latest_cycle_summary)
                        logging.exception("Robot closed-loop replan failed.")

            _write_summary(
                cycle_root=cycle_root,
                latest_cycle_summary=latest_cycle_summary,
                current_target_base_xyz=current_target_base_xyz,
                current_aux_object_positions=current_aux_object_positions,
                stable_target_base_xyz=stable_target_base_xyz,
                replan_distance_threshold_m=float(args.replan_distance_threshold_m),
                stream_interval_s=float(args.stream_interval_s),
                capture_cycle_count=capture_cycle_count,
                replan_count=replan_count,
                stability=stability,
                stream_status=stream_status,
                status="running",
            )
            if int(args.max_cycles) > 0 and capture_cycle_count >= int(args.max_cycles):
                logging.info("Stopping robot closed loop after %s capture cycles.", capture_cycle_count)
                break

            active_segment_index = current_index
            if current_points and now - last_step_time >= max(float(args.step_duration_s), 1e-3):
                point = current_points[current_index]
                joint_positions = np.asarray(point["joint_positions"], dtype=float)
                current_gripper_raw = _segment_gripper_command(
                    point.get("segment_label"),
                    current_gripper_raw,
                    open_raw=float(robot_config.gripper_open_pos),
                    close_raw=float(robot_config.gripper_closed_pos),
                )
                if not args.dry_run and robot.is_connected:
                    robot.send_action(_robot_action_dict(joint_positions, current_gripper_raw))
                current_joint_positions = joint_positions
                if current_index < len(current_points) - 1:
                    current_index += 1
                elif stop_after_completion and trajectory_completed_at is None:
                    trajectory_completed_at = now
                    logging.info(
                        "Robot reached final return-home point; holding for %.2f seconds before stopping.",
                        float(args.final_hold_s),
                    )
                last_step_time = now

            if (
                stop_after_completion
                and trajectory_completed_at is not None
                and now - trajectory_completed_at >= max(float(args.final_hold_s), 0.0)
            ):
                logging.info("Robot completed final home hold. Stopping closed-loop execution.")
                break

            current_segment = _current_segment_state(current_points, active_segment_index)
            if current_segment is None:
                last_segment_key = None
            else:
                segment_key = (
                    str(current_segment.get("segment_id", "unknown_segment")),
                    str(current_segment.get("invariance", "unknown")),
                    str(current_segment.get("source", "unknown")),
                )
                if segment_key != last_segment_key:
                    logging.info(
                        "Active robot segment: %s label=%s invariance=%s source=%s",
                        current_segment.get("segment_id", "unknown_segment"),
                        current_segment.get("segment_label", ""),
                        current_segment.get("invariance", "unknown"),
                        current_segment.get("source", "unknown"),
                    )
                    last_segment_key = segment_key
            time.sleep(0.005)
    finally:
        _stop_capture_process(stop_event, capture_process)
        if robot.is_connected:
            robot.disconnect()

    if latest_cycle_summary is not None or capture_cycle_count > 0:
        summary_payload = _write_summary(
            cycle_root=cycle_root,
            latest_cycle_summary=latest_cycle_summary,
            current_target_base_xyz=current_target_base_xyz,
            current_aux_object_positions=current_aux_object_positions,
            stable_target_base_xyz=stable_target_base_xyz,
            replan_distance_threshold_m=float(args.replan_distance_threshold_m),
            stream_interval_s=float(args.stream_interval_s),
            capture_cycle_count=capture_cycle_count,
            replan_count=replan_count,
            stability=stability,
            stream_status=stream_status,
            status="ok" if latest_cycle_summary is not None else "warn",
        )
        print(json.dumps(summary_payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
