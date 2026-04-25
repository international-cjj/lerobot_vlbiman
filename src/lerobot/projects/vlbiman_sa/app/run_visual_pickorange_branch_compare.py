#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

import cv2
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

from lerobot.projects.vlbiman_sa.runtime_env import (
    prepare_mujoco_runtime,
    require_mujoco_viewer_backend,
    strip_user_site_packages,
)

prepare_mujoco_runtime(argv=sys.argv[1:])
strip_user_site_packages()

import mujoco
import mujoco.viewer

from lerobot.projects.vlbiman_sa.app.run_live_grasp_preview import (
    _capture_live_result,
    _load_json,
    _load_task_config,
    _run_t5_t6,
)
from lerobot.projects.vlbiman_sa.app.run_visual_closed_loop_validation import (
    _apply_camera,
    _build_stage_overlay_texts,
    _build_execution_points,
    _current_segment_state,
    _joint_qpos_indices,
    _load_segment_display_labels,
    _resolve_repo_path,
    _set_robot_qpos,
    _update_target_markers,
)
from lerobot.projects.vlbiman_sa.app.run_visual_pickorange_frrg_validation import (
    _aux_object_positions,
    _build_coarse_summary,
    _build_renderer,
    _compile_frrg_segment,
    _default_capture_config_path,
    _default_frrg_config_path,
    _default_live_output_root,
    _default_model_path,
    _default_task_config_path,
    _default_vision_config_path,
    _gripper_qpos_indices,
    _gripper_target_from_raw,
    _phrase_key,
    _point_gripper_raw,
    _render_frame,
    _save_json,
    _segment_semantic_states,
    _set_gripper_qpos,
    _split_pick_segment,
    _target_object_payload,
)
from lerobot.projects.vlbiman_sa.grasp.contracts import load_frrg_config
from lerobot_robot_cjjarm.config_cjjarm_sim import CjjArmSimConfig
from lerobot_robot_cjjarm.kinematics import CjjArmKinematics


def _default_output_root() -> Path:
    return REPO_ROOT / "outputs" / "vlbiman_sa" / "visual_pickorange_compare"


def _timestamp_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start from a mid-trajectory anchor and compare the original VLBiMan replay "
            "against the FRRG replacement branch inside the same pick-orange sim scene."
        )
    )
    parser.add_argument("--task-config", type=Path, default=_default_task_config_path())
    parser.add_argument("--capture-config", type=Path, default=_default_capture_config_path())
    parser.add_argument("--vision-config", type=Path, default=_default_vision_config_path())
    parser.add_argument("--frrg-config", type=Path, default=_default_frrg_config_path())
    parser.add_argument("--output-root", type=Path, default=_default_output_root())
    parser.add_argument("--live-output-root", type=Path, default=_default_live_output_root())
    parser.add_argument("--reuse-live-result", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=_default_model_path())
    parser.add_argument("--display", type=str, default=None)
    parser.add_argument("--target-phrase", type=str, default=None)
    parser.add_argument("--aux-target-phrase", action="append", default=None)
    parser.add_argument("--camera-serial-number", type=str, default=None)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--camera-timeout-ms", type=int, default=2500)
    parser.add_argument("--bridge-max-joint-step-rad", type=float, default=0.08)
    parser.add_argument("--frrg-max-steps", type=int, default=80)
    parser.add_argument("--start-index", type=int, default=47)
    parser.add_argument(
        "--jump-to-pick-after-index",
        type=int,
        default=None,
        help="If set before the grasp segment, keep points up to this index and then jump directly to the pick stage.",
    )
    parser.add_argument(
        "--stop-after-pick",
        action="store_true",
        help="Stop after the grasp segment instead of continuing into transfer/place/retreat.",
    )
    parser.add_argument(
        "--frrg-playback-repeat",
        type=int,
        default=1,
        help="Repeat each FRRG playback point this many times. 3 means FRRG runs at one third speed.",
    )
    parser.add_argument("--step-duration-s", type=float, default=0.04)
    parser.add_argument("--final-hold-s", type=float, default=1.0)
    parser.add_argument(
        "--branch",
        choices=("both", "original_replay", "frrg_replacement"),
        default="both",
        help="Select which branch to render. 'both' keeps the A/B compare behavior.",
    )
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--video-fps", type=float, default=12.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--show-left-ui", action="store_true")
    parser.add_argument("--show-right-ui", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _renumber_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    renumbered: list[dict[str, Any]] = []
    for idx, point in enumerate(points):
        row = dict(point)
        row["trajectory_index"] = idx
        renumbered.append(row)
    return renumbered


def _hold_pick_points_from_current_pose(
    pick_points: list[dict[str, Any]],
    *,
    current_joint_positions: np.ndarray,
    anchor_index: int,
) -> list[dict[str, Any]]:
    held_points: list[dict[str, Any]] = []
    current_joint_positions = np.asarray(current_joint_positions, dtype=float).tolist()
    for point_offset, point in enumerate(pick_points):
        row = dict(point)
        row["joint_positions"] = list(current_joint_positions)
        row["source"] = "hold_arm_from_current"
        row["hold_from_anchor_index"] = int(anchor_index)
        row["original_segment_source"] = str(point.get("source", ""))
        row["original_trajectory_index"] = point.get("trajectory_index")
        held_points.append(row)
    return held_points


def _repeat_playback_points(
    points: list[dict[str, Any]],
    *,
    repeat_count: int,
    playback_source: str,
) -> list[dict[str, Any]]:
    repeat_count = max(int(repeat_count), 1)
    if repeat_count == 1:
        return [dict(point) for point in points]

    repeated: list[dict[str, Any]] = []
    for point in points:
        for repeat_index in range(repeat_count):
            row = dict(point)
            row["playback_repeat_index"] = int(repeat_index)
            row["playback_repeat_count"] = int(repeat_count)
            row["playback_source"] = playback_source
            repeated.append(row)
    return repeated


def _gripper_raw_before_index(points: list[dict[str, Any]], index: int) -> float:
    previous_raw = 0.0
    for point in points[: max(int(index), 0)]:
        previous_raw = _point_gripper_raw(point, previous_raw)
    return float(previous_raw)


def _segment_counter(points: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for point in points:
        counter[str(point.get("segment_label", "unknown"))] += 1
    return dict(counter)


def _contiguous_segments(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    if not points:
        return segments
    start = 0
    while start < len(points):
        segment_id = str(points[start].get("segment_id", ""))
        segment_label = str(points[start].get("segment_label", ""))
        end = start + 1
        while end < len(points) and str(points[end].get("segment_id", "")) == segment_id:
            end += 1
        joint_rows = np.asarray([points[idx]["joint_positions"] for idx in range(start, end)], dtype=float)
        segments.append(
            {
                "segment_id": segment_id,
                "segment_label": segment_label,
                "source": str(points[start].get("source", "")),
                "invariance": str(points[start].get("invariance", "")),
                "point_count": int(end - start),
                "start_index": int(start),
                "end_index": int(end - 1),
                "joint_span_rad_inf": float(np.max(np.ptp(joint_rows, axis=0))) if len(joint_rows) > 0 else 0.0,
            }
        )
        start = end
    return segments


def _overlay_text(frame_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    out = frame_bgr.copy()
    y = 28
    for line in lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)
        y += 28
    return out


def _render_branch(
    *,
    branch_name: str,
    branch_points: list[dict[str, Any]],
    full_planned_points: list[dict[str, Any]],
    anchor_index: int,
    model_path: Path,
    kinematics: CjjArmKinematics,
    output_dir: Path,
    render_width: int,
    render_height: int,
    video_fps: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    joint_qpos = _joint_qpos_indices(model)
    gripper_qpos = _gripper_qpos_indices(model)
    sim_config = CjjArmSimConfig()

    previous_gripper_raw = _gripper_raw_before_index(full_planned_points, anchor_index)
    previous_joint_positions: np.ndarray | None = None
    previous_pose: np.ndarray | None = None
    max_joint_step_rad = 0.0
    total_joint_step_rad_inf = 0.0
    ee_path_translation_mm = 0.0
    step_rows: list[dict[str, Any]] = []

    renderer, camera = _build_renderer(model, width=int(render_width), height=int(render_height))
    video_path = output_dir / "validation.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(video_fps),
        (
            int(min(int(render_width), int(model.vis.global_.offwidth))),
            int(min(int(render_height), int(model.vis.global_.offheight))),
        ),
    )
    if not writer.isOpened():
        renderer.close()
        raise RuntimeError(f"Failed to open video writer: {video_path}")

    try:
        for point_index, point in enumerate(branch_points):
            joint_positions = np.asarray(point["joint_positions"], dtype=float)
            _set_robot_qpos(data, joint_qpos, joint_positions)
            previous_gripper_raw = _point_gripper_raw(point, previous_gripper_raw)
            _set_gripper_qpos(data, gripper_qpos, _gripper_target_from_raw(previous_gripper_raw, sim_config))

            frame_rgb = _render_frame(model, data, renderer, camera)

            current_pose = np.asarray(kinematics.compute_fk(joint_positions), dtype=float)
            if previous_joint_positions is None:
                joint_step_rad = 0.0
                ee_step_translation_mm = 0.0
            else:
                joint_step_rad = float(np.max(np.abs(joint_positions - previous_joint_positions)))
                ee_step_translation_mm = float(np.linalg.norm(current_pose[:3] - previous_pose[:3]) * 1000.0)
            max_joint_step_rad = max(max_joint_step_rad, joint_step_rad)
            total_joint_step_rad_inf += joint_step_rad
            ee_path_translation_mm += ee_step_translation_mm
            previous_joint_positions = joint_positions
            previous_pose = current_pose

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_bgr = _overlay_text(
                frame_bgr,
                [
                    f"branch={branch_name}",
                    f"point={point_index + 1}/{len(branch_points)} anchor={anchor_index}",
                    f"segment={point.get('segment_id', '')}:{point.get('segment_label', '')}",
                    f"source={point.get('source', '')}",
                ],
            )
            writer.write(frame_bgr)
            if point_index == 0:
                cv2.imwrite(str(output_dir / "frame_000.png"), frame_bgr)
            if point_index == len(branch_points) - 1:
                cv2.imwrite(str(output_dir / "frame_last.png"), frame_bgr)

            step_rows.append(
                {
                    "point_index": int(point_index),
                    "segment_id": str(point.get("segment_id", "")),
                    "segment_label": str(point.get("segment_label", "")),
                    "source": str(point.get("source", "")),
                    "invariance": str(point.get("invariance", "")),
                    "joint_step_rad_inf": joint_step_rad,
                    "ee_step_translation_mm": ee_step_translation_mm,
                    "gripper_raw": float(previous_gripper_raw),
                }
            )
    finally:
        writer.release()
        renderer.close()

    summary = {
        "status": "ok",
        "branch_name": branch_name,
        "output_dir": str(output_dir.resolve()),
        "video_path": str(video_path.resolve()),
        "frame0_path": str((output_dir / "frame_000.png").resolve()),
        "frame_last_path": str((output_dir / "frame_last.png").resolve()),
        "point_count": int(len(branch_points)),
        "executed_steps_after_anchor": int(max(len(branch_points) - 1, 0)),
        "segment_label_counts": _segment_counter(branch_points),
        "contiguous_segments": _contiguous_segments(branch_points),
        "max_joint_step_rad_inf": float(max_joint_step_rad),
        "total_joint_step_rad_inf": float(total_joint_step_rad_inf),
        "ee_path_translation_mm": float(ee_path_translation_mm),
        "start_segment_id": None if not branch_points else str(branch_points[0].get("segment_id", "")),
        "start_segment_label": None if not branch_points else str(branch_points[0].get("segment_label", "")),
        "end_segment_id": None if not branch_points else str(branch_points[-1].get("segment_id", "")),
        "end_segment_label": None if not branch_points else str(branch_points[-1].get("segment_label", "")),
    }
    _save_json(output_dir / "summary.json", summary)
    (output_dir / "step_metrics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in step_rows),
        encoding="utf-8",
    )
    return summary


def _play_branch_in_viewer(
    *,
    branch_name: str,
    branch_points: list[dict[str, Any]],
    full_planned_points: list[dict[str, Any]],
    anchor_index: int,
    model_path: Path,
    kinematics: CjjArmKinematics,
    output_dir: Path,
    target_base_xyz: np.ndarray,
    aux_object_positions: dict[str, np.ndarray],
    stage_labels: dict[str, str],
    step_duration_s: float,
    final_hold_s: float,
    show_left_ui: bool,
    show_right_ui: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    joint_qpos = _joint_qpos_indices(model)
    gripper_qpos = _gripper_qpos_indices(model)
    sim_config = CjjArmSimConfig()

    current_index = 0
    current_gripper_raw = _gripper_raw_before_index(full_planned_points, anchor_index)
    last_step_time = 0.0
    hold_started_at: float | None = None
    last_segment_key: tuple[str, str, str] | None = None
    previous_joint_positions: np.ndarray | None = None
    previous_pose: np.ndarray | None = None
    max_joint_step_rad = 0.0
    total_joint_step_rad_inf = 0.0
    ee_path_translation_mm = 0.0
    step_rows: list[dict[str, Any]] = []

    require_mujoco_viewer_backend()
    viewer = mujoco.viewer.launch_passive(
        model,
        data,
        show_left_ui=bool(show_left_ui),
        show_right_ui=bool(show_right_ui),
    )
    _apply_camera(viewer)
    try:
        while viewer.is_running():
            now = time.monotonic()
            if current_index < len(branch_points) and now - last_step_time >= max(float(step_duration_s), 1e-3):
                point = branch_points[current_index]
                joint_positions = np.asarray(point["joint_positions"], dtype=float)
                _set_robot_qpos(data, joint_qpos, joint_positions)
                current_gripper_raw = _point_gripper_raw(point, current_gripper_raw)
                _set_gripper_qpos(data, gripper_qpos, _gripper_target_from_raw(current_gripper_raw, sim_config))

                current_pose = np.asarray(kinematics.compute_fk(joint_positions), dtype=float)
                if previous_joint_positions is None:
                    joint_step_rad = 0.0
                    ee_step_translation_mm = 0.0
                else:
                    joint_step_rad = float(np.max(np.abs(joint_positions - previous_joint_positions)))
                    ee_step_translation_mm = float(np.linalg.norm(current_pose[:3] - previous_pose[:3]) * 1000.0)
                max_joint_step_rad = max(max_joint_step_rad, joint_step_rad)
                total_joint_step_rad_inf += joint_step_rad
                ee_path_translation_mm += ee_step_translation_mm
                previous_joint_positions = joint_positions
                previous_pose = current_pose

                step_rows.append(
                    {
                        "point_index": int(current_index),
                        "segment_id": str(point.get("segment_id", "")),
                        "segment_label": str(point.get("segment_label", "")),
                        "source": str(point.get("source", "")),
                        "invariance": str(point.get("invariance", "")),
                        "joint_step_rad_inf": joint_step_rad,
                        "ee_step_translation_mm": ee_step_translation_mm,
                        "gripper_raw": float(current_gripper_raw),
                    }
                )
                last_step_time = now
                current_index += 1
            elif current_index >= len(branch_points):
                if hold_started_at is None:
                    hold_started_at = now
                elif now - hold_started_at >= max(float(final_hold_s), 0.0):
                    break

            active_segment_index = max(0, min(current_index - 1, max(len(branch_points) - 1, 0)))
            current_segment = _current_segment_state(branch_points, active_segment_index) if branch_points else None
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
                        "Viewer branch=%s active segment: %s label=%s invariance=%s source=%s",
                        branch_name,
                        current_segment.get("segment_id", "unknown_segment"),
                        current_segment.get("segment_label", ""),
                        current_segment.get("invariance", "unknown"),
                        current_segment.get("source", "unknown"),
                    )
                    last_segment_key = segment_key

            mujoco.mj_forward(model, data)
            _update_target_markers(viewer, target_base_xyz, aux_object_positions, current_segment)
            viewer.set_texts(
                _build_stage_overlay_texts(
                    current_segment,
                    stage_labels=stage_labels,
                    point_index=active_segment_index,
                    point_count=len(branch_points),
                )
            )
            viewer.sync()
            time.sleep(0.005)
    finally:
        try:
            viewer.clear_texts()
        except Exception:
            logging.exception("Failed to clear viewer texts.")
        viewer.close()

    summary = {
        "status": "ok",
        "branch_name": branch_name,
        "render_mode": "viewer",
        "output_dir": str(output_dir.resolve()),
        "video_path": None,
        "frame0_path": None,
        "frame_last_path": None,
        "point_count": int(len(branch_points)),
        "executed_steps_after_anchor": int(max(len(branch_points) - 1, 0)),
        "segment_label_counts": _segment_counter(branch_points),
        "contiguous_segments": _contiguous_segments(branch_points),
        "max_joint_step_rad_inf": float(max_joint_step_rad),
        "total_joint_step_rad_inf": float(total_joint_step_rad_inf),
        "ee_path_translation_mm": float(ee_path_translation_mm),
        "start_segment_id": None if not branch_points else str(branch_points[0].get("segment_id", "")),
        "start_segment_label": None if not branch_points else str(branch_points[0].get("segment_label", "")),
        "end_segment_id": None if not branch_points else str(branch_points[-1].get("segment_id", "")),
        "end_segment_label": None if not branch_points else str(branch_points[-1].get("segment_label", "")),
    }
    _save_json(output_dir / "summary.json", summary)
    (output_dir / "step_metrics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in step_rows),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)
    if args.display is not None:
        os.environ["DISPLAY"] = args.display

    task_config = _load_task_config(args.task_config)
    task_config.handeye_result_path = _resolve_repo_path(task_config.handeye_result_path)
    task_config.recording_session_dir = _resolve_repo_path(task_config.recording_session_dir)
    task_config.intrinsics_path = _resolve_repo_path(task_config.intrinsics_path)
    task_config.transforms_path = _resolve_repo_path(task_config.transforms_path)
    task_config.skill_bank_path = _resolve_repo_path(task_config.skill_bank_path)
    task_config.target_phrase = args.target_phrase or task_config.target_phrase

    aux_target_phrases: list[str] = []
    configured_aux_phrase = str(getattr(task_config, "secondary_target_phrase", "") or "").strip()
    if configured_aux_phrase:
        aux_target_phrases.append(configured_aux_phrase)
    for phrase in list(args.aux_target_phrase or []):
        phrase = str(phrase).strip()
        if phrase and _phrase_key(phrase) not in {_phrase_key(item) for item in aux_target_phrases}:
            aux_target_phrases.append(phrase)

    args.output_root.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_root / _timestamp_name("pickorange_compare")
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_live_result is not None:
        live_result_path = _resolve_repo_path(args.reuse_live_result)
    else:
        live_result_path = _capture_live_result(
            capture_config_path=args.capture_config,
            vision_config_path=args.vision_config,
            handeye_result_path=task_config.handeye_result_path,
            output_root=args.live_output_root,
            camera_serial_number=args.camera_serial_number or task_config.camera_serial_number,
            target_phrase=task_config.target_phrase,
            warmup_frames=int(args.warmup_frames),
            camera_timeout_ms=int(args.camera_timeout_ms),
        )
    live_result = _load_json(live_result_path)

    pose_summary, trajectory_summary = _run_t5_t6(
        task_config=task_config,
        live_result_path=live_result_path,
        run_dir=run_dir,
        aux_target_phrases=list(aux_target_phrases),
    )

    trajectory_points_path = run_dir / "analysis" / "t6_trajectory" / "trajectory_points.json"
    trajectory_payload = _load_json(trajectory_points_path)
    planned_points = list(trajectory_payload.get("points", []))
    if not planned_points:
        raise ValueError(f"No points found in {trajectory_points_path}")

    anchor_index = int(args.start_index)
    if anchor_index < 0 or anchor_index >= len(planned_points):
        raise ValueError(f"start-index={anchor_index} is outside trajectory range [0, {len(planned_points) - 1}]")

    semantic_states = _segment_semantic_states(task_config.skill_bank_path)
    prefix_points, pick_points, suffix_points = _split_pick_segment(planned_points, semantic_states)
    pick_start = len(prefix_points)
    if anchor_index >= pick_start:
        raise ValueError(
            f"start-index={anchor_index} is not before pick segment start={pick_start}; "
            "this compare flow expects an anchor before pickorange."
        )
    jump_to_pick_after_index = None if args.jump_to_pick_after_index is None else int(args.jump_to_pick_after_index)
    if jump_to_pick_after_index is not None:
        if jump_to_pick_after_index < anchor_index:
            raise ValueError(
                f"jump-to-pick-after-index={jump_to_pick_after_index} is before start-index={anchor_index}."
            )
        if jump_to_pick_after_index >= pick_start:
            raise ValueError(
                f"jump-to-pick-after-index={jump_to_pick_after_index} must be before pick segment start={pick_start}."
            )

    target_object_payload = _target_object_payload(live_result, task_config.target_phrase)
    current_aux_object_positions = _aux_object_positions(live_result, task_config.target_phrase)

    sim_config = CjjArmSimConfig()
    kinematics = CjjArmKinematics(
        urdf_path=sim_config.urdf_path,
        end_effector_frame=sim_config.end_effector_frame,
        joint_names=[sim_config.urdf_joint_map[name] for name in sim_config.joint_action_order],
    )

    selected_branch = str(args.branch)
    if not bool(args.headless) and selected_branch == "both":
        raise ValueError("GUI mode only supports a single branch. Pass --branch original_replay or --branch frrg_replacement.")
    need_original_branch = selected_branch in {"both", "original_replay"}
    need_frrg_branch = selected_branch in {"both", "frrg_replacement"}
    current_target_base_xyz = np.asarray(target_object_payload["base_xyz_m"], dtype=float).reshape(3)
    stage_labels = _load_segment_display_labels(task_config.skill_bank_path)
    stage_labels.update(
        {
            "frrg_handoff": "frrg_handoff",
            "frrg_capture_build": "frrg_capture_build",
            "frrg_close_hold": "frrg_close_hold",
            "frrg_lift_test": "frrg_lift_test",
        }
    )

    if jump_to_pick_after_index is None:
        branch_prefix_points = list(planned_points[anchor_index:pick_start])
    else:
        branch_prefix_points = list(planned_points[anchor_index : jump_to_pick_after_index + 1])
    skipped_prepick_points = (
        [] if jump_to_pick_after_index is None else list(planned_points[jump_to_pick_after_index + 1 : pick_start])
    )

    branch_grasp_start_q = np.asarray(
        branch_prefix_points[-1]["joint_positions"] if branch_prefix_points else planned_points[anchor_index]["joint_positions"],
        dtype=float,
    )
    original_pick_points = _hold_pick_points_from_current_pose(
        pick_points,
        current_joint_positions=branch_grasp_start_q,
        anchor_index=int(jump_to_pick_after_index if jump_to_pick_after_index is not None else pick_start - 1),
    )
    original_suffix_execution_points: list[dict[str, Any]] = []
    original_suffix_bridge_decision: dict[str, Any] | None = None
    if suffix_points and not bool(args.stop_after_pick):
        original_suffix_execution_points, original_suffix_bridge_decision = _build_execution_points(
            current_joint_positions=branch_grasp_start_q,
            planned_points=suffix_points,
            bridge_max_joint_step_rad=float(args.bridge_max_joint_step_rad),
            trajectory_summary=None,
        )

    original_branch_points = _renumber_points(branch_prefix_points + original_pick_points + original_suffix_execution_points)

    coarse_summary: dict[str, Any] | None = None
    frrg_output_dir: Path | None = None
    frrg_points: list[dict[str, Any]] = []
    frrg_summary: dict[str, Any] | None = None
    suffix_bridge_decision: dict[str, Any] | None = None
    frrg_branch_points: list[dict[str, Any]] = []
    if need_frrg_branch:
        coarse_start_q = branch_grasp_start_q.copy()
        coarse_summary = _build_coarse_summary(
            pregrasp_joint_positions=coarse_start_q,
            target_object_payload=target_object_payload,
            kinematics=kinematics,
            handoff_open_width_m=load_frrg_config(args.frrg_config).handoff.handoff_open_width_m,
        )
        frrg_output_dir = run_dir / "analysis" / "frrg_pickorange"
        frrg_points, frrg_end_q, frrg_summary = _compile_frrg_segment(
            coarse_summary=coarse_summary,
            frrg_config_path=args.frrg_config,
            kinematics=kinematics,
            start_joint_positions=coarse_start_q,
            max_steps=int(args.frrg_max_steps),
            output_dir=frrg_output_dir,
        )

        prefix_remainder_points = list(branch_prefix_points)
        repeated_frrg_points = _repeat_playback_points(
            frrg_points,
            repeat_count=int(args.frrg_playback_repeat),
            playback_source="frrg_repeat",
        )
        suffix_execution_points = []
        suffix_bridge_decision = None
        if suffix_points and not bool(args.stop_after_pick):
            suffix_execution_points, suffix_bridge_decision = _build_execution_points(
                current_joint_positions=frrg_end_q,
                planned_points=suffix_points,
                bridge_max_joint_step_rad=float(args.bridge_max_joint_step_rad),
                trajectory_summary=None,
            )
        frrg_branch_points = _renumber_points(prefix_remainder_points + repeated_frrg_points + suffix_execution_points)

    branch_root = run_dir / "branches"
    original_summary: dict[str, Any] | None = None
    if need_original_branch:
        if bool(args.headless):
            original_summary = _render_branch(
                branch_name="original_replay",
                branch_points=original_branch_points,
                full_planned_points=planned_points,
                anchor_index=anchor_index,
                model_path=args.model_path,
                kinematics=kinematics,
                output_dir=branch_root / "original_replay",
                render_width=int(args.render_width),
                render_height=int(args.render_height),
                video_fps=float(args.video_fps),
            )
        else:
            original_summary = _play_branch_in_viewer(
                branch_name="original_replay",
                branch_points=original_branch_points,
                full_planned_points=planned_points,
                anchor_index=anchor_index,
                model_path=args.model_path,
                kinematics=kinematics,
                output_dir=branch_root / "original_replay",
                target_base_xyz=current_target_base_xyz,
                aux_object_positions=current_aux_object_positions,
                stage_labels=stage_labels,
                step_duration_s=float(args.step_duration_s),
                final_hold_s=float(args.final_hold_s),
                show_left_ui=bool(args.show_left_ui),
                show_right_ui=bool(args.show_right_ui),
            )

    frrg_branch_summary: dict[str, Any] | None = None
    if need_frrg_branch:
        if bool(args.headless):
            frrg_branch_summary = _render_branch(
                branch_name="frrg_replacement",
                branch_points=frrg_branch_points,
                full_planned_points=planned_points,
                anchor_index=anchor_index,
                model_path=args.model_path,
                kinematics=kinematics,
                output_dir=branch_root / "frrg_replacement",
                render_width=int(args.render_width),
                render_height=int(args.render_height),
                video_fps=float(args.video_fps),
            )
        else:
            frrg_branch_summary = _play_branch_in_viewer(
                branch_name="frrg_replacement",
                branch_points=frrg_branch_points,
                full_planned_points=planned_points,
                anchor_index=anchor_index,
                model_path=args.model_path,
                kinematics=kinematics,
                output_dir=branch_root / "frrg_replacement",
                target_base_xyz=current_target_base_xyz,
                aux_object_positions=current_aux_object_positions,
                stage_labels=stage_labels,
                step_duration_s=float(args.step_duration_s),
                final_hold_s=float(args.final_hold_s),
                show_left_ui=bool(args.show_left_ui),
                show_right_ui=bool(args.show_right_ui),
            )

    original_pick_joint_rows = np.asarray([point["joint_positions"] for point in pick_points], dtype=float)
    frrg_joint_rows = np.asarray([point["joint_positions"] for point in frrg_points], dtype=float) if frrg_points else None

    compare_metrics: dict[str, Any] | None = None
    if need_original_branch and need_frrg_branch and original_summary is not None and frrg_branch_summary is not None:
        compare_metrics = {
            "original_branch_point_count": int(len(original_branch_points)),
            "frrg_branch_point_count": int(len(frrg_branch_points)),
            "branch_point_count_delta": int(len(frrg_branch_points) - len(original_branch_points)),
            "original_branch_ee_path_translation_mm": float(original_summary["ee_path_translation_mm"]),
            "frrg_branch_ee_path_translation_mm": float(frrg_branch_summary["ee_path_translation_mm"]),
            "ee_path_translation_delta_mm": float(
                frrg_branch_summary["ee_path_translation_mm"] - original_summary["ee_path_translation_mm"]
            ),
            "original_branch_total_joint_step_rad_inf": float(original_summary["total_joint_step_rad_inf"]),
            "frrg_branch_total_joint_step_rad_inf": float(frrg_branch_summary["total_joint_step_rad_inf"]),
            "total_joint_step_delta_rad_inf": float(
                frrg_branch_summary["total_joint_step_rad_inf"] - original_summary["total_joint_step_rad_inf"]
            ),
            "original_pick_joint_span_rad_inf": (
                float(np.max(np.ptp(original_pick_joint_rows, axis=0))) if len(original_pick_joint_rows) > 0 else 0.0
            ),
            "frrg_pick_joint_span_rad_inf": (
                float(np.max(np.ptp(frrg_joint_rows, axis=0))) if frrg_joint_rows is not None and len(frrg_joint_rows) > 0 else 0.0
            ),
            "original_pick_point_count": int(len(pick_points)),
            "frrg_pick_point_count": int(len(frrg_points)),
            "frrg_status": str(frrg_summary["status"]) if frrg_summary is not None else None,
            "frrg_final_phase": str(frrg_summary["final_phase"]) if frrg_summary is not None else None,
            "frrg_steps_run": int(frrg_summary["steps_run"]) if frrg_summary is not None else None,
        }

    compare_summary = {
        "status": "ok",
        "run_dir": str(run_dir.resolve()),
        "selected_branch": selected_branch,
        "task_config_path": str(args.task_config.resolve()),
        "frrg_config_path": str(args.frrg_config.resolve()),
        "live_result_path": str(live_result_path.resolve()),
        "trajectory_points_path": str(trajectory_points_path.resolve()),
        "anchor_trajectory_index": int(anchor_index),
        "anchor_segment_id": str(planned_points[anchor_index].get("segment_id", "")),
        "anchor_segment_label": str(planned_points[anchor_index].get("segment_label", "")),
        "anchor_joint_positions": [float(value) for value in planned_points[anchor_index]["joint_positions"]],
        "jump_to_pick_after_index": jump_to_pick_after_index,
        "jump_to_pick_segment_id": None if jump_to_pick_after_index is None else str(planned_points[jump_to_pick_after_index].get("segment_id", "")),
        "jump_to_pick_segment_label": None if jump_to_pick_after_index is None else str(planned_points[jump_to_pick_after_index].get("segment_label", "")),
        "pick_segment_id": str(pick_points[0].get("segment_id", "")) if pick_points else None,
        "pick_segment_label": str(pick_points[0].get("segment_label", "")) if pick_points else None,
        "pick_segment_start_index": int(pick_start),
        "pick_segment_point_count": int(len(pick_points)),
        "skipped_prepick_point_count": int(len(skipped_prepick_points)),
        "skipped_prepick_range": (
            None
            if not skipped_prepick_points
            else {
                "start_index": int(jump_to_pick_after_index + 1),
                "end_index": int(pick_start - 1),
            }
        ),
        "target_phrase": task_config.target_phrase,
        "aux_target_phrases": list(aux_target_phrases),
        "target_base_xyz_m": np.asarray(target_object_payload["base_xyz_m"], dtype=float).reshape(3).tolist(),
        "stop_after_pick": bool(args.stop_after_pick),
        "frrg_playback_repeat": int(args.frrg_playback_repeat),
        "frrg_playback_speed_ratio": 1.0 / max(int(args.frrg_playback_repeat), 1),
        "pose_summary_path": str((run_dir / "analysis" / "t5_pose" / "summary.json").resolve()),
        "trajectory_summary_path": str((run_dir / "analysis" / "t6_trajectory" / "summary.json").resolve()),
        "frrg_summary_path": None if frrg_output_dir is None else str((frrg_output_dir / "summary.json").resolve()),
        "original_suffix_bridge_decision": original_suffix_bridge_decision,
        "suffix_bridge_decision": suffix_bridge_decision,
        "original_branch": original_summary,
        "frrg_branch": frrg_branch_summary,
        "compare_metrics": compare_metrics,
        "pick_metrics": {
            "original_pick_point_count": int(len(pick_points)),
            "original_pick_joint_span_rad_inf": (
                float(np.max(np.ptp(original_pick_joint_rows, axis=0))) if len(original_pick_joint_rows) > 0 else 0.0
            ),
            "frrg_pick_point_count": int(len(frrg_points)),
            "frrg_pick_joint_span_rad_inf": (
                float(np.max(np.ptp(frrg_joint_rows, axis=0))) if frrg_joint_rows is not None and len(frrg_joint_rows) > 0 else 0.0
            ),
            "frrg_status": None if frrg_summary is None else str(frrg_summary["status"]),
            "frrg_final_phase": None if frrg_summary is None else str(frrg_summary["final_phase"]),
            "frrg_steps_run": None if frrg_summary is None else int(frrg_summary["steps_run"]),
            "coarse_summary_path": None if coarse_summary is None else str((run_dir / "analysis" / "coarse_summary.json").resolve()),
        },
        "pose_summary": pose_summary,
        "trajectory_summary": trajectory_summary,
        "aux_object_positions_m": {
            str(name): value.astype(float).tolist()
            for name, value in current_aux_object_positions.items()
        },
    }
    if coarse_summary is not None:
        _save_json(run_dir / "analysis" / "coarse_summary.json", coarse_summary)
    _save_json(run_dir / "summary.json", compare_summary)
    print(json.dumps(compare_summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
