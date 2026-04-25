#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    _build_execution_points,
    _build_stage_overlay_texts,
    _current_segment_state,
    _joint_qpos_indices,
    _load_segment_display_labels,
    _resolve_repo_path,
    _set_robot_qpos,
    _update_target_markers,
)
from lerobot.projects.vlbiman_sa.grasp.closed_loop_controller import FRRGClosedLoopController
from lerobot.projects.vlbiman_sa.grasp.coarse_handoff import COARSE_HANDOFF_SOURCE, build_frrg_input_from_coarse_summary
from lerobot.projects.vlbiman_sa.grasp.contracts import load_frrg_config
from lerobot.projects.vlbiman_sa.grasp.frame_math import matrix_to_pose6d, pose6d_to_matrix
from lerobot.utils.rotation import Rotation
from lerobot_robot_cjjarm.config_cjjarm_sim import CjjArmSimConfig
from lerobot_robot_cjjarm.kinematics import CjjArmKinematics, compose_pose_delta


def _default_task_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "task_grasp_one_shot_full_20260411T061326.yaml"


def _default_capture_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "one_shot_record.yaml"


def _default_vision_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "vision_analysis.yaml"


def _default_frrg_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "frrg_grasp.yaml"


def _default_output_root() -> Path:
    return REPO_ROOT / "outputs" / "vlbiman_sa" / "visual_pickorange_frrg"


def _default_live_output_root() -> Path:
    return REPO_ROOT / "outputs" / "vlbiman_sa" / "live_orange_pose"


def _default_model_path() -> Path:
    return (
        REPO_ROOT
        / "lerobot_robot_cjjarm"
        / "lerobot_robot_cjjarm"
        / "cjjarm_urdf"
        / "TRLC-DK1-Follower-home.mjcf"
    )


def _timestamp_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the VLBiMan pick-orange sim scene, but replace the grasp replay segment with FRRG."
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
    parser.add_argument("--step-duration-s", type=float, default=0.04)
    parser.add_argument("--frrg-max-steps", type=int, default=80)
    parser.add_argument("--final-hold-s", type=float, default=1.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--show-left-ui", action="store_true")
    parser.add_argument("--show-right-ui", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def _phrase_key(target_phrase: str) -> str:
    return "_".join(part for part in str(target_phrase).strip().lower().replace("-", " ").split() if part)


def _gripper_qpos_indices(model: mujoco.MjModel) -> list[int]:
    addrs: list[int] = []
    for joint_name in ("gripper_left", "gripper_right"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            addrs.append(int(model.jnt_qposadr[joint_id]))
    return addrs


def _set_gripper_qpos(data: mujoco.MjData, gripper_qpos: list[int], gripper_target: float) -> None:
    for addr in gripper_qpos:
        data.qpos[addr] = float(gripper_target)


def _build_renderer(
    model: mujoco.MjModel,
    *,
    width: int,
    height: int,
) -> tuple[mujoco.Renderer, mujoco.MjvCamera]:
    render_width = min(int(width), int(model.vis.global_.offwidth))
    render_height = min(int(height), int(model.vis.global_.offheight))
    renderer = mujoco.Renderer(model, render_height, render_width)
    camera = mujoco.MjvCamera()
    camera.azimuth = 148.0
    camera.elevation = -20.0
    camera.distance = 1.75
    camera.lookat[:] = np.asarray([0.0, 0.0, 0.24], dtype=float)
    return renderer, camera


def _render_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    camera: mujoco.MjvCamera,
) -> np.ndarray:
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera)
    return np.asarray(renderer.render(), dtype=np.uint8)


def _segment_semantic_states(skill_bank_path: Path | None) -> dict[str, str]:
    if skill_bank_path is None or not skill_bank_path.exists():
        return {}
    payload = _load_json(skill_bank_path)
    semantic_states: dict[str, str] = {}
    for segment in payload.get("segments", []):
        if not isinstance(segment, dict):
            continue
        segment_id = str(segment.get("segment_id", "")).strip()
        if not segment_id:
            continue
        metrics = segment.get("metrics") if isinstance(segment.get("metrics"), dict) else {}
        semantic_state = metrics.get("semantic_state")
        if semantic_state is not None:
            semantic_states[segment_id] = str(semantic_state)
    return semantic_states


def _is_pick_point(point: dict[str, Any], semantic_states: dict[str, str]) -> bool:
    segment_id = str(point.get("segment_id", "")).strip()
    segment_label = str(point.get("segment_label", "")).strip().lower().replace("-", "_")
    semantic_state = str(semantic_states.get(segment_id, "")).strip().lower().replace("-", "_")
    semantic_action = semantic_state.split("_", 1)[0] if semantic_state else ""
    if semantic_action in {"grasp", "pick", "pickup"}:
        return True
    return segment_label == "gripper_close"


def _split_pick_segment(
    points: list[dict[str, Any]],
    semantic_states: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    start_index: int | None = None
    for idx, point in enumerate(points):
        if _is_pick_point(point, semantic_states):
            start_index = idx
            break
    if start_index is None:
        raise ValueError("Could not find a pick/grasp segment in the planned trajectory.")

    pick_segment_id = str(points[start_index].get("segment_id", "")).strip()
    end_index = start_index
    while end_index < len(points) and str(points[end_index].get("segment_id", "")).strip() == pick_segment_id:
        end_index += 1
    return list(points[:start_index]), list(points[start_index:end_index]), list(points[end_index:])


def _target_object_payload(live_result: dict[str, Any], target_phrase: str) -> dict[str, Any]:
    target_key = _phrase_key(target_phrase)
    objects = live_result.get("objects")
    if isinstance(objects, dict):
        if target_key in objects and isinstance(objects[target_key], dict):
            return dict(objects[target_key])
        for payload in objects.values():
            if isinstance(payload, dict) and _phrase_key(str(payload.get("target_phrase", ""))) == target_key:
                return dict(payload)
    if _phrase_key(str(live_result.get("target_phrase", ""))) == target_key:
        return dict(live_result)
    raise ValueError(f"Live result does not contain target payload for '{target_phrase}'.")


def _aux_object_positions(live_result: dict[str, Any], target_phrase: str) -> dict[str, np.ndarray]:
    target_key = _phrase_key(target_phrase)
    objects = live_result.get("objects")
    out: dict[str, np.ndarray] = {}
    if not isinstance(objects, dict):
        return out
    for object_key, payload in objects.items():
        if str(object_key) == target_key or not isinstance(payload, dict):
            continue
        base_xyz = payload.get("base_xyz_m")
        if base_xyz is None:
            continue
        out[str(object_key)] = np.asarray(base_xyz, dtype=float).reshape(3)
    return out


def _bbox_size_px(object_payload: dict[str, Any]) -> tuple[float, float]:
    for container_key in ("anchor", "segmentation"):
        container = object_payload.get(container_key)
        if not isinstance(container, dict):
            continue
        bbox = container.get("bbox_xyxy")
        if isinstance(bbox, list) and len(bbox) == 4:
            width = max(float(bbox[2]) - float(bbox[0]), 1.0)
            height = max(float(bbox[3]) - float(bbox[1]), 1.0)
            return width, height
    orientation = object_payload.get("orientation")
    if isinstance(orientation, dict):
        major_axis = float(orientation.get("major_axis_px", 40.0))
        minor_axis = float(orientation.get("minor_axis_px", major_axis))
        return max(major_axis, 1.0), max(minor_axis, 1.0)
    return 80.0, 80.0


def _rotvec_pose_to_pose6d_dict(pose_rotvec: np.ndarray) -> dict[str, list[float]]:
    pose_rotvec = np.asarray(pose_rotvec, dtype=float).reshape(6)
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = pose_rotvec[:3]
    transform[:3, :3] = Rotation.from_rotvec(pose_rotvec[3:]).as_matrix()
    pose6d = matrix_to_pose6d(transform)
    return {
        "xyz": [float(value) for value in pose6d.xyz],
        "rpy": [float(value) for value in pose6d.rpy],
    }


def _object_delta_to_base(delta_pose_object: list[float], object_pose_payload: dict[str, Any]) -> np.ndarray:
    object_transform = pose6d_to_matrix(object_pose_payload)
    object_rotation = object_transform[:3, :3]
    delta = np.asarray(delta_pose_object, dtype=float)
    base_delta = np.zeros(6, dtype=float)
    base_delta[:3] = object_rotation @ delta[:3]
    base_delta[3:] = object_rotation @ delta[3:]
    return base_delta


def _translation_error_mm(target_pose: np.ndarray, achieved_pose: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(target_pose[:3]) - np.asarray(achieved_pose[:3])) * 1000.0)


def _rotation_error_deg(target_pose: np.ndarray, achieved_pose: np.ndarray) -> float:
    target_rot = Rotation.from_rotvec(np.asarray(target_pose[3:], dtype=float)).as_matrix()
    achieved_rot = Rotation.from_rotvec(np.asarray(achieved_pose[3:], dtype=float)).as_matrix()
    delta = Rotation.from_matrix(target_rot @ achieved_rot.T).as_rotvec()
    return float(np.rad2deg(np.linalg.norm(delta)))


def _gripper_raw_for_width(gripper_width_m: float, handoff_open_width_m: float) -> float:
    return -1.0 if float(gripper_width_m) <= float(handoff_open_width_m) * 0.5 else 0.0


def _point_gripper_raw(point: dict[str, Any], previous_raw: float) -> float:
    if "gripper_raw" in point:
        return float(point["gripper_raw"])
    segment_label = str(point.get("segment_label", "")).strip().lower()
    if segment_label == "gripper_close":
        return -1.0
    if segment_label == "gripper_open":
        return 0.0
    return float(previous_raw)


def _gripper_target_from_raw(gripper_raw: float, sim_config: CjjArmSimConfig) -> float:
    if float(gripper_raw) < float(sim_config.gripper_trigger_threshold):
        return float(sim_config.gripper_closed_pos)
    return float(sim_config.gripper_open_pos)


def _build_coarse_summary(
    *,
    pregrasp_joint_positions: np.ndarray,
    target_object_payload: dict[str, Any],
    kinematics: CjjArmKinematics,
    handoff_open_width_m: float,
) -> dict[str, Any]:
    pregrasp_pose_base = _rotvec_pose_to_pose6d_dict(kinematics.compute_fk(pregrasp_joint_positions))
    target_base_xyz = np.asarray(target_object_payload["base_xyz_m"], dtype=float).reshape(3)
    target_pose_base = {
        "xyz": [float(value) for value in target_base_xyz],
        "rpy": [0.0, 0.0, 0.0],
    }
    pregrasp_object_transform = np.linalg.inv(pose6d_to_matrix(target_pose_base)) @ pose6d_to_matrix(pregrasp_pose_base)
    pregrasp_pose_object = matrix_to_pose6d(pregrasp_object_transform)
    orientation = target_object_payload.get("orientation") if isinstance(target_object_payload.get("orientation"), dict) else {}
    anchor = target_object_payload.get("anchor") if isinstance(target_object_payload.get("anchor"), dict) else {}
    segmentation = (
        target_object_payload.get("segmentation") if isinstance(target_object_payload.get("segmentation"), dict) else {}
    )
    centroid = anchor.get("centroid_px") or anchor.get("contact_px") or orientation.get("centroid_px") or [320.0, 240.0]
    object_proj_width_px, object_proj_height_px = _bbox_size_px(target_object_payload)
    object_axis_angle = float(pregrasp_pose_object.rpy[2])
    vision_conf = float(segmentation.get("score", anchor.get("score", 1.0)) or 1.0)
    target_visible = str(target_object_payload.get("status", "ok")) == "ok"

    return {
        "timestamp": 0.0,
        "target_pose_base": target_pose_base,
        "pregrasp_pose_base": pregrasp_pose_base,
        "gripper_initial_width": float(handoff_open_width_m),
        "gripper_current_proxy": 0.35,
        "vision_summary": {
            "target_visible": bool(target_visible),
            "vision_conf": float(vision_conf),
            "corridor_center_px": [float(centroid[0]), float(centroid[1])],
            "object_center_px": [float(centroid[0]), float(centroid[1])],
            "object_axis_angle": float(object_axis_angle),
            "object_proj_width_px": float(object_proj_width_px),
            "object_proj_height_px": float(object_proj_height_px),
        },
    }


def _compile_frrg_segment(
    *,
    coarse_summary: dict[str, Any],
    frrg_config_path: Path,
    kinematics: CjjArmKinematics,
    start_joint_positions: np.ndarray,
    max_steps: int,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    config = load_frrg_config(frrg_config_path)
    input_payload = build_frrg_input_from_coarse_summary(coarse_summary)
    for field_name in ("e_dep", "e_lat", "e_vert", "e_ang", "e_sym", "occ_corridor", "drift_obj"):
        input_payload.pop(field_name, None)
    controller = FRRGClosedLoopController(
        config,
        input_payload,
        input_mode=COARSE_HANDOFF_SOURCE,
    )
    run_result = controller.run(max_steps=max_steps)

    _write_jsonl(output_dir / "phase_trace.jsonl", run_result.phase_trace)
    _write_jsonl(output_dir / "actions.jsonl", run_result.actions)
    _write_jsonl(output_dir / "states.jsonl", run_result.states)
    _write_jsonl(output_dir / "guards.jsonl", run_result.guards)

    summary = run_result.summary_dict(
        config_path=str(frrg_config_path.resolve()),
        mock_state_path="vlbiman_pickorange_splice",
        output_dir=str(output_dir.resolve()),
        max_steps=int(max_steps),
        input_mode=COARSE_HANDOFF_SOURCE,
        input_summary={"source": "vlbiman_pickorange_frrg"},
    )
    summary["coarse_summary"] = coarse_summary
    _save_json(output_dir / "summary.json", summary)

    target_pose = np.asarray(kinematics.compute_fk(start_joint_positions), dtype=float)
    current_gripper_width = float(coarse_summary["gripper_initial_width"])
    current_q = np.asarray(start_joint_positions, dtype=float).copy()
    compiled_points: list[dict[str, Any]] = []
    for step_index, action_record in enumerate(run_result.actions):
        safe_action = dict(action_record["safe_action"])
        phase = str(action_record["phase"])
        base_delta = _object_delta_to_base(list(safe_action["delta_pose_object"]), dict(coarse_summary["target_pose_base"]))
        target_pose = compose_pose_delta(target_pose, base_delta, rotation_frame="world")
        current_gripper_width = max(0.0, current_gripper_width + float(safe_action["delta_gripper"]))
        solved_q = kinematics.compute_ik(
            target_pose,
            current_q,
            position_weight=1.0,
            orientation_weight=0.05,
            keep_pointing_only=True,
        )
        achieved_pose = kinematics.compute_fk(solved_q)
        compiled_points.append(
            {
                "trajectory_index": step_index,
                "frame_index": None,
                "relative_time_s": float(step_index) / max(float(config.runtime.control_hz), 1e-6),
                "segment_id": f"frrg_{phase.lower()}",
                "segment_label": phase.lower(),
                "invariance": "frrg",
                "source": "frrg_safe_action",
                "joint_positions": np.asarray(solved_q, dtype=float).tolist(),
                "gripper_raw": float(_gripper_raw_for_width(current_gripper_width, config.handoff.handoff_open_width_m)),
                "gripper_width_m": float(current_gripper_width),
                "frrg_phase": phase,
                "frrg_safe_action": safe_action,
                "target_pose_6d": _rotvec_pose_to_pose6d_dict(target_pose),
                "translation_error_mm": _translation_error_mm(target_pose, achieved_pose),
                "rotation_error_deg": _rotation_error_deg(target_pose, achieved_pose),
                "max_joint_step_rad": float(np.max(np.abs(np.asarray(solved_q, dtype=float) - current_q))),
            }
        )
        current_q = np.asarray(solved_q, dtype=float)

    _save_json(output_dir / "compiled_points.json", {"points": compiled_points})
    return compiled_points, current_q, summary


def _renumber_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    renumbered: list[dict[str, Any]] = []
    for idx, point in enumerate(points):
        row = dict(point)
        row["trajectory_index"] = idx
        renumbered.append(row)
    return renumbered


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
    run_dir = args.output_root / _timestamp_name("pickorange_frrg")
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

    model = mujoco.MjModel.from_xml_path(str(args.model_path))
    data = mujoco.MjData(model)
    joint_qpos = _joint_qpos_indices(model)
    gripper_qpos = _gripper_qpos_indices(model)
    reset_joint_positions = np.asarray([data.qpos[joint_qpos[name]] for name in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")], dtype=float)

    sim_config = CjjArmSimConfig()
    kinematics = CjjArmKinematics(
        urdf_path=sim_config.urdf_path,
        end_effector_frame=sim_config.end_effector_frame,
        joint_names=[sim_config.urdf_joint_map[name] for name in sim_config.joint_action_order],
    )

    execution_points, bridge_decision = _build_execution_points(
        current_joint_positions=reset_joint_positions,
        planned_points=planned_points,
        bridge_max_joint_step_rad=float(args.bridge_max_joint_step_rad),
        trajectory_summary=trajectory_summary,
    )
    semantic_states = _segment_semantic_states(task_config.skill_bank_path)
    prefix_points, pick_points, suffix_points = _split_pick_segment(execution_points, semantic_states)

    coarse_start_q = (
        np.asarray(prefix_points[-1]["joint_positions"], dtype=float) if prefix_points else reset_joint_positions.copy()
    )
    target_object_payload = _target_object_payload(live_result, task_config.target_phrase)
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

    suffix_execution_points: list[dict[str, Any]] = []
    suffix_bridge_decision: dict[str, Any] | None = None
    if frrg_summary["status"] == "success" and suffix_points:
        suffix_execution_points, suffix_bridge_decision = _build_execution_points(
            current_joint_positions=frrg_end_q,
            planned_points=suffix_points,
            bridge_max_joint_step_rad=float(args.bridge_max_joint_step_rad),
            trajectory_summary=None,
        )

    final_points = _renumber_points(prefix_points + frrg_points + suffix_execution_points)
    execution_payload = {
        "points": final_points,
        "bridge_decision": bridge_decision,
        "suffix_bridge_decision": suffix_bridge_decision,
        "pick_segment_replaced": {
            "replaced_segment_id": str(pick_points[0].get("segment_id", "")) if pick_points else None,
            "original_pick_point_count": len(pick_points),
            "frrg_point_count": len(frrg_points),
            "prefix_point_count": len(prefix_points),
            "suffix_point_count": len(suffix_execution_points),
            "frrg_status": frrg_summary["status"],
            "frrg_final_phase": frrg_summary["final_phase"],
        },
    }
    _save_json(run_dir / "execution" / "spliced_points.json", execution_payload)
    _save_json(run_dir / "analysis" / "coarse_summary.json", coarse_summary)

    current_target_base_xyz = np.asarray(target_object_payload["base_xyz_m"], dtype=float).reshape(3)
    current_aux_object_positions = _aux_object_positions(live_result, task_config.target_phrase)
    segment_display_labels = _load_segment_display_labels(task_config.skill_bank_path)
    segment_display_labels.update(
        {
            "frrg_handoff": "frrg_handoff",
            "frrg_capture_build": "frrg_capture_build",
            "frrg_close_hold": "frrg_close_hold",
            "frrg_lift_test": "frrg_lift_test",
        }
    )

    current_index = 0
    current_gripper_raw = 0.0
    last_step_time = 0.0
    hold_started_at: float | None = None
    last_segment_key: tuple[str, str, str] | None = None
    status = "ok"
    frame0_path: Path | None = None
    frame_last_path: Path | None = None
    if args.headless:
        renderer, camera = _build_renderer(
            model,
            width=int(args.render_width),
            height=int(args.render_height),
        )
        try:
            for point_index, point in enumerate(final_points):
                joint_positions = np.asarray(point["joint_positions"], dtype=float)
                _set_robot_qpos(data, joint_qpos, joint_positions)
                current_gripper_raw = _point_gripper_raw(point, current_gripper_raw)
                _set_gripper_qpos(data, gripper_qpos, _gripper_target_from_raw(current_gripper_raw, sim_config))
                current_segment = _current_segment_state(final_points, point_index)
                segment_key = (
                    str(current_segment.get("segment_id", "unknown_segment")),
                    str(current_segment.get("invariance", "unknown")),
                    str(current_segment.get("source", "unknown")),
                ) if current_segment is not None else None
                if segment_key is not None and segment_key != last_segment_key:
                    logging.info(
                        "Active segment: %s label=%s invariance=%s source=%s",
                        current_segment.get("segment_id", "unknown_segment"),
                        current_segment.get("segment_label", ""),
                        current_segment.get("invariance", "unknown"),
                        current_segment.get("source", "unknown"),
                    )
                    last_segment_key = segment_key

                frame_rgb = _render_frame(model, data, renderer, camera)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                if point_index == 0:
                    frame0_path = run_dir / "preview" / "frame_000.png"
                    frame0_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(frame0_path), frame_bgr)
                if point_index == len(final_points) - 1:
                    frame_last_path = run_dir / "preview" / "frame_last.png"
                    frame_last_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(frame_last_path), frame_bgr)
        finally:
            renderer.close()
    else:
        require_mujoco_viewer_backend()
        viewer = mujoco.viewer.launch_passive(
            model,
            data,
            show_left_ui=bool(args.show_left_ui),
            show_right_ui=bool(args.show_right_ui),
        )
        _apply_camera(viewer)
        try:
            while viewer.is_running():
                now = time.monotonic()
                if current_index < len(final_points) and now - last_step_time >= max(float(args.step_duration_s), 1e-3):
                    point = final_points[current_index]
                    joint_positions = np.asarray(point["joint_positions"], dtype=float)
                    _set_robot_qpos(data, joint_qpos, joint_positions)
                    current_gripper_raw = _point_gripper_raw(point, current_gripper_raw)
                    _set_gripper_qpos(data, gripper_qpos, _gripper_target_from_raw(current_gripper_raw, sim_config))
                    last_step_time = now
                    current_index += 1
                elif current_index >= len(final_points):
                    if hold_started_at is None:
                        hold_started_at = now
                    elif now - hold_started_at >= max(float(args.final_hold_s), 0.0):
                        break

                active_segment_index = max(0, min(current_index - 1, max(len(final_points) - 1, 0)))
                current_segment = _current_segment_state(final_points, active_segment_index) if final_points else None
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
                            "Active segment: %s label=%s invariance=%s source=%s",
                            current_segment.get("segment_id", "unknown_segment"),
                            current_segment.get("segment_label", ""),
                            current_segment.get("invariance", "unknown"),
                            current_segment.get("source", "unknown"),
                        )
                        last_segment_key = segment_key

                mujoco.mj_forward(model, data)
                _update_target_markers(viewer, current_target_base_xyz, current_aux_object_positions, current_segment)
                viewer.set_texts(
                    _build_stage_overlay_texts(
                        current_segment,
                        stage_labels=segment_display_labels,
                        point_index=active_segment_index,
                        point_count=len(final_points),
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
        "status": status,
        "run_dir": str(run_dir),
        "task_config_path": str(args.task_config.resolve()),
        "frrg_config_path": str(args.frrg_config.resolve()),
        "live_result_path": str(live_result_path.resolve()),
        "pose_summary_path": str((run_dir / "analysis" / "t5_pose" / "summary.json").resolve()),
        "trajectory_summary_path": str((run_dir / "analysis" / "t6_trajectory" / "summary.json").resolve()),
        "trajectory_points_path": str(trajectory_points_path.resolve()),
        "spliced_points_path": str((run_dir / "execution" / "spliced_points.json").resolve()),
        "coarse_summary_path": str((run_dir / "analysis" / "coarse_summary.json").resolve()),
        "frrg_summary_path": str((frrg_output_dir / "summary.json").resolve()),
        "headless": bool(args.headless),
        "frame0_path": None if frame0_path is None else str(frame0_path.resolve()),
        "frame_last_path": None if frame_last_path is None else str(frame_last_path.resolve()),
        "target_phrase": task_config.target_phrase,
        "aux_target_phrases": list(aux_target_phrases),
        "target_base_xyz_m": current_target_base_xyz.astype(float).tolist(),
        "bridge_decision": bridge_decision,
        "suffix_bridge_decision": suffix_bridge_decision,
        "original_t6_point_count": len(planned_points),
        "execution_point_count_before_splice": len(execution_points),
        "final_execution_point_count": len(final_points),
        "prefix_point_count": len(prefix_points),
        "original_pick_point_count": len(pick_points),
        "frrg_point_count": len(frrg_points),
        "suffix_point_count": len(suffix_execution_points),
        "frrg_status": frrg_summary["status"],
        "frrg_final_phase": frrg_summary["final_phase"],
        "frrg_steps_run": frrg_summary["steps_run"],
        "frrg_failure_reason": frrg_summary["failure_reason"],
        "pick_segment_id": str(pick_points[0].get("segment_id", "")) if pick_points else None,
        "pick_segment_label": str(pick_points[0].get("segment_label", "")) if pick_points else None,
        "pose_summary": pose_summary,
        "trajectory_summary": trajectory_summary,
    }
    _save_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
