#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
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
    _current_segment_state,
    _load_segment_display_labels,
    _resolve_repo_path,
)
from lerobot.projects.vlbiman_sa.app.run_visual_pickorange_branch_compare import (
    _gripper_raw_before_index,
    _hold_pick_points_from_current_pose,
)
from lerobot.projects.vlbiman_sa.app.run_visual_pickorange_frrg_validation import (
    _aux_object_positions,
    _build_renderer,
    _default_capture_config_path,
    _default_frrg_config_path,
    _default_live_output_root,
    _default_model_path,
    _default_task_config_path,
    _default_vision_config_path,
    _gripper_qpos_indices,
    _phrase_key,
    _point_gripper_raw,
    _render_frame,
    _save_json,
    _segment_semantic_states,
    _split_pick_segment,
    _target_object_payload,
)
from lerobot.projects.vlbiman_sa.grasp.closed_loop_controller import FRRGClosedLoopController
from lerobot.projects.vlbiman_sa.grasp.contracts import load_frrg_config
from lerobot.projects.vlbiman_sa.grasp.coarse_handoff import COARSE_HANDOFF_SOURCE
from lerobot.projects.vlbiman_sa.grasp.frame_math import pose6d_to_matrix
from lerobot.projects.vlbiman_sa.grasp.state_machine import PHASE_FAILURE, PHASE_SUCCESS
from lerobot_robot_cjjarm.config_cjjarm_sim import CjjArmSimConfig
from lerobot_robot_cjjarm.kinematics import CjjArmKinematics, compose_pose_delta


JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


def _default_output_root() -> Path:
    return REPO_ROOT / "outputs" / "vlbiman_sa" / "visual_pickorange_true_closed_loop"


def _timestamp_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pickorange scene with physical MuJoCo objects and compare the original "
            "hold-and-close replay against the FRRG true closed-loop grasp branch."
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
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--jump-to-pick-after-index", type=int, default=40)
    parser.add_argument("--frrg-max-steps", type=int, default=140)
    parser.add_argument("--step-duration-s", type=float, default=0.12)
    parser.add_argument("--final-hold-s", type=float, default=1.0)
    parser.add_argument(
        "--branch",
        choices=("both", "original_replay", "frrg_true_closed_loop"),
        default="both",
        help="Select which branch to render. GUI mode supports a single branch only.",
    )
    parser.add_argument("--orange-radius-m", type=float, default=0.022)
    parser.add_argument("--orange-mass-kg", type=float, default=0.12)
    parser.add_argument("--pink-cup-radius-m", type=float, default=0.035)
    parser.add_argument("--pink-cup-height-m", type=float, default=0.09)
    parser.add_argument("--physics-interp-step-rad", type=float, default=0.02)
    parser.add_argument("--physics-settle-steps", type=int, default=2)
    parser.add_argument("--physical-success-lift-mm", type=float, default=8.0)
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--show-left-ui", action="store_true")
    parser.add_argument("--show-right-ui", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _scene_with_objects_xml(
    base_xml: str,
    *,
    asset_root: Path,
    orange_xyz: np.ndarray,
    orange_radius_m: float,
    orange_mass_kg: float,
    pink_cup_xyz: np.ndarray | None,
    pink_cup_radius_m: float,
    pink_cup_height_m: float,
) -> str:
    def _rewrite_asset_path(match: re.Match[str]) -> str:
        original_path = Path(match.group(1))
        resolved_path = original_path if original_path.is_absolute() else (asset_root / original_path).resolve()
        return f'file="{resolved_path}"'

    base_xml = re.sub(r'file="([^"]+)"', _rewrite_asset_path, base_xml)
    insert = f"""
    <body name="orange_body" pos="{float(orange_xyz[0]):.6f} {float(orange_xyz[1]):.6f} {float(orange_xyz[2]):.6f}">
      <freejoint name="orange_free"/>
      <geom
        name="orange_geom"
        type="sphere"
        size="{float(orange_radius_m):.6f}"
        mass="{float(orange_mass_kg):.6f}"
        rgba="1.0 0.48 0.05 1.0"
        friction="1.4 0.03 0.001"
        solref="0.005 1"
        solimp="0.93 0.98 0.001"
      />
    </body>
"""
    if pink_cup_xyz is not None:
        insert += f"""
    <body name="pink_cup_body" pos="{float(pink_cup_xyz[0]):.6f} {float(pink_cup_xyz[1]):.6f} {float(max(pink_cup_xyz[2], pink_cup_height_m * 0.5 - 0.01)):.6f}">
      <geom
        name="pink_cup_geom"
        type="cylinder"
        size="{float(pink_cup_radius_m):.6f} {float(pink_cup_height_m * 0.5):.6f}"
        rgba="1.0 0.36 0.72 0.55"
        contype="0"
        conaffinity="0"
      />
    </body>
"""
    marker = "</worldbody>"
    if marker not in base_xml:
        raise ValueError("Base MuJoCo scene does not contain </worldbody>.")
    return base_xml.replace(marker, insert + "\n  </worldbody>", 1)


def _write_physical_scene(
    *,
    base_model_path: Path,
    output_dir: Path,
    orange_xyz: np.ndarray,
    orange_radius_m: float,
    orange_mass_kg: float,
    pink_cup_xyz: np.ndarray | None,
    pink_cup_radius_m: float,
    pink_cup_height_m: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_xml = base_model_path.read_text(encoding="utf-8")
    scene_xml = _scene_with_objects_xml(
        base_xml,
        asset_root=base_model_path.parent,
        orange_xyz=orange_xyz,
        orange_radius_m=orange_radius_m,
        orange_mass_kg=orange_mass_kg,
        pink_cup_xyz=pink_cup_xyz,
        pink_cup_radius_m=pink_cup_radius_m,
        pink_cup_height_m=pink_cup_height_m,
    )
    scene_path = output_dir / "pickorange_true_closed_loop_scene.mjcf"
    scene_path.write_text(scene_xml, encoding="utf-8")
    return scene_path


def _joint_qpos_indices(model: mujoco.MjModel) -> dict[str, int]:
    indices: dict[str, int] = {}
    for joint_name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint not found in MuJoCo model: {joint_name}")
        indices[joint_name] = int(model.jnt_qposadr[joint_id])
    return indices


def _joint_positions_from_qpos(data: mujoco.MjData, joint_qpos: dict[str, int]) -> np.ndarray:
    return np.asarray([data.qpos[joint_qpos[name]] for name in JOINT_NAMES], dtype=float)


def _set_joint_qpos(data: mujoco.MjData, joint_qpos: dict[str, int], joint_positions: np.ndarray) -> None:
    for joint_index, joint_name in enumerate(JOINT_NAMES):
        data.qpos[joint_qpos[joint_name]] = float(joint_positions[joint_index])


def _gripper_qpos_from_width(
    *,
    width_m: float,
    handoff_open_width_m: float,
    sim_config: CjjArmSimConfig,
) -> float:
    normalized = 0.0 if handoff_open_width_m <= 1e-9 else float(width_m) / float(handoff_open_width_m)
    normalized = min(max(normalized, 0.0), 1.0)
    return float(sim_config.gripper_closed_pos) + normalized * (
        float(sim_config.gripper_open_pos) - float(sim_config.gripper_closed_pos)
    )


def _point_target_width_m(point: dict[str, Any], previous_width_m: float, handoff_open_width_m: float) -> float:
    if "gripper_width_m" in point:
        return float(point["gripper_width_m"])
    previous_raw = -1.0 if float(previous_width_m) <= float(handoff_open_width_m) * 0.5 else 0.0
    raw = _point_gripper_raw(point, previous_raw)
    return 0.0 if float(raw) < 0.0 else float(handoff_open_width_m)


def _apply_joint_targets_physical(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_qpos: dict[str, int],
    gripper_qpos: list[int],
    target_joint_positions: np.ndarray,
    gripper_qpos_target: float,
    interp_step_rad: float,
    settle_steps: int,
) -> np.ndarray:
    current_joint_positions = _joint_positions_from_qpos(data, joint_qpos)
    max_delta = float(np.max(np.abs(np.asarray(target_joint_positions, dtype=float) - current_joint_positions)))
    num_interp = max(1, int(math.ceil(max_delta / max(float(interp_step_rad), 1e-6))))
    for interp_index in range(1, num_interp + 1):
        alpha = float(interp_index) / float(num_interp)
        interp_joint_positions = current_joint_positions + alpha * (
            np.asarray(target_joint_positions, dtype=float) - current_joint_positions
        )
        _set_joint_qpos(data, joint_qpos, interp_joint_positions)
        for qpos_addr in gripper_qpos:
            data.qpos[qpos_addr] = float(gripper_qpos_target)
        mujoco.mj_step(model, data)
    for _ in range(max(int(settle_steps), 0)):
        mujoco.mj_step(model, data)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return _joint_positions_from_qpos(data, joint_qpos)


def _body_xyz(data: mujoco.MjData, body_id: int) -> np.ndarray:
    return np.asarray(data.xpos[body_id], dtype=float).reshape(3)


def _body_point_world(data: mujoco.MjData, body_id: int, point_local_xyz: np.ndarray) -> np.ndarray:
    position = np.asarray(data.xpos[body_id], dtype=float).reshape(3)
    rotation = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
    return position + rotation @ np.asarray(point_local_xyz, dtype=float).reshape(3)


def _default_collision_catalog_path() -> Path:
    return REPO_ROOT / "lerobot_robot_cjjarm" / "lerobot_robot_cjjarm" / "cjjarm_urdf" / "collision_trlc-dk1.yml"


def _load_distal_grasp_center_local(catalog_path: Path | None = None) -> np.ndarray:
    catalog_path = _default_collision_catalog_path() if catalog_path is None else Path(catalog_path)
    payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    collision_spheres = dict(payload.get("collision_spheres", {}))
    left_chain = list(collision_spheres.get("finger_left", []))
    right_chain = list(collision_spheres.get("finger_right", []))
    if not left_chain or not right_chain:
        raise ValueError(f"Finger collision spheres missing in {catalog_path}")
    left_tip = np.asarray(left_chain[-1]["center"], dtype=float).reshape(3)
    right_tip = np.asarray(right_chain[-1]["center"], dtype=float).reshape(3)
    return 0.5 * (left_tip + right_tip)


def _orange_contact_detected(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    orange_geom_id: int,
    finger_body_ids: set[int],
) -> bool:
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        if int(contact.geom1) == int(orange_geom_id):
            other_geom = int(contact.geom2)
        elif int(contact.geom2) == int(orange_geom_id):
            other_geom = int(contact.geom1)
        else:
            continue
        other_body_id = int(model.geom_bodyid[other_geom])
        if other_body_id in finger_body_ids:
            return True
    return False


def _rotvec_pose_to_pose6d_dict(pose_rotvec: np.ndarray) -> dict[str, list[float]]:
    pose_rotvec = np.asarray(pose_rotvec, dtype=float).reshape(6)
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = pose_rotvec[:3]
    transform[:3, :3] = Rotation.from_rotvec(pose_rotvec[3:]).as_matrix()
    rotation = Rotation.from_matrix(transform[:3, :3]).as_euler("xyz", degrees=False)
    return {
        "xyz": [float(value) for value in transform[:3, 3]],
        "rpy": [float(value) for value in rotation],
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


def _normalize_vector(vector: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return np.asarray(fallback, dtype=float).reshape(3)
    return vector / norm


def _corridor_axes_from_forward_axis(corridor_forward_axis_base: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward_axis = _normalize_vector(
        np.asarray(corridor_forward_axis_base, dtype=float).reshape(3),
        fallback=np.asarray([1.0, 0.0, 0.0], dtype=float),
    )
    outward_axis = -forward_axis
    global_up = np.asarray([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(outward_axis, global_up))) >= 0.98:
        global_up = np.asarray([0.0, 1.0, 0.0], dtype=float)
    lateral_axis = _normalize_vector(np.cross(global_up, outward_axis), fallback=np.asarray([0.0, 1.0, 0.0], dtype=float))
    vertical_axis = _normalize_vector(np.cross(outward_axis, lateral_axis), fallback=np.asarray([0.0, 0.0, 1.0], dtype=float))
    return lateral_axis, vertical_axis, outward_axis


def _corridor_object_pose(
    *,
    orange_xyz: np.ndarray,
    corridor_forward_axis_base: np.ndarray,
) -> tuple[dict[str, list[float]], np.ndarray, np.ndarray, np.ndarray]:
    lateral_axis, vertical_axis, outward_axis = _corridor_axes_from_forward_axis(corridor_forward_axis_base)
    rotation_matrix = np.column_stack([lateral_axis, vertical_axis, outward_axis])
    rpy = Rotation.from_matrix(rotation_matrix).as_euler("xyz", degrees=False)
    pose = {
        "xyz": [float(value) for value in np.asarray(orange_xyz, dtype=float).reshape(3)],
        "rpy": [float(value) for value in rpy],
    }
    return pose, lateral_axis, vertical_axis, outward_axis


def _capture_occupancy(*, e_lat_m: float, corridor_width_m: float) -> float:
    width = max(float(corridor_width_m), 1e-6)
    overlap = max(0.0, width - abs(float(e_lat_m)))
    return float(max(0.0, min(1.0, overlap / width)))


def _build_frrg_sim_payload(
    *,
    joint_positions: np.ndarray,
    orange_xyz: np.ndarray,
    previous_orange_xyz: np.ndarray | None,
    current_gripper_width_m: float,
    orange_contact_detected: bool,
    initial_orange_z_m: float,
    control_payload: dict[str, Any],
    kinematics: CjjArmKinematics,
    corridor_forward_axis_base: np.ndarray,
    orange_radius_m: float,
    control_point_world_xyz: np.ndarray,
    target_width_px: float = 90.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ee_pose_rotvec = np.asarray(kinematics.compute_fk(joint_positions), dtype=float)
    ee_pose_base = _rotvec_pose_to_pose6d_dict(ee_pose_rotvec)
    ee_xyz = np.asarray(control_point_world_xyz, dtype=float).reshape(3)
    ee_pose_base["xyz"] = [float(value) for value in ee_xyz]
    object_pose_base, lateral_axis, _vertical_axis, outward_axis = _corridor_object_pose(
        orange_xyz=orange_xyz,
        corridor_forward_axis_base=corridor_forward_axis_base,
    )
    correction_xyz = np.asarray(orange_xyz, dtype=float).reshape(3) - ee_xyz
    e_lat_m = float(np.dot(correction_xyz, lateral_axis))
    e_vert_m = float(np.dot(correction_xyz, _vertical_axis))
    ee_depth_m = float(np.dot(ee_xyz - np.asarray(orange_xyz, dtype=float).reshape(3), outward_axis))
    object_jump_m = 0.0 if previous_orange_xyz is None else float(np.linalg.norm(orange_xyz - previous_orange_xyz))
    object_lift_m = max(0.0, float(orange_xyz[2]) - float(initial_orange_z_m))
    drift_obj = object_jump_m
    occupancy = _capture_occupancy(e_lat_m=e_lat_m, corridor_width_m=max(2.0 * float(orange_radius_m), 0.03))
    payload = {
        "timestamp": float(control_payload.get("timestamp", 0.0)),
        "phase": str(control_payload.get("phase", "HANDOFF")),
        "mode": COARSE_HANDOFF_SOURCE,
        "retry_count": int(control_payload.get("retry_count", 0)),
        "stable_count": int(control_payload.get("stable_count", 0)),
        "phase_elapsed_s": float(control_payload.get("phase_elapsed_s", 0.0)),
        "ee_pose_base": ee_pose_base,
        "object_pose_base": object_pose_base,
        "gripper_width": float(current_gripper_width_m),
        "gripper_cmd": float(current_gripper_width_m),
        "gripper_current_proxy": 0.35 if orange_contact_detected else 0.0,
        "vision_conf": 1.0,
        "target_visible": True,
        "corridor_center_px": [0.0, 0.0],
        "object_center_px": [0.0, 0.0],
        "object_axis_angle": 0.0,
        "object_proj_width_px": float(target_width_px),
        "object_proj_height_px": float(target_width_px),
        "e_lat": float(e_lat_m),
        "e_vert": float(e_vert_m),
        "e_ang": 0.0,
        "e_sym": 0.0,
        "occ_corridor": float(occupancy),
        "drift_obj": float(drift_obj),
        "object_lift_m": float(object_lift_m),
        "capture_score": 0.0,
        "hold_score": 0.0,
        "lift_score": 0.0,
        "object_jump_m": float(object_jump_m),
        "feature_debug_terms": {
            "lateral_error_unit": "m",
            "drift_unit": "m",
            "object_width_px": float(target_width_px),
            "corridor_width_px": float(target_width_px),
            "vertical_enabled": True,
            "vertical_tol_m": 0.015,
        },
    }
    metrics = {
        "orange_xyz_m": [float(value) for value in orange_xyz],
        "ee_xyz_m": [float(value) for value in ee_xyz],
        "tool0_xyz_m": [float(value) for value in np.asarray(ee_pose_rotvec[:3], dtype=float)],
        "orange_contact_detected": bool(orange_contact_detected),
        "object_lift_m": float(object_lift_m),
        "object_jump_m": float(object_jump_m),
        "occupancy": float(occupancy),
        "e_lat_m": float(e_lat_m),
        "e_vert_m": float(e_vert_m),
        "ee_depth_m": float(ee_depth_m),
        "gripper_width_m": float(current_gripper_width_m),
        "corridor_forward_axis_base": [float(value) for value in np.asarray(corridor_forward_axis_base, dtype=float)],
        "corridor_lateral_axis_base": [float(value) for value in lateral_axis],
        "corridor_outward_axis_base": [float(value) for value in outward_axis],
    }
    return payload, metrics


def _overlay_text(frame_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    overlay = frame_bgr.copy()
    y = 28
    for line in lines:
        cv2.putText(overlay, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
        y += 28
    return overlay


@dataclass
class BranchPhysicsStats:
    previous_joint_positions: np.ndarray | None = None
    previous_ee_pose: np.ndarray | None = None
    max_joint_step_rad: float = 0.0
    total_joint_step_rad_inf: float = 0.0
    ee_path_translation_mm: float = 0.0
    max_orange_lift_m: float = 0.0
    contact_step_count: int = 0


class _BranchRecorder:
    def __init__(
        self,
        *,
        branch_name: str,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        output_dir: Path,
        headless: bool,
        render_width: int,
        render_height: int,
        video_fps: float,
        step_duration_s: float,
        final_hold_s: float,
        show_left_ui: bool,
        show_right_ui: bool,
    ) -> None:
        self.branch_name = branch_name
        self.model = model
        self.data = data
        self.output_dir = output_dir
        self.headless = bool(headless)
        self.step_duration_s = float(step_duration_s)
        self.final_hold_s = float(final_hold_s)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame0_written = False
        self.last_frame_bgr: np.ndarray | None = None
        self.writer = None
        self.renderer = None
        self.camera = None
        self.viewer = None
        if self.headless:
            self.renderer, self.camera = _build_renderer(
                model,
                width=int(render_width),
                height=int(render_height),
            )
            video_path = self.output_dir / "validation.mp4"
            self.writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(video_fps),
                (
                    int(min(int(render_width), int(model.vis.global_.offwidth))),
                    int(min(int(render_height), int(model.vis.global_.offheight))),
                ),
            )
            if not self.writer.isOpened():
                self.renderer.close()
                raise RuntimeError(f"Failed to open video writer: {video_path}")
        else:
            require_mujoco_viewer_backend()
            self.viewer = mujoco.viewer.launch_passive(
                model,
                data,
                show_left_ui=bool(show_left_ui),
                show_right_ui=bool(show_right_ui),
            )
            _apply_camera(self.viewer)

    def capture(self, *, lines: list[str], current_segment: dict[str, Any] | None) -> None:
        if self.headless:
            frame_rgb = _render_frame(self.model, self.data, self.renderer, self.camera)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_bgr = _overlay_text(frame_bgr, lines)
            self.writer.write(frame_bgr)
            self.last_frame_bgr = frame_bgr
            if not self.frame0_written:
                cv2.imwrite(str(self.output_dir / "frame_000.png"), frame_bgr)
                self.frame0_written = True
            return

        if self.viewer is None or not self.viewer.is_running():
            return
        self.viewer.set_texts(
            _build_stage_overlay_texts(
                current_segment,
                stage_labels={},
                point_index=0,
                point_count=1,
            )
        )
        self.viewer.sync()
        time.sleep(max(self.step_duration_s, 1e-3))

    def hold(self) -> None:
        if self.headless:
            return
        if self.viewer is None or not self.viewer.is_running():
            return
        started_at = time.monotonic()
        while self.viewer.is_running() and time.monotonic() - started_at < max(self.final_hold_s, 0.0):
            self.viewer.sync()
            time.sleep(0.01)

    def close(self) -> None:
        if self.headless:
            if self.last_frame_bgr is not None:
                cv2.imwrite(str(self.output_dir / "frame_last.png"), self.last_frame_bgr)
            if self.writer is not None:
                self.writer.release()
            if self.renderer is not None:
                self.renderer.close()
            return
        if self.viewer is not None:
            try:
                self.viewer.clear_texts()
            except Exception:
                logging.exception("Failed to clear viewer texts.")
            self.viewer.close()


def _segment_counter(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter[str(row.get("segment_label", "unknown"))] += 1
    return dict(counter)


def _contiguous_segments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    if not rows:
        return segments
    start = 0
    while start < len(rows):
        segment_id = str(rows[start].get("segment_id", ""))
        segment_label = str(rows[start].get("segment_label", ""))
        end = start + 1
        while end < len(rows) and str(rows[end].get("segment_id", "")) == segment_id:
            end += 1
        joint_rows = np.asarray([rows[idx]["joint_positions"] for idx in range(start, end)], dtype=float)
        segments.append(
            {
                "segment_id": segment_id,
                "segment_label": segment_label,
                "source": str(rows[start].get("source", "")),
                "point_count": int(end - start),
                "start_index": int(start),
                "end_index": int(end - 1),
                "joint_span_rad_inf": float(np.max(np.ptp(joint_rows, axis=0))) if len(joint_rows) > 0 else 0.0,
            }
        )
        start = end
    return segments


def _corridor_forward_axis_from_points(
    *,
    anchor_joint_positions: np.ndarray,
    skipped_prepick_points: list[dict[str, Any]],
    pick_points: list[dict[str, Any]],
    kinematics: CjjArmKinematics,
) -> np.ndarray:
    target_point: dict[str, Any] | None = skipped_prepick_points[-1] if skipped_prepick_points else None
    if target_point is None and pick_points:
        target_point = pick_points[0]
    anchor_pose = np.asarray(kinematics.compute_fk(anchor_joint_positions), dtype=float)
    if target_point is None:
        return np.asarray([1.0, 0.0, 0.0], dtype=float)
    target_pose = np.asarray(kinematics.compute_fk(np.asarray(target_point["joint_positions"], dtype=float)), dtype=float)
    forward_axis = target_pose[:3] - anchor_pose[:3]
    return _normalize_vector(forward_axis, fallback=np.asarray([1.0, 0.0, 0.0], dtype=float))


def _update_branch_stats(
    *,
    stats: BranchPhysicsStats,
    joint_positions: np.ndarray,
    ee_pose_rotvec: np.ndarray,
    orange_lift_m: float,
    orange_contact_detected: bool,
) -> tuple[float, float]:
    if stats.previous_joint_positions is None:
        joint_step_rad = 0.0
        ee_step_translation_mm = 0.0
    else:
        joint_step_rad = float(np.max(np.abs(joint_positions - stats.previous_joint_positions)))
        ee_step_translation_mm = float(np.linalg.norm(ee_pose_rotvec[:3] - stats.previous_ee_pose[:3]) * 1000.0)
    stats.previous_joint_positions = np.asarray(joint_positions, dtype=float)
    stats.previous_ee_pose = np.asarray(ee_pose_rotvec, dtype=float)
    stats.max_joint_step_rad = max(stats.max_joint_step_rad, joint_step_rad)
    stats.total_joint_step_rad_inf += joint_step_rad
    stats.ee_path_translation_mm += ee_step_translation_mm
    stats.max_orange_lift_m = max(stats.max_orange_lift_m, float(orange_lift_m))
    if orange_contact_detected:
        stats.contact_step_count += 1
    return joint_step_rad, ee_step_translation_mm


def _run_original_branch(
    *,
    model_path: Path,
    output_dir: Path,
    prefix_points: list[dict[str, Any]],
    pick_points: list[dict[str, Any]],
    handoff_open_width_m: float,
    orange_body_id: int | None = None,
    physical_success_lift_m: float,
    headless: bool,
    render_width: int,
    render_height: int,
    video_fps: float,
    step_duration_s: float,
    final_hold_s: float,
    show_left_ui: bool,
    show_right_ui: bool,
    interp_step_rad: float,
    settle_steps: int,
    orange_body_name: str,
    orange_geom_name: str,
) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    joint_qpos = _joint_qpos_indices(model)
    gripper_qpos = _gripper_qpos_indices(model)
    orange_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, orange_body_name)
    orange_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, orange_geom_name)
    link6_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6-7")
    grasp_center_local = _load_distal_grasp_center_local()
    finger_body_ids = {
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "finger_left")),
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "finger_right")),
    }
    sim_config = CjjArmSimConfig()
    kinematics = CjjArmKinematics(
        urdf_path=sim_config.urdf_path,
        end_effector_frame=sim_config.end_effector_frame,
        joint_names=[sim_config.urdf_joint_map[name] for name in sim_config.joint_action_order],
    )
    recorder = _BranchRecorder(
        branch_name="original_replay",
        model=model,
        data=data,
        output_dir=output_dir,
        headless=headless,
        render_width=render_width,
        render_height=render_height,
        video_fps=video_fps,
        step_duration_s=step_duration_s,
        final_hold_s=final_hold_s,
        show_left_ui=show_left_ui,
        show_right_ui=show_right_ui,
    )
    stats = BranchPhysicsStats()
    step_rows: list[dict[str, Any]] = []
    current_gripper_width_m = float(handoff_open_width_m)
    try:
        for _ in range(10):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        initial_orange_xyz = _body_xyz(data, orange_body_id)
        current_gripper_target = _gripper_qpos_from_width(
            width_m=current_gripper_width_m,
            handoff_open_width_m=handoff_open_width_m,
            sim_config=sim_config,
        )
        for point_index, point in enumerate(prefix_points):
            target_joint_positions = np.asarray(point["joint_positions"], dtype=float)
            current_gripper_width_m = _point_target_width_m(point, current_gripper_width_m, handoff_open_width_m)
            current_gripper_target = _gripper_qpos_from_width(
                width_m=current_gripper_width_m,
                handoff_open_width_m=handoff_open_width_m,
                sim_config=sim_config,
            )
            executed_joint_positions = _apply_joint_targets_physical(
                model=model,
                data=data,
                joint_qpos=joint_qpos,
                gripper_qpos=gripper_qpos,
                target_joint_positions=target_joint_positions,
                gripper_qpos_target=current_gripper_target,
                interp_step_rad=interp_step_rad,
                settle_steps=settle_steps,
            )
            orange_xyz = _body_xyz(data, orange_body_id)
            orange_contact = _orange_contact_detected(
                model=model,
                data=data,
                orange_geom_id=orange_geom_id,
                finger_body_ids=finger_body_ids,
            )
            ee_pose_rotvec = np.asarray(kinematics.compute_fk(executed_joint_positions), dtype=float)
            orange_lift_m = max(0.0, float(orange_xyz[2]) - float(initial_orange_xyz[2]))
            joint_step_rad, ee_step_translation_mm = _update_branch_stats(
                stats=stats,
                joint_positions=executed_joint_positions,
                ee_pose_rotvec=ee_pose_rotvec,
                orange_lift_m=orange_lift_m,
                orange_contact_detected=orange_contact,
            )
            segment_state = {
                "segment_id": str(point.get("segment_id", "")),
                "segment_label": str(point.get("segment_label", "")),
                "source": str(point.get("source", "")),
            }
            recorder.capture(
                lines=[
                    "branch=original_replay",
                    f"stage={segment_state['segment_label']} point={point_index + 1}/{len(prefix_points) + len(pick_points)}",
                    f"orange_lift_mm={orange_lift_m * 1000.0:.2f} contact={str(orange_contact).lower()}",
                    f"gripper_width={current_gripper_width_m:.4f}m",
                ],
                current_segment=segment_state,
            )
            step_rows.append(
                {
                    "point_index": int(point_index),
                    "segment_id": segment_state["segment_id"],
                    "segment_label": segment_state["segment_label"],
                    "source": segment_state["source"],
                    "joint_positions": executed_joint_positions.astype(float).tolist(),
                    "joint_step_rad_inf": float(joint_step_rad),
                    "ee_step_translation_mm": float(ee_step_translation_mm),
                    "orange_xyz_m": orange_xyz.astype(float).tolist(),
                    "orange_contact_detected": bool(orange_contact),
                    "orange_lift_m": float(orange_lift_m),
                    "gripper_width_m": float(current_gripper_width_m),
                }
            )

        anchor_joint_positions = _joint_positions_from_qpos(data, joint_qpos)
        held_pick_points = _hold_pick_points_from_current_pose(
            pick_points,
            current_joint_positions=anchor_joint_positions,
            anchor_index=len(prefix_points) - 1,
        )
        for pick_offset, point in enumerate(held_pick_points):
            target_joint_positions = np.asarray(point["joint_positions"], dtype=float)
            current_gripper_width_m = _point_target_width_m(point, current_gripper_width_m, handoff_open_width_m)
            current_gripper_target = _gripper_qpos_from_width(
                width_m=current_gripper_width_m,
                handoff_open_width_m=handoff_open_width_m,
                sim_config=sim_config,
            )
            executed_joint_positions = _apply_joint_targets_physical(
                model=model,
                data=data,
                joint_qpos=joint_qpos,
                gripper_qpos=gripper_qpos,
                target_joint_positions=target_joint_positions,
                gripper_qpos_target=current_gripper_target,
                interp_step_rad=interp_step_rad,
                settle_steps=settle_steps,
            )
            orange_xyz = _body_xyz(data, orange_body_id)
            orange_contact = _orange_contact_detected(
                model=model,
                data=data,
                orange_geom_id=orange_geom_id,
                finger_body_ids=finger_body_ids,
            )
            ee_pose_rotvec = np.asarray(kinematics.compute_fk(executed_joint_positions), dtype=float)
            orange_lift_m = max(0.0, float(orange_xyz[2]) - float(initial_orange_xyz[2]))
            joint_step_rad, ee_step_translation_mm = _update_branch_stats(
                stats=stats,
                joint_positions=executed_joint_positions,
                ee_pose_rotvec=ee_pose_rotvec,
                orange_lift_m=orange_lift_m,
                orange_contact_detected=orange_contact,
            )
            segment_state = {
                "segment_id": str(point.get("segment_id", "")),
                "segment_label": str(point.get("segment_label", "")),
                "source": str(point.get("source", "")),
            }
            recorder.capture(
                lines=[
                    "branch=original_replay",
                    f"stage={segment_state['segment_label']} point={len(prefix_points) + pick_offset + 1}/{len(prefix_points) + len(held_pick_points)}",
                    f"orange_lift_mm={orange_lift_m * 1000.0:.2f} contact={str(orange_contact).lower()}",
                    f"gripper_width={current_gripper_width_m:.4f}m",
                ],
                current_segment=segment_state,
            )
            step_rows.append(
                {
                    "point_index": int(len(prefix_points) + pick_offset),
                    "segment_id": segment_state["segment_id"],
                    "segment_label": segment_state["segment_label"],
                    "source": segment_state["source"],
                    "joint_positions": executed_joint_positions.astype(float).tolist(),
                    "joint_step_rad_inf": float(joint_step_rad),
                    "ee_step_translation_mm": float(ee_step_translation_mm),
                    "orange_xyz_m": orange_xyz.astype(float).tolist(),
                    "orange_contact_detected": bool(orange_contact),
                    "orange_lift_m": float(orange_lift_m),
                    "gripper_width_m": float(current_gripper_width_m),
                }
            )
        recorder.hold()
        final_orange_xyz = _body_xyz(data, orange_body_id)
        final_orange_lift_m = max(0.0, float(final_orange_xyz[2]) - float(initial_orange_xyz[2]))
        summary = {
            "status": "ok",
            "branch_name": "original_replay",
            "render_mode": "headless" if headless else "viewer",
            "output_dir": str(output_dir.resolve()),
            "video_path": None if not headless else str((output_dir / "validation.mp4").resolve()),
            "frame0_path": None if not headless else str((output_dir / "frame_000.png").resolve()),
            "frame_last_path": None if not headless else str((output_dir / "frame_last.png").resolve()),
            "point_count": int(len(step_rows)),
            "segment_label_counts": _segment_counter(step_rows),
            "contiguous_segments": _contiguous_segments(step_rows),
            "max_joint_step_rad_inf": float(stats.max_joint_step_rad),
            "total_joint_step_rad_inf": float(stats.total_joint_step_rad_inf),
            "ee_path_translation_mm": float(stats.ee_path_translation_mm),
            "initial_orange_xyz_m": initial_orange_xyz.astype(float).tolist(),
            "final_orange_xyz_m": final_orange_xyz.astype(float).tolist(),
            "max_orange_lift_mm": float(stats.max_orange_lift_m * 1000.0),
            "final_orange_lift_mm": float(final_orange_lift_m * 1000.0),
            "orange_contact_step_count": int(stats.contact_step_count),
            "physical_success": bool(stats.max_orange_lift_m >= float(physical_success_lift_m)),
            "physical_success_lift_threshold_mm": float(physical_success_lift_m * 1000.0),
        }
        _save_json(output_dir / "summary.json", summary)
        (output_dir / "step_metrics.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in step_rows),
            encoding="utf-8",
        )
        return summary
    finally:
        recorder.close()


def _run_frrg_branch(
    *,
    model_path: Path,
    output_dir: Path,
    prefix_points: list[dict[str, Any]],
    corridor_forward_axis_base: np.ndarray,
    handoff_open_width_m: float,
    frrg_config_path: Path,
    frrg_max_steps: int,
    physical_success_lift_m: float,
    headless: bool,
    render_width: int,
    render_height: int,
    video_fps: float,
    step_duration_s: float,
    final_hold_s: float,
    show_left_ui: bool,
    show_right_ui: bool,
    interp_step_rad: float,
    settle_steps: int,
    orange_radius_m: float,
    orange_body_name: str,
    orange_geom_name: str,
) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    joint_qpos = _joint_qpos_indices(model)
    gripper_qpos = _gripper_qpos_indices(model)
    orange_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, orange_body_name)
    orange_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, orange_geom_name)
    link6_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6-7")
    grasp_center_local = _load_distal_grasp_center_local()
    finger_body_ids = {
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "finger_left")),
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "finger_right")),
    }
    sim_config = CjjArmSimConfig()
    kinematics = CjjArmKinematics(
        urdf_path=sim_config.urdf_path,
        end_effector_frame=sim_config.end_effector_frame,
        joint_names=[sim_config.urdf_joint_map[name] for name in sim_config.joint_action_order],
    )
    recorder = _BranchRecorder(
        branch_name="frrg_true_closed_loop",
        model=model,
        data=data,
        output_dir=output_dir,
        headless=headless,
        render_width=render_width,
        render_height=render_height,
        video_fps=video_fps,
        step_duration_s=step_duration_s,
        final_hold_s=final_hold_s,
        show_left_ui=show_left_ui,
        show_right_ui=show_right_ui,
    )
    stats = BranchPhysicsStats()
    step_rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    guards: list[dict[str, Any]] = []
    phase_trace: list[dict[str, Any]] = []
    max_raw_action_norm = 0.0
    max_safe_action_norm = 0.0
    max_residual_norm = 0.0
    current_gripper_width_m = float(handoff_open_width_m)
    config = load_frrg_config(frrg_config_path)
    config = replace(
        config,
        capture_build=replace(
            config.capture_build,
            vert_gain=max(float(config.capture_build.vert_gain), 1.0),
            target_depth_max_m=min(
                float(config.capture_build.target_depth_max_m),
                max(float(config.capture_build.target_depth_goal_m) + 0.003, 0.015),
            ),
        ),
    )
    controller: FRRGClosedLoopController | None = None
    control_payload: dict[str, Any] | None = None
    previous_orange_xyz: np.ndarray | None = None
    final_phase = "HANDOFF"
    failure_reason: str | None = None
    try:
        for _ in range(10):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        initial_orange_xyz = _body_xyz(data, orange_body_id)
        current_gripper_target = _gripper_qpos_from_width(
            width_m=current_gripper_width_m,
            handoff_open_width_m=handoff_open_width_m,
            sim_config=sim_config,
        )
        for point_index, point in enumerate(prefix_points):
            target_joint_positions = np.asarray(point["joint_positions"], dtype=float)
            current_gripper_width_m = _point_target_width_m(point, current_gripper_width_m, handoff_open_width_m)
            current_gripper_target = _gripper_qpos_from_width(
                width_m=current_gripper_width_m,
                handoff_open_width_m=handoff_open_width_m,
                sim_config=sim_config,
            )
            executed_joint_positions = _apply_joint_targets_physical(
                model=model,
                data=data,
                joint_qpos=joint_qpos,
                gripper_qpos=gripper_qpos,
                target_joint_positions=target_joint_positions,
                gripper_qpos_target=current_gripper_target,
                interp_step_rad=interp_step_rad,
                settle_steps=settle_steps,
            )
            orange_xyz = _body_xyz(data, orange_body_id)
            orange_contact = _orange_contact_detected(
                model=model,
                data=data,
                orange_geom_id=orange_geom_id,
                finger_body_ids=finger_body_ids,
            )
            ee_pose_rotvec = np.asarray(kinematics.compute_fk(executed_joint_positions), dtype=float)
            orange_lift_m = max(0.0, float(orange_xyz[2]) - float(initial_orange_xyz[2]))
            joint_step_rad, ee_step_translation_mm = _update_branch_stats(
                stats=stats,
                joint_positions=executed_joint_positions,
                ee_pose_rotvec=ee_pose_rotvec,
                orange_lift_m=orange_lift_m,
                orange_contact_detected=orange_contact,
            )
            recorder.capture(
                lines=[
                    "branch=frrg_true_closed_loop",
                    f"stage={point.get('segment_label', '')} point={point_index + 1}/{len(prefix_points)}",
                    f"orange_lift_mm={orange_lift_m * 1000.0:.2f} contact={str(orange_contact).lower()}",
                    f"gripper_width={current_gripper_width_m:.4f}m",
                ],
                current_segment={
                    "segment_id": str(point.get("segment_id", "")),
                    "segment_label": str(point.get("segment_label", "")),
                    "source": str(point.get("source", "")),
                },
            )
            step_rows.append(
                {
                    "point_index": int(point_index),
                    "segment_id": str(point.get("segment_id", "")),
                    "segment_label": str(point.get("segment_label", "")),
                    "source": str(point.get("source", "")),
                    "joint_positions": executed_joint_positions.astype(float).tolist(),
                    "joint_step_rad_inf": float(joint_step_rad),
                    "ee_step_translation_mm": float(ee_step_translation_mm),
                    "orange_xyz_m": orange_xyz.astype(float).tolist(),
                    "orange_contact_detected": bool(orange_contact),
                    "orange_lift_m": float(orange_lift_m),
                    "gripper_width_m": float(current_gripper_width_m),
                }
            )
        prefix_count = len(step_rows)
        for step_index in range(max(int(frrg_max_steps), 1)):
            joint_positions = _joint_positions_from_qpos(data, joint_qpos)
            orange_xyz = _body_xyz(data, orange_body_id)
            orange_contact = _orange_contact_detected(
                model=model,
                data=data,
                orange_geom_id=orange_geom_id,
                finger_body_ids=finger_body_ids,
            )
            if control_payload is None:
                control_payload = {
                    "timestamp": 0.0,
                    "phase": "CAPTURE_BUILD",
                    "retry_count": 0,
                    "stable_count": 0,
                    "phase_elapsed_s": 0.0,
                }
            grasp_center_world_xyz = _body_point_world(data, link6_body_id, grasp_center_local)
            payload, sim_metrics = _build_frrg_sim_payload(
                joint_positions=joint_positions,
                orange_xyz=orange_xyz,
                previous_orange_xyz=previous_orange_xyz,
                current_gripper_width_m=current_gripper_width_m,
                orange_contact_detected=orange_contact,
                initial_orange_z_m=float(initial_orange_xyz[2]),
                control_payload=control_payload,
                kinematics=kinematics,
                corridor_forward_axis_base=corridor_forward_axis_base,
                orange_radius_m=orange_radius_m,
                control_point_world_xyz=grasp_center_world_xyz,
            )
            if controller is None:
                controller = FRRGClosedLoopController(config, payload, input_mode=COARSE_HANDOFF_SOURCE)
            step_result = controller.step(step_index, state_payload=payload)
            action_record = step_result.action_record()
            state_record = step_result.state_record()
            guard_record = step_result.guards_record()
            phase_record = step_result.phase_trace_record()
            actions.append(action_record)
            states.append(state_record)
            guards.append(guard_record)
            phase_trace.append(phase_record)
            max_raw_action_norm = max(max_raw_action_norm, float(step_result.safety_result.raw_action_norm))
            max_safe_action_norm = max(max_safe_action_norm, float(step_result.safety_result.safe_action_norm))
            max_residual_norm = max(max_residual_norm, float(step_result.safety_result.residual_norm))
            final_phase = str(step_result.phase_transition.next_phase)
            failure_reason = step_result.phase_transition.reason or step_result.safety_result.reason

            safe_action = dict(action_record["safe_action"])
            current_pose_rotvec = np.asarray(kinematics.compute_fk(joint_positions), dtype=float)
            object_pose_payload = dict(payload["object_pose_base"])
            base_delta = _object_delta_to_base(list(safe_action["delta_pose_object"]), object_pose_payload)
            target_pose_rotvec = compose_pose_delta(current_pose_rotvec, base_delta, rotation_frame="world")
            target_gripper_width_m = max(
                0.0,
                min(float(handoff_open_width_m), float(current_gripper_width_m) + float(safe_action["delta_gripper"])),
            )
            target_gripper_qpos = _gripper_qpos_from_width(
                width_m=target_gripper_width_m,
                handoff_open_width_m=handoff_open_width_m,
                sim_config=sim_config,
            )
            target_joint_positions = np.asarray(
                kinematics.compute_ik(
                    target_pose_rotvec,
                    joint_positions,
                    position_weight=1.0,
                    orientation_weight=0.05,
                    keep_pointing_only=True,
                ),
                dtype=float,
            )
            if not bool(action_record["stop"]) and final_phase != PHASE_FAILURE:
                executed_joint_positions = _apply_joint_targets_physical(
                    model=model,
                    data=data,
                    joint_qpos=joint_qpos,
                    gripper_qpos=gripper_qpos,
                    target_joint_positions=target_joint_positions,
                    gripper_qpos_target=target_gripper_qpos,
                    interp_step_rad=interp_step_rad,
                    settle_steps=settle_steps,
                )
                current_gripper_width_m = target_gripper_width_m
            else:
                executed_joint_positions = joint_positions

            achieved_pose_rotvec = np.asarray(kinematics.compute_fk(executed_joint_positions), dtype=float)
            orange_xyz_after = _body_xyz(data, orange_body_id)
            orange_contact_after = _orange_contact_detected(
                model=model,
                data=data,
                orange_geom_id=orange_geom_id,
                finger_body_ids=finger_body_ids,
            )
            orange_lift_m = max(0.0, float(orange_xyz_after[2]) - float(initial_orange_xyz[2]))
            joint_step_rad, ee_step_translation_mm = _update_branch_stats(
                stats=stats,
                joint_positions=executed_joint_positions,
                ee_pose_rotvec=achieved_pose_rotvec,
                orange_lift_m=orange_lift_m,
                orange_contact_detected=orange_contact_after,
            )
            translation_error_mm = _translation_error_mm(target_pose_rotvec, achieved_pose_rotvec)
            rotation_error_deg = _rotation_error_deg(target_pose_rotvec, achieved_pose_rotvec)
            segment_state = {
                "segment_id": f"frrg_{str(step_result.scored.state.phase).lower()}",
                "segment_label": str(step_result.scored.state.phase).lower(),
                "source": "frrg_true_closed_loop",
            }
            recorder.capture(
                lines=[
                    "branch=frrg_true_closed_loop",
                    f"phase={step_result.scored.state.phase} next={final_phase} step={step_index + 1}/{int(frrg_max_steps)}",
                    f"orange_lift_mm={orange_lift_m * 1000.0:.2f} contact={str(orange_contact_after).lower()}",
                    f"track_err={translation_error_mm:.2f}mm rot_err={rotation_error_deg:.2f}deg",
                    f"gripper_width={current_gripper_width_m:.4f}m",
                ],
                current_segment=segment_state,
            )
            step_rows.append(
                {
                    "point_index": int(prefix_count + step_index),
                    "segment_id": segment_state["segment_id"],
                    "segment_label": segment_state["segment_label"],
                    "source": segment_state["source"],
                    "joint_positions": executed_joint_positions.astype(float).tolist(),
                    "joint_step_rad_inf": float(joint_step_rad),
                    "ee_step_translation_mm": float(ee_step_translation_mm),
                    "orange_xyz_m": orange_xyz_after.astype(float).tolist(),
                    "orange_contact_detected": bool(orange_contact_after),
                    "orange_lift_m": float(orange_lift_m),
                    "gripper_width_m": float(current_gripper_width_m),
                    "controller_phase": str(step_result.scored.state.phase),
                    "controller_next_phase": final_phase,
                    "safe_action": safe_action,
                    "raw_action_norm": float(step_result.safety_result.raw_action_norm),
                    "safe_action_norm": float(step_result.safety_result.safe_action_norm),
                    "residual_norm": float(step_result.safety_result.residual_norm),
                    "target_pose_6d": _rotvec_pose_to_pose6d_dict(target_pose_rotvec),
                    "translation_error_mm": float(translation_error_mm),
                    "rotation_error_deg": float(rotation_error_deg),
                    "payload_metrics": sim_metrics,
                }
            )
            previous_orange_xyz = orange_xyz_after.copy()
            next_object_pose_base, _lateral_axis, _vertical_axis, _outward_axis = _corridor_object_pose(
                orange_xyz=orange_xyz_after,
                corridor_forward_axis_base=corridor_forward_axis_base,
            )
            control_payload = {
                **step_result.next_payload,
                "ee_pose_base": _rotvec_pose_to_pose6d_dict(achieved_pose_rotvec),
                "object_pose_base": next_object_pose_base,
                "gripper_width": float(current_gripper_width_m),
                "gripper_cmd": float(current_gripper_width_m),
                "object_jump_m": float(sim_metrics["object_jump_m"]),
            }
            if final_phase in {PHASE_SUCCESS, PHASE_FAILURE}:
                break

        recorder.hold()
        final_orange_xyz = _body_xyz(data, orange_body_id)
        final_orange_lift_m = max(0.0, float(final_orange_xyz[2]) - float(initial_orange_xyz[2]))
        status = "success" if final_phase == PHASE_SUCCESS else "failure"
        summary = {
            "status": status,
            "branch_name": "frrg_true_closed_loop",
            "render_mode": "headless" if headless else "viewer",
            "output_dir": str(output_dir.resolve()),
            "video_path": None if not headless else str((output_dir / "validation.mp4").resolve()),
            "frame0_path": None if not headless else str((output_dir / "frame_000.png").resolve()),
            "frame_last_path": None if not headless else str((output_dir / "frame_last.png").resolve()),
            "point_count": int(len(step_rows)),
            "prefix_point_count": int(prefix_count),
            "frrg_control_step_count": int(len(actions)),
            "segment_label_counts": _segment_counter(step_rows),
            "contiguous_segments": _contiguous_segments(step_rows),
            "max_joint_step_rad_inf": float(stats.max_joint_step_rad),
            "total_joint_step_rad_inf": float(stats.total_joint_step_rad_inf),
            "ee_path_translation_mm": float(stats.ee_path_translation_mm),
            "initial_orange_xyz_m": initial_orange_xyz.astype(float).tolist(),
            "final_orange_xyz_m": final_orange_xyz.astype(float).tolist(),
            "max_orange_lift_mm": float(stats.max_orange_lift_m * 1000.0),
            "final_orange_lift_mm": float(final_orange_lift_m * 1000.0),
            "orange_contact_step_count": int(stats.contact_step_count),
            "physical_success": bool(stats.max_orange_lift_m >= float(physical_success_lift_m)),
            "physical_success_lift_threshold_mm": float(physical_success_lift_m * 1000.0),
            "controller_status": status,
            "controller_final_phase": final_phase,
            "controller_failure_reason": None if final_phase == PHASE_SUCCESS else failure_reason,
            "controller_steps_run": int(len(actions)),
            "max_raw_action_norm": float(max_raw_action_norm),
            "max_safe_action_norm": float(max_safe_action_norm),
            "max_residual_norm": float(max_residual_norm),
        }
        _save_json(output_dir / "summary.json", summary)
        (output_dir / "step_metrics.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in step_rows),
            encoding="utf-8",
        )
        analysis_dir = output_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        (analysis_dir / "actions.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in actions),
            encoding="utf-8",
        )
        (analysis_dir / "states.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in states),
            encoding="utf-8",
        )
        (analysis_dir / "guards.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in guards),
            encoding="utf-8",
        )
        (analysis_dir / "phase_trace.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in phase_trace),
            encoding="utf-8",
        )
        return summary
    finally:
        recorder.close()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)
    if args.display is not None:
        os.environ["DISPLAY"] = args.display

    selected_branch = str(args.branch)
    if not bool(args.headless) and selected_branch == "both":
        raise ValueError("GUI mode only supports a single branch. Pass --branch original_replay or --branch frrg_true_closed_loop.")

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
    run_dir = args.output_root / _timestamp_name("pickorange_true_closed_loop")
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
    prefix_points, pick_points, _suffix_points = _split_pick_segment(planned_points, semantic_states)
    pick_start = len(prefix_points)
    if anchor_index >= pick_start:
        raise ValueError(
            f"start-index={anchor_index} is not before pick segment start={pick_start}; "
            "this true closed-loop flow expects an anchor before pickorange."
        )
    jump_to_pick_after_index = int(args.jump_to_pick_after_index)
    if jump_to_pick_after_index < anchor_index:
        raise ValueError(
            f"jump-to-pick-after-index={jump_to_pick_after_index} is before start-index={anchor_index}."
        )
    if jump_to_pick_after_index >= pick_start:
        raise ValueError(
            f"jump-to-pick-after-index={jump_to_pick_after_index} must be before pick segment start={pick_start}."
        )
    branch_prefix_points = list(planned_points[anchor_index : jump_to_pick_after_index + 1])
    skipped_prepick_points = list(planned_points[jump_to_pick_after_index + 1 : pick_start])

    target_object_payload = _target_object_payload(live_result, task_config.target_phrase)
    current_aux_object_positions = _aux_object_positions(live_result, task_config.target_phrase)
    orange_xyz = np.asarray(target_object_payload["base_xyz_m"], dtype=float).reshape(3)
    pink_cup_xyz = None
    if "pink_cup" in current_aux_object_positions:
        pink_cup_xyz = np.asarray(current_aux_object_positions["pink_cup"], dtype=float).reshape(3)

    scene_model_path = _write_physical_scene(
        base_model_path=args.model_path,
        output_dir=run_dir / "analysis" / "scene",
        orange_xyz=orange_xyz,
        orange_radius_m=float(args.orange_radius_m),
        orange_mass_kg=float(args.orange_mass_kg),
        pink_cup_xyz=pink_cup_xyz,
        pink_cup_radius_m=float(args.pink_cup_radius_m),
        pink_cup_height_m=float(args.pink_cup_height_m),
    )

    sim_config = CjjArmSimConfig()
    kinematics = CjjArmKinematics(
        urdf_path=sim_config.urdf_path,
        end_effector_frame=sim_config.end_effector_frame,
        joint_names=[sim_config.urdf_joint_map[name] for name in sim_config.joint_action_order],
    )
    corridor_forward_axis_base = _corridor_forward_axis_from_points(
        anchor_joint_positions=np.asarray(branch_prefix_points[-1]["joint_positions"], dtype=float),
        skipped_prepick_points=skipped_prepick_points,
        pick_points=pick_points,
        kinematics=kinematics,
    )

    frrg_config = load_frrg_config(args.frrg_config)
    stage_labels = _load_segment_display_labels(task_config.skill_bank_path)
    stage_labels.update(
        {
            "frrg_handoff": "frrg_handoff",
            "frrg_capture_build": "frrg_capture_build",
            "frrg_close_hold": "frrg_close_hold",
            "frrg_lift_test": "frrg_lift_test",
        }
    )

    branch_root = run_dir / "branches"
    original_summary: dict[str, Any] | None = None
    frrg_summary: dict[str, Any] | None = None
    if selected_branch in {"both", "original_replay"}:
        original_summary = _run_original_branch(
            model_path=scene_model_path,
            output_dir=branch_root / "original_replay",
            prefix_points=branch_prefix_points,
            pick_points=pick_points,
            handoff_open_width_m=float(frrg_config.handoff.handoff_open_width_m),
            physical_success_lift_m=float(args.physical_success_lift_mm) / 1000.0,
            headless=bool(args.headless),
            render_width=int(args.render_width),
            render_height=int(args.render_height),
            video_fps=float(args.video_fps),
            step_duration_s=float(args.step_duration_s),
            final_hold_s=float(args.final_hold_s),
            show_left_ui=bool(args.show_left_ui),
            show_right_ui=bool(args.show_right_ui),
            interp_step_rad=float(args.physics_interp_step_rad),
            settle_steps=int(args.physics_settle_steps),
            orange_body_name="orange_body",
            orange_geom_name="orange_geom",
        )

    if selected_branch in {"both", "frrg_true_closed_loop"}:
        frrg_summary = _run_frrg_branch(
            model_path=scene_model_path,
            output_dir=branch_root / "frrg_true_closed_loop",
            prefix_points=branch_prefix_points,
            corridor_forward_axis_base=corridor_forward_axis_base,
            handoff_open_width_m=float(frrg_config.handoff.handoff_open_width_m),
            frrg_config_path=args.frrg_config,
            frrg_max_steps=int(args.frrg_max_steps),
            physical_success_lift_m=float(args.physical_success_lift_mm) / 1000.0,
            headless=bool(args.headless),
            render_width=int(args.render_width),
            render_height=int(args.render_height),
            video_fps=float(args.video_fps),
            step_duration_s=float(args.step_duration_s),
            final_hold_s=float(args.final_hold_s),
            show_left_ui=bool(args.show_left_ui),
            show_right_ui=bool(args.show_right_ui),
            interp_step_rad=float(args.physics_interp_step_rad),
            settle_steps=int(args.physics_settle_steps),
            orange_radius_m=float(args.orange_radius_m),
            orange_body_name="orange_body",
            orange_geom_name="orange_geom",
        )

    compare_metrics: dict[str, Any] | None = None
    if original_summary is not None and frrg_summary is not None:
        compare_metrics = {
            "original_physical_success": bool(original_summary["physical_success"]),
            "frrg_physical_success": bool(frrg_summary["physical_success"]),
            "original_max_orange_lift_mm": float(original_summary["max_orange_lift_mm"]),
            "frrg_max_orange_lift_mm": float(frrg_summary["max_orange_lift_mm"]),
            "orange_lift_delta_mm": float(frrg_summary["max_orange_lift_mm"] - original_summary["max_orange_lift_mm"]),
            "original_contact_step_count": int(original_summary["orange_contact_step_count"]),
            "frrg_contact_step_count": int(frrg_summary["orange_contact_step_count"]),
            "frrg_controller_status": str(frrg_summary["controller_status"]),
            "frrg_controller_final_phase": str(frrg_summary["controller_final_phase"]),
            "frrg_controller_steps_run": int(frrg_summary["controller_steps_run"]),
        }

    compare_summary = {
        "status": "ok",
        "run_dir": str(run_dir.resolve()),
        "selected_branch": selected_branch,
        "task_config_path": str(args.task_config.resolve()),
        "frrg_config_path": str(args.frrg_config.resolve()),
        "live_result_path": str(live_result_path.resolve()),
        "trajectory_points_path": str(trajectory_points_path.resolve()),
        "scene_model_path": str(scene_model_path.resolve()),
        "anchor_trajectory_index": int(anchor_index),
        "jump_to_pick_after_index": int(jump_to_pick_after_index),
        "pick_segment_start_index": int(pick_start),
        "pick_segment_point_count": int(len(pick_points)),
        "skipped_prepick_point_count": int(len(skipped_prepick_points)),
        "corridor_forward_axis_base": [float(value) for value in corridor_forward_axis_base],
        "target_phrase": task_config.target_phrase,
        "aux_target_phrases": list(aux_target_phrases),
        "target_base_xyz_m": orange_xyz.astype(float).tolist(),
        "aux_object_positions_m": {
            str(name): value.astype(float).tolist()
            for name, value in current_aux_object_positions.items()
        },
        "orange_radius_m": float(args.orange_radius_m),
        "orange_mass_kg": float(args.orange_mass_kg),
        "physical_success_lift_threshold_mm": float(args.physical_success_lift_mm),
        "pose_summary": pose_summary,
        "trajectory_summary": trajectory_summary,
        "original_branch": original_summary,
        "frrg_branch": frrg_summary,
        "compare_metrics": compare_metrics,
    }
    _save_json(run_dir / "summary.json", compare_summary)
    print(json.dumps(compare_summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
