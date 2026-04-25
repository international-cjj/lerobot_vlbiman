#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
import json
import logging
import multiprocessing as mp
import os
import queue
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
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
    _capture_frame,
    _load_json,
    _load_task_config,
    _load_yaml,
    _load_base_from_camera,
    _run_t5_t6,
    _write_single_frame_session,
)
from lerobot.projects.vlbiman_sa.app.run_pose_adaptation import resolve_pose_target_phrases
from lerobot.projects.vlbiman_sa.app.run_one_shot_record import _build_camera
from lerobot.projects.vlbiman_sa.demo.schema import FrameRecord
from lerobot.projects.vlbiman_sa.geometry.transforms import apply_transform_points
from lerobot.projects.vlbiman_sa.sim import (
    DEFAULT_SCENE_PRESET_NAME,
    DualCameraSceneConfig,
    ScenePrimitiveObjectConfig,
    TargetSphereConfig,
    build_dual_camera_scene,
    load_base_from_camera_transform,
    load_wrist_camera_mount_pose,
    scene_preset_names,
    scene_preset_objects,
    scene_object_body_name,
)
from lerobot.projects.vlbiman_sa.vision import (
    AnchorEstimator,
    AnchorEstimatorConfig,
    CameraIntrinsics,
    OrientationMomentsEstimator,
    VLMObjectSegmentor,
    VLMObjectSegmentorConfig,
)


def _default_task_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "task_grasp.yaml"


def _default_capture_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "one_shot_record.yaml"


def _default_vision_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "vision_analysis.yaml"


def _default_output_root() -> Path:
    return REPO_ROOT / "outputs" / "vlbiman_sa" / "visual_closed_loop"


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


def _default_urdf_path() -> Path:
    return (
        REPO_ROOT
        / "lerobot_robot_cjjarm"
        / "lerobot_robot_cjjarm"
        / "cjjarm_urdf"
        / "TRLC-DK1-Follower.urdf"
    )


def _default_handeye_path() -> Path:
    return REPO_ROOT / "outputs" / "vlbiman_sa" / "calib" / "handeye_result.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously localize the orange, update it in MuJoCo, and replan the simulated grasp."
    )
    parser.add_argument("--task-config", type=Path, default=_default_task_config_path())
    parser.add_argument("--capture-config", type=Path, default=_default_capture_config_path())
    parser.add_argument("--vision-config", type=Path, default=_default_vision_config_path())
    parser.add_argument("--output-root", type=Path, default=_default_output_root())
    parser.add_argument("--live-output-root", type=Path, default=_default_live_output_root())
    parser.add_argument(
        "--model-path",
        type=Path,
        default=_default_model_path(),
        help="Base MJCF when using the unified dual-camera scene, or the raw MuJoCo XML path in --legacy-scene mode.",
    )
    parser.add_argument("--urdf-path", type=Path, default=_default_urdf_path())
    parser.add_argument("--handeye-result", type=Path, default=_default_handeye_path())
    parser.add_argument("--display", type=str, default=":1")
    parser.add_argument("--target-phrase", type=str, default=None)
    parser.add_argument("--container-target-phrase", type=str, default=None, help="Legacy alias for a single auxiliary target phrase.")
    parser.add_argument("--aux-target-phrase", action="append", default=None, help="Additional live target phrase to localize alongside the primary target. Repeat for multiple targets.")
    parser.add_argument("--camera-serial-number", type=str, default=None)
    parser.add_argument("--capture-mode", choices=("single_shot", "stream"), default="single_shot")
    parser.add_argument("--single-capture-timeout-s", type=float, default=20.0)
    parser.add_argument("--stream-interval-s", type=float, default=1.0)
    parser.add_argument("--replan-distance-threshold-m", type=float, default=0.015)
    parser.add_argument("--step-duration-s", type=float, default=0.04)
    parser.add_argument("--stability-window-size", type=int, default=None)
    parser.add_argument("--position-variance-threshold-mm2", type=float, default=None)
    parser.add_argument("--orientation-variance-threshold-deg2", type=float, default=None)
    parser.add_argument("--bridge-max-joint-step-rad", type=float, default=0.08)
    parser.add_argument("--max-cycles", type=int, default=0, help="Stop after this many successful capture cycles. 0 means unlimited.")
    parser.add_argument("--max-runtime-s", type=float, default=0.0, help="Stop after this many seconds. 0 means unlimited.")
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--camera-timeout-ms", type=int, default=2500)
    parser.add_argument("--target-radius-m", type=float, default=0.022)
    parser.add_argument("--target-mass-kg", type=float, default=0.12)
    parser.add_argument(
        "--scene-preset",
        type=str,
        choices=scene_preset_names(),
        default=DEFAULT_SCENE_PRESET_NAME,
        help="Optional physical-object preset to append to the canonical dual-camera scene.",
    )
    parser.add_argument(
        "--legacy-scene",
        action="store_true",
        help="Use the legacy arm-only MuJoCo scene and render the orange as a non-colliding marker.",
    )
    parser.add_argument("--show-left-ui", action="store_true")
    parser.add_argument("--show-right-ui", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _timestamp_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _apply_camera(handle: mujoco.viewer.Handle) -> None:
    handle.cam.azimuth = 148.0
    handle.cam.elevation = -20.0
    handle.cam.distance = 1.75
    handle.cam.lookat[:] = np.asarray([0.0, 0.0, 0.24], dtype=float)


def _joint_qpos_indices(model: mujoco.MjModel) -> dict[str, int]:
    joint_qpos: dict[str, int] = {}
    for joint_name in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint not found in MuJoCo model: {joint_name}")
        joint_qpos[joint_name] = int(model.jnt_qposadr[joint_id])
    return joint_qpos


def _set_robot_qpos(data: mujoco.MjData, joint_qpos: dict[str, int], joint_positions: np.ndarray) -> None:
    for joint_idx, joint_name in enumerate(("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")):
        data.qpos[joint_qpos[joint_name]] = float(joint_positions[joint_idx])


def _target_freejoint_indices(model: mujoco.MjModel) -> tuple[int | None, int | None]:
    target_joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "virtual_target_body_free"))
    if target_joint_id < 0:
        return None, None
    return int(model.jnt_qposadr[target_joint_id]), int(model.jnt_dofadr[target_joint_id])


def _set_target_freejoint_pose(
    data: mujoco.MjData,
    *,
    qpos_addr: int | None,
    qvel_addr: int | None,
    position_xyz_m: np.ndarray | None,
) -> None:
    if qpos_addr is None or position_xyz_m is None:
        return
    position = np.asarray(position_xyz_m, dtype=float).reshape(3)
    data.qpos[qpos_addr : qpos_addr + 3] = position
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
    if qvel_addr is not None:
        data.qvel[qvel_addr : qvel_addr + 6] = 0.0


def _object_freejoint_indices(model: mujoco.MjModel, object_key: str) -> tuple[int | None, int | None]:
    joint_name = f"{scene_object_body_name(object_key)}_free"
    joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
    if joint_id < 0:
        return None, None
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def _physical_object_config(object_key: str, position_xyz_m: tuple[float, float, float]) -> ScenePrimitiveObjectConfig:
    normalized = _phrase_key(object_key)
    if normalized == "pink_cup":
        return ScenePrimitiveObjectConfig(
            object_key=object_key,
            shape="cylinder",
            position_xyz_m=position_xyz_m,
            size_xyz_m=(0.035, 0.05, 0.0),
            mass_kg=0.18,
            rgba=(1.0, 0.36, 0.7, 1.0),
        )
    return ScenePrimitiveObjectConfig(
        object_key=object_key,
        shape="box",
        position_xyz_m=position_xyz_m,
        size_xyz_m=(0.035, 0.035, 0.03),
        mass_kg=0.12,
        rgba=(0.65, 0.75, 0.98, 1.0),
    )


def _build_validation_model(
    *,
    base_mjcf_path: Path,
    urdf_path: Path,
    handeye_result_path: Path,
    aux_object_keys: list[str],
    target_radius_m: float,
    target_mass_kg: float,
    scene_preset: str,
    legacy_scene: bool,
) -> tuple[mujoco.MjModel, bool]:
    if legacy_scene:
        return mujoco.MjModel.from_xml_path(str(base_mjcf_path)), True
    target_position = (0.45, 0.0, 0.08)
    base_from_camera = load_base_from_camera_transform(handeye_result_path)
    wrist_xyz, wrist_rpy = load_wrist_camera_mount_pose(urdf_path)
    artifacts = build_dual_camera_scene(
        base_mjcf_path=base_mjcf_path,
        base_from_external_camera=base_from_camera,
        wrist_camera_xyz_m=wrist_xyz,
        wrist_camera_rpy_rad=wrist_rpy,
        config=DualCameraSceneConfig(
            include_target_cube=str(scene_preset).strip().lower() == DEFAULT_SCENE_PRESET_NAME,
            target=TargetSphereConfig(
                position_xyz_m=target_position,
                radius_m=float(target_radius_m),
                mass_kg=float(target_mass_kg),
                rgba=(1.0, 1.0, 0.0, 0.0)
                if str(scene_preset).strip().lower() != DEFAULT_SCENE_PRESET_NAME
                else (0.93, 0.36, 0.08, 0.35),
            ),
            objects=tuple(
                list(
                    _physical_object_config(object_key, (1.5 + 0.15 * index, 0.0, 0.08))
                    for index, object_key in enumerate(aux_object_keys)
                )
                + list(
                    scene_preset_objects(
                        scene_preset,
                        center_xy_m=(float(target_position[0]), float(target_position[1])),
                    )
                )
            ),
        ),
    )
    return mujoco.MjModel.from_xml_string(artifacts.xml_text), False


def _segment_badge_style(invariance: str) -> tuple[mujoco.mjtGeom, np.ndarray, np.ndarray]:
    if invariance == "var":
        return (
            mujoco.mjtGeom.mjGEOM_BOX,
            np.asarray([0.028, 0.028, 0.028], dtype=np.float64),
            np.asarray([0.24, 0.51, 1.0, 0.95], dtype=np.float32),
        )
    if invariance == "inv":
        return (
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.asarray([0.018, 0.042, 0.0], dtype=np.float64),
            np.asarray([1.0, 0.6, 0.27, 0.95], dtype=np.float32),
        )
    if invariance == "bridge":
        return (
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            np.asarray([0.02, 0.04, 0.0], dtype=np.float64),
            np.asarray([0.82, 0.82, 0.82, 0.95], dtype=np.float32),
        )
    return (
        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        np.asarray([0.022, 0.022, 0.022], dtype=np.float64),
        np.asarray([0.92, 0.92, 0.92, 0.95], dtype=np.float32),
    )


def _current_segment_state(current_points: list[dict[str, Any]], current_index: int) -> dict[str, Any] | None:
    if not current_points:
        return None
    point = dict(current_points[min(current_index, len(current_points) - 1)])
    point.setdefault("segment_id", "unknown_segment")
    point.setdefault("segment_label", "")
    point.setdefault("invariance", "unknown")
    point.setdefault("source", "unknown")
    return point


def _load_segment_display_labels(skill_bank_path: Path | None) -> dict[str, str]:
    if skill_bank_path is None or not skill_bank_path.exists():
        return {}
    try:
        payload = _load_json(skill_bank_path)
    except Exception:
        logging.exception("Failed to load skill bank display labels from %s.", skill_bank_path)
        return {}
    labels: dict[str, str] = {}
    for segment in payload.get("segments", []):
        if not isinstance(segment, dict):
            continue
        segment_id = str(segment.get("segment_id", "")).strip()
        if not segment_id:
            continue
        metrics = segment.get("metrics") if isinstance(segment.get("metrics"), dict) else {}
        display_state = metrics.get("semantic_state") or segment.get("label") or segment_id
        labels[segment_id] = str(display_state)
    return labels


def _build_stage_overlay_texts(
    current_segment: dict[str, Any] | None,
    *,
    stage_labels: dict[str, str],
    point_index: int,
    point_count: int,
) -> list[tuple[int, int, str, str]]:
    font = int(mujoco.mjtFontScale.mjFONTSCALE_150)
    gridpos = int(mujoco.mjtGridPos.mjGRID_TOPLEFT)
    if current_segment is None:
        return [
            (font, gridpos, "stage", "waiting_for_plan"),
            (font, gridpos, "progress", f"0 / {int(point_count)}"),
        ]

    segment_id = str(current_segment.get("segment_id", "unknown_segment"))
    segment_label = str(current_segment.get("segment_label", ""))
    invariance = str(current_segment.get("invariance", "unknown"))
    source = str(current_segment.get("source", "unknown"))
    display_stage = stage_labels.get(segment_id) or segment_label or segment_id
    return [
        (font, gridpos, "stage", display_stage),
        (font, gridpos, "segment", segment_id),
        (font, gridpos, "mode", f"{segment_label} | {invariance}"),
        (font, gridpos, "source", source),
        (font, gridpos, "progress", f"{min(point_index + 1, max(point_count, 1))} / {int(point_count)}"),
    ]


def _aux_marker_style(object_key: str) -> tuple[mujoco.mjtGeom, np.ndarray, np.ndarray]:
    normalized = _phrase_key(object_key)
    if normalized == "pink_cup":
        return (
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            np.asarray([0.035, 0.05, 0.0], dtype=np.float64),
            np.asarray([1.0, 0.36, 0.7, 0.92], dtype=np.float32),
        )
    return (
        mujoco.mjtGeom.mjGEOM_BOX,
        np.asarray([0.035, 0.035, 0.03], dtype=np.float64),
        np.asarray([0.65, 0.75, 0.98, 0.92], dtype=np.float32),
    )


def _update_target_markers(
    handle: mujoco.viewer.Handle,
    orange_base_xyz: np.ndarray | None,
    aux_object_positions: dict[str, np.ndarray],
    current_segment: dict[str, Any] | None,
    *,
    include_object_markers: bool,
) -> None:
    handle.user_scn.ngeom = 0
    geom_count = 0
    if include_object_markers and orange_base_xyz is not None:
        mujoco.mjv_initGeom(
            handle.user_scn.geoms[geom_count],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.asarray([0.03, 0.03, 0.03], dtype=np.float64),
            np.asarray(orange_base_xyz, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.asarray([1.0, 0.55, 0.0, 0.95], dtype=np.float32),
        )
        geom_count += 1
    if include_object_markers:
        for object_key, object_base_xyz in sorted(aux_object_positions.items()):
            geom_type, geom_size, geom_color = _aux_marker_style(object_key)
            mujoco.mjv_initGeom(
                handle.user_scn.geoms[geom_count],
                geom_type,
                geom_size,
                np.asarray(object_base_xyz, dtype=np.float64),
                np.eye(3, dtype=np.float64).reshape(-1),
                geom_color,
            )
            geom_count += 1
    if orange_base_xyz is None:
        handle.user_scn.ngeom = geom_count
        return
    if current_segment is not None:
        badge_geom, badge_size, badge_color = _segment_badge_style(str(current_segment.get("invariance", "unknown")))
        badge_pos = np.asarray(orange_base_xyz, dtype=np.float64) + np.asarray([0.0, 0.0, 0.09], dtype=np.float64)
        mujoco.mjv_initGeom(
            handle.user_scn.geoms[geom_count],
            badge_geom,
            badge_size,
            badge_pos,
            np.eye(3, dtype=np.float64).reshape(-1),
            badge_color,
        )
        geom_count += 1
    handle.user_scn.ngeom = geom_count


def _resolve_repo_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (REPO_ROOT / path)


@dataclass(slots=True)
class StabilityConfig:
    window_size: int
    position_variance_threshold_mm2: float
    orientation_variance_threshold_deg2: float


def _load_stability_config(args: argparse.Namespace, vision_config_path: Path) -> StabilityConfig:
    payload = _load_yaml(vision_config_path)
    tracker_payload = dict(payload.get("tracker", {}))
    return StabilityConfig(
        window_size=max(1, int(args.stability_window_size or tracker_payload.get("stability_window_size", 5))),
        position_variance_threshold_mm2=float(
            args.position_variance_threshold_mm2
            if args.position_variance_threshold_mm2 is not None
            else tracker_payload.get("position_variance_threshold_mm2", 100.0)
        ),
        orientation_variance_threshold_deg2=float(
            args.orientation_variance_threshold_deg2
            if args.orientation_variance_threshold_deg2 is not None
            else tracker_payload.get("orientation_variance_threshold_deg2", 225.0)
        ),
    )


def _joint_positions_from_qpos(data: mujoco.MjData, joint_qpos: dict[str, int]) -> np.ndarray:
    return np.asarray(
        [data.qpos[joint_qpos[joint_name]] for joint_name in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")],
        dtype=float,
    )


def _offer_latest(queue_obj: mp.Queue, payload: dict[str, Any]) -> None:
    while True:
        try:
            queue_obj.put_nowait(payload)
            return
        except queue.Full:
            try:
                queue_obj.get_nowait()
            except queue.Empty:
                return


def _drain_latest_sample_from_queue(queue_obj: mp.Queue) -> tuple[int, dict[str, Any] | None]:
    drained_samples = 0
    latest_sample: dict[str, Any] | None = None
    while True:
        try:
            latest_sample = queue_obj.get_nowait()
            drained_samples += 1
        except queue.Empty:
            break
    return drained_samples, latest_sample


def _await_single_capture_sample(
    queue_obj: mp.Queue,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(float(timeout_s), 1e-3)
    last_error: dict[str, Any] | None = None
    while True:
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            break
        try:
            sample = queue_obj.get(timeout=min(0.5, remaining_s))
        except queue.Empty:
            continue
        if sample.get("status") == "ok" and sample.get("base_xyz_m") is not None:
            return sample
        last_error = sample
        if sample.get("status") == "error":
            logging.warning("Pose stream sample failed during single capture: %s", sample.get("error"))
    if last_error is not None and last_error.get("status") == "error":
        raise RuntimeError(f"Single capture failed: {last_error.get('error')}")
    raise TimeoutError(f"Timed out waiting for a valid single capture after {float(timeout_s):.2f} seconds.")


def _single_capture_stability(sample: dict[str, Any]) -> dict[str, Any]:
    base_xyz = np.asarray(sample["base_xyz_m"], dtype=np.float64).reshape(3)
    orientation_deg = sample.get("orientation_deg")
    return {
        "stable": True,
        "sample_count": 1,
        "required_sample_count": 1,
        "position_variance_mm2": 0.0,
        "position_std_mm": 0.0,
        "orientation_variance_deg2": 0.0 if orientation_deg is not None else None,
        "orientation_std_deg": 0.0 if orientation_deg is not None else None,
        "mean_position_m": base_xyz.astype(float).tolist(),
    }


def _stop_capture_process(stop_event: mp.Event, capture_process: mp.Process) -> None:
    stop_event.set()
    capture_process.join(timeout=5.0)
    if capture_process.is_alive():
        capture_process.terminate()
        capture_process.join(timeout=1.0)


def _single_frame_record() -> FrameRecord:
    return FrameRecord(
        frame_index=0,
        slot_index=0,
        wall_time_ns=0,
        relative_time_s=0.0,
        scheduled_time_s=0.0,
        capture_started_ns=0,
        capture_ended_ns=0,
        capture_latency_ms=0.0,
        camera_timestamp_ns=0,
        robot_timestamp_ns=0,
        time_skew_ms=0.0,
        color_path=Path("rgb/frame_000000.png"),
        depth_path=Path("depth/frame_000000.npy"),
    )


def _phrase_key(target_phrase: str) -> str:
    return "_".join(part for part in str(target_phrase).strip().lower().replace("-", " ").split() if part)


def _estimate_live_object_with_resources(
    *,
    temp_dir: Path,
    color_rgb: np.ndarray,
    depth_map: np.ndarray,
    segmentor: VLMObjectSegmentor,
    anchor_estimator: AnchorEstimator,
    orientation_estimator: OrientationMomentsEstimator,
    base_from_camera: np.ndarray,
    target_phrase: str,
) -> dict[str, Any]:
    session_dir = _write_single_frame_session(temp_dir / _phrase_key(target_phrase), color_rgb, depth_map)
    segmentor_result = segmentor.segment_video(
        session_dir=session_dir,
        records=[_single_frame_record()],
        frame_indices=[0],
        output_dir=temp_dir / f"vision_{_phrase_key(target_phrase)}",
        target_phrase=target_phrase,
        keep_artifacts=False,
    )
    mask = segmentor_result.masks[0][1]
    frame_result = segmentor_result.frame_results[0]
    orientation = orientation_estimator.estimate(mask)
    anchor = anchor_estimator.estimate(
        frame_index=0,
        mask=mask,
        depth_map=depth_map,
        orientation_deg=orientation.angle_deg,
        score=frame_result.score,
    )
    base_xyz_m = None
    if anchor.camera_xyz_m is not None:
        base_xyz_m = apply_transform_points(base_from_camera, np.asarray(anchor.camera_xyz_m, dtype=float))[0].tolist()
    status = "ok" if base_xyz_m is not None else "warn"
    return {
        "status": status,
        "target_phrase": target_phrase,
        "seed_detection": segmentor_result.seed_detection.to_dict(),
        "segmentation": frame_result.to_dict(),
        "anchor": anchor.to_dict(),
        "orientation": orientation.to_dict(),
        "base_xyz_m": base_xyz_m,
    }


def _capture_live_result_with_resources(
    *,
    camera: Any,
    segmentor: VLMObjectSegmentor,
    intrinsics: CameraIntrinsics,
    anchor_estimator: AnchorEstimator,
    orientation_estimator: OrientationMomentsEstimator,
    base_from_camera: np.ndarray,
    handeye_payload: dict[str, Any],
    handeye_result_path: Path,
    output_root: Path,
    target_phrase: str,
    aux_target_phrases: list[str],
    warmup_frames: int,
    camera_timeout_ms: int,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    color_rgb, depth_map = _capture_frame(camera, warmup_frames=warmup_frames, timeout_ms=camera_timeout_ms)
    with tempfile.TemporaryDirectory(prefix="vlbiman_live_stream_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        primary_result = _estimate_live_object_with_resources(
            temp_dir=temp_dir,
            color_rgb=color_rgb,
            depth_map=depth_map,
            segmentor=segmentor,
            anchor_estimator=anchor_estimator,
            orientation_estimator=orientation_estimator,
            base_from_camera=base_from_camera,
            target_phrase=target_phrase,
        )
        object_results = {
            _phrase_key(target_phrase): primary_result,
        }
        for aux_phrase in aux_target_phrases:
            aux_phrase = str(aux_phrase).strip()
            if not aux_phrase:
                continue
            try:
                object_results[_phrase_key(aux_phrase)] = _estimate_live_object_with_resources(
                    temp_dir=temp_dir,
                    color_rgb=color_rgb,
                    depth_map=depth_map,
                    segmentor=segmentor,
                    anchor_estimator=anchor_estimator,
                    orientation_estimator=orientation_estimator,
                    base_from_camera=base_from_camera,
                    target_phrase=aux_phrase,
                )
            except Exception as exc:
                logging.warning("Aux target detection failed for %s: %s", aux_phrase, exc)
                object_results[_phrase_key(aux_phrase)] = {
                    "status": "error",
                    "target_phrase": aux_phrase,
                    "error": repr(exc),
                    "base_xyz_m": None,
                }

    metrics = dict(handeye_payload.get("metrics", {})) if isinstance(handeye_payload.get("metrics"), dict) else {}
    result_path = output_root / f"{_timestamp_name('live_scene')}.json"
    result = {
        "status": primary_result["status"],
        "target_phrase": target_phrase,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "stream_mode": "live_video",
        "result_path": str(result_path),
        "artifacts_retained": False,
        "seed_detection": primary_result["seed_detection"],
        "segmentation": primary_result["segmentation"],
        "anchor": primary_result["anchor"],
        "orientation": primary_result["orientation"],
        "base_xyz_m": primary_result["base_xyz_m"],
        "objects": object_results,
        "handeye_status": {
            "path": str(handeye_result_path),
            "passed": handeye_payload.get("passed"),
            "accepted_without_passing_thresholds": handeye_payload.get("accepted_without_passing_thresholds"),
            "translation_mean_mm": metrics.get("translation_mean_mm"),
            "rotation_mean_deg": metrics.get("rotation_mean_deg"),
        },
    }
    _save_json(result_path, result)
    _save_json(output_root / "latest_result.json", result)
    return result_path


def _pose_stream_worker(
    *,
    output_queue: mp.Queue,
    stop_event: mp.Event,
    capture_config_path: str,
    vision_config_path: str,
    handeye_result_path: str,
    output_root: str,
    camera_serial_number: str | None,
    target_phrase: str,
    aux_target_phrases: list[str],
    warmup_frames: int,
    camera_timeout_ms: int,
    stream_interval_s: float,
    log_level: str,
) -> None:
    logging.basicConfig(level=getattr(logging, str(log_level).upper(), logging.INFO), force=True)
    capture_payload = _load_yaml(Path(capture_config_path))
    camera_cfg = dict(capture_payload.get("camera", {}))
    if camera_serial_number is not None:
        camera_cfg["serial_number_or_name"] = camera_serial_number

    vision_payload = _load_yaml(Path(vision_config_path))
    segmentor_cfg = VLMObjectSegmentorConfig(**dict(vision_payload.get("segmentor", {})))
    anchor_cfg = AnchorEstimatorConfig(**dict(vision_payload.get("anchor", {})))
    intrinsics = CameraIntrinsics.from_json(Path(vision_payload["intrinsics_path"]))
    segmentor = VLMObjectSegmentor(segmentor_cfg)
    orientation_estimator = OrientationMomentsEstimator()
    anchor_estimator = AnchorEstimator(intrinsics, anchor_cfg)
    base_from_camera, handeye_payload = _load_base_from_camera(Path(handeye_result_path))
    camera = _build_camera(camera_cfg)
    interval_s = max(float(stream_interval_s), 1e-3)
    consecutive_timeout_failures = 0
    try:
        camera.connect()
        # Keep camera open and warm models once so per-frame latency stays low.
        segmentor._get_predictor()
        segmentor._get_florence_processor_model()
        while not stop_event.is_set():
            started_at = time.monotonic()
            try:
                live_result_path = _capture_live_result_with_resources(
                    camera=camera,
                    segmentor=segmentor,
                    intrinsics=intrinsics,
                    anchor_estimator=anchor_estimator,
                    orientation_estimator=orientation_estimator,
                    base_from_camera=base_from_camera,
                    handeye_payload=handeye_payload,
                    handeye_result_path=Path(handeye_result_path),
                    output_root=Path(output_root),
                    target_phrase=target_phrase,
                    aux_target_phrases=list(aux_target_phrases),
                    warmup_frames=int(warmup_frames),
                    camera_timeout_ms=int(camera_timeout_ms),
                )
                live_result = _load_json(live_result_path)
                orientation = live_result.get("orientation") if isinstance(live_result.get("orientation"), dict) else {}
                objects = live_result.get("objects") if isinstance(live_result.get("objects"), dict) else {}
                sample = {
                    "status": str(live_result.get("status", "warn")),
                    "captured_at_utc": str(live_result.get("captured_at_utc", datetime.now(timezone.utc).isoformat())),
                    "live_result_path": str(live_result_path),
                    "base_xyz_m": live_result.get("base_xyz_m"),
                    "orientation_deg": orientation.get("angle_deg"),
                    "object_positions_m": {
                        str(name): payload.get("base_xyz_m")
                        for name, payload in objects.items()
                        if isinstance(payload, dict)
                    },
                }
                _offer_latest(output_queue, sample)
                consecutive_timeout_failures = 0
            except Exception as exc:
                logging.exception("Live pose stream capture failed.")
                if isinstance(exc, TimeoutError):
                    consecutive_timeout_failures += 1
                else:
                    consecutive_timeout_failures = 0
                _offer_latest(
                    output_queue,
                    {
                        "status": "error",
                        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                        "error": repr(exc),
                    },
                )
                if consecutive_timeout_failures >= 3:
                    logging.warning(
                        "Pose stream saw %d consecutive camera timeouts. Reconnecting camera.",
                        consecutive_timeout_failures,
                    )
                    try:
                        camera.disconnect()
                    except Exception:
                        logging.exception("Camera disconnect failed during timeout recovery.")
                    try:
                        camera.connect()
                        consecutive_timeout_failures = 0
                        logging.info("Camera reconnected after timeout recovery.")
                    except Exception:
                        logging.exception("Camera reconnect failed during timeout recovery.")
            remaining_s = interval_s - (time.monotonic() - started_at)
            if remaining_s > 0.0:
                stop_event.wait(remaining_s)
    finally:
        try:
            camera.disconnect()
        except Exception:
            logging.exception("Camera disconnect raised an exception in pose stream worker.")


def _axial_variance_deg2(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    radians = np.radians(np.asarray(values, dtype=np.float64))
    doubled = 2.0 * radians
    mean_sin = float(np.mean(np.sin(doubled)))
    mean_cos = float(np.mean(np.cos(doubled)))
    if np.hypot(mean_sin, mean_cos) <= 1e-9:
        return 8100.0, 90.0
    mean_angle_deg = float(np.degrees(np.arctan2(mean_sin, mean_cos) / 2.0))
    diffs_deg = np.asarray([((value - mean_angle_deg + 90.0) % 180.0) - 90.0 for value in values], dtype=np.float64)
    variance_deg2 = float(np.mean(np.square(diffs_deg)))
    return variance_deg2, float(np.sqrt(max(variance_deg2, 0.0)))


def _stability_metrics(
    samples: deque[dict[str, Any]],
    config: StabilityConfig,
) -> dict[str, Any]:
    if len(samples) < config.window_size:
        return {
            "stable": False,
            "sample_count": len(samples),
            "required_sample_count": config.window_size,
            "position_variance_mm2": None,
            "position_std_mm": None,
            "orientation_variance_deg2": None,
            "orientation_std_deg": None,
            "mean_position_m": None,
        }
    window = list(samples)[-config.window_size :]
    positions = []
    orientations = []
    for sample in window:
        base_xyz = sample.get("base_xyz_m")
        if base_xyz is None:
            return {
                "stable": False,
                "sample_count": len(window),
                "required_sample_count": config.window_size,
                "position_variance_mm2": None,
                "position_std_mm": None,
                "orientation_variance_deg2": None,
                "orientation_std_deg": None,
                "mean_position_m": None,
            }
        positions.append(np.asarray(base_xyz, dtype=np.float64).reshape(3))
        orientation_deg = sample.get("orientation_deg")
        if orientation_deg is not None:
            orientations.append(float(orientation_deg))
    stacked_positions = np.stack(positions, axis=0)
    mean_position = np.mean(stacked_positions, axis=0)
    position_variance_mm2 = float(np.max(np.var(stacked_positions, axis=0)) * 1_000_000.0)
    position_std_mm = float(np.sqrt(max(position_variance_mm2, 0.0)))
    orientation_variance_deg2, orientation_std_deg = _axial_variance_deg2(orientations)
    stable = position_variance_mm2 <= float(config.position_variance_threshold_mm2)
    if orientation_variance_deg2 is not None:
        stable = stable and orientation_variance_deg2 <= float(config.orientation_variance_threshold_deg2)
    return {
        "stable": bool(stable),
        "sample_count": len(window),
        "required_sample_count": config.window_size,
        "position_variance_mm2": position_variance_mm2,
        "position_std_mm": position_std_mm,
        "orientation_variance_deg2": orientation_variance_deg2,
        "orientation_std_deg": orientation_std_deg,
        "mean_position_m": mean_position.astype(float).tolist(),
    }


def _build_continuous_points(
    current_joint_positions: np.ndarray | None,
    new_points: list[dict[str, Any]],
    *,
    bridge_max_joint_step_rad: float,
) -> list[dict[str, Any]]:
    if current_joint_positions is None or not new_points:
        return list(new_points)
    first_target = np.asarray(new_points[0]["joint_positions"], dtype=float)
    delta = first_target - np.asarray(current_joint_positions, dtype=float)
    max_delta = float(np.max(np.abs(delta)))
    if max_delta <= 1e-6:
        return list(new_points)

    step_limit = max(float(bridge_max_joint_step_rad), 1e-3)
    bridge_steps = max(1, int(np.ceil(max_delta / step_limit)))
    template = dict(new_points[0])
    bridged_points: list[dict[str, Any]] = []
    for step_idx in range(bridge_steps):
        alpha = float(step_idx + 1) / float(bridge_steps + 1)
        point = dict(template)
        point["joint_positions"] = (
            (1.0 - alpha) * np.asarray(current_joint_positions, dtype=float) + alpha * first_target
        ).astype(float).tolist()
        point["segment_id"] = "bridge_from_current"
        point["source"] = "replan_bridge"
        point["invariance"] = "bridge"
        point["translation_error_mm"] = 0.0
        point["rotation_error_deg"] = 0.0
        point["max_joint_step_rad"] = max_delta / float(bridge_steps + 1)
        bridged_points.append(point)
    bridged_points.extend(new_points)
    return bridged_points


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
    planned_points: list[dict[str, Any]],
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
    if planned_points:
        first_point = planned_points[0]
        source = str(first_point.get("source", "")).strip().lower()
        segment_id = str(first_point.get("segment_id", "")).strip().lower()
        for token in ("continuity", "current_state", "start_state", "state_seed"):
            if token in source:
                return True, f"first_point_source:{source}"
            if token in segment_id:
                return True, f"first_point_segment:{segment_id}"
    return False, "none"


def _build_execution_points(
    current_joint_positions: np.ndarray | None,
    planned_points: list[dict[str, Any]],
    *,
    bridge_max_joint_step_rad: float,
    trajectory_summary: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if current_joint_positions is None or not planned_points:
        return list(planned_points), {
            "bridge_injected": False,
            "bridge_point_count": 0,
            "bridge_reason": "missing_state_or_points",
            "bridge_fallback_triggered": False,
            "continuity_declared_by_t6": False,
            "continuity_signal": "none",
            "initial_joint_gap_rad_inf": None,
            "bridge_step_limit_rad": float(max(float(bridge_max_joint_step_rad), 1e-3)),
        }

    first_target = np.asarray(planned_points[0]["joint_positions"], dtype=float)
    current_q = np.asarray(current_joint_positions, dtype=float)
    max_delta = float(np.max(np.abs(first_target - current_q)))
    step_limit = float(max(float(bridge_max_joint_step_rad), 1e-3))
    continuity_declared, continuity_signal = _continuity_signal(trajectory_summary, planned_points)

    if max_delta <= 1e-6:
        return list(planned_points), {
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
        return list(planned_points), {
            "bridge_injected": False,
            "bridge_point_count": 0,
            "bridge_reason": "continuity_declared_safe_gap",
            "bridge_fallback_triggered": False,
            "continuity_declared_by_t6": True,
            "continuity_signal": continuity_signal,
            "initial_joint_gap_rad_inf": max_delta,
            "bridge_step_limit_rad": step_limit,
        }

    bridged_points = _build_continuous_points(
        current_joint_positions=current_q,
        new_points=planned_points,
        bridge_max_joint_step_rad=step_limit,
    )
    fallback_triggered = continuity_declared and max_delta > step_limit
    bridge_reason = (
        "continuity_declared_gap_too_large_fallback_bridge"
        if fallback_triggered
        else "continuity_not_declared_bridge"
    )
    return bridged_points, {
        "bridge_injected": True,
        "bridge_point_count": max(0, len(bridged_points) - len(planned_points)),
        "bridge_reason": bridge_reason,
        "bridge_fallback_triggered": fallback_triggered,
        "continuity_declared_by_t6": continuity_declared,
        "continuity_signal": continuity_signal,
        "initial_joint_gap_rad_inf": max_delta,
        "bridge_step_limit_rad": step_limit,
    }


def _write_summary(
    *,
    cycle_root: Path,
    latest_cycle_summary: dict[str, Any] | None,
    current_target_base_xyz: np.ndarray | None,
    current_aux_object_positions: dict[str, np.ndarray],
    replan_distance_threshold_m: float,
    stream_interval_s: float,
    capture_cycle_count: int,
    replan_count: int,
    stability: dict[str, Any],
    stable_target_base_xyz: np.ndarray | None,
    stream_status: str,
    status: str,
) -> dict[str, Any]:
    payload = {
        "status": status,
        "cycle_root": str(cycle_root),
        "latest_cycle": latest_cycle_summary,
        "current_target_base_xyz_m": (
            current_target_base_xyz.astype(float).tolist() if current_target_base_xyz is not None else None
        ),
        "current_aux_object_positions_m": {
            str(key): value.astype(float).tolist()
            for key, value in current_aux_object_positions.items()
        },
        "stable_target_base_xyz_m": (
            stable_target_base_xyz.astype(float).tolist() if stable_target_base_xyz is not None else None
        ),
        "replan_distance_threshold_m": float(replan_distance_threshold_m),
        "stream_interval_s": float(stream_interval_s),
        "capture_cycle_count": int(capture_cycle_count),
        "replan_count": int(replan_count),
        "stability": stability,
        "stream_status": stream_status,
    }
    _save_json(cycle_root / "summary.json", payload)
    return payload


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)
    os.environ["DISPLAY"] = args.display

    task_config = _load_task_config(args.task_config)
    task_config.handeye_result_path = _resolve_repo_path(task_config.handeye_result_path)
    task_config.recording_session_dir = _resolve_repo_path(task_config.recording_session_dir)
    task_config.intrinsics_path = _resolve_repo_path(task_config.intrinsics_path)
    task_config.transforms_path = _resolve_repo_path(task_config.transforms_path)
    task_config.skill_bank_path = _resolve_repo_path(task_config.skill_bank_path)
    segment_display_labels = _load_segment_display_labels(task_config.skill_bank_path)
    task_config.target_phrase = args.target_phrase or task_config.target_phrase
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
    capture_config = _load_yaml(args.capture_config)
    stability_config = _load_stability_config(args, args.vision_config)

    logging.warning(
        "Validation mode only updates MuJoCo. It does not send actions to the physical robot. "
        "Use run_visual_closed_loop_on_robot.py for hardware motion."
    )
    logging.info(
        "Closed-loop planning targets: primary=%s secondary=%s aux=%s",
        planning_primary_target,
        planning_secondary_target or "<none>",
        aux_target_phrases,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    cycle_root = args.output_root / _timestamp_name("visual_loop")
    cycle_root.mkdir(parents=True, exist_ok=True)

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

    aux_object_keys = [_phrase_key(phrase) for phrase in aux_target_phrases]
    model, include_primary_target_marker = _build_validation_model(
        base_mjcf_path=args.model_path,
        urdf_path=args.urdf_path,
        handeye_result_path=args.handeye_result,
        aux_object_keys=aux_object_keys,
        target_radius_m=float(args.target_radius_m),
        target_mass_kg=float(args.target_mass_kg),
        scene_preset=str(args.scene_preset),
        legacy_scene=bool(args.legacy_scene),
    )
    data = mujoco.MjData(model)
    joint_qpos = _joint_qpos_indices(model)
    target_qpos_addr, target_qvel_addr = _target_freejoint_indices(model)
    aux_object_freejoints = {
        object_key: _object_freejoint_indices(model, object_key) for object_key in aux_object_keys
    }
    reset_joint_positions = _joint_positions_from_qpos(data, joint_qpos)
    keyboard_state = {"exit_requested": False, "reset_requested": False}

    def _key_callback(keycode: int) -> None:
        if keycode == 256:
            keyboard_state["exit_requested"] = True
        elif keycode == 32:
            keyboard_state["reset_requested"] = True

    require_mujoco_viewer_backend()
    viewer = mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=_key_callback,
        show_left_ui=bool(args.show_left_ui),
        show_right_ui=bool(args.show_right_ui),
    )
    _apply_camera(viewer)

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
    single_capture_mode = str(args.capture_mode) == "single_shot"

    logging.info("MuJoCo segment badge: blue box=var, orange capsule=inv, gray cylinder=bridge.")

    try:
        if single_capture_mode:
            initial_sample = _await_single_capture_sample(
                sample_queue,
                timeout_s=float(args.single_capture_timeout_s),
            )
            stream_status = str(initial_sample.get("status", "ok"))
            current_target_base_xyz = np.asarray(initial_sample["base_xyz_m"], dtype=float).reshape(3)
            object_positions = (
                initial_sample.get("object_positions_m")
                if isinstance(initial_sample.get("object_positions_m"), dict)
                else {}
            )
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
        while viewer.is_running():
            now = time.monotonic()
            if keyboard_state["exit_requested"]:
                logging.info("Stopping visual closed loop on ESC.")
                break
            if float(args.max_runtime_s) > 0.0 and now - loop_started_at >= float(args.max_runtime_s):
                logging.info("Stopping visual closed loop after %.2f seconds.", now - loop_started_at)
                break
            if keyboard_state["reset_requested"]:
                current_points = []
                current_index = 0
                last_replan_base_xyz = None
                last_segment_key = None
                _set_robot_qpos(data, joint_qpos, reset_joint_positions)
                current_joint_positions = reset_joint_positions.copy()
                keyboard_state["reset_requested"] = False
                logging.info("Reset robot to home pose on SPACE.")
            drained_samples = 0
            latest_sample: dict[str, Any] | None = None
            if not single_capture_mode:
                drained_samples, latest_sample = _drain_latest_sample_from_queue(sample_queue)
            if latest_sample is not None:
                stream_status = str(latest_sample.get("status", "warn"))
                if latest_sample.get("status") == "ok" and latest_sample.get("base_xyz_m") is not None:
                    current_target_base_xyz = np.asarray(latest_sample["base_xyz_m"], dtype=float).reshape(3)
                    object_positions = (
                        latest_sample.get("object_positions_m")
                        if isinstance(latest_sample.get("object_positions_m"), dict)
                        else {}
                    )
                    current_aux_object_positions = {
                        str(name): np.asarray(position, dtype=float).reshape(3)
                        for name, position in object_positions.items()
                        if position is not None and str(name) != _phrase_key(task_config.target_phrase)
                    }
                    latest_live_result_path = Path(str(latest_sample["live_result_path"]))
                    pose_window.append(latest_sample)
                    stability = _stability_metrics(pose_window, stability_config)
                    stable_target = stability.get("mean_position_m")
                    stable_target_base_xyz = (
                        np.asarray(stable_target, dtype=float).reshape(3) if stable_target is not None else None
                    )
                    capture_cycle_count += drained_samples
                elif latest_sample.get("status") == "error":
                    logging.warning("Pose stream sample failed: %s", latest_sample.get("error"))

            if (
                latest_live_result_path is not None
                and stability.get("stable")
                and stable_target_base_xyz is not None
            ):
                should_replan = (
                    last_replan_base_xyz is None
                    or np.linalg.norm(stable_target_base_xyz - last_replan_base_xyz)
                    >= float(args.replan_distance_threshold_m)
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
                            current_joint_positions = _joint_positions_from_qpos(data, joint_qpos)
                            current_points, bridge_decision = _build_execution_points(
                                current_joint_positions=current_joint_positions,
                                planned_points=planned_points,
                                bridge_max_joint_step_rad=float(args.bridge_max_joint_step_rad),
                                trajectory_summary=trajectory_summary,
                            )
                            current_index = 0
                            last_replan_base_xyz = stable_target_base_xyz.copy()
                            replan_count += 1
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
                                "Replanned trajectory for %s at %s with %s points (%s bridge, reason=%s).",
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
                            logging.warning("Closed-loop replan produced no trajectory points.")
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
                        logging.exception("Closed-loop replan failed.")

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
                logging.info("Stopping visual closed loop after %s capture cycles.", capture_cycle_count)
                break

            active_segment_index = current_index
            if current_points and now - last_step_time >= max(float(args.step_duration_s), 1e-3):
                joint_positions = np.asarray(current_points[current_index]["joint_positions"], dtype=float)
                _set_robot_qpos(data, joint_qpos, joint_positions)
                current_joint_positions = joint_positions
                if current_index < len(current_points) - 1:
                    current_index += 1
                last_step_time = now

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
                        "Active trajectory segment: %s label=%s invariance=%s source=%s",
                        current_segment.get("segment_id", "unknown_segment"),
                        current_segment.get("segment_label", ""),
                        current_segment.get("invariance", "unknown"),
                        current_segment.get("source", "unknown"),
                    )
                    last_segment_key = segment_key

            _set_target_freejoint_pose(
                data,
                qpos_addr=target_qpos_addr,
                qvel_addr=target_qvel_addr,
                position_xyz_m=current_target_base_xyz,
            )
            for object_key, object_base_xyz in current_aux_object_positions.items():
                qpos_addr, qvel_addr = aux_object_freejoints.get(object_key, (None, None))
                _set_target_freejoint_pose(
                    data,
                    qpos_addr=qpos_addr,
                    qvel_addr=qvel_addr,
                    position_xyz_m=object_base_xyz,
                )
            mujoco.mj_forward(model, data)
            _update_target_markers(
                viewer,
                current_target_base_xyz,
                current_aux_object_positions,
                current_segment,
                include_object_markers=bool(include_primary_target_marker),
            )
            viewer.set_texts(
                _build_stage_overlay_texts(
                    current_segment,
                    stage_labels=segment_display_labels,
                    point_index=active_segment_index,
                    point_count=len(current_points),
                )
            )
            viewer.sync()
            time.sleep(0.005)
    finally:
        _stop_capture_process(stop_event, capture_process)
        try:
            viewer.clear_texts()
        except Exception:
            logging.exception("Failed to clear viewer overlay texts.")
        viewer.close()

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
