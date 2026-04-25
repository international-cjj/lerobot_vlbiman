#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
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
    for path in (repo_root / "src", repo_root, repo_root / "lerobot_robot_cjjarm"):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo_root


_maybe_reexec_in_repo_venv()
REPO_ROOT = _bootstrap_paths()

from lerobot.projects.vlbiman_sa.runtime_env import prepare_mujoco_runtime, require_mujoco_viewer_backend

prepare_mujoco_runtime(argv=sys.argv[1:])

import mujoco
import mujoco.viewer

from lerobot.projects.vlbiman_sa.demo.io import load_frame_records
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
)


def _default_trajectory_path() -> Path:
    return (
        REPO_ROOT
        / "outputs"
        / "vlbiman_sa"
        / "live_grasp_preview"
        / "preview_20260326T142935Z"
        / "analysis"
        / "t6_trajectory"
        / "trajectory_points.json"
    )


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
    parser = argparse.ArgumentParser(description="Open MuJoCo viewer and loop a saved T6 trajectory.")
    parser.add_argument("--trajectory-points", type=Path, default=_default_trajectory_path())
    parser.add_argument(
        "--demo-session-dir",
        type=Path,
        default=None,
        help="Replay the recorded demo joint sequence instead of a generated T6 trajectory.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=_default_model_path(),
        help="Base MJCF when using the unified dual-camera scene, or the raw MuJoCo XML path in --legacy-scene mode.",
    )
    parser.add_argument("--urdf-path", type=Path, default=_default_urdf_path())
    parser.add_argument("--handeye-result", type=Path, default=_default_handeye_path())
    parser.add_argument(
        "--orange-base-xyz",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional orange position in base frame. If omitted, infer from paired T5/live outputs.",
    )
    parser.add_argument(
        "--pink-cup-base-xyz",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional pink cup position in base frame. If omitted, infer from paired T5/live outputs.",
    )
    parser.add_argument(
        "--target-base-xyz",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Legacy alias for --orange-base-xyz.",
    )
    parser.add_argument("--display", type=str, default=None, help="Override DISPLAY, e.g. :1")
    parser.add_argument("--target-phrase", type=str, default="orange", help="Primary object phrase for execution context.")
    parser.add_argument(
        "--aux-target-phrase",
        action="append",
        default=[],
        help="Auxiliary object phrase for execution context; can be repeated.",
    )
    parser.add_argument("--step-duration-s", type=float, default=0.04)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--loop", action="store_true", default=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--target-radius-m", type=float, default=0.022)
    parser.add_argument("--target-mass-kg", type=float, default=0.12)
    parser.add_argument(
        "--scene-preset",
        type=str,
        choices=scene_preset_names(),
        default=DEFAULT_SCENE_PRESET_NAME,
        help="Optional physical-object preset to append to the canonical scene.",
    )
    parser.add_argument(
        "--legacy-scene",
        action="store_true",
        help="Use the legacy arm-only MuJoCo scene and render the orange as a non-colliding marker.",
    )
    parser.add_argument("--show-left-ui", action="store_true")
    parser.add_argument("--show-right-ui", action="store_true")
    parser.add_argument(
        "--preview-diagnostics-json",
        type=Path,
        default=None,
        help="Optional output path for machine-readable preview diagnostics JSON.",
    )
    parser.add_argument(
        "--skip-preview-diagnostics-write",
        action="store_true",
        help="Do not write preview diagnostics JSON sidecar.",
    )
    parser.add_argument(
        "--preview-diagnostics-only",
        action="store_true",
        help="Extract and print diagnostics without opening the MuJoCo viewer.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _apply_camera(handle: mujoco.viewer.Handle) -> None:
    handle.cam.azimuth = 148.0
    handle.cam.elevation = -20.0
    handle.cam.distance = 1.75
    handle.cam.lookat[:] = np.asarray([0.0, 0.0, 0.24], dtype=float)


def _infer_object_base_positions(trajectory_points_path: Path) -> dict[str, np.ndarray]:
    analysis_dir = trajectory_points_path.parent.parent
    t5_summary_path = analysis_dir / "t5_pose" / "summary.json"
    object_positions: dict[str, np.ndarray] = {}

    if t5_summary_path.exists():
        payload = _load_json(t5_summary_path)
        for object_key, meta in dict(payload.get("object_meta", {})).items():
            base_xyz = meta.get("live_base_xyz_m")
            if base_xyz is None:
                continue
            object_positions[str(object_key)] = np.asarray(base_xyz, dtype=float).reshape(3)
        if object_positions:
            return object_positions

    run_dir = trajectory_points_path.parents[2]
    summary_candidates = [
        run_dir / "preview" / "summary.json",
        run_dir / "summary.json",
    ]
    live_result_candidates = [
        run_dir / "image_pose" / "result.json",
    ]

    for summary_path in summary_candidates:
        if not summary_path.exists():
            continue
        payload = _load_json(summary_path)
        live_result_path = payload.get("live_result_path")
        if live_result_path:
            live_result_candidates.insert(0, _resolve_path(live_result_path))

    for live_result_path in live_result_candidates:
        if not live_result_path.exists():
            continue
        payload = _load_json(live_result_path)
        objects = payload.get("objects")
        if isinstance(objects, dict):
            for object_key, object_payload in objects.items():
                if not isinstance(object_payload, dict):
                    continue
                base_xyz = object_payload.get("base_xyz_m")
                if base_xyz is None:
                    continue
                object_positions[str(object_key)] = np.asarray(base_xyz, dtype=float).reshape(3)
        base_xyz = payload.get("base_xyz_m")
        if base_xyz is not None and "orange" not in object_positions:
            object_positions["orange"] = np.asarray(base_xyz, dtype=float).reshape(3)
        if object_positions:
            return object_positions
    return object_positions


def _infer_demo_object_positions(session_dir: Path) -> dict[str, np.ndarray]:
    summary_path = _resolve_path(session_dir) / "analysis" / "t5_pose" / "summary.json"
    object_positions: dict[str, np.ndarray] = {}
    if not summary_path.exists():
        return object_positions
    payload = _load_json(summary_path)
    for object_key, object_payload in dict(payload.get("object_adaptations", {})).items():
        if not isinstance(object_payload, dict):
            continue
        demo_anchor = object_payload.get("demo_anchor_base_xyz_m")
        if demo_anchor is None:
            continue
        object_positions[str(object_key)] = np.asarray(demo_anchor, dtype=float).reshape(3)
    return object_positions


def _aux_marker_style(object_key: str) -> tuple[mujoco.mjtGeom, np.ndarray, np.ndarray]:
    normalized = str(object_key).strip().lower().replace(" ", "_")
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


def _object_key_for_phrase(phrase: str) -> str:
    normalized = str(phrase).strip().lower().replace(" ", "_")
    if normalized in {"yellow", "yellow_ball", "ball", "orange"}:
        return normalized if normalized == "orange" else "yellow_ball"
    if normalized in {"pink_cup", "cup"}:
        return "pink_cup"
    if normalized in {"green", "green_plate", "plate"}:
        return "green_plate"
    return normalized or "object"


def _update_target_markers(
    handle: mujoco.viewer.Handle,
    object_positions: dict[str, np.ndarray],
    *,
    include_markers: bool,
) -> None:
    handle.user_scn.ngeom = 0
    if not include_markers:
        return
    geom_count = 0
    orange_base_xyz = object_positions.get("orange")
    if orange_base_xyz is not None:
        mujoco.mjv_initGeom(
            handle.user_scn.geoms[geom_count],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.asarray([0.03, 0.03, 0.03], dtype=np.float64),
            np.asarray(orange_base_xyz, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.asarray([1.0, 0.55, 0.0, 0.95], dtype=np.float32),
        )
        geom_count += 1
    for object_key in sorted(object_positions):
        if object_key == "orange":
            continue
        geom_type, geom_size, geom_color = _aux_marker_style(object_key)
        mujoco.mjv_initGeom(
            handle.user_scn.geoms[geom_count],
            geom_type,
            geom_size,
            np.asarray(object_positions[object_key], dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            geom_color,
        )
        geom_count += 1
    handle.user_scn.ngeom = geom_count


def _joint_qpos_indices(model: mujoco.MjModel) -> dict[str, int]:
    joint_qpos: dict[str, int] = {}
    for joint_name in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint not found in MuJoCo model: {joint_name}")
        joint_qpos[joint_name] = int(model.jnt_qposadr[joint_id])
    return joint_qpos


def _set_robot_qpos(
    data: mujoco.MjData,
    joint_qpos: dict[str, int],
    joint_positions: np.ndarray,
) -> None:
    for joint_idx, joint_name in enumerate(("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")):
        data.qpos[joint_qpos[joint_name]] = float(joint_positions[joint_idx])


def _physical_object_config(object_key: str, position_xyz: np.ndarray) -> ScenePrimitiveObjectConfig:
    normalized = str(object_key).strip().lower().replace(" ", "_")
    position = tuple(float(value) for value in np.asarray(position_xyz, dtype=float).reshape(3))
    if normalized == "pink_cup":
        return ScenePrimitiveObjectConfig(
            object_key=object_key,
            shape="cylinder",
            position_xyz_m=position,
            size_xyz_m=(0.035, 0.05, 0.0),
            mass_kg=0.18,
            rgba=(1.0, 0.36, 0.7, 1.0),
        )
    return ScenePrimitiveObjectConfig(
        object_key=object_key,
        shape="box",
        position_xyz_m=position,
        size_xyz_m=(0.035, 0.035, 0.03),
        mass_kg=0.12,
        rgba=(0.65, 0.75, 0.98, 1.0),
    )


def _build_sim_model(
    *,
    base_mjcf_path: Path,
    urdf_path: Path,
    handeye_result_path: Path,
    object_positions: dict[str, np.ndarray],
    orange_base_xyz: np.ndarray | None,
    target_phrase: str,
    target_radius_m: float,
    target_mass_kg: float,
    scene_preset: str,
    legacy_scene: bool,
) -> tuple[mujoco.MjModel, bool]:
    if legacy_scene:
        return mujoco.MjModel.from_xml_path(str(base_mjcf_path)), True
    primary_object_key = _object_key_for_phrase(target_phrase)
    primary_base_xyz = object_positions.get(primary_object_key)
    if primary_base_xyz is None:
        primary_base_xyz = orange_base_xyz
    target_position = (
        tuple(float(value) for value in np.asarray(primary_base_xyz, dtype=float).reshape(3))
        if primary_base_xyz is not None
        else (0.45, 0.0, 0.08)
    )
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
                    _physical_object_config(object_key, object_xyz)
                    for object_key, object_xyz in sorted(object_positions.items())
                    if str(object_key) not in {"orange", primary_object_key}
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


def _load_demo_points(session_dir: Path) -> list[dict[str, Any]]:
    records = load_frame_records(_resolve_path(session_dir))
    joint_keys = [f"joint_{index}.pos" for index in range(1, 7)]
    points: list[dict[str, Any]] = []
    for record in records:
        joint_positions = [float(record.joint_positions[key]) for key in joint_keys]
        points.append(
            {
                "joint_positions": joint_positions,
                "relative_time_s": float(record.relative_time_s),
                "segment_id": "demo_sequence",
                "segment_label": "demo_sequence",
                "invariance": "demo",
                "source": "demo_record",
            }
        )
    return points


def _default_preview_diagnostics_path(trajectory_points_path: Path) -> Path:
    return trajectory_points_path.parent / "preview_diagnostics.json"


def _mode_category(planning_mode: str) -> str:
    normalized = str(planning_mode).strip().lower()
    if "framewise" in normalized and "retarget" in normalized:
        return "framewise_retarget"
    if normalized == "hold_arm_gripper_only":
        return "hold_arm"
    if "bridge" in normalized:
        return "bridge"
    if normalized == "return_to_home_blend":
        return "return_home"
    if normalized == "demo_joint_replay":
        return "demo_replay"
    return "other"


def _build_preview_diagnostics(trajectory_points_path: Path, trajectory_payload: dict[str, Any]) -> dict[str, Any]:
    points = [point for point in list(trajectory_payload.get("points", [])) if isinstance(point, dict)]
    summary = trajectory_payload.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}

    raw_segment_modes = summary.get("segment_planning_modes", {})
    segment_planning_modes = (
        {str(segment_id): str(mode) for segment_id, mode in raw_segment_modes.items()}
        if isinstance(raw_segment_modes, dict)
        else {}
    )
    segment_warp_mode = {segment_id: _mode_category(mode) for segment_id, mode in segment_planning_modes.items()}

    raw_fusion_diag = summary.get("segment_fusion_diagnostics", {})
    segment_fusion_diagnostics = raw_fusion_diag if isinstance(raw_fusion_diag, dict) else {}

    source_counts = Counter(str(point.get("source") or "unknown") for point in points)
    bridge_point_count = sum(count for source, count in source_counts.items() if "bridge" in source.lower())
    first_motion_source = None
    for point in points:
        source = point.get("source")
        if source is not None:
            first_motion_source = str(source)
            break

    mode_counts = Counter(segment_warp_mode.values())
    framewise_segments = sorted(
        [segment_id for segment_id, mode in segment_planning_modes.items() if "framewise" in mode.lower()]
    )
    hold_arm_segments = sorted(
        [segment_id for segment_id, mode in segment_planning_modes.items() if mode == "hold_arm_gripper_only"]
    )

    return {
        "trajectory_points_path": str(trajectory_points_path),
        "status": summary.get("status"),
        "point_count": int(len(points)),
        "ik_success_rate": summary.get("ik_success_rate"),
        "segment_planning_modes": segment_planning_modes,
        "segment_warp_mode": segment_warp_mode,
        "segment_mode_counts": dict(mode_counts),
        "framewise_retarget_segments": framewise_segments,
        "framewise_retarget_count": int(len(framewise_segments)),
        "hold_arm_segments": hold_arm_segments,
        "bridge_injected": bool(bridge_point_count > 0),
        "bridge_point_count": int(bridge_point_count),
        "first_motion_source": first_motion_source,
        "source_counts": dict(source_counts),
        "segment_fusion_diagnostics": segment_fusion_diagnostics,
    }


def _print_preview_diagnostics(payload: dict[str, Any]) -> None:
    framewise_segments = payload.get("framewise_retarget_segments", [])
    hold_arm_segments = payload.get("hold_arm_segments", [])
    print("[preview] trajectory_status:", payload.get("status"))
    print("[preview] ik_success_rate:", payload.get("ik_success_rate"))
    print("[preview] framewise_retarget_segments:", ",".join(framewise_segments) if framewise_segments else "none")
    print("[preview] hold_arm_segments:", ",".join(hold_arm_segments) if hold_arm_segments else "none")
    print("[preview] bridge_injected:", payload.get("bridge_injected"))
    print("[preview] bridge_point_count:", payload.get("bridge_point_count"))
    print("[preview] first_motion_source:", payload.get("first_motion_source"))
    print("[preview] segment_mode_counts:", json.dumps(payload.get("segment_mode_counts", {}), ensure_ascii=False))


def main() -> int:
    args = _parse_args()
    if args.display:
        os.environ["DISPLAY"] = args.display

    preview_diagnostics_payload: dict[str, Any] | None = None
    if args.demo_session_dir is not None:
        points = _load_demo_points(args.demo_session_dir)[:: max(1, int(args.frame_stride))]
        object_positions = _infer_demo_object_positions(args.demo_session_dir)
        if args.preview_diagnostics_only:
            raise ValueError("--preview-diagnostics-only requires trajectory mode (omit --demo-session-dir).")
    else:
        trajectory_payload = _load_json(args.trajectory_points)
        points = trajectory_payload["points"][:: max(1, int(args.frame_stride))]
        object_positions = _infer_object_base_positions(args.trajectory_points)
        preview_diagnostics_payload = _build_preview_diagnostics(args.trajectory_points, trajectory_payload)
        _print_preview_diagnostics(preview_diagnostics_payload)
        if not bool(args.skip_preview_diagnostics_write):
            diagnostics_path = (
                _resolve_path(args.preview_diagnostics_json)
                if args.preview_diagnostics_json is not None
                else _default_preview_diagnostics_path(args.trajectory_points)
            )
            _save_json(diagnostics_path, preview_diagnostics_payload)
            print(f"[preview] diagnostics_json: {diagnostics_path}")
        if args.preview_diagnostics_only:
            return 0

    require_mujoco_viewer_backend()
    if not points:
        if args.demo_session_dir is not None:
            raise ValueError(f"No demo points found in {args.demo_session_dir}")
        raise ValueError(f"No trajectory points found in {args.trajectory_points}")
    if args.target_base_xyz is not None:
        object_positions["orange"] = np.asarray(args.target_base_xyz, dtype=float).reshape(3)
    if args.orange_base_xyz is not None:
        object_positions["orange"] = np.asarray(args.orange_base_xyz, dtype=float).reshape(3)
    if args.pink_cup_base_xyz is not None:
        object_positions["pink_cup"] = np.asarray(args.pink_cup_base_xyz, dtype=float).reshape(3)

    model, include_primary_target_marker = _build_sim_model(
        base_mjcf_path=args.model_path,
        urdf_path=args.urdf_path,
        handeye_result_path=args.handeye_result,
        object_positions=object_positions,
        orange_base_xyz=object_positions.get("orange"),
        target_phrase=str(args.target_phrase),
        target_radius_m=float(args.target_radius_m),
        target_mass_kg=float(args.target_mass_kg),
        scene_preset=str(args.scene_preset),
        legacy_scene=bool(args.legacy_scene),
    )
    data = mujoco.MjData(model)
    joint_qpos = _joint_qpos_indices(model)

    step_duration = max(float(args.step_duration_s), 1e-3)
    loop_forever = not bool(args.once)

    viewer = mujoco.viewer.launch_passive(
        model,
        data,
        show_left_ui=bool(args.show_left_ui),
        show_right_ui=bool(args.show_right_ui),
    )
    _apply_camera(viewer)
    _update_target_markers(
        viewer,
        object_positions,
        include_markers=bool(include_primary_target_marker),
    )

    try:
        while viewer.is_running():
            for point in points:
                if not viewer.is_running():
                    break
                tick_start = time.perf_counter()
                joint_positions = np.asarray(point["joint_positions"], dtype=float)
                _set_robot_qpos(data, joint_qpos, joint_positions)
                mujoco.mj_forward(model, data)
                _update_target_markers(
                    viewer,
                    object_positions,
                    include_markers=bool(include_primary_target_marker),
                )
                viewer.sync()
                remaining = step_duration - (time.perf_counter() - tick_start)
                if remaining > 0:
                    time.sleep(remaining)
            if not loop_forever:
                break
    finally:
        viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
