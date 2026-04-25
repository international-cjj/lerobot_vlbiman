#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
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
    extra_paths = [
        repo_root / "src",
        repo_root,
        repo_root / "lerobot_robot_cjjarm",
        repo_root / "lerobot_camera_gemini335l",
    ]
    for path in extra_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo_root


_maybe_reexec_in_repo_venv()
REPO_ROOT = _bootstrap_paths()

from lerobot.projects.vlbiman_sa.runtime_env import strip_user_site_packages

strip_user_site_packages()

from lerobot.projects.vlbiman_sa.app.run_one_shot_record import _build_camera
from lerobot.projects.vlbiman_sa.app.run_pose_adaptation import build_pose_pipeline_config, run_pose_adaptation_pipeline
from lerobot.projects.vlbiman_sa.app.run_trajectory_generation import (
    TrajectoryPipelineConfig,
    run_trajectory_generation_pipeline,
)
from lerobot.projects.vlbiman_sa.core.contracts import TaskGraspConfig
from lerobot.projects.vlbiman_sa.demo.io import append_frame_metadata, save_frame_assets
from lerobot.projects.vlbiman_sa.demo.schema import FrameRecord
from lerobot.projects.vlbiman_sa.geometry.transforms import apply_transform_points
from lerobot.projects.vlbiman_sa.trajectory.progressive_ik import (
    IKPyState,
    build_ikpy_state,
    forward_kinematics_tool,
    full_q_from_arm_q,
)
from lerobot.projects.vlbiman_sa.vision import (
    AnchorEstimator,
    AnchorEstimatorConfig,
    CameraIntrinsics,
    OrientationMomentsEstimator,
    VLMObjectSegmentor,
    VLMObjectSegmentorConfig,
)


@dataclass(slots=True)
class PreviewArtifacts:
    run_dir: Path
    live_result_path: Path
    pose_summary_path: Path
    trajectory_summary_path: Path
    trajectory_points_path: Path
    rerun_path: Path
    video_path: Path
    summary_path: Path


def _default_task_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "task_grasp.yaml"


def _default_capture_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "one_shot_record.yaml"


def _default_vision_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "vision_analysis.yaml"


def _default_output_root() -> Path:
    return REPO_ROOT / "outputs" / "vlbiman_sa" / "live_grasp_preview"


def _default_live_output_root() -> Path:
    return REPO_ROOT / "outputs" / "vlbiman_sa" / "live_orange_pose"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a live orange image and preview the planned grasp trajectory.")
    parser.add_argument("--task-config", type=Path, default=_default_task_config_path())
    parser.add_argument("--capture-config", type=Path, default=_default_capture_config_path())
    parser.add_argument("--vision-config", type=Path, default=_default_vision_config_path())
    parser.add_argument("--output-root", type=Path, default=_default_output_root())
    parser.add_argument("--live-output-root", type=Path, default=_default_live_output_root())
    parser.add_argument("--reuse-live-result", type=Path, default=None)
    parser.add_argument("--camera-serial-number", type=str, default=None)
    parser.add_argument("--target-phrase", type=str, default=None)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--camera-timeout-ms", type=int, default=2500)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument("--video-stride", type=int, default=1)
    parser.add_argument("--spawn-rerun", action="store_true")
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


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _capture_frame(camera: Any, warmup_frames: int, timeout_ms: int) -> tuple[np.ndarray, np.ndarray]:
    retry_attempts = 3
    last_error: Exception | None = None
    for attempt_idx in range(retry_attempts):
        color_rgb = None
        depth_map = None
        try:
            for _ in range(max(1, warmup_frames)):
                color_rgb, depth_map = camera.read_rgbd(timeout_ms=timeout_ms)
                time.sleep(0.03)
            assert color_rgb is not None and depth_map is not None
            return np.asarray(color_rgb), np.asarray(depth_map)
        except TimeoutError as exc:
            last_error = exc
            if attempt_idx < retry_attempts - 1:
                logging.warning(
                    "RGBD capture timed out (attempt %d/%d, timeout=%d ms). Retrying.",
                    attempt_idx + 1,
                    retry_attempts,
                    int(timeout_ms),
                )
                time.sleep(0.2 * float(attempt_idx + 1))

    if last_error is not None:
        raise TimeoutError(
            f"RGBD capture failed after {retry_attempts} attempts with timeout {int(timeout_ms)} ms. "
            f"Last error: {last_error}"
        ) from last_error
    raise RuntimeError("RGBD capture failed for an unexpected reason.")


def _write_single_frame_session(output_dir: Path, color_rgb: np.ndarray, depth_map: np.ndarray) -> Path:
    session_dir = output_dir / "capture_session"
    if session_dir.exists():
        for path in sorted(session_dir.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    (session_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (session_dir / "depth").mkdir(parents=True, exist_ok=True)

    now_ns = time.time_ns()
    frame = FrameRecord(
        frame_index=0,
        slot_index=0,
        wall_time_ns=now_ns,
        relative_time_s=0.0,
        scheduled_time_s=0.0,
        capture_started_ns=now_ns,
        capture_ended_ns=now_ns,
        capture_latency_ms=0.0,
        camera_timestamp_ns=now_ns,
        robot_timestamp_ns=now_ns,
        time_skew_ms=0.0,
    )
    frame = save_frame_assets(session_dir, frame, color_rgb, depth_map)
    append_frame_metadata(session_dir / "metadata.jsonl", frame)
    return session_dir


def _load_base_from_camera(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    payload = _load_json(path)
    matrix = np.asarray(payload.get("base_from_camera"), dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"base_from_camera in {path} is not a 4x4 matrix.")
    return matrix, payload


def _overlay_live_result(
    color_rgb: np.ndarray,
    mask: np.ndarray,
    bbox_xyxy: list[int],
    centroid_px: list[float] | None,
    contact_px: list[float] | None,
    camera_xyz_m: list[float] | None,
    base_xyz_m: list[float] | None,
) -> np.ndarray:
    canvas = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    tint = canvas.copy()
    tint[mask > 0] = (20, 100, 245)
    canvas = cv2.addWeighted(canvas, 0.72, tint, 0.28, 0.0)
    x0, y0, x1, y1 = bbox_xyxy
    if x1 > x0 and y1 > y0:
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 0), 2)
    if centroid_px is not None:
        cv2.circle(canvas, (int(round(centroid_px[0])), int(round(centroid_px[1]))), 5, (255, 255, 0), -1)
    if contact_px is not None:
        cv2.circle(canvas, (int(round(contact_px[0])), int(round(contact_px[1]))), 6, (0, 255, 255), -1)
    cv2.putText(canvas, "live orange pose", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    if camera_xyz_m is not None:
        cv2.putText(
            canvas,
            f"camera xyz=({camera_xyz_m[0]:.3f}, {camera_xyz_m[1]:.3f}, {camera_xyz_m[2]:.3f}) m",
            (24, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if base_xyz_m is not None:
        cv2.putText(
            canvas,
            f"base xyz=({base_xyz_m[0]:.3f}, {base_xyz_m[1]:.3f}, {base_xyz_m[2]:.3f}) m",
            (24, 114),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return canvas


def _capture_live_result(
    *,
    capture_config_path: Path,
    vision_config_path: Path,
    handeye_result_path: Path,
    output_root: Path,
    camera_serial_number: str | None,
    target_phrase: str,
    warmup_frames: int,
    camera_timeout_ms: int,
) -> Path:
    capture_payload = _load_yaml(capture_config_path)
    camera_cfg = dict(capture_payload.get("camera", {}))
    if camera_serial_number is not None:
        camera_cfg["serial_number_or_name"] = camera_serial_number

    vision_payload = _load_yaml(vision_config_path)
    segmentor_cfg = VLMObjectSegmentorConfig(**dict(vision_payload.get("segmentor", {})))
    anchor_cfg = AnchorEstimatorConfig(**dict(vision_payload.get("anchor", {})))

    output_dir = output_root / _timestamp_name("live_orange")
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = _build_camera(camera_cfg)
    try:
        camera.connect()
        color_rgb, depth_map = _capture_frame(camera, warmup_frames=warmup_frames, timeout_ms=camera_timeout_ms)
    finally:
        try:
            camera.disconnect()
        except Exception:
            logging.exception("Camera disconnect raised an exception.")

    snapshot_rgb_path = output_dir / "snapshot_rgb.png"
    snapshot_depth_path = output_dir / "snapshot_depth.npy"
    cv2.imwrite(str(snapshot_rgb_path), cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR))
    np.save(snapshot_depth_path, depth_map)

    session_dir = _write_single_frame_session(output_dir, color_rgb, depth_map)
    record = FrameRecord(
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

    segmentor = VLMObjectSegmentor(segmentor_cfg)
    segmentor_result = segmentor.segment_video(
        session_dir=session_dir,
        records=[record],
        frame_indices=[0],
        output_dir=output_dir / "vision",
        target_phrase=target_phrase,
    )
    mask = segmentor_result.masks[0][1]
    frame_result = segmentor_result.frame_results[0]

    orientation = OrientationMomentsEstimator().estimate(mask)
    intrinsics = CameraIntrinsics.from_json(Path(vision_payload["intrinsics_path"]))
    anchor = AnchorEstimator(intrinsics, anchor_cfg).estimate(
        frame_index=0,
        mask=mask,
        depth_map=depth_map,
        orientation_deg=orientation.angle_deg,
        score=frame_result.score,
    )

    base_from_camera, handeye_payload = _load_base_from_camera(handeye_result_path)
    base_xyz_m = None
    if anchor.camera_xyz_m is not None:
        base_xyz_m = apply_transform_points(base_from_camera, np.asarray(anchor.camera_xyz_m, dtype=float))[0].tolist()

    overlay = _overlay_live_result(
        color_rgb=color_rgb,
        mask=mask,
        bbox_xyxy=frame_result.bbox_xyxy,
        centroid_px=anchor.centroid_px,
        contact_px=anchor.contact_px,
        camera_xyz_m=anchor.camera_xyz_m,
        base_xyz_m=base_xyz_m,
    )
    overlay_path = output_dir / "overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    metrics = dict(handeye_payload.get("metrics", {})) if isinstance(handeye_payload.get("metrics"), dict) else {}
    result = {
        "status": "ok" if base_xyz_m is not None else "warn",
        "target_phrase": target_phrase,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "snapshot_rgb_path": str(snapshot_rgb_path),
        "snapshot_depth_path": str(snapshot_depth_path),
        "overlay_path": str(overlay_path),
        "seed_detection": segmentor_result.seed_detection.to_dict(),
        "segmentation": asdict(frame_result),
        "anchor": anchor.to_dict(),
        "orientation": orientation.to_dict(),
        "base_xyz_m": base_xyz_m,
        "handeye_status": {
            "path": str(handeye_result_path),
            "passed": handeye_payload.get("passed"),
            "accepted_without_passing_thresholds": handeye_payload.get("accepted_without_passing_thresholds"),
            "translation_mean_mm": metrics.get("translation_mean_mm"),
            "rotation_mean_deg": metrics.get("rotation_mean_deg"),
        },
    }
    result_path = output_dir / "result.json"
    _save_json(result_path, result)
    _save_json(output_root / "latest_result.json", result)
    return result_path


def _full_q(joint_positions: np.ndarray, state: IKPyState) -> np.ndarray:
    return full_q_from_arm_q(state, joint_positions)


def _link_positions(state: IKPyState, joint_positions: np.ndarray) -> np.ndarray:
    frames = state.chain.forward_kinematics(_full_q(joint_positions, state), full_kinematics=True)
    positions = [np.zeros(3, dtype=float)]
    positions.extend(np.asarray(frame[:3, 3], dtype=float) for frame in frames)
    return np.asarray(positions, dtype=float)


def _log_pose_axes(rr: Any, entity_path: str, transform: np.ndarray, axis_length: float) -> None:
    origin = np.asarray(transform[:3, 3], dtype=float)
    basis = np.asarray(transform[:3, :3], dtype=float)
    rr.log(
        entity_path,
        rr.Arrows3D(
            origins=np.repeat(origin[None, :], 3, axis=0),
            vectors=np.asarray([basis[:, 0], basis[:, 1], basis[:, 2]], dtype=float) * float(axis_length),
            colors=np.asarray(
                [
                    [255, 70, 70, 255],
                    [70, 220, 120, 255],
                    [70, 140, 255, 255],
                ],
                dtype=np.uint8,
            ),
        ),
        static=True,
    )


def _export_rerun_preview(
    *,
    live_result_path: Path,
    adapted_pose_path: Path,
    trajectory_points_path: Path,
    output_path: Path,
    spawn_viewer: bool,
) -> None:
    import rerun as rr

    live_result = _load_json(live_result_path)
    adapted_pose = _load_json(adapted_pose_path)
    trajectory_payload = _load_json(trajectory_points_path)
    points = trajectory_payload["points"]
    state = build_ikpy_state()

    rr.init("vlbiman_live_grasp_preview", spawn=spawn_viewer)
    rgb_path = Path(str(live_result["snapshot_rgb_path"]))
    overlay_path = Path(str(live_result["overlay_path"]))
    if rgb_path.exists():
        rgb = cv2.cvtColor(cv2.imread(str(rgb_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        rr.log("capture/rgb", rr.Image(rgb), static=True)
    if overlay_path.exists():
        overlay = cv2.cvtColor(cv2.imread(str(overlay_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        rr.log("capture/overlay", rr.Image(overlay), static=True)

    orange_anchor = np.asarray(live_result["base_xyz_m"], dtype=float).reshape(1, 3)
    rr.log(
        "world/live_anchor",
        rr.Points3D(orange_anchor, colors=np.asarray([[255, 165, 0, 255]], dtype=np.uint8), radii=[0.02]),
        static=True,
    )
    target_pose = np.asarray(adapted_pose["adapted_gripper_matrix"], dtype=float)
    rr.log(
        "world/target/gripper_origin",
        rr.Points3D(np.asarray([target_pose[:3, 3]], dtype=float), colors=np.asarray([[255, 0, 0, 255]], dtype=np.uint8), radii=[0.018]),
        static=True,
    )
    _log_pose_axes(rr, "world/target/gripper_axes", target_pose, axis_length=0.06)

    ee_path = np.asarray(
        [np.asarray(point["solved_pose_matrix"], dtype=float)[:3, 3] for point in points if point["solved_pose_matrix"] is not None],
        dtype=float,
    )
    if len(ee_path) > 1:
        rr.log(
            "world/path/ee_full",
            rr.LineStrips3D([ee_path], colors=np.asarray([[70, 140, 255, 255]], dtype=np.uint8), radii=[0.003]),
            static=True,
        )

    for idx, point in enumerate(points):
        rr.set_time("trajectory_step", sequence=idx)
        rr.set_time("trajectory_time", duration=float(idx))

        joint_positions = np.asarray(point["joint_positions"], dtype=float)
        joint_xyz = _link_positions(state, joint_positions)
        rr.log(
            "world/robot/joints",
            rr.Points3D(joint_xyz, colors=np.asarray([[30, 30, 30, 255]] * len(joint_xyz), dtype=np.uint8), radii=[0.01] * len(joint_xyz)),
        )
        rr.log(
            "world/robot/links",
            rr.LineStrips3D([joint_xyz], colors=np.asarray([[50, 110, 220, 255]], dtype=np.uint8), radii=[0.005]),
        )

        if point["solved_pose_matrix"] is not None:
            current_pose = np.asarray(point["solved_pose_matrix"], dtype=float)
        else:
            current_pose = forward_kinematics_tool(state, joint_positions)
        ee_xyz = current_pose[:3, 3].reshape(1, 3)
        rr.log(
            "world/robot/ee",
            rr.Points3D(ee_xyz, colors=np.asarray([[220, 40, 40, 255]], dtype=np.uint8), radii=[0.016]),
        )
        basis = current_pose[:3, :3]
        rr.log(
            "world/robot/ee_axes",
            rr.Arrows3D(
                origins=np.repeat(ee_xyz, 3, axis=0),
                vectors=np.asarray([basis[:, 0], basis[:, 1], basis[:, 2]], dtype=float) * 0.05,
                colors=np.asarray(
                    [
                        [255, 70, 70, 255],
                        [70, 220, 120, 255],
                        [70, 140, 255, 255],
                    ],
                    dtype=np.uint8,
                ),
            ),
        )

        rr.log("metrics/source_index", rr.Scalars(float(idx)))
        rr.log("metrics/source_kind", rr.TextLog(f"{point['segment_id']} | {point['source']} | {point['invariance']}"))
        if point.get("translation_error_mm") is not None:
            rr.log("metrics/translation_error_mm", rr.Scalars(float(point["translation_error_mm"])))
        if point.get("rotation_error_deg") is not None:
            rr.log("metrics/rotation_error_deg", rr.Scalars(float(point["rotation_error_deg"])))
        if point.get("max_joint_step_rad") is not None:
            rr.log("metrics/max_joint_step_rad", rr.Scalars(float(point["max_joint_step_rad"])))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rr.save(output_path)


def _render_mujoco_preview(
    *,
    trajectory_points_path: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    frame_stride: int,
) -> None:
    import mujoco

    trajectory_payload = _load_json(trajectory_points_path)
    points = trajectory_payload["points"]

    model_path = (
        REPO_ROOT
        / "lerobot_robot_cjjarm"
        / "lerobot_robot_cjjarm"
        / "cjjarm_urdf"
        / "TRLC-DK1-Follower-home.mjcf"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    render_width = min(int(width), int(model.vis.global_.offwidth))
    render_height = min(int(height), int(model.vis.global_.offheight))
    renderer = mujoco.Renderer(model, render_height, render_width)
    camera = mujoco.MjvCamera()
    camera.azimuth = 148.0
    camera.elevation = -20.0
    camera.distance = 1.75
    camera.lookat[:] = np.asarray([0.0, 0.0, 0.24], dtype=float)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (render_width, render_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    try:
        joint_qpos = {}
        for joint_name in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            joint_qpos[joint_name] = int(model.jnt_qposadr[joint_id])

        for idx, point in enumerate(points[:: max(1, int(frame_stride))]):
            joint_positions = np.asarray(point["joint_positions"], dtype=float)
            for joint_idx, joint_name in enumerate(("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")):
                data.qpos[joint_qpos[joint_name]] = float(joint_positions[joint_idx])
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(
                bgr,
                f"step={idx * max(1, int(frame_stride))} segment={point['segment_id']} source={point['source']}",
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(bgr)
    finally:
        writer.release()
        renderer.close()


def _run_t5_t6(
    *,
    task_config: TaskGraspConfig,
    live_result_path: Path,
    run_dir: Path,
    aux_target_phrases: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if task_config.recording_session_dir is None:
        raise ValueError("task_config.recording_session_dir must point to the reference demo session.")

    session_dir = task_config.recording_session_dir
    analysis_dir = session_dir / "analysis"
    skill_bank_path = task_config.skill_bank_path or analysis_dir / "t3_skill_bank" / "skill_bank.json"
    pose_output_dir = run_dir / "analysis" / "t5_pose"
    trajectory_output_dir = run_dir / "analysis" / "t6_trajectory"
    effective_aux_target_phrases = list(aux_target_phrases) if aux_target_phrases is not None else []
    if aux_target_phrases is None:
        live_result = _load_json(live_result_path)
        live_objects = live_result.get("objects") if isinstance(live_result, dict) else {}
        primary_key = "".join(part for part in str(task_config.target_phrase).strip().lower().replace("-", " ").split())
        seen_keys: set[str] = set()
        if isinstance(live_objects, dict):
            for payload in live_objects.values():
                if not isinstance(payload, dict):
                    continue
                phrase = str(payload.get("target_phrase", "")).strip()
                phrase_key = "".join(part for part in phrase.lower().replace("-", " ").split())
                if not phrase_key or phrase_key == primary_key or phrase_key in seen_keys:
                    continue
                effective_aux_target_phrases.append(phrase)
                seen_keys.add(phrase_key)

    pose_summary = run_pose_adaptation_pipeline(
        build_pose_pipeline_config(
            task_config=task_config,
            session_dir=session_dir,
            analysis_dir=analysis_dir,
            output_dir=pose_output_dir,
            live_result_path=live_result_path,
            auxiliary_target_phrases=effective_aux_target_phrases,
            allow_configured_secondary_fallback=bool(effective_aux_target_phrases),
        )
    )
    trajectory_summary = run_trajectory_generation_pipeline(
        TrajectoryPipelineConfig(
            session_dir=session_dir,
            analysis_dir=analysis_dir,
            output_dir=trajectory_output_dir,
            skill_bank_path=skill_bank_path,
            adapted_pose_path=pose_output_dir / "adapted_pose.json",
        )
    )
    return pose_summary, trajectory_summary


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)

    task_config = _load_task_config(args.task_config)
    target_phrase = args.target_phrase or task_config.target_phrase
    task_config.target_phrase = target_phrase
    run_dir = args.output_root / _timestamp_name("preview")
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_live_result is not None:
        live_result_path = args.reuse_live_result
    else:
        live_result_path = _capture_live_result(
            capture_config_path=args.capture_config,
            vision_config_path=args.vision_config,
            handeye_result_path=task_config.handeye_result_path,
            output_root=args.live_output_root,
            camera_serial_number=args.camera_serial_number or task_config.camera_serial_number,
            target_phrase=target_phrase,
            warmup_frames=args.warmup_frames,
            camera_timeout_ms=args.camera_timeout_ms,
        )

    pose_summary, trajectory_summary = _run_t5_t6(
        task_config=task_config,
        live_result_path=live_result_path,
        run_dir=run_dir,
    )

    artifacts = PreviewArtifacts(
        run_dir=run_dir,
        live_result_path=live_result_path,
        pose_summary_path=run_dir / "analysis" / "t5_pose" / "summary.json",
        trajectory_summary_path=run_dir / "analysis" / "t6_trajectory" / "summary.json",
        trajectory_points_path=run_dir / "analysis" / "t6_trajectory" / "trajectory_points.json",
        rerun_path=run_dir / "preview" / "trajectory_preview.rrd",
        video_path=run_dir / "preview" / "trajectory_preview.mp4",
        summary_path=run_dir / "preview" / "summary.json",
    )

    _export_rerun_preview(
        live_result_path=artifacts.live_result_path,
        adapted_pose_path=run_dir / "analysis" / "t5_pose" / "adapted_pose.json",
        trajectory_points_path=artifacts.trajectory_points_path,
        output_path=artifacts.rerun_path,
        spawn_viewer=bool(args.spawn_rerun),
    )
    _render_mujoco_preview(
        trajectory_points_path=artifacts.trajectory_points_path,
        output_path=artifacts.video_path,
        width=int(args.video_width),
        height=int(args.video_height),
        fps=float(args.video_fps),
        frame_stride=max(1, int(args.video_stride)),
    )

    payload = {
        "status": "ok",
        "target_phrase": target_phrase,
        "run_dir": str(artifacts.run_dir),
        "live_result_path": str(artifacts.live_result_path),
        "pose_summary_path": str(artifacts.pose_summary_path),
        "trajectory_summary_path": str(artifacts.trajectory_summary_path),
        "trajectory_points_path": str(artifacts.trajectory_points_path),
        "rerun_path": str(artifacts.rerun_path),
        "video_path": str(artifacts.video_path),
        "pose_summary": pose_summary,
        "trajectory_summary": trajectory_summary,
    }
    _save_json(artifacts.summary_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
