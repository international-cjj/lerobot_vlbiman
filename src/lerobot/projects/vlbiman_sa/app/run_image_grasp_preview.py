#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
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
    ]
    for path in extra_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo_root


_maybe_reexec_in_repo_venv()
REPO_ROOT = _bootstrap_paths()

from lerobot.projects.vlbiman_sa.runtime_env import strip_user_site_packages

strip_user_site_packages()

from lerobot.projects.vlbiman_sa.app.run_pose_adaptation import build_pose_pipeline_config, run_pose_adaptation_pipeline
from lerobot.projects.vlbiman_sa.app.run_trajectory_generation import (
    TrajectoryPipelineConfig,
    run_trajectory_generation_pipeline,
)
from lerobot.projects.vlbiman_sa.core.contracts import TaskGraspConfig
from lerobot.projects.vlbiman_sa.demo.io import append_frame_metadata, save_frame_assets
from lerobot.projects.vlbiman_sa.demo.schema import FrameRecord
from lerobot.projects.vlbiman_sa.vision import (
    CameraIntrinsics,
    OrientationMomentsEstimator,
    VLMObjectSegmentor,
    VLMObjectSegmentorConfig,
)


def _default_task_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "task_grasp.yaml"


def _default_vision_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "vision_analysis.yaml"


def _default_image_path() -> Path:
    return Path("outputs/vlbiman_sa/recordings/one_shot_20260323T185847Z/analysis/t6_trajectory/image.png")


def _default_output_root() -> Path:
    return REPO_ROOT / "outputs" / "vlbiman_sa" / "image_grasp_preview"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run T5/T6 preview from a single RGB image under tabletop-height assumption.")
    parser.add_argument("--image-path", type=Path, default=_default_image_path())
    parser.add_argument("--task-config", type=Path, default=_default_task_config_path())
    parser.add_argument("--vision-config", type=Path, default=_default_vision_config_path())
    parser.add_argument("--output-root", type=Path, default=_default_output_root())
    parser.add_argument("--target-z-m", type=float, default=None, help="Override assumed object anchor z in base frame.")
    parser.add_argument("--target-phrase", type=str, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _timestamp_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _load_yaml(path: Path) -> dict[str, Any]:
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


def _build_single_frame_session(output_dir: Path, image_path: Path) -> tuple[Path, list[FrameRecord], np.ndarray]:
    color_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    depth_map = np.zeros(color_rgb.shape[:2], dtype=np.float32)

    session_dir = output_dir / "capture_session"
    if session_dir.exists():
        shutil.rmtree(session_dir)
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
    return session_dir, [frame], color_rgb


def _load_base_from_camera(path: Path) -> np.ndarray:
    payload = _load_json(path)
    matrix = np.asarray(payload.get("base_from_camera"), dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"base_from_camera in {path} is not a 4x4 matrix.")
    return matrix


def _scaled_intrinsics(intrinsics: CameraIntrinsics, width: int, height: int) -> tuple[float, float, float, float]:
    scale_x = width / intrinsics.width if intrinsics.width else 1.0
    scale_y = height / intrinsics.height if intrinsics.height else 1.0
    return (
        intrinsics.fx * scale_x,
        intrinsics.fy * scale_y,
        intrinsics.cx * scale_x,
        intrinsics.cy * scale_y,
    )


def _project_contact_to_base(
    *,
    contact_px: list[float],
    intrinsics: CameraIntrinsics,
    image_shape: tuple[int, int],
    base_from_camera: np.ndarray,
    target_z_m: float,
) -> tuple[list[float], list[float]]:
    height, width = image_shape
    fx, fy, cx, cy = _scaled_intrinsics(intrinsics, width=width, height=height)
    ray_camera = np.asarray(
        [
            (float(contact_px[0]) - cx) / fx,
            (float(contact_px[1]) - cy) / fy,
            1.0,
        ],
        dtype=float,
    )
    camera_origin_base = np.asarray(base_from_camera[:3, 3], dtype=float)
    ray_base = np.asarray(base_from_camera[:3, :3], dtype=float) @ ray_camera
    if abs(float(ray_base[2])) < 1e-9:
        raise ValueError("Ray is parallel to assumed tabletop plane.")
    distance = (float(target_z_m) - float(camera_origin_base[2])) / float(ray_base[2])
    if distance <= 0:
        raise ValueError("Projected point lies behind the camera for the chosen tabletop height.")
    base_xyz = camera_origin_base + ray_base * distance
    camera_xyz = ray_camera * distance
    return base_xyz.astype(float).tolist(), camera_xyz.astype(float).tolist()


def _median_demo_anchor_z(session_dir: Path) -> float:
    demo_frames_path = session_dir / "analysis" / "t5_pose" / "demo_frames.json"
    if demo_frames_path.exists():
        payload = _load_json(demo_frames_path)
        values = sorted(float(frame["anchor_base_xyz_m"][2]) for frame in payload if frame.get("anchor_base_xyz_m"))
        if values:
            return values[len(values) // 2]

    anchors_path = session_dir / "analysis" / "t4_vision" / "anchors.json"
    if anchors_path.exists():
        payload = _load_json(anchors_path)
        values = sorted(float(item["camera_xyz_m"][2]) for item in payload if item.get("camera_xyz_m"))
        if values:
            return values[len(values) // 2]
    return 0.1


def _overlay_result(
    color_rgb: np.ndarray,
    mask: np.ndarray,
    bbox_xyxy: list[int],
    centroid_px: list[float] | None,
    contact_px: list[float] | None,
    base_xyz_m: list[float] | None,
    target_z_m: float,
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
    cv2.putText(canvas, "image-based tabletop projection", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"assumed anchor z={target_z_m:.3f}m", (24, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
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


def _build_image_live_result(
    *,
    image_path: Path,
    task_config: TaskGraspConfig,
    vision_config_path: Path,
    output_dir: Path,
    target_phrase: str,
    target_z_m: float,
) -> Path:
    vision_payload = _load_yaml(vision_config_path)
    segmentor_cfg = VLMObjectSegmentorConfig(**dict(vision_payload.get("segmentor", {})))
    intrinsics = CameraIntrinsics.from_json(task_config.intrinsics_path)
    base_from_camera = _load_base_from_camera(task_config.handeye_result_path)

    session_dir, records, color_rgb = _build_single_frame_session(output_dir, image_path)
    segmentor = VLMObjectSegmentor(segmentor_cfg)
    segmentor_result = segmentor.segment_video(
        session_dir=session_dir,
        records=records,
        frame_indices=[0],
        output_dir=output_dir / "vision",
        target_phrase=target_phrase,
    )
    frame_result = segmentor_result.frame_results[0]
    mask = segmentor_result.masks[0][1]
    orientation = OrientationMomentsEstimator().estimate(mask)

    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0:
        raise ValueError("Target mask is empty for the provided image.")
    centroid_px = [float(xs.mean()), float(ys.mean())]
    contact_px = centroid_px
    base_xyz_m, camera_xyz_m = _project_contact_to_base(
        contact_px=contact_px,
        intrinsics=intrinsics,
        image_shape=color_rgb.shape[:2],
        base_from_camera=base_from_camera,
        target_z_m=float(target_z_m),
    )

    overlay = _overlay_result(
        color_rgb=color_rgb,
        mask=mask,
        bbox_xyxy=frame_result.bbox_xyxy,
        centroid_px=centroid_px,
        contact_px=contact_px,
        base_xyz_m=base_xyz_m,
        target_z_m=float(target_z_m),
    )
    overlay_path = output_dir / "overlay.png"
    snapshot_path = output_dir / "snapshot_rgb.png"
    cv2.imwrite(str(snapshot_path), cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(overlay_path), overlay)

    result = {
        "status": "ok",
        "target_phrase": target_phrase,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "snapshot_rgb_path": str(snapshot_path),
        "snapshot_depth_path": None,
        "overlay_path": str(overlay_path),
        "seed_detection": segmentor_result.seed_detection.to_dict(),
        "segmentation": asdict(frame_result),
        "anchor": {
            "frame_index": 0,
            "mask_area_px": int(xs.size),
            "bbox_xyxy": list(frame_result.bbox_xyxy),
            "centroid_px": centroid_px,
            "contact_px": contact_px,
            "depth_m": None,
            "camera_xyz_m": camera_xyz_m,
            "orientation_deg": orientation.angle_deg,
            "score": float(frame_result.score),
        },
        "orientation": orientation.to_dict(),
        "base_xyz_m": base_xyz_m,
        "plane_assumption": {
            "target_anchor_z_m": float(target_z_m),
            "method": "camera_ray_intersection_with_fixed_base_z",
        },
    }
    result_path = output_dir / "result.json"
    _save_json(result_path, result)
    return result_path


def _run_t5_t6(
    *,
    task_config: TaskGraspConfig,
    live_result_path: Path,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if task_config.recording_session_dir is None:
        raise ValueError("task_config.recording_session_dir must point to the reference demo session.")
    session_dir = task_config.recording_session_dir
    analysis_dir = session_dir / "analysis"
    skill_bank_path = task_config.skill_bank_path or analysis_dir / "t3_skill_bank" / "skill_bank.json"
    pose_output_dir = run_dir / "analysis" / "t5_pose"
    trajectory_output_dir = run_dir / "analysis" / "t6_trajectory"
    live_result = _load_json(live_result_path)
    live_objects = live_result.get("objects") if isinstance(live_result, dict) else {}
    primary_key = "".join(part for part in str(task_config.target_phrase).strip().lower().replace("-", " ").split())
    effective_aux_target_phrases: list[str] = []
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
    task_config = _load_task_config(args.task_config)
    target_phrase = args.target_phrase or task_config.target_phrase
    task_config.target_phrase = target_phrase
    run_dir = args.output_root / _timestamp_name("image_preview")
    image_pose_dir = run_dir / "image_pose"
    image_pose_dir.mkdir(parents=True, exist_ok=True)

    target_z_m = (
        float(args.target_z_m)
        if args.target_z_m is not None
        else float(_median_demo_anchor_z(task_config.recording_session_dir))
    )

    live_result_path = _build_image_live_result(
        image_path=args.image_path,
        task_config=task_config,
        vision_config_path=args.vision_config,
        output_dir=image_pose_dir,
        target_phrase=target_phrase,
        target_z_m=target_z_m,
    )
    pose_summary, trajectory_summary = _run_t5_t6(
        task_config=task_config,
        live_result_path=live_result_path,
        run_dir=run_dir,
    )

    payload = {
        "status": "ok",
        "run_dir": str(run_dir),
        "image_path": str(args.image_path),
        "live_result_path": str(live_result_path),
        "assumed_target_anchor_z_m": float(target_z_m),
        "pose_summary": pose_summary,
        "trajectory_summary": trajectory_summary,
    }
    _save_json(run_dir / "summary.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
