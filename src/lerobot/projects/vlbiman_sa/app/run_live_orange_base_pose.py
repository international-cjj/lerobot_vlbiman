#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
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


_maybe_reexec_in_repo_venv()


def _bootstrap_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[5]
    extra_paths = [
        repo_root / "src",
        repo_root,
        repo_root / "lerobot_camera_gemini335l",
        repo_root / "lerobot_robot_cjjarm",
    ]
    for path in extra_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo_root


REPO_ROOT = _bootstrap_paths()

from lerobot.projects.vlbiman_sa.runtime_env import strip_user_site_packages

strip_user_site_packages()

from lerobot.projects.vlbiman_sa.app.run_one_shot_record import _build_camera
from lerobot.projects.vlbiman_sa.demo.io import append_frame_metadata, save_frame_assets
from lerobot.projects.vlbiman_sa.demo.schema import FrameRecord
from lerobot.projects.vlbiman_sa.geometry.transforms import apply_transform_points
from lerobot.projects.vlbiman_sa.vision import (
    AnchorEstimator,
    AnchorEstimatorConfig,
    CameraIntrinsics,
    OrientationMomentsEstimator,
    VLMObjectSegmentor,
    VLMObjectSegmentorConfig,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _default_capture_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "one_shot_record.yaml"


def _default_vision_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "vision_analysis.yaml"


def _default_handeye_path() -> Path:
    return _repo_root() / "outputs" / "vlbiman_sa" / "calib" / "handeye_result.json"


def _default_output_root() -> Path:
    return _repo_root() / "outputs" / "vlbiman_sa" / "live_orange_pose"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one RGBD frame and estimate orange position in base frame.")
    parser.add_argument("--capture-config", type=Path, default=_default_capture_config_path())
    parser.add_argument("--vision-config", type=Path, default=_default_vision_config_path())
    parser.add_argument("--handeye-result", type=Path, default=_default_handeye_path())
    parser.add_argument("--output-root", type=Path, default=_default_output_root())
    parser.add_argument("--camera-serial-number", type=str, default=None)
    parser.add_argument("--target-phrase", type=str, default="orange")
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--camera-timeout-ms", type=int, default=1000)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(payload).__name__}.")
    return payload


def _timestamp_name() -> str:
    return datetime.now(timezone.utc).strftime("live_orange_%Y%m%dT%H%M%SZ")


def _capture_frame(camera: Any, warmup_frames: int, timeout_ms: int) -> tuple[np.ndarray, np.ndarray]:
    color_rgb = None
    depth_map = None
    for _ in range(max(1, warmup_frames)):
        color_rgb, depth_map = camera.read_rgbd(timeout_ms=timeout_ms)
        time.sleep(0.03)
    assert color_rgb is not None and depth_map is not None
    return np.asarray(color_rgb), np.asarray(depth_map)


def _write_single_frame_session(
    output_dir: Path,
    color_rgb: np.ndarray,
    depth_map: np.ndarray,
) -> tuple[Path, list[FrameRecord]]:
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
    return session_dir, [frame]


def _load_base_from_camera(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.asarray(payload.get("base_from_camera"), dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"base_from_camera in {path} is not a 4x4 matrix.")
    return matrix, payload


def _overlay_result(
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
    cv2.putText(canvas, "target=orange", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
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


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)

    capture_payload = _load_yaml(args.capture_config)
    camera_cfg = dict(capture_payload.get("camera", {}))
    if args.camera_serial_number is not None:
        camera_cfg["serial_number_or_name"] = args.camera_serial_number

    vision_payload = _load_yaml(args.vision_config)
    segmentor_cfg = VLMObjectSegmentorConfig(**dict(vision_payload.get("segmentor", {})))
    anchor_cfg = AnchorEstimatorConfig(**dict(vision_payload.get("anchor", {})))

    output_dir = args.output_root / _timestamp_name()
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = _build_camera(camera_cfg)
    try:
        camera.connect()
        color_rgb, depth_map = _capture_frame(camera, args.warmup_frames, args.camera_timeout_ms)
    finally:
        try:
            camera.disconnect()
        except Exception:
            logging.exception("Camera disconnect raised an exception.")

    cv2.imwrite(str(output_dir / "snapshot_rgb.png"), cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR))
    np.save(output_dir / "snapshot_depth.npy", depth_map)

    session_dir, records = _write_single_frame_session(output_dir, color_rgb, depth_map)
    segmentor = VLMObjectSegmentor(segmentor_cfg)
    segmentor_result = segmentor.segment_video(
        session_dir=session_dir,
        records=records,
        frame_indices=[0],
        output_dir=output_dir / "vision",
        target_phrase=args.target_phrase,
    )
    mask = segmentor_result.masks[0][1]
    frame_result = segmentor_result.frame_results[0]

    orientation = OrientationMomentsEstimator().estimate(mask)
    anchor = AnchorEstimator(CameraIntrinsics.from_json(Path(vision_payload["intrinsics_path"])), anchor_cfg).estimate(
        frame_index=0,
        mask=mask,
        depth_map=depth_map,
        orientation_deg=orientation.angle_deg,
        score=frame_result.score,
    )

    base_from_camera, handeye_payload = _load_base_from_camera(args.handeye_result)
    base_xyz_m = None
    if anchor.camera_xyz_m is not None:
        base_xyz_m = apply_transform_points(base_from_camera, np.asarray(anchor.camera_xyz_m, dtype=float))[0].tolist()

    overlay = _overlay_result(
        color_rgb=color_rgb,
        mask=mask,
        bbox_xyxy=frame_result.bbox_xyxy,
        centroid_px=anchor.centroid_px,
        contact_px=anchor.contact_px,
        camera_xyz_m=anchor.camera_xyz_m,
        base_xyz_m=base_xyz_m,
    )
    cv2.imwrite(str(output_dir / "overlay.png"), overlay)

    metrics = dict(handeye_payload.get("metrics", {})) if isinstance(handeye_payload.get("metrics"), dict) else {}
    result = {
        "status": "ok" if base_xyz_m is not None else "warn",
        "target_phrase": args.target_phrase,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "snapshot_rgb_path": str(output_dir / "snapshot_rgb.png"),
        "snapshot_depth_path": str(output_dir / "snapshot_depth.npy"),
        "overlay_path": str(output_dir / "overlay.png"),
        "seed_detection": segmentor_result.seed_detection.to_dict(),
        "segmentation": asdict(frame_result),
        "anchor": anchor.to_dict(),
        "orientation": orientation.to_dict(),
        "base_xyz_m": base_xyz_m,
        "handeye_status": {
            "path": str(args.handeye_result),
            "passed": handeye_payload.get("passed"),
            "accepted_without_passing_thresholds": handeye_payload.get("accepted_without_passing_thresholds"),
            "translation_mean_mm": metrics.get("translation_mean_mm"),
            "rotation_mean_deg": metrics.get("rotation_mean_deg"),
        },
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    latest_path = args.output_root / "latest_result.json"
    latest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if base_xyz_m is not None else 1


if __name__ == "__main__":
    sys.exit(main())
