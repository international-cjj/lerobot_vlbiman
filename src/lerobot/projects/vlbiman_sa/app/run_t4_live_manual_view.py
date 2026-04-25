#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
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


def _bootstrap_paths() -> None:
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


_maybe_reexec_in_repo_venv()
_bootstrap_paths()

from lerobot.projects.vlbiman_sa.runtime_env import strip_user_site_packages  # noqa: E402

strip_user_site_packages()

from lerobot.projects.vlbiman_sa.app.run_live_orange_base_pose import (  # noqa: E402
    _capture_frame,
    _default_capture_config_path,
    _default_handeye_path,
    _default_vision_config_path,
    _load_base_from_camera,
    _load_yaml,
    _write_single_frame_session,
)
from lerobot.projects.vlbiman_sa.app.run_one_shot_record import _build_camera  # noqa: E402
from lerobot.projects.vlbiman_sa.geometry.transforms import apply_transform_points  # noqa: E402
from lerobot.projects.vlbiman_sa.vision import (  # noqa: E402
    AnchorEstimator,
    AnchorEstimatorConfig,
    CameraIntrinsics,
    OrientationMomentsEstimator,
    VLMObjectSegmentor,
    VLMObjectSegmentorConfig,
)


def _default_prompt_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "t4_live_manual_view.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live T4 manual viewer for Florence-2 + SAM2 segmentation.")
    parser.add_argument("--capture-config", type=Path, default=_default_capture_config_path())
    parser.add_argument("--vision-config", type=Path, default=_default_vision_config_path())
    parser.add_argument("--prompt-config", type=Path, default=_default_prompt_config_path())
    parser.add_argument("--handeye-result", type=Path, default=_default_handeye_path())
    parser.add_argument("--camera-serial-number", type=str, default=None)
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--camera-timeout-ms", type=int, default=1000)
    parser.add_argument("--display-scale", type=float, default=1.0)
    parser.add_argument("--min-loop-interval-s", type=float, default=0.0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _load_prompt_settings(path: Path, fallback_phrase: str, fallback_window_name: str) -> tuple[str, str]:
    payload = _load_yaml(path)
    phrase = str(payload.get("target_phrase", fallback_phrase)).strip() or fallback_phrase
    window_name = str(payload.get("window_name", fallback_window_name)).strip() or fallback_window_name
    return phrase, window_name


def _draw_text_block(canvas: np.ndarray, lines: list[str], *, origin: tuple[int, int]) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28


def _overlay_frame(
    *,
    color_rgb: np.ndarray,
    target_phrase: str,
    prompt_config_path: Path,
    fps: float,
    mask: np.ndarray | None,
    bbox_xyxy: list[int] | None,
    centroid_px: list[float] | None,
    contact_px: list[float] | None,
    camera_xyz_m: list[float] | None,
    base_xyz_m: list[float] | None,
    status_text: str,
    detection_label: str | None,
) -> np.ndarray:
    canvas = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    if mask is not None:
        tint = canvas.copy()
        tint[mask > 0] = (20, 100, 245)
        canvas = cv2.addWeighted(canvas, 0.72, tint, 0.28, 0.0)
    if bbox_xyxy is not None:
        x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
        if x1 > x0 and y1 > y0:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 0), 2)
    if centroid_px is not None:
        cv2.circle(canvas, (int(round(centroid_px[0])), int(round(centroid_px[1]))), 5, (255, 255, 0), -1)
    if contact_px is not None:
        cv2.circle(canvas, (int(round(contact_px[0])), int(round(contact_px[1]))), 6, (0, 255, 255), -1)

    lines = [
        f"phrase={target_phrase}",
        f"status={status_text}",
        f"seed_label={detection_label or '-'}",
        f"fps={fps:.2f}",
        f"edit_prompt={prompt_config_path}",
        "keys: q/esc quit",
    ]
    if camera_xyz_m is not None:
        lines.append(
            f"camera_xyz=({camera_xyz_m[0]:.3f}, {camera_xyz_m[1]:.3f}, {camera_xyz_m[2]:.3f}) m"
        )
    if base_xyz_m is not None:
        lines.append(f"base_xyz=({base_xyz_m[0]:.3f}, {base_xyz_m[1]:.3f}, {base_xyz_m[2]:.3f}) m")
    _draw_text_block(canvas, lines, origin=(20, 32))
    return canvas


def _resize_for_display(image: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return image
    height, width = image.shape[:2]
    return cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)

    capture_payload = _load_yaml(args.capture_config)
    camera_cfg = dict(capture_payload.get("camera", {}))
    if args.camera_serial_number is not None:
        camera_cfg["serial_number_or_name"] = args.camera_serial_number

    vision_payload = _load_yaml(args.vision_config)
    segmentor = VLMObjectSegmentor(VLMObjectSegmentorConfig(**dict(vision_payload.get("segmentor", {}))))
    anchor_estimator = AnchorEstimator(
        CameraIntrinsics.from_json(Path(vision_payload["intrinsics_path"])),
        AnchorEstimatorConfig(**dict(vision_payload.get("anchor", {}))),
    )
    orientation_estimator = OrientationMomentsEstimator()
    base_from_camera, _ = _load_base_from_camera(args.handeye_result)

    current_phrase = "orange"
    current_window_name = "T4 Live Manual View"
    last_phrase: str | None = None
    prompt_mtime_ns: int | None = None
    prompt_error: str | None = None

    logging.info("Prompt config: %s", args.prompt_config)
    logging.info("Edit target_phrase in that YAML file; the viewer reloads it automatically.")

    camera = _build_camera(camera_cfg)
    temp_root_obj = tempfile.TemporaryDirectory(prefix="vlbiman_t4_live_")
    temp_root = Path(temp_root_obj.name)
    try:
        camera.connect()
        while True:
            loop_started = time.perf_counter()

            try:
                stat = args.prompt_config.stat()
                if prompt_mtime_ns != stat.st_mtime_ns:
                    current_phrase, current_window_name = _load_prompt_settings(
                        args.prompt_config,
                        current_phrase,
                        current_window_name,
                    )
                    prompt_mtime_ns = stat.st_mtime_ns
                    prompt_error = None
            except Exception as exc:
                prompt_error = str(exc)

            if current_phrase != last_phrase:
                logging.info("Current target phrase: %s", current_phrase)
                last_phrase = current_phrase

            color_rgb, depth_map = _capture_frame(camera, args.warmup_frames, args.camera_timeout_ms)
            iteration_dir = temp_root / "current_frame"
            shutil.rmtree(iteration_dir, ignore_errors=True)
            session_dir, records = _write_single_frame_session(iteration_dir, color_rgb, depth_map)

            mask = None
            bbox_xyxy = None
            centroid_px = None
            contact_px = None
            camera_xyz_m = None
            base_xyz_m = None
            detection_label = None
            status_text = "ok"
            try:
                segmentor_result = segmentor.segment_video(
                    session_dir=session_dir,
                    records=records,
                    frame_indices=[0],
                    output_dir=iteration_dir / "vision",
                    target_phrase=current_phrase,
                    keep_artifacts=False,
                )
                frame_result = segmentor_result.frame_results[0]
                detection_label = segmentor_result.seed_detection.label
                bbox_xyxy = frame_result.bbox_xyxy
                mask = segmentor_result.masks[0][1]
                orientation = orientation_estimator.estimate(mask)
                anchor = anchor_estimator.estimate(
                    frame_index=0,
                    mask=mask,
                    depth_map=depth_map,
                    orientation_deg=orientation.angle_deg,
                    score=frame_result.score,
                )
                centroid_px = anchor.centroid_px
                contact_px = anchor.contact_px
                camera_xyz_m = anchor.camera_xyz_m
                if camera_xyz_m is not None:
                    base_xyz_m = apply_transform_points(base_from_camera, np.asarray(camera_xyz_m, dtype=float))[0].tolist()
            except Exception as exc:
                status_text = f"error: {exc}"
                logging.warning("Live segmentation failed for phrase '%s': %s", current_phrase, exc)
            finally:
                shutil.rmtree(iteration_dir, ignore_errors=True)

            if prompt_error:
                status_text = f"{status_text} | prompt_config_error={prompt_error}"

            fps = 1.0 / max(time.perf_counter() - loop_started, 1e-6)
            overlay = _overlay_frame(
                color_rgb=color_rgb,
                target_phrase=current_phrase,
                prompt_config_path=args.prompt_config,
                fps=fps,
                mask=mask,
                bbox_xyxy=bbox_xyxy,
                centroid_px=centroid_px,
                contact_px=contact_px,
                camera_xyz_m=camera_xyz_m,
                base_xyz_m=base_xyz_m,
                status_text=status_text,
                detection_label=detection_label,
            )
            cv2.imshow(current_window_name, _resize_for_display(overlay, args.display_scale))

            elapsed = time.perf_counter() - loop_started
            wait_ms = max(1, int(max(0.0, args.min_loop_interval_s - elapsed) * 1000.0))
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        try:
            camera.disconnect()
        except Exception:
            logging.exception("Camera disconnect raised an exception.")
        cv2.destroyAllWindows()
        temp_root_obj.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
