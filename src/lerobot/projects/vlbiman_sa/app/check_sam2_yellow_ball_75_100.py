from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


DEFAULT_SAM2_REPO = Path("/home/cjj/lerobot_2026_1/third_party/sam2_official_20260324")
CONDA_LEROBOT_ROOT = Path("/home/cjj/miniconda3/envs/lerobot")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _maybe_reexec_in_conda_lerobot() -> None:
    repo_root = _repo_root()
    conda_root = Path(os.environ.get("VLBIMAN_CONDA_LEROBOT_PREFIX", CONDA_LEROBOT_ROOT))
    conda_python = conda_root / "bin" / "python"
    already_conda = Path(sys.prefix).resolve() == conda_root.resolve() or Path(sys.executable).resolve() == conda_python.resolve()
    if already_conda and os.environ.get("PYTHONNOUSERSITE") == "1":
        return
    if not conda_python.exists():
        return
    env = os.environ.copy()
    env["VLBIMAN_CONDA_LEROBOT_REEXEC"] = "1"
    env["CONDA_PREFIX"] = str(conda_root)
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("VIRTUAL_ENV", None)
    env["PATH"] = os.pathsep.join([str(conda_root / "bin"), env.get("PATH", "")])
    pythonpath = [str(DEFAULT_SAM2_REPO), str(repo_root / "src"), str(repo_root)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    os.execve(str(conda_python), [str(conda_python), __file__, *sys.argv[1:]], env)


_maybe_reexec_in_conda_lerobot()

import cv2
import numpy as np


def _bootstrap_paths(sam2_repo: Path | None = None) -> None:
    repo_root = _repo_root()
    paths = [repo_root / "src", repo_root, sam2_repo or DEFAULT_SAM2_REPO]
    for path in paths:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


_bootstrap_paths()

from lerobot.projects.vlbiman_sa.demo.io import load_frame_records
from lerobot.projects.vlbiman_sa.demo.schema import FrameRecord
from lerobot.projects.vlbiman_sa.inv_servo.config import (
    default_inv_rgb_servo_config_path,
    load_inv_rgb_servo_config,
)
from lerobot.projects.vlbiman_sa.inv_servo.sam2_live_tracker import SAM2LiveTracker, SAM2TrackedFrame


def _runtime_environment() -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "expected_conda_prefix": str(CONDA_LEROBOT_ROOT),
        "used_conda_lerobot": Path(sys.prefix).resolve() == CONDA_LEROBOT_ROOT.resolve(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM2 yellow-ball tracking from frame 75 to 100.")
    parser.add_argument("--config", type=Path, default=default_inv_rgb_servo_config_path())
    parser.add_argument("--data-dir", "--run-dir", dest="data_dir", type=Path, default=None)
    parser.add_argument("--session-name", default="sim_one_shot")
    parser.add_argument("--camera", default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--phrase", default=None)
    parser.add_argument("--groundingdino-result", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sam2-repo", type=Path, default=DEFAULT_SAM2_REPO)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def _resolve_camera_asset(record: FrameRecord, preferred: str, aliases: list[str]) -> tuple[str | None, dict[str, Any] | None]:
    assets = dict(record.camera_assets)
    candidates = [preferred, *aliases]
    for value in list(candidates):
        if value.endswith("_camera"):
            candidates.append(value.removesuffix("_camera"))
        else:
            candidates.append(f"{value}_camera")
    for candidate in dict.fromkeys(candidates):
        if candidate in assets:
            return candidate, assets[candidate]
    return None, None


def _read_camera_frame(
    *,
    session_dir: Path,
    record: FrameRecord,
    camera: str,
    aliases: list[str],
) -> tuple[str, Path, np.ndarray]:
    camera_name, camera_asset = _resolve_camera_asset(record, camera, aliases)
    if camera_name is None or camera_asset is None:
        raise FileNotFoundError(f"Camera asset {camera!r} not found on frame {record.frame_index}.")
    image_path = session_dir / camera_asset["color_path"]
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"RGB frame not found: {image_path}")
    return camera_name, image_path, cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _default_groundingdino_result(config_output_dir: Path) -> Path:
    preferred = config_output_dir / "groundingdino_frame75_result.json"
    if preferred.exists():
        return preferred
    fallback = config_output_dir / "detection_000075.json"
    return fallback


def _load_groundingdino_result(path: Path) -> tuple[dict[str, Any], Path]:
    actual_path = path
    if not actual_path.exists() and actual_path.name == "groundingdino_frame75_result.json":
        fallback = actual_path.with_name("detection_000075.json")
        if fallback.exists():
            actual_path = fallback
    if not actual_path.exists():
        raise FileNotFoundError(f"GroundingDINO result not found: {path}")
    payload = json.loads(actual_path.read_text(encoding="utf-8"))
    if not payload.get("ok", False):
        raise RuntimeError(f"GroundingDINO result is not ok: {actual_path}")
    return payload, actual_path


def _extract_bbox(payload: dict[str, Any]) -> list[float]:
    bbox = payload.get("bbox_xyxy")
    if bbox is None:
        bbox = (
            payload.get("detection_result", {})
            .get("state", {})
            .get("bbox_xyxy")
        )
    if bbox is None or len(bbox) != 4:
        raise RuntimeError("GroundingDINO result does not contain bbox_xyxy.")
    return [float(value) for value in bbox]


def _prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("mask_*.png", "overlay_*.png", "sam2_75_100_summary.json", "sam2_75_100_trace.jsonl"):
        for path in output_dir.glob(pattern):
            path.unlink()


def _write_mask_and_overlay(
    *,
    output_dir: Path,
    frame: SAM2TrackedFrame,
    rgb_frame: np.ndarray,
    phrase: str,
) -> tuple[Path, Path]:
    mask_path = output_dir / f"mask_{frame.frame_index:06d}.png"
    overlay_path = output_dir / f"overlay_{frame.frame_index:06d}.png"
    cv2.imwrite(str(mask_path), frame.mask)

    base_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    overlay = base_bgr.copy()
    mask_bool = frame.mask > 0
    tint = np.zeros_like(base_bgr)
    tint[mask_bool] = (0, 255, 255)
    blended = cv2.addWeighted(base_bgr, 0.55, tint, 0.45, 0.0)
    overlay[mask_bool] = blended[mask_bool]
    if frame.mask_state.bbox_xyxy is not None:
        x0, y0, x1, y1 = [int(round(float(value))) for value in frame.mask_state.bbox_xyxy]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 255), 2)
    if frame.mask_state.centroid_uv is not None:
        u, v = [int(round(float(value))) for value in frame.mask_state.centroid_uv]
        cv2.circle(overlay, (u, v), 4, (0, 0, 255), -1)
    cv2.putText(
        overlay,
        f"{phrase} {frame.frame_index}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(overlay_path), overlay)
    return mask_path, overlay_path


def _write_trace(trace_path: Path, records: list[dict[str, Any]]) -> None:
    with trace_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    _bootstrap_paths(args.sam2_repo)
    config = load_inv_rgb_servo_config(args.config)
    data_dir = args.data_dir or config.data.original_flow_dir
    camera = args.camera or config.data.camera
    phrase = args.phrase or config.target.phrase
    start_frame = config.sam2.validation_start_frame if args.start_frame is None else int(args.start_frame)
    end_frame = config.sam2.validation_end_frame if args.end_frame is None else int(args.end_frame)
    output_dir = args.output_dir or config.output.sam2_check_dir
    groundingdino_result_path = args.groundingdino_result or _default_groundingdino_result(config.output.groundingdino_check_dir)

    if end_frame < start_frame:
        raise ValueError("--end-frame must be >= --start-frame.")

    groundingdino_payload, actual_groundingdino_path = _load_groundingdino_result(groundingdino_result_path)
    init_bbox = _extract_bbox(groundingdino_payload)

    session_dir = data_dir / "recordings" / args.session_name
    records_by_index = {int(record.frame_index): record for record in load_frame_records(session_dir)}
    frame_indices = list(range(start_frame, end_frame + 1))
    frames: list[np.ndarray] = []
    image_paths: dict[int, Path] = {}
    camera_name = camera
    for frame_index in frame_indices:
        if frame_index not in records_by_index:
            raise RuntimeError(f"Frame {frame_index} is missing from {session_dir / 'metadata.jsonl'}")
        camera_name, image_path, rgb_frame = _read_camera_frame(
            session_dir=session_dir,
            record=records_by_index[frame_index],
            camera=camera,
            aliases=config.data.camera_aliases,
        )
        frames.append(rgb_frame)
        image_paths[frame_index] = image_path

    _prepare_output_dir(output_dir)
    tracker_config = config.sam2.to_tracker_config()
    tracker_config.repo_path = args.sam2_repo
    tracker = SAM2LiveTracker(tracker_config)
    environment = tracker.check_environment()
    if not environment.ok:
        raise RuntimeError(json.dumps(environment.to_dict(), ensure_ascii=False))

    tracked_frames = tracker.track_sequence(
        frames,
        frame_indices=frame_indices,
        init_bbox_xyxy=init_bbox,
        work_dir=output_dir / "sam2_input_jpegs",
    )

    trace_records: list[dict[str, Any]] = []
    failed_frames: list[int] = []
    for rgb_frame, tracked in zip(frames, tracked_frames, strict=True):
        mask_path, overlay_path = _write_mask_and_overlay(
            output_dir=output_dir,
            frame=tracked,
            rgb_frame=rgb_frame,
            phrase=phrase,
        )
        frame_payload = tracked.to_dict(include_mask=False)
        frame_payload.update(
            {
                "phrase": phrase,
                "camera": camera_name,
                "image_path": str(image_paths[tracked.frame_index]),
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
            }
        )
        trace_records.append(frame_payload)
        if not tracked.mask_state.visible:
            failed_frames.append(tracked.frame_index)

    trace_path = output_dir / "sam2_75_100_trace.jsonl"
    summary_path = output_dir / "sam2_75_100_summary.json"
    _write_trace(trace_path, trace_records)

    mask_count = len(list(output_dir.glob("mask_*.png")))
    overlay_count = len(list(output_dir.glob("overlay_*.png")))
    fps_values = [float(record["fps"]) for record in trace_records if float(record["fps"]) > 0.0]
    update_ms_values = [float(record["update_ms"]) for record in trace_records]
    expected_count = end_frame - start_frame + 1
    ok = (
        environment.ok
        and len(tracked_frames) == expected_count
        and mask_count == expected_count
        and overlay_count == expected_count
        and not failed_frames
        and (output_dir / f"mask_{end_frame:06d}.png").exists()
    )
    summary: dict[str, Any] = {
        "ok": ok,
        "phrase": phrase,
        "camera": camera_name,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "expected_frame_count": expected_count,
        "mask_count": mask_count,
        "overlay_count": overlay_count,
        "init_bbox_source": str(actual_groundingdino_path),
        "init_bbox_xyxy": init_bbox,
        "avg_fps": float(sum(fps_values) / len(fps_values)) if fps_values else 0.0,
        "min_fps": float(min(fps_values)) if fps_values else 0.0,
        "max_update_ms": float(max(update_ms_values)) if update_ms_values else 0.0,
        "failed_frames": failed_frames,
        "failure_reason": None if ok else "sam2_tracking_validation_failed",
        "runtime_environment": _runtime_environment(),
        "sam2_environment": environment.to_dict(),
        "sam2_predictor": tracker.predictor_info,
        "trace_path": str(trace_path),
        "output_dir": str(output_dir),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if ok:
        print("SAM2_YELLOW_BALL_75_100_MASKS_OK")
    if args.strict and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
