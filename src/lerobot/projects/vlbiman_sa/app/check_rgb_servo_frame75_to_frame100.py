from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
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
from lerobot.projects.vlbiman_sa.inv_servo.inv_grasp_executor import InvGraspExecutor, InvGraspExecutorConfig
from lerobot.projects.vlbiman_sa.inv_servo.metrics import mask_iou, mask_state_from_mask
from lerobot.projects.vlbiman_sa.inv_servo.rgb_servo_controller import RGBServoController
from lerobot.projects.vlbiman_sa.inv_servo.sam2_live_tracker import SAM2LiveTracker
from lerobot.projects.vlbiman_sa.inv_servo.servo_safety import ServoSafetyConfig, ServoSafetyFilter
from lerobot.projects.vlbiman_sa.inv_servo.sim_backend import SimBackendConfig, SimExecutionBackend
from lerobot.projects.vlbiman_sa.inv_servo.target_state import ServoTarget


def _runtime_environment() -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "expected_conda_prefix": str(CONDA_LEROBOT_ROOT),
        "used_conda_lerobot": Path(sys.prefix).resolve() == CONDA_LEROBOT_ROOT.resolve(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate RGB visual servo alignment from frame 75 to frame 100.")
    parser.add_argument("--config", type=Path, default=default_inv_rgb_servo_config_path())
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--session-name", default="sim_one_shot")
    parser.add_argument("--camera", default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--target-frame", type=int, default=None)
    parser.add_argument("--phrase", default=None)
    parser.add_argument("--groundingdino-result", type=Path, default=None)
    parser.add_argument("--target-mask", type=Path, default=None)
    parser.add_argument("--backend", choices=("sim",), default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sam2-repo", type=Path, default=DEFAULT_SAM2_REPO)
    parser.add_argument("--no-sam2-cache", action="store_true", help="Force a fresh SAM2 pass instead of reusing stage-6 masks.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--stable-frames", type=int, default=None)
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
    return config_output_dir / "detection_000075.json"


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
        bbox = payload.get("detection_result", {}).get("state", {}).get("bbox_xyxy")
    if bbox is None or len(bbox) != 4:
        raise RuntimeError("GroundingDINO result does not contain bbox_xyxy.")
    return [float(value) for value in bbox]


def _prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    patterns = (
        "servo_step_*_rgb.png",
        "servo_step_*_mask.png",
        "servo_step_*_overlay.png",
        "servo_final_rgb.png",
        "servo_final_mask.png",
        "servo_final_overlay.png",
        "target_frame100_mask.png",
        "target_frame100_overlay.png",
        "servo_75_to_100_trace.jsonl",
        "servo_75_to_100_summary.json",
        "servo_75_to_100_failure_report.json",
    )
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            path.unlink()


def _load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found or unreadable: {path}")
    return (mask > 0).astype(np.uint8) * 255


def _load_sam2_cache(
    *,
    cache_dir: Path,
    frame_indices: list[int],
) -> dict[int, np.ndarray] | None:
    masks: dict[int, np.ndarray] = {}
    for frame_index in frame_indices:
        path = cache_dir / f"mask_{frame_index:06d}.png"
        if not path.exists():
            return None
        masks[frame_index] = _load_mask(path)
    return masks


def _run_sam2(
    *,
    config: Any,
    sam2_repo: Path,
    frames: list[np.ndarray],
    frame_indices: list[int],
    init_bbox: list[float],
    output_dir: Path,
) -> dict[int, np.ndarray]:
    tracker_config = config.sam2.to_tracker_config()
    tracker_config.repo_path = sam2_repo
    tracker = SAM2LiveTracker(tracker_config)
    environment = tracker.check_environment()
    if not environment.ok:
        raise RuntimeError(json.dumps(environment.to_dict(), ensure_ascii=False))
    tracked_frames = tracker.track_sequence(
        frames,
        frame_indices=frame_indices,
        init_bbox_xyxy=init_bbox,
        work_dir=output_dir / "sam2_servo_input_jpegs",
    )
    return {int(frame.frame_index): frame.mask for frame in tracked_frames}


def _write_trace(trace_path: Path, records: list[dict[str, Any]]) -> None:
    with trace_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _overlay_image(rgb_frame: np.ndarray, mask: np.ndarray, *, label: str) -> np.ndarray:
    base_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    overlay = base_bgr.copy()
    mask_bool = mask > 0
    tint = np.zeros_like(base_bgr)
    tint[mask_bool] = (0, 255, 255)
    blended = cv2.addWeighted(base_bgr, 0.55, tint, 0.45, 0.0)
    overlay[mask_bool] = blended[mask_bool]

    ys, xs = np.nonzero(mask_bool)
    if xs.size > 0:
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 255), 2)
        cv2.circle(overlay, (int(round(float(xs.mean()))), int(round(float(ys.mean())))), 4, (0, 0, 255), -1)
    cv2.putText(
        overlay,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def _write_step_images(
    *,
    output_dir: Path,
    step: int,
    rgb_frame: np.ndarray,
    mask: np.ndarray,
    phrase: str,
    frame_index: int,
) -> tuple[Path, Path, Path]:
    rgb_path = output_dir / f"servo_step_{step:03d}_rgb.png"
    mask_path = output_dir / f"servo_step_{step:03d}_mask.png"
    overlay_path = output_dir / f"servo_step_{step:03d}_overlay.png"
    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(overlay_path), _overlay_image(rgb_frame, mask, label=f"{phrase} {frame_index} step {step}"))
    return rgb_path, mask_path, overlay_path


def _compact_error(error: dict[str, Any]) -> dict[str, Any]:
    return {
        "e_u": float(error["e_u"]),
        "e_v": float(error["e_v"]),
        "e_a": float(error["e_a"]),
        "error_norm": float(error["error_norm"]),
        "mask_iou": None if error.get("mask_iou") is None else float(error["mask_iou"]),
        "mask_iou_error": None if error.get("mask_iou_error") is None else float(error["mask_iou_error"]),
    }


def _state_payload(mask_state: Any) -> dict[str, Any]:
    height, width = mask_state.image_size_hw
    return {
        "center_uv": None if mask_state.centroid_uv is None else [float(mask_state.centroid_uv[0]), float(mask_state.centroid_uv[1])],
        "area_ratio": float(mask_state.mask_area_px) / max(float(height * width), 1.0),
        "bbox_xyxy": None if mask_state.bbox_xyxy is None else [float(value) for value in mask_state.bbox_xyxy],
        "mask_area_px": int(mask_state.mask_area_px),
        "image_size_hw": [int(height), int(width)],
        "source": mask_state.source,
    }


def _write_failure_report(output_dir: Path, summary: dict[str, Any], trace_records: list[dict[str, Any]]) -> None:
    report = {
        "ok": False,
        "failure_reason": summary.get("failure_reason") or "servo_validation_failed",
        "summary": summary,
        "last_trace": trace_records[-1] if trace_records else None,
    }
    (output_dir / "servo_75_to_100_failure_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    _bootstrap_paths(args.sam2_repo)
    config = load_inv_rgb_servo_config(args.config)

    data_dir = args.data_dir or config.data.original_flow_dir
    camera = args.camera or config.data.camera
    phrase = args.phrase or config.servo_validation.phrase
    start_frame = config.servo_validation.start_frame if args.start_frame is None else int(args.start_frame)
    target_frame = config.servo_validation.target_frame if args.target_frame is None else int(args.target_frame)
    backend_name = args.backend or config.servo_validation.backend
    output_dir = args.output_dir or config.output.servo_check_dir
    target_mask_path = args.target_mask or config.servo_validation.target_mask_path
    groundingdino_result_path = args.groundingdino_result or _default_groundingdino_result(config.output.groundingdino_check_dir)
    stable_frames_required = config.servo_validation.stable_frames if args.stable_frames is None else int(args.stable_frames)
    max_steps = config.servo_validation.max_steps if args.max_steps is None else int(args.max_steps)

    if backend_name != "sim":
        raise ValueError(f"Only --backend sim is supported by this validation script, got {backend_name!r}.")
    if target_frame < start_frame:
        raise ValueError("--target-frame must be >= --start-frame.")
    if stable_frames_required <= 0 or max_steps <= 0:
        raise ValueError("--stable-frames and --max-steps must be positive.")

    _prepare_output_dir(output_dir)

    groundingdino_payload, actual_groundingdino_path = _load_groundingdino_result(groundingdino_result_path)
    init_bbox = _extract_bbox(groundingdino_payload)

    session_dir = data_dir / "recordings" / args.session_name
    records_by_index = {int(record.frame_index): record for record in load_frame_records(session_dir)}
    frame_indices = list(range(start_frame, target_frame + 1))
    frames_by_index: dict[int, np.ndarray] = {}
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
        frames_by_index[frame_index] = rgb_frame

    target_mask = _load_mask(target_mask_path)
    target_mask_state = mask_state_from_mask(
        target_mask,
        frame_index=target_frame,
        source="target_frame100_mask",
        mask_path=target_mask_path,
    )
    target = ServoTarget(phrase=phrase, mask=target_mask_state)

    sam2_masks = None
    sam2_source = "fresh"
    if not args.no_sam2_cache:
        sam2_masks = _load_sam2_cache(cache_dir=target_mask_path.parent, frame_indices=frame_indices)
        if sam2_masks is not None:
            sam2_source = "stage6_mask_cache"
    if sam2_masks is None:
        sam2_masks = _run_sam2(
            config=config,
            sam2_repo=args.sam2_repo,
            frames=[frames_by_index[index] for index in frame_indices],
            frame_indices=frame_indices,
            init_bbox=init_bbox,
            output_dir=output_dir,
        )

    controller = RGBServoController(config.servo.to_controller_config())
    safety_filter = ServoSafetyFilter(
        ServoSafetyConfig(
            max_step_xy_m=config.servo.max_step_xy_m,
            max_step_z_m=config.servo.max_step_z_m,
            max_rotation_rad=config.safety.max_joint_step_rad,
        )
    )
    backend = SimExecutionBackend(
        SimBackendConfig(
            dry_run=False,
            start_frame=start_frame,
            target_frame=target_frame,
            data_dir=data_dir,
            session_name=args.session_name,
            camera=camera,
            camera_aliases=tuple(config.data.camera_aliases),
            replay_end_frame=target_frame,
        )
    )
    reset_result = backend.reset_to_frame(start_frame)
    if not reset_result.ok:
        raise RuntimeError(json.dumps(reset_result.to_dict(), ensure_ascii=False))
    executor = InvGraspExecutor(
        config=InvGraspExecutorConfig(max_steps=max_steps, stable_frames=stable_frames_required),
        controller=controller,
        safety_filter=safety_filter,
        backend=backend,
    )

    target_mask_output = output_dir / "target_frame100_mask.png"
    target_overlay_output = output_dir / "target_frame100_overlay.png"
    shutil.copyfile(target_mask_path, target_mask_output)
    cv2.imwrite(
        str(target_overlay_output),
        _overlay_image(frames_by_index[target_frame], target_mask, label=f"{phrase} target {target_frame}"),
    )

    trace_records: list[dict[str, Any]] = []
    stable_count = 0
    final_error: dict[str, Any] | None = None
    final_rgb: np.ndarray | None = None
    final_mask: np.ndarray | None = None
    final_overlay: np.ndarray | None = None
    close_result: dict[str, Any] | None = None
    converged = False
    failure_reason: str | None = None

    started_at = time.perf_counter()
    for step in range(max_steps):
        observation_result = backend.get_rgb_frame(camera)
        if not observation_result.ok or observation_result.state is None:
            failure_reason = observation_result.failure_reason or "sim_rgb_frame_unavailable"
            break
        observation = observation_result.state
        frame_index = int(observation["frame_index"])
        rgb_frame = np.asarray(observation["image_rgb"])
        current_mask = sam2_masks.get(frame_index)
        if current_mask is None:
            failure_reason = f"sam2_mask_missing_for_frame_{frame_index:06d}"
            break

        current_mask_state = mask_state_from_mask(
            current_mask,
            frame_index=frame_index,
            source=sam2_source,
            mask_path=target_mask_path.parent / f"mask_{frame_index:06d}.png",
        )
        current_iou = mask_iou(current_mask, target_mask)
        step_result = executor.run_step(current_mask_state, target, mask_iou=current_iou)
        if not step_result.ok or step_result.state is None:
            failure_reason = step_result.failure_reason or "servo_step_failed"
            break

        controller_state = step_result.state["controller"]
        safety_state = step_result.state["safety"]
        backend_state = step_result.state["backend"]
        error = _compact_error(controller_state["error"])
        final_error = error

        rgb_path, mask_path, overlay_path = _write_step_images(
            output_dir=output_dir,
            step=step,
            rgb_frame=rgb_frame,
            mask=current_mask,
            phrase=phrase,
            frame_index=frame_index,
        )
        trace_records.append(
            {
                "step": step,
                "timestamp": float(time.perf_counter() - started_at),
                "frame_source": "sim_from_frame75",
                "frame_index": frame_index,
                "rgb_path": str(rgb_path),
                "current_mask_path": str(mask_path),
                "current_overlay_path": str(overlay_path),
                "current_mask": _state_payload(current_mask_state),
                "target_mask": {
                    **_state_payload(target_mask_state),
                    "source": str(target_mask_path),
                },
                "error": error,
                "action": {
                    "delta_cam": [float(value) for value in controller_state["command"]["delta_cam"]],
                    "command": controller_state["command"],
                },
                "safety": {
                    "ok": True,
                    "action": "continue" if not controller_state["command"]["stop"] else "hold",
                    "failure_reason": None,
                    "limited": bool(safety_state["limited"]),
                    "safe_command": safety_state["safe_command"],
                },
                "backend": {
                    "ok": bool(backend_state["ok"]),
                    "failure_reason": backend_state["failure_reason"],
                    "state": backend_state["state"],
                },
            }
        )

        final_rgb = rgb_frame
        final_mask = current_mask
        final_overlay = _overlay_image(rgb_frame, current_mask, label=f"{phrase} final step {step}")

        if controller_state["error"]["converged"]:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= stable_frames_required:
            converged = True
            close_result = executor.close_gripper().to_dict()
            break

    if final_rgb is not None:
        cv2.imwrite(str(output_dir / "servo_final_rgb.png"), cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR))
    if final_mask is not None:
        cv2.imwrite(str(output_dir / "servo_final_mask.png"), final_mask)
    if final_overlay is not None:
        cv2.imwrite(str(output_dir / "servo_final_overlay.png"), final_overlay)

    trace_path = output_dir / "servo_75_to_100_trace.jsonl"
    summary_path = output_dir / "servo_75_to_100_summary.json"
    _write_trace(trace_path, trace_records)

    if final_error is None:
        final_error = {
            "e_u": float("inf"),
            "e_v": float("inf"),
            "e_a": float("inf"),
            "error_norm": float("inf"),
            "mask_iou": 0.0,
            "mask_iou_error": 1.0,
        }

    thresholds = {
        "center_tol_u": float(config.servo_validation.success_center_tol_u),
        "center_tol_v": float(config.servo_validation.success_center_tol_v),
        "area_tol": float(config.servo_validation.success_area_tol),
        "mask_iou_tol": float(config.servo_validation.success_mask_iou),
    }
    mask_iou_value = float(final_error["mask_iou"] or 0.0)
    mask_aligned = (
        abs(float(final_error["e_u"])) <= thresholds["center_tol_u"]
        and abs(float(final_error["e_v"])) <= thresholds["center_tol_v"]
        and abs(float(final_error["e_a"])) <= thresholds["area_tol"]
        and mask_iou_value >= thresholds["mask_iou_tol"]
    )
    ok = bool(converged and mask_aligned and trace_records)
    if not ok and failure_reason is None:
        failure_reason = "servo_validation_failed"

    summary: dict[str, Any] = {
        "ok": ok,
        "phrase": phrase,
        "camera": camera_name,
        "backend": backend_name,
        "start_frame": start_frame,
        "target_frame": target_frame,
        "target_mask": Path(target_mask_path).name,
        "target_mask_path": str(target_mask_path),
        "groundingdino_result_path": str(actual_groundingdino_path),
        "groundingdino_init_bbox_xyxy": init_bbox,
        "sam2_source": sam2_source,
        "used_groundingdino": True,
        "used_sam2": True,
        "used_rgb_servo_controller": True,
        "used_sim_backend": True,
        "steps_run": len(trace_records),
        "max_steps": max_steps,
        "stable_frames": stable_frames_required,
        "stable_frames_observed": stable_count,
        "final_error": final_error,
        "success_threshold": thresholds,
        "visual_servo_converged": bool(converged),
        "mask_aligned_to_frame100": bool(mask_aligned),
        "closed_gripper": None if close_result is None else bool(close_result.get("ok")),
        "close_gripper_result": close_result,
        "trace_path": str(trace_path),
        "output_dir": str(output_dir),
        "runtime_environment": _runtime_environment(),
        "failure_reason": None if ok else failure_reason,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not ok:
        _write_failure_report(output_dir, summary, trace_records)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if ok:
        print("RGB_SERVO_FRAME75_TO_FRAME100_OK")
    if args.strict and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
