from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any


DEFAULT_SAM2_REPO = Path("/home/cjj/lerobot_2026_1/third_party/sam2_official_20260324")
CONDA_LEROBOT_ROOT = Path("/home/cjj/miniconda3/envs/lerobot")
BASE_JOINT_STEP_SIZES_RAD = (0.05, 0.035, 0.025, 0.015, 0.009, 0.005, 0.003)
DEFAULT_JOINT_STEP_SCALE = 1.0


for _offline_env_name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ.setdefault(_offline_env_name, "1")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _maybe_reexec_in_conda_lerobot() -> None:
    repo_root = _repo_root()
    conda_root = Path(os.environ.get("VLBIMAN_CONDA_LEROBOT_PREFIX", CONDA_LEROBOT_ROOT))
    conda_python = conda_root / "bin" / "python"
    use_viewer = "--use-viewer" in sys.argv[1:]
    display = _display_from_argv(sys.argv[1:])
    if display:
        os.environ["DISPLAY"] = display
    if use_viewer:
        os.environ["MUJOCO_GL"] = "glfw"
    already_conda = Path(sys.prefix).resolve() == conda_root.resolve() or Path(sys.executable).resolve() == conda_python.resolve()
    if already_conda and os.environ.get("PYTHONNOUSERSITE") == "1":
        os.environ.setdefault("CONDA_PREFIX", str(conda_root))
        return
    if not conda_python.exists():
        return
    env = os.environ.copy()
    env["VLBIMAN_CONDA_LEROBOT_REEXEC"] = "1"
    env["CONDA_PREFIX"] = str(conda_root)
    env["PYTHONNOUSERSITE"] = "1"
    if display:
        env["DISPLAY"] = display
    if use_viewer:
        env["MUJOCO_GL"] = "glfw"
    else:
        env.setdefault("MUJOCO_GL", "egl")
    env.pop("VIRTUAL_ENV", None)
    env["PATH"] = os.pathsep.join([str(conda_root / "bin"), env.get("PATH", "")])
    pythonpath = [
        str(DEFAULT_SAM2_REPO),
        str(repo_root / "src"),
        str(repo_root),
        str(repo_root / "lerobot_robot_cjjarm"),
    ]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    os.execve(str(conda_python), [str(conda_python), str(Path(__file__).resolve()), *sys.argv[1:]], env)


def _display_from_argv(argv: list[str]) -> str | None:
    for index, token in enumerate(argv):
        if token == "--display" and index + 1 < len(argv):
            display = str(argv[index + 1]).strip()
            return display or None
        if token.startswith("--display="):
            display = str(token.split("=", 1)[1]).strip()
            return display or None
    return None


_maybe_reexec_in_conda_lerobot()

import cv2
import numpy as np


def _bootstrap_paths(sam2_repo: Path | None = None) -> None:
    repo_root = _repo_root()
    paths = [repo_root / "src", repo_root, repo_root / "lerobot_robot_cjjarm", sam2_repo or DEFAULT_SAM2_REPO]
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
from lerobot.projects.vlbiman_sa.inv_servo.detector import GroundingDINODetector
from lerobot.projects.vlbiman_sa.inv_servo.metrics import mask_iou, mask_state_from_mask
from lerobot.projects.vlbiman_sa.inv_servo.rgb_servo_controller import RGBServoController
from lerobot.projects.vlbiman_sa.inv_servo.sam2_live_tracker import SAM2LiveTracker
from lerobot.projects.vlbiman_sa.inv_servo.sim_backend import MujocoSimBackendConfig, MujocoSimExecutionBackend
from lerobot.projects.vlbiman_sa.inv_servo.target_state import MaskState, ServoTarget


def _runtime_environment() -> dict[str, Any]:
    torch_state: dict[str, Any]
    try:
        import torch

        torch_state = {
            "torch_version": torch.__version__,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda_device_count": int(torch.cuda.device_count()),
            "torch_cuda_version": torch.version.cuda,
            "torch_cuda_current_device": int(torch.cuda.current_device()) if torch.cuda.is_available() else None,
            "torch_cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        torch_state = {"torch_error": f"{type(exc).__name__}: {exc}"}
    return {
        "python_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "effective_conda_prefix": os.environ.get("CONDA_PREFIX") or sys.prefix,
        "expected_conda_prefix": str(CONDA_LEROBOT_ROOT),
        "used_conda_lerobot": Path(sys.prefix).resolve() == CONDA_LEROBOT_ROOT.resolve(),
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        **torch_state,
    }


def _assert_cuda_available_for_inference(config: Any) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "--require-cuda was set, but torch.cuda.is_available() is false in "
            f"{sys.executable}. Check nvidia-smi, driver/runtime visibility, and CUDA_VISIBLE_DEVICES."
        )
    blocked: list[str] = []
    for name, device in (("detector.device", config.detector.device), ("sam2.device", config.sam2.device)):
        normalized = str(device or "").strip().lower()
        if normalized and normalized != "auto" and not normalized.startswith("cuda"):
            blocked.append(f"{name}={device!r}")
    if blocked:
        raise RuntimeError("--require-cuda forbids non-GPU inference config: " + ", ".join(blocked))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate independent MuJoCo RGB visual servo alignment from frame 75 to frame 100.")
    parser.add_argument("--config", type=Path, default=default_inv_rgb_servo_config_path())
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--session-name", default="sim_one_shot")
    parser.add_argument("--camera", default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--target-frame", type=int, default=None)
    parser.add_argument("--phrase", default=None)
    parser.add_argument("--target-mask", type=Path, default=None)
    parser.add_argument("--scene-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sam2-repo", type=Path, default=DEFAULT_SAM2_REPO)
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--control-substeps", type=int, default=160)
    parser.add_argument("--candidate-substeps", type=int, default=160)
    parser.add_argument(
        "--control-mode",
        choices=("pose_ik", "joint_search"),
        default="pose_ik",
        help="Main control path. pose_ik uses visual 6DoF pose deltas plus MuJoCo Jacobian IK; joint_search is the legacy candidate search.",
    )
    parser.add_argument("--ee-body-name", default="link6-7", help="MuJoCo body used as the end-effector for pose IK.")
    parser.add_argument(
        "--delta-pose-frame",
        choices=("camera", "ee", "world"),
        default="camera",
        help="Frame for RGBServoController delta_pose translation. camera uses image-camera [right, down, forward] and converts through MuJoCo camera pose.",
    )
    parser.add_argument("--ik-max-iters", type=int, default=80)
    parser.add_argument("--ik-pos-tol", type=float, default=0.002)
    parser.add_argument("--ik-rot-tol", type=float, default=0.08)
    parser.add_argument("--ik-damping", type=float, default=1e-4)
    parser.add_argument("--ik-step-scale", type=float, default=0.65)
    parser.add_argument("--ik-max-step-rad", type=float, default=0.08)
    parser.add_argument("--ik-position-weight", type=float, default=1.0)
    parser.add_argument("--ik-rotation-weight", type=float, default=0.05)
    parser.add_argument(
        "--servo-delta-scale",
        type=float,
        default=0.8,
        help=(
            "Scale RGBServoController gains and per-step delta limits before control starts. "
            "The controller still emits the final scaled delta_pose used by IK. "
            "Default 0.8 is one tenth of the previous fast validation scale 8.0."
        ),
    )
    parser.add_argument(
        "--arm-position-kp",
        type=float,
        default=None,
        help="Optional MuJoCo arm position-actuator kp override for independent simulation validation.",
    )
    parser.add_argument(
        "--joint-step-scale",
        type=float,
        default=DEFAULT_JOINT_STEP_SCALE,
        help="Fraction of the planned visual-servo joint delta to execute each loop. Default restores direct planned execution at 100%%.",
    )
    parser.add_argument(
        "--planning-step-scale",
        type=float,
        default=1.0,
        help="Scale applied to the online joint-search lookahead step sizes.",
    )
    parser.add_argument("--max-delta-per-step", type=float, default=0.08)
    parser.add_argument(
        "--motion-smoothing-segments",
        type=int,
        default=1,
        help="Split each accepted MuJoCo joint target into N interpolated execution segments. 1 disables smoothing.",
    )
    parser.add_argument(
        "--motion-smoothing-profile",
        choices=("linear", "smoothstep", "smootherstep"),
        default="smootherstep",
        help="Interpolation profile used when --motion-smoothing-segments is greater than 1.",
    )
    parser.add_argument(
        "--motion-smoothing-segment-delay",
        type=float,
        default=0.0,
        help="Seconds to pause after each interpolated execution segment, mainly for smoother live viewer playback.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--stable-frames", type=int, default=None)
    parser.add_argument("--video-fps", type=float, default=6.0)
    parser.add_argument("--display", default=None, help="X display to use with --use-viewer, for example :1.")
    parser.add_argument("--viewer-step-delay", type=float, default=0.35, help="Seconds to pause after each control step when --use-viewer is enabled.")
    parser.add_argument("--use-viewer", action="store_true")
    parser.add_argument("--require-cuda", action="store_true", help="Fail early unless visual inference can run on CUDA.")
    parser.add_argument("--progress", action="store_true", help="Print per-step servo progress.")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def _resolve_path(path: Path | None, *, default: Path | None = None) -> Path:
    raw = path if path is not None else default
    if raw is None:
        raise ValueError("path is required")
    return raw if raw.is_absolute() else _repo_root() / raw


def _default_scene_path() -> Path:
    return _repo_root() / "outputs" / "vlbiman_sa" / "mujoco_dual_camera_scene" / "pure_sim_plate_ball" / "scene" / "dual_camera_target_scene.mjcf"


def _prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    patterns = (
        "servo_step_*_rgb.png",
        "servo_step_*_mask.png",
        "servo_step_*_overlay.png",
        "servo_step_*_control_mask.png",
        "servo_final_rgb.png",
        "servo_final_mask.png",
        "servo_final_overlay.png",
        "target_frame100_mask.png",
        "target_frame100_overlay.png",
        "servo_75_to_100_trace.jsonl",
        "servo_75_to_100_summary.json",
        "servo_75_to_100_failure_report.json",
        "servo_75_to_100_real_mujoco.mp4",
    )
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            path.unlink()
    sam2_work_dir = output_dir / "sam2_live_input_jpegs"
    if sam2_work_dir.exists():
        shutil.rmtree(sam2_work_dir)


def _load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found or unreadable: {path}")
    return (mask > 0).astype(np.uint8) * 255


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


def _read_recorded_camera_frame(
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
    cv2.putText(overlay, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    return overlay


def _write_step_images(
    *,
    output_dir: Path,
    step: int,
    rgb_frame: np.ndarray,
    sam2_mask: np.ndarray,
    control_mask: np.ndarray,
    phrase: str,
    frame_index: int,
) -> tuple[Path, Path, Path, Path]:
    rgb_path = output_dir / f"servo_step_{step:03d}_rgb.png"
    mask_path = output_dir / f"servo_step_{step:03d}_mask.png"
    overlay_path = output_dir / f"servo_step_{step:03d}_overlay.png"
    control_mask_path = output_dir / f"servo_step_{step:03d}_control_mask.png"
    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_path), sam2_mask)
    cv2.imwrite(str(control_mask_path), control_mask)
    cv2.imwrite(str(overlay_path), _overlay_image(rgb_frame, sam2_mask, label=f"{phrase} mujoco {frame_index} step {step}"))
    return rgb_path, mask_path, overlay_path, control_mask_path


def _write_trace(trace_path: Path, records: list[dict[str, Any]]) -> None:
    with trace_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_failure_report(output_dir: Path, summary: dict[str, Any], trace_records: list[dict[str, Any]]) -> None:
    report = {
        "ok": False,
        "failure_reason": summary.get("failure_reason") or "real_mujoco_servo_validation_failed",
        "summary": summary,
        "last_trace": trace_records[-1] if trace_records else None,
    }
    (output_dir / "servo_75_to_100_failure_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _state_payload(mask_state: MaskState) -> dict[str, Any]:
    height, width = mask_state.image_size_hw
    return {
        "center_uv": None if mask_state.centroid_uv is None else [float(mask_state.centroid_uv[0]), float(mask_state.centroid_uv[1])],
        "area_ratio": float(mask_state.mask_area_px) / max(float(height * width), 1.0),
        "bbox_xyxy": None if mask_state.bbox_xyxy is None else [float(value) for value in mask_state.bbox_xyxy],
        "mask_area_px": int(mask_state.mask_area_px),
        "image_size_hw": [int(height), int(width)],
        "source": mask_state.source,
    }


def _compact_error(error: dict[str, Any]) -> dict[str, Any]:
    return {
        "e_u": float(error["e_u"]),
        "e_v": float(error["e_v"]),
        "e_a": float(error["e_a"]),
        "error_norm": float(error["error_norm"]),
        "mask_iou": None if error.get("mask_iou") is None else float(error["mask_iou"]),
        "mask_iou_error": None if error.get("mask_iou_error") is None else float(error["mask_iou_error"]),
    }


def _yellow_hsv_mask(rgb_frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)
    raw = cv2.inRange(hsv, np.array([18, 60, 80], dtype=np.uint8), np.array([42, 255, 255], dtype=np.uint8))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    if n_labels <= 1:
        return raw
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = np.zeros_like(raw)
    mask[labels == largest_label] = 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _joint_step_sizes(step_scale: float) -> tuple[float, ...]:
    if not math.isfinite(float(step_scale)) or float(step_scale) <= 0.0:
        raise ValueError(f"--planning-step-scale must be positive, got {step_scale!r}")
    return tuple(float(step) * float(step_scale) for step in BASE_JOINT_STEP_SIZES_RAD)


def _joint_step_fraction(value: float) -> float:
    fraction = float(value)
    if not math.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
        raise ValueError(f"--joint-step-scale must be in (0, 1], got {value!r}")
    return fraction


def _mask_feature(mask: np.ndarray) -> dict[str, Any] | None:
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0:
        return None
    height, width = mask.shape
    area_ratio = float(xs.size) / max(float(height * width), 1.0)
    return {
        "u_norm": float(xs.mean()) / max(float(width), 1.0),
        "v_norm": float(ys.mean()) / max(float(height), 1.0),
        "log_area": float(math.log(area_ratio + 1e-9)),
        "center_uv": [float(xs.mean()), float(ys.mean())],
        "mask_area_px": int(xs.size),
        "area_ratio": area_ratio,
    }


def _control_objective(mask: np.ndarray, *, target_mask: np.ndarray, target_feature: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    feature = _mask_feature(mask)
    if feature is None:
        return float("inf"), {"visible": False}
    e_u = float(feature["u_norm"] - target_feature["u_norm"])
    e_v = float(feature["v_norm"] - target_feature["v_norm"])
    e_a = float(target_feature["log_area"] - feature["log_area"])
    iou = float(mask_iou(mask, target_mask) or 0.0)
    cost = (e_u / 0.035) ** 2 + (e_v / 0.035) ** 2 + (e_a / 0.12) ** 2 + 1.5 * (1.0 - iou)
    return float(cost), {
        "visible": True,
        "e_u": e_u,
        "e_v": e_v,
        "e_a": e_a,
        "mask_iou": iou,
        "cost": float(cost),
        "feature": feature,
    }


def _evaluate_joint_target(
    *,
    backend: MujocoSimExecutionBackend,
    joint_target: np.ndarray,
    target_mask: np.ndarray,
    target_feature: dict[str, Any],
    candidate_substeps: int,
    camera: str,
) -> tuple[float, dict[str, Any]]:
    if backend.robot is None:
        raise RuntimeError("MuJoCo backend is not connected.")
    snapshot = backend.snapshot_state()
    try:
        targets = backend.control_targets_for_joints(joint_target, gripper_position=0.0)
        backend.robot.set_control_targets(targets, substeps=candidate_substeps)
        observation = backend.get_rgb_frame(camera)
        if not observation.ok or observation.state is None:
            return float("inf"), {"visible": False, "failure_reason": observation.failure_reason}
        control_mask = _yellow_hsv_mask(np.asarray(observation.state["image_rgb"]))
        return _control_objective(control_mask, target_mask=target_mask, target_feature=target_feature)
    finally:
        backend.restore_state(snapshot)


def _choose_joint_target(
    *,
    backend: MujocoSimExecutionBackend,
    current_rgb: np.ndarray,
    target_mask: np.ndarray,
    target_feature: dict[str, Any],
    initial_joints: np.ndarray,
    candidate_substeps: int,
    joint_step_sizes: tuple[float, ...],
    camera: str,
) -> dict[str, Any]:
    if backend.robot is None:
        raise RuntimeError("MuJoCo backend is not connected.")
    current_control_mask = _yellow_hsv_mask(current_rgb)
    current_cost, current_metrics = _control_objective(current_control_mask, target_mask=target_mask, target_feature=target_feature)
    base_q = backend.current_joint_positions()
    best_cost = current_cost
    best_q: np.ndarray | None = None
    best_metrics: dict[str, Any] | None = None
    best_delta: np.ndarray | None = None
    joint_limits = np.asarray(backend.robot._arm_joint_limits_logical, dtype=float)

    for step_size in joint_step_sizes:
        deltas: list[np.ndarray] = []
        for axis in (1, 2):
            for sign in (-1.0, 1.0):
                delta = np.zeros(6, dtype=float)
                delta[axis] = sign * step_size
                deltas.append(delta)
        for sign_2 in (-1.0, 1.0):
            for sign_3 in (-1.0, 1.0):
                delta = np.zeros(6, dtype=float)
                delta[1] = sign_2 * step_size
                delta[2] = sign_3 * step_size
                deltas.append(delta)
                delta = np.zeros(6, dtype=float)
                delta[1] = sign_2 * step_size
                delta[2] = sign_3 * 0.7 * step_size
                deltas.append(delta)
        if abs(float(current_metrics.get("e_u", 0.0))) > 0.01:
            for axis in (0, 4):
                for sign in (-1.0, 1.0):
                    delta = np.zeros(6, dtype=float)
                    delta[axis] = sign * min(step_size, 0.01)
                    deltas.append(delta)

        for delta in deltas:
            candidate = base_q + delta
            candidate[3] = initial_joints[3]
            candidate[5] = initial_joints[5]
            candidate = np.clip(candidate, joint_limits[:, 0], joint_limits[:, 1])
            cost, metrics = _evaluate_joint_target(
                backend=backend,
                joint_target=candidate,
                target_mask=target_mask,
                target_feature=target_feature,
                candidate_substeps=candidate_substeps,
                camera=camera,
            )
            if cost < best_cost - 1e-6:
                best_cost = cost
                best_q = candidate
                best_metrics = metrics
                best_delta = candidate - base_q
        if best_q is not None:
            break

    return {
        "ok": best_q is not None,
        "joint_target": None if best_q is None else best_q.tolist(),
        "joint_delta": None if best_delta is None else best_delta.tolist(),
        "current_cost": float(current_cost),
        "current_metrics": current_metrics,
        "best_cost": float(best_cost),
        "best_metrics": best_metrics,
        "joint_step_sizes": [float(value) for value in joint_step_sizes],
        "control_mask": current_control_mask,
        "source": "online_mujoco_visual_joint_search",
    }


def _save_video(output_dir: Path, trace_records: list[dict[str, Any]], *, fps: float = 6.0) -> str | None:
    if not trace_records:
        return None
    first_overlay = cv2.imread(str(trace_records[0]["current_overlay_path"]), cv2.IMREAD_COLOR)
    if first_overlay is None:
        return None
    height, width = first_overlay.shape[:2]
    video_path = output_dir / "servo_75_to_100_real_mujoco.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        return None
    try:
        for record in trace_records:
            frame = cv2.imread(str(record["current_overlay_path"]), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()
    return str(video_path)


def main() -> None:
    args = parse_args()
    _bootstrap_paths(args.sam2_repo)
    config = load_inv_rgb_servo_config(args.config)
    if args.require_cuda:
        _assert_cuda_available_for_inference(config)

    data_dir = _resolve_path(args.data_dir, default=config.data.original_flow_dir)
    camera = args.camera or config.data.camera
    phrase = args.phrase or config.servo_validation.phrase
    start_frame = config.servo_validation.start_frame if args.start_frame is None else int(args.start_frame)
    target_frame = config.servo_validation.target_frame if args.target_frame is None else int(args.target_frame)
    target_mask_path = _resolve_path(args.target_mask, default=config.servo_validation.target_mask_path)
    scene_path = _resolve_path(args.scene_path, default=config.servo_validation.scene_path or _default_scene_path())
    output_dir = _resolve_path(
        args.output_dir,
        default=Path("outputs/vlbiman_sa/inv_rgb_servo/check_servo_75_to_100_real_mujoco"),
    )
    stable_frames_required = config.servo_validation.stable_frames if args.stable_frames is None else int(args.stable_frames)
    max_steps = config.servo_validation.max_steps if args.max_steps is None else int(args.max_steps)

    if target_frame < start_frame:
        raise ValueError("--target-frame must be >= --start-frame.")
    if stable_frames_required <= 0 or max_steps <= 0:
        raise ValueError("--stable-frames and --max-steps must be positive.")
    if args.video_fps <= 0.0:
        raise ValueError("--video-fps must be positive.")
    if args.viewer_step_delay < 0.0:
        raise ValueError("--viewer-step-delay must be non-negative.")
    if args.motion_smoothing_segments <= 0:
        raise ValueError("--motion-smoothing-segments must be positive.")
    if args.motion_smoothing_segment_delay < 0.0:
        raise ValueError("--motion-smoothing-segment-delay must be non-negative.")
    if not math.isfinite(float(args.servo_delta_scale)) or float(args.servo_delta_scale) <= 0.0:
        raise ValueError("--servo-delta-scale must be finite and positive.")
    if args.arm_position_kp is not None and (
        not math.isfinite(float(args.arm_position_kp)) or float(args.arm_position_kp) <= 0.0
    ):
        raise ValueError("--arm-position-kp must be finite and positive when provided.")
    if args.ik_max_iters <= 0:
        raise ValueError("--ik-max-iters must be positive.")
    for name in ("ik_pos_tol", "ik_rot_tol", "ik_damping", "ik_step_scale", "ik_max_step_rad", "ik_position_weight", "ik_rotation_weight"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative.")
    if not scene_path.exists():
        raise FileNotFoundError(f"MuJoCo scene not found: {scene_path}")
    joint_step_fraction = _joint_step_fraction(args.joint_step_scale)
    joint_step_sizes = _joint_step_sizes(args.planning_step_scale)

    _prepare_output_dir(output_dir)

    session_dir = data_dir / "recordings" / args.session_name
    records_by_index = {int(record.frame_index): record for record in load_frame_records(session_dir)}
    if start_frame not in records_by_index:
        raise RuntimeError(f"Frame {start_frame} is missing from {session_dir / 'metadata.jsonl'}")
    if target_frame not in records_by_index:
        raise RuntimeError(f"Frame {target_frame} is missing from {session_dir / 'metadata.jsonl'}")

    target_mask = _load_mask(target_mask_path)
    target_mask_state = mask_state_from_mask(
        target_mask,
        frame_index=target_frame,
        source="target_frame100_sam2_mask",
        mask_path=target_mask_path,
    )
    target_feature = _mask_feature(target_mask)
    if target_feature is None:
        raise RuntimeError("target mask is empty")
    target = ServoTarget(phrase=phrase, mask=target_mask_state)

    camera_name, _, target_rgb = _read_recorded_camera_frame(
        session_dir=session_dir,
        record=records_by_index[target_frame],
        camera=camera,
        aliases=config.data.camera_aliases,
    )
    target_mask_output = output_dir / "target_frame100_mask.png"
    target_overlay_output = output_dir / "target_frame100_overlay.png"
    shutil.copyfile(target_mask_path, target_mask_output)
    cv2.imwrite(str(target_overlay_output), _overlay_image(target_rgb, target_mask, label=f"{phrase} target {target_frame}"))

    backend = MujocoSimExecutionBackend(
        MujocoSimBackendConfig(
            start_frame=start_frame,
            target_frame=target_frame,
            data_dir=data_dir,
            session_name=args.session_name,
            camera=camera,
            camera_aliases=tuple(config.data.camera_aliases),
            mujoco_model_path=scene_path,
            render_width=args.render_width,
            render_height=args.render_height,
            scene_settle_steps=0,
            action_substeps=args.control_substeps,
            max_delta_per_step=args.max_delta_per_step,
            use_viewer=args.use_viewer,
            motion_smoothing_segments=args.motion_smoothing_segments,
            motion_smoothing_profile=args.motion_smoothing_profile,
            motion_smoothing_segment_delay_s=args.motion_smoothing_segment_delay,
            arm_position_kp=args.arm_position_kp,
        )
    )
    reset_result = backend.reset_to_frame(start_frame)
    if not reset_result.ok:
        raise RuntimeError(json.dumps(reset_result.to_dict(), ensure_ascii=False))
    initial_joints = backend.current_joint_positions().copy()

    detector = GroundingDINODetector(config.detector.to_detector_config())
    controller_config = config.servo.to_controller_config()
    servo_delta_scale = float(args.servo_delta_scale)
    controller_config.k_u *= servo_delta_scale
    controller_config.k_v *= servo_delta_scale
    controller_config.k_a *= servo_delta_scale
    controller_config.max_step_xy_m *= servo_delta_scale
    controller_config.max_step_z_m *= servo_delta_scale
    controller = RGBServoController(controller_config)
    tracker_config = config.sam2.to_tracker_config()
    tracker_config.repo_path = args.sam2_repo
    tracker = SAM2LiveTracker(tracker_config)
    tracker_environment = tracker.check_environment()
    if not tracker_environment.ok:
        raise RuntimeError(json.dumps(tracker_environment.to_dict(), ensure_ascii=False))

    first_observation = backend.get_rgb_frame(camera)
    if not first_observation.ok or first_observation.state is None:
        raise RuntimeError(json.dumps(first_observation.to_dict(), ensure_ascii=False))
    first_rgb = np.asarray(first_observation.state["image_rgb"])
    detection_result = detector.detect(first_rgb, phrase=phrase, frame_index=start_frame)
    if not detection_result.ok or detection_result.state is None:
        raise RuntimeError(json.dumps(detection_result.to_dict(), ensure_ascii=False))
    init_bbox = [float(value) for value in detection_result.state["bbox_xyxy"]]
    sam2_payload: dict[str, Any] | None = tracker.initialize(first_rgb, init_bbox)

    trace_records: list[dict[str, Any]] = []
    stable_count = 0
    final_error: dict[str, Any] | None = None
    final_rgb: np.ndarray | None = None
    final_mask: np.ndarray | None = None
    final_overlay: np.ndarray | None = None
    close_result: dict[str, Any] | None = None
    converged = False
    failure_reason: str | None = None
    last_joint_target: np.ndarray | None = None
    last_target_ee_pose: dict[str, Any] | None = None

    started_at = time.perf_counter()
    for step in range(max_steps):
        observation_result = backend.get_rgb_frame(camera)
        if not observation_result.ok or observation_result.state is None:
            failure_reason = observation_result.failure_reason or "mujoco_rgb_frame_unavailable"
            break
        observation = observation_result.state
        frame_index = int(observation["frame_index"])
        rgb_frame = np.asarray(observation["image_rgb"])
        if step == 0:
            payload = sam2_payload
        else:
            payload = tracker.update(rgb_frame, frame_index=frame_index)
        if payload is None or not payload.get("ok", False) or payload.get("mask") is None:
            failure_reason = str(None if payload is None else payload.get("failure_reason")) or "sam2_update_failed"
            break

        current_mask = (np.asarray(payload["mask"]) > 0).astype(np.uint8) * 255
        current_mask_state = mask_state_from_mask(
            current_mask,
            frame_index=frame_index,
            source="sam2_live_real_mujoco",
            debug={
                "local_frame_index": payload.get("local_frame_index"),
                "update_ms": payload.get("update_ms"),
                "fps": payload.get("fps"),
            },
        )
        current_iou = mask_iou(current_mask, target_mask)
        controller_result = controller.compute_command(current_mask_state, target, mask_iou=current_iou)
        if not controller_result.ok or controller_result.state is None:
            failure_reason = controller_result.failure_reason or "rgb_servo_controller_failed"
            break
        controller_state = controller_result.state
        error = _compact_error(controller_state["error"])
        final_error = error

        if controller_state["error"]["converged"]:
            stable_count += 1
        else:
            stable_count = 0

        action_record: dict[str, Any] = {
            "delta_cam": [float(value) for value in controller_state["command"]["delta_cam"]],
            "delta_pose": controller_state["command"].get("delta_pose"),
            "control_mode": args.control_mode,
            "command": controller_state["command"],
            "executed_action_source": "none",
            "joint_target": None,
            "joint_delta": None,
            "current_ee_pose": None,
            "target_ee_pose": None,
            "ik_success": None,
            "ik_error": None,
            "target_q": None,
            "executed_q": None,
            "accepted": False,
        }
        backend_state: dict[str, Any] = {"ok": True, "failure_reason": None, "state": backend.robot_state()}

        control_mask = _yellow_hsv_mask(rgb_frame)
        if stable_count >= stable_frames_required:
            converged = True
            close_result = backend.close_gripper().to_dict()
            action_record["executed_action_source"] = "hold_close_gripper"
        else:
            if args.control_mode == "pose_ik":
                current_joints = backend.current_joint_positions()
                if controller_state["error"]["converged"] and last_joint_target is not None:
                    hold_pose_result = backend.get_current_ee_pose(args.ee_body_name)
                    if hold_pose_result.ok and hold_pose_result.state is not None:
                        action_record["current_ee_pose"] = hold_pose_result.state
                    if last_target_ee_pose is not None:
                        action_record["target_ee_pose"] = last_target_ee_pose
                    exec_result = backend.execute_joint_target(
                        last_joint_target,
                        gripper_position=0.0,
                        substeps=args.control_substeps,
                        debug={
                            "control_source": "hold_last_pose_ik_target",
                            "reason": "visual_error_converged",
                            "delta_pose": controller_state["command"]["delta_pose"],
                            "ee_body_name": args.ee_body_name,
                        },
                    )
                    executed_q = None
                    if exec_result.state is not None:
                        record_state = exec_result.state.get("record", {}).get("robot_state", {})
                        joint_positions = record_state.get("joint_positions", {})
                        executed_q = [
                            float(joint_positions.get(f"joint_{index}.pos", float("nan")))
                            for index in range(1, 7)
                        ]
                    action_record.update(
                        {
                            "executed_action_source": "hold_last_pose_ik_target",
                            "joint_target": [float(value) for value in last_joint_target],
                            "joint_delta": [float(value) for value in last_joint_target - current_joints],
                            "target_q": [float(value) for value in last_joint_target],
                            "ik_success": True,
                            "ik_error": {"method": "hold_previous_successful_ik_target", "failure_reason": None},
                            "accepted": bool(exec_result.ok),
                            "executed_q": executed_q,
                        }
                    )
                    backend_state = {
                        "ok": bool(exec_result.ok),
                        "failure_reason": exec_result.failure_reason,
                        "state": None if exec_result.state is None else exec_result.state,
                    }
                    if not exec_result.ok:
                        failure_reason = exec_result.failure_reason or "mujoco_hold_pose_ik_target_failed"
                else:
                    current_pose_result = backend.get_current_ee_pose(args.ee_body_name)
                    if not current_pose_result.ok or current_pose_result.state is None:
                        failure_reason = current_pose_result.failure_reason or "current_ee_pose_failed"
                        backend_state = current_pose_result.to_dict()
                    else:
                        current_ee_pose = current_pose_result.state
                        target_pose_result = backend.compose_target_ee_pose(
                            current_ee_pose,
                            controller_state["command"]["delta_pose"],
                            translation_frame=args.delta_pose_frame,
                            camera=camera,
                        )
                        if not target_pose_result.ok or target_pose_result.state is None:
                            failure_reason = target_pose_result.failure_reason or "target_ee_pose_compose_failed"
                            backend_state = target_pose_result.to_dict()
                            action_record["current_ee_pose"] = current_ee_pose
                        else:
                            target_ee_pose = target_pose_result.state
                            ik_result = backend.solve_ik_to_pose(
                                target_ee_pose,
                                ee_body_name=args.ee_body_name,
                                q_init=current_joints,
                                max_iters=args.ik_max_iters,
                                pos_tol=args.ik_pos_tol,
                                rot_tol=args.ik_rot_tol,
                                damping=args.ik_damping,
                                step_scale=args.ik_step_scale,
                                max_step_rad=args.ik_max_step_rad,
                                position_weight=args.ik_position_weight,
                                rotation_weight=args.ik_rotation_weight,
                            )
                            ik_state = ik_result.state if ik_result.state is not None else {}
                            action_record.update(
                                {
                                    "executed_action_source": "pose_delta_mujoco_jacobian_ik",
                                    "current_ee_pose": current_ee_pose,
                                    "target_ee_pose": target_ee_pose,
                                    "ik_success": bool(ik_result.ok),
                                    "ik_error": {
                                        "failure_reason": ik_result.failure_reason,
                                        "position_error": ik_state.get("position_error"),
                                        "rotation_error": ik_state.get("rotation_error"),
                                        "position_error_norm": ik_state.get("position_error_norm"),
                                        "rotation_error_norm": ik_state.get("rotation_error_norm"),
                                        "iterations": ik_state.get("iterations"),
                                        "method": ik_state.get("method"),
                                    },
                                    "target_q": ik_state.get("target_q"),
                                }
                            )
                            if not ik_result.ok or ik_result.state is None:
                                failure_reason = ik_result.failure_reason or "pose_delta_ik_failed"
                                backend_state = ik_result.to_dict()
                            else:
                                joint_target = np.asarray(ik_result.state["target_q"], dtype=float)
                                exec_result = backend.execute_joint_target(
                                    joint_target,
                                    gripper_position=0.0,
                                    substeps=args.control_substeps,
                                    debug={
                                        "control_source": "pose_delta_mujoco_jacobian_ik",
                                        "delta_pose": controller_state["command"]["delta_pose"],
                                        "delta_pose_frame": args.delta_pose_frame,
                                        "ee_body_name": args.ee_body_name,
                                        "current_ee_pose": current_ee_pose,
                                        "target_ee_pose": target_ee_pose,
                                        "ik": ik_result.state,
                                    },
                                )
                                last_joint_target = joint_target
                                last_target_ee_pose = target_ee_pose
                                executed_q = None
                                if exec_result.state is not None:
                                    record_state = exec_result.state.get("record", {}).get("robot_state", {})
                                    joint_positions = record_state.get("joint_positions", {})
                                    executed_q = [
                                        float(joint_positions.get(f"joint_{index}.pos", float("nan")))
                                        for index in range(1, 7)
                                    ]
                                action_record.update(
                                    {
                                        "joint_target": [float(value) for value in joint_target],
                                        "joint_delta": [float(value) for value in joint_target - current_joints],
                                        "executed_joint_delta_fraction": 1.0,
                                        "accepted": bool(exec_result.ok),
                                        "executed_q": executed_q,
                                    }
                                )
                                backend_state = {
                                    "ok": bool(exec_result.ok),
                                    "failure_reason": exec_result.failure_reason,
                                    "state": None if exec_result.state is None else exec_result.state,
                                }
                                if not exec_result.ok:
                                    failure_reason = exec_result.failure_reason or "mujoco_pose_ik_joint_action_failed"
            else:
                search = _choose_joint_target(
                    backend=backend,
                    current_rgb=rgb_frame,
                    target_mask=target_mask,
                    target_feature=target_feature,
                    initial_joints=initial_joints,
                    candidate_substeps=args.candidate_substeps,
                    joint_step_sizes=joint_step_sizes,
                    camera=camera,
                )
                control_mask = np.asarray(search["control_mask"])
                if search["ok"]:
                    planned_joint_target = np.asarray(search["joint_target"], dtype=float)
                    current_joints = backend.current_joint_positions()
                    joint_target = current_joints + joint_step_fraction * (planned_joint_target - current_joints)
                    exec_result = backend.execute_joint_target(
                        joint_target,
                        gripper_position=0.0,
                        substeps=args.control_substeps,
                        debug={
                                "search_source": search["source"],
                                "current_cost": search["current_cost"],
                                "best_cost": search["best_cost"],
                                "joint_step_fraction": joint_step_fraction,
                                "planning_step_scale": args.planning_step_scale,
                                "planning_step_sizes": [float(value) for value in joint_step_sizes],
                            },
                        )
                    last_joint_target = joint_target
                    action_record.update(
                        {
                            "executed_action_source": search["source"],
                            "joint_target": [float(value) for value in joint_target],
                            "joint_delta": [float(value) for value in joint_target - current_joints],
                            "planned_joint_target": [float(value) for value in planned_joint_target],
                            "planned_joint_delta": search["joint_delta"],
                            "executed_joint_delta_fraction": float(joint_step_fraction),
                            "accepted": bool(exec_result.ok),
                            "search": {
                                "current_cost": search["current_cost"],
                                "current_metrics": search["current_metrics"],
                                "best_cost": search["best_cost"],
                                "best_metrics": search["best_metrics"],
                            },
                        }
                    )
                    backend_state = {
                        "ok": bool(exec_result.ok),
                        "failure_reason": exec_result.failure_reason,
                        "state": None if exec_result.state is None else exec_result.state,
                    }
                    if not exec_result.ok:
                        failure_reason = exec_result.failure_reason or "mujoco_joint_action_failed"
                elif controller_state["error"]["converged"] and last_joint_target is not None:
                    exec_result = backend.execute_joint_target(
                        last_joint_target,
                        gripper_position=0.0,
                        substeps=args.control_substeps,
                        debug={"search_source": "hold_last_converged_joint_target"},
                    )
                    action_record.update(
                        {
                            "executed_action_source": "hold_last_converged_joint_target",
                            "joint_target": [float(value) for value in last_joint_target],
                            "joint_delta": [0.0] * 6,
                            "accepted": bool(exec_result.ok),
                        }
                    )
                    backend_state = {
                        "ok": bool(exec_result.ok),
                        "failure_reason": exec_result.failure_reason,
                        "state": None if exec_result.state is None else exec_result.state,
                    }
                    if not exec_result.ok:
                        failure_reason = exec_result.failure_reason or "mujoco_hold_action_failed"
                else:
                    failure_reason = "mujoco_visual_joint_search_no_improvement"

        rgb_path, mask_path, overlay_path, control_mask_path = _write_step_images(
            output_dir=output_dir,
            step=step,
            rgb_frame=rgb_frame,
            sam2_mask=current_mask,
            control_mask=control_mask,
            phrase=phrase,
            frame_index=frame_index,
        )
        trace_records.append(
            {
                "step": step,
                "timestamp": float(time.perf_counter() - started_at),
                "frame_source": "real_mujoco_render_from_frame75_state",
                "frame_index": frame_index,
                "rgb_path": str(rgb_path),
                "current_mask_path": str(mask_path),
                "current_overlay_path": str(overlay_path),
                "control_mask_path": str(control_mask_path),
                "current_mask": _state_payload(current_mask_state),
                "target_mask": {
                    **_state_payload(target_mask_state),
                    "source": str(target_mask_path),
                },
                "error": error,
                "action": action_record,
                "safety": {
                    "ok": True,
                    "action": "hold" if action_record["executed_action_source"] == "hold_close_gripper" else "continue",
                    "failure_reason": None,
                    "limited": False,
                },
                "backend": backend_state,
            }
        )
        if args.progress or args.use_viewer:
            print(
                "REAL_MUJOCO_SERVO_STEP "
                f"step={step} frame={frame_index} stable={stable_count}/{stable_frames_required} "
                f"iou={float(error['mask_iou'] or 0.0):.3f} "
                f"e_u={float(error['e_u']):.4f} e_v={float(error['e_v']):.4f} e_a={float(error['e_a']):.4f} "
                f"action={action_record['executed_action_source']}",
                flush=True,
            )
        if args.use_viewer and args.viewer_step_delay > 0.0:
            time.sleep(float(args.viewer_step_delay))

        final_rgb = rgb_frame
        final_mask = current_mask
        final_overlay = _overlay_image(rgb_frame, current_mask, label=f"{phrase} real mujoco final step {step}")
        if converged or failure_reason is not None:
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
    video_path = None if args.no_video else _save_video(output_dir, trace_records, fps=float(args.video_fps))

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
        failure_reason = "real_mujoco_servo_validation_failed"

    summary: dict[str, Any] = {
        "ok": ok,
        "phrase": phrase,
        "camera": camera_name,
        "backend": "real_mujoco",
        "start_frame": start_frame,
        "target_frame": target_frame,
        "target_mask": Path(target_mask_path).name,
        "target_mask_path": str(target_mask_path),
        "mujoco_scene_path": str(scene_path),
        "groundingdino_init_bbox_xyxy": init_bbox,
        "groundingdino_detection": detection_result.state,
        "sam2_source": "live_real_mujoco_sequence",
        "sam2_predictor": tracker.predictor_info,
        "used_groundingdino": True,
        "used_sam2": True,
        "used_rgb_servo_controller": True,
        "used_sim_backend": True,
        "used_independent_mujoco": True,
        "used_recorded_observation_sequence": False,
        "used_target_frame_robot_state": False,
        "require_cuda": bool(args.require_cuda),
        "control_loop": (
            "groundingdino_sam2_rgb_servo_pose_delta_mujoco_jacobian_ik"
            if args.control_mode == "pose_ik"
            else "groundingdino_sam2_rgb_servo_real_mujoco_joint_search"
        ),
        "control_mode": str(args.control_mode),
        "used_joint_candidate_search_in_main_loop": bool(args.control_mode == "joint_search"),
        "ee_body_name": str(args.ee_body_name),
        "delta_pose_frame": str(args.delta_pose_frame),
        "servo_delta_scale": float(args.servo_delta_scale),
        "effective_servo_controller": {
            "k_u": float(controller_config.k_u),
            "k_v": float(controller_config.k_v),
            "k_a": float(controller_config.k_a),
            "max_step_xy_m": float(controller_config.max_step_xy_m),
            "max_step_z_m": float(controller_config.max_step_z_m),
            "center_tol_u": float(controller_config.center_tol_u),
            "center_tol_v": float(controller_config.center_tol_v),
            "area_tol": float(controller_config.area_tol),
            "mask_iou_tol": float(controller_config.mask_iou_tol),
        },
        "ik": {
            "method": "mujoco_body_jacobian_damped_least_squares",
            "max_iters": int(args.ik_max_iters),
            "pos_tol": float(args.ik_pos_tol),
            "rot_tol": float(args.ik_rot_tol),
            "damping": float(args.ik_damping),
            "step_scale": float(args.ik_step_scale),
            "max_step_rad": float(args.ik_max_step_rad),
            "position_weight": float(args.ik_position_weight),
            "rotation_weight": float(args.ik_rotation_weight),
        },
        "motion_profile": "direct_planned_adjustment" if joint_step_fraction >= 0.999999 else "scaled_micro_adjustment",
        "execution_motion_smoothing": {
            "segments": int(args.motion_smoothing_segments),
            "profile": str(args.motion_smoothing_profile),
            "segment_delay_s": float(args.motion_smoothing_segment_delay),
        },
        "mujoco_position_actuator": {
            "control_type": "position",
            "arm_position_kp_override": None if args.arm_position_kp is None else float(args.arm_position_kp),
        },
        "joint_step_scale": float(args.joint_step_scale),
        "executed_joint_delta_fraction": float(joint_step_fraction),
        "planning_step_scale": float(args.planning_step_scale),
        "base_joint_step_sizes_rad": [float(value) for value in BASE_JOINT_STEP_SIZES_RAD],
        "planning_joint_step_sizes_rad": [float(value) for value in joint_step_sizes],
        "video_fps": float(args.video_fps),
        "candidate_evaluation_mask_source": "mujoco_rendered_hsv_yellow_mask_for_fast_joint_search",
        "official_alignment_mask_source": "sam2_live_real_mujoco",
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
        "video_path": video_path,
        "output_dir": str(output_dir),
        "runtime_environment": _runtime_environment(),
        "failure_reason": None if ok else failure_reason,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not ok:
        _write_failure_report(output_dir, summary, trace_records)

    backend.disconnect()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if ok:
        print("RGB_SERVO_REAL_MUJOCO_FRAME75_TO_FRAME100_OK")
    if args.strict and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
