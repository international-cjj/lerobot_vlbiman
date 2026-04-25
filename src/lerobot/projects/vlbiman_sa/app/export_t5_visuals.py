#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
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


_maybe_reexec_in_repo_venv()

try:
    from lerobot.projects.vlbiman_sa.geometry import FrameManager
    from lerobot.projects.vlbiman_sa.vision import CameraIntrinsics
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[5]
    extra_paths = [
        repo_root / "src",
        repo_root,
    ]
    for path in extra_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from lerobot.projects.vlbiman_sa.geometry import FrameManager
    from lerobot.projects.vlbiman_sa.vision import CameraIntrinsics


@dataclass(slots=True)
class ExportT5Config:
    session_dir: Path
    analysis_dir: Path
    t5_dir: Path
    output_dir: Path
    live_result_path: Path
    intrinsics_path: Path
    transforms_path: Path
    handeye_result_path: Path


def _default_session_dir() -> Path:
    return Path("outputs/vlbiman_sa/recordings/one_shot_20260323T185847Z")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export T5 pose-adaptation validation visuals.")
    parser.add_argument("--session-dir", type=Path, default=_default_session_dir())
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--t5-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--live-result-path",
        type=Path,
        default=Path("outputs/vlbiman_sa/live_orange_pose/latest_result.json"),
    )
    parser.add_argument(
        "--intrinsics-path",
        type=Path,
        default=Path("src/lerobot/projects/vlbiman_sa/configs/camera_intrinsics.json"),
    )
    parser.add_argument(
        "--transforms-path",
        type=Path,
        default=Path("src/lerobot/projects/vlbiman_sa/configs/transforms.yaml"),
    )
    parser.add_argument(
        "--handeye-result-path",
        type=Path,
        default=Path("outputs/vlbiman_sa/calib/handeye_result.json"),
    )
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> ExportT5Config:
    analysis_dir = args.analysis_dir or (args.session_dir / "analysis")
    t5_dir = args.t5_dir or (analysis_dir / "t5_pose")
    output_dir = args.output_dir or (t5_dir / "visuals")
    return ExportT5Config(
        session_dir=args.session_dir,
        analysis_dir=analysis_dir,
        t5_dir=t5_dir,
        output_dir=output_dir,
        live_result_path=args.live_result_path,
        intrinsics_path=args.intrinsics_path,
        transforms_path=args.transforms_path,
        handeye_result_path=args.handeye_result_path,
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_frame_manager(config: ExportT5Config) -> tuple[FrameManager, dict[str, Any] | None]:
    manager = FrameManager.from_yaml(config.transforms_path)
    handeye_payload: dict[str, Any] | None = None
    try:
        manager.get_transform("base", "camera")
        return manager, None
    except KeyError:
        pass

    if not config.handeye_result_path.exists():
        raise FileNotFoundError(
            "No base<-camera transform is available in transforms.yaml and handeye_result.json is missing."
        )

    handeye_payload = _load_json(config.handeye_result_path)
    base_from_camera = np.asarray(handeye_payload.get("base_from_camera"), dtype=float)
    if base_from_camera.shape != (4, 4):
        raise ValueError(f"base_from_camera in {config.handeye_result_path} must be 4x4.")
    manager.set_transform("base", "camera", base_from_camera)
    return manager, handeye_payload


def _project_base_point(
    base_xyz_m: list[float] | np.ndarray,
    *,
    frame_manager: FrameManager,
    intrinsics: CameraIntrinsics,
    image_shape: tuple[int, int, int],
) -> tuple[int, int] | None:
    point = np.asarray(base_xyz_m, dtype=float).reshape(1, 3)
    camera_xyz = frame_manager.transform_points("camera", "base", point)[0]
    if camera_xyz[2] <= 1e-6:
        return None

    height, width = image_shape[:2]
    scale_x = width / intrinsics.width if intrinsics.width else 1.0
    scale_y = height / intrinsics.height if intrinsics.height else 1.0
    fx = intrinsics.fx * scale_x
    fy = intrinsics.fy * scale_y
    cx = intrinsics.cx * scale_x
    cy = intrinsics.cy * scale_y
    x_px = fx * camera_xyz[0] / camera_xyz[2] + cx
    y_px = fy * camera_xyz[1] / camera_xyz[2] + cy
    if not (0 <= x_px < width and 0 <= y_px < height):
        return None
    return int(round(x_px)), int(round(y_px))


def _fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    scale = min(width / max(src_w, 1), height / max(src_h, 1))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _draw_card(title: str, lines: list[str], image: np.ndarray) -> np.ndarray:
    header_h = 140
    canvas = np.full((header_h + image.shape[0], image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(canvas, title, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    for idx, line in enumerate(lines):
        y = 72 + idx * 26
        cv2.putText(canvas, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (45, 45, 45), 2, cv2.LINE_AA)
    canvas[header_h:, :] = image
    return canvas


def _save_montage(images: list[np.ndarray], output_path: Path, title: str, cols: int = 2) -> None:
    if not images:
        return
    cols = min(cols, len(images))
    rows = int(math.ceil(len(images) / cols))
    card_h, card_w = images[0].shape[:2]
    grid = np.full((rows * card_h, cols * card_w, 3), 255, dtype=np.uint8)
    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        y0 = row * card_h
        x0 = col * card_w
        grid[y0 : y0 + card_h, x0 : x0 + card_w] = image
    header = np.full((64, grid.shape[1], 3), 250, dtype=np.uint8)
    cv2.putText(header, title, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.vstack([header, grid]))


def _draw_pose_overlay(
    image: np.ndarray,
    *,
    anchor_px: tuple[int, int] | None,
    gripper_px: tuple[int, int] | None,
    pose_axis_px: tuple[int, int] | None,
    anchor_label: str,
    gripper_label: str,
) -> np.ndarray:
    canvas = image.copy()
    if anchor_px is not None:
        cv2.circle(canvas, anchor_px, 8, (0, 220, 255), -1)
        cv2.putText(canvas, anchor_label, (anchor_px[0] + 12, anchor_px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
    if gripper_px is not None:
        cv2.drawMarker(canvas, gripper_px, (40, 40, 255), cv2.MARKER_CROSS, 22, 3)
        cv2.putText(canvas, gripper_label, (gripper_px[0] + 12, gripper_px[1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 255), 2, cv2.LINE_AA)
    if anchor_px is not None and gripper_px is not None:
        cv2.arrowedLine(canvas, anchor_px, gripper_px, (80, 160, 255), 3, cv2.LINE_AA, tipLength=0.08)
    if gripper_px is not None and pose_axis_px is not None:
        cv2.arrowedLine(canvas, gripper_px, pose_axis_px, (70, 200, 70), 3, cv2.LINE_AA, tipLength=0.12)
    return canvas


def _pose_axis_tip(base_pose_matrix: list[list[float]] | np.ndarray, length_m: float = 0.05) -> np.ndarray:
    transform = np.asarray(base_pose_matrix, dtype=float)
    origin = transform[:3, 3]
    axis = transform[:3, 0]
    return origin + axis * float(length_m)


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def _load_reference_image(config: ExportT5Config, frame_index: int) -> np.ndarray:
    overlay_path = config.analysis_dir / "t4_vision" / "overlays" / f"frame_{frame_index:06d}.png"
    if overlay_path.exists():
        return _read_image(overlay_path)
    rgb_path = config.session_dir / "rgb" / f"frame_{frame_index:06d}.png"
    return _read_image(rgb_path)


def export_t5_visuals(config: ExportT5Config) -> dict[str, Any]:
    intrinsics = CameraIntrinsics.from_json(config.intrinsics_path)
    frame_manager, handeye_payload = _load_frame_manager(config)
    adapted_pose = _load_json(config.t5_dir / "adapted_pose.json")
    reference_frames = _load_json(config.t5_dir / "reference_frames.json")
    summary = _load_json(config.t5_dir / "summary.json")
    live_result = _load_json(config.live_result_path)

    reference_dir = config.output_dir / "reference_cards"
    reference_dir.mkdir(parents=True, exist_ok=True)
    reference_cards: list[np.ndarray] = []
    reference_payload = []

    for item in reference_frames:
        frame_index = int(item["frame_index"])
        raw = _load_reference_image(config, frame_index)
        anchor_px = _project_base_point(
            item["anchor_base_xyz_m"],
            frame_manager=frame_manager,
            intrinsics=intrinsics,
            image_shape=raw.shape,
        )
        gripper_px = _project_base_point(
            np.asarray(item["gripper_pose_matrix"], dtype=float)[:3, 3],
            frame_manager=frame_manager,
            intrinsics=intrinsics,
            image_shape=raw.shape,
        )
        axis_px = _project_base_point(
            _pose_axis_tip(item["gripper_pose_matrix"]),
            frame_manager=frame_manager,
            intrinsics=intrinsics,
            image_shape=raw.shape,
        )
        overlay = _draw_pose_overlay(
            raw,
            anchor_px=anchor_px,
            gripper_px=gripper_px,
            pose_axis_px=axis_px,
            anchor_label="anchor",
            gripper_label="demo grip",
        )
        fitted = _fit_image(overlay, 1280, 720)
        lines = [
            f"frame={frame_index} segment={item.get('segment_id')} label={item.get('segment_label')}",
            f"stable={item.get('stable')} ref_vec={np.linalg.norm(np.asarray(item['relative_xyz_m'], dtype=float))*1000.0:.1f} mm",
            f"object=orange | yellow=anchor, red=demo gripper, green=tool x-axis",
        ]
        card = _draw_card(f"T5 Demo Reference | frame {frame_index}", lines, fitted)
        card_path = reference_dir / f"frame_{frame_index:06d}.png"
        cv2.imwrite(str(card_path), card)
        reference_cards.append(card)
        reference_payload.append({"frame_index": frame_index, "card_path": str(card_path)})

    reference_montage_path = config.output_dir / "t5_reference_montage.png"
    _save_montage(reference_cards, reference_montage_path, "T5 Demo Reference Frames", cols=2)

    live_overlay_path = Path(live_result["overlay_path"])
    live_raw = _read_image(live_overlay_path)
    live_anchor_px = _project_base_point(
        adapted_pose["target_anchor_base_xyz_m"],
        frame_manager=frame_manager,
        intrinsics=intrinsics,
        image_shape=live_raw.shape,
    )
    target_gripper_px = _project_base_point(
        np.asarray(adapted_pose["adapted_gripper_matrix"], dtype=float)[:3, 3],
        frame_manager=frame_manager,
        intrinsics=intrinsics,
        image_shape=live_raw.shape,
    )
    target_axis_px = _project_base_point(
        _pose_axis_tip(adapted_pose["adapted_gripper_matrix"]),
        frame_manager=frame_manager,
        intrinsics=intrinsics,
        image_shape=live_raw.shape,
    )
    live_overlay = _draw_pose_overlay(
        live_raw,
        anchor_px=live_anchor_px,
        gripper_px=target_gripper_px,
        pose_axis_px=target_axis_px,
        anchor_label="orange",
        gripper_label="target grip",
    )
    live_card = _draw_card(
        "T5 Live Target Projection",
        [
            f"ref_frames={adapted_pose['reference_frame_indices']}",
            f"delta_x={np.round(np.asarray(adapted_pose['delta_x_m'], dtype=float), 4).tolist()} m",
            f"delta_theta={adapted_pose['delta_theta_deg']:.2f} deg  delta_h={adapted_pose['delta_h_m']:.4f} m",
            "yellow=orange anchor, red=adapted gripper, green=target x-axis",
        ],
        _fit_image(live_overlay, 1280, 720),
    )
    live_card_path = config.output_dir / "t5_live_target_projection.png"
    cv2.imwrite(str(live_card_path), live_card)

    overview_images = reference_cards + [live_card]
    overview_path = config.output_dir / "t5_validation_overview.png"
    _save_montage(overview_images, overview_path, "T5 Validation Overview", cols=2)

    payload = {
        "status": "ok",
        "reference_cards": reference_payload,
        "reference_montage_path": str(reference_montage_path),
        "live_target_projection_path": str(live_card_path),
        "overview_path": str(overview_path),
        "summary_path": str(config.t5_dir / "summary.json"),
        "handeye_status": {
            "path": str(config.handeye_result_path),
            "passed": handeye_payload.get("passed") if isinstance(handeye_payload, dict) else None,
            "accepted_without_passing_thresholds": (
                handeye_payload.get("accepted_without_passing_thresholds") if isinstance(handeye_payload, dict) else None
            ),
        },
        "reference_selection": summary.get("reference_selection"),
    }
    _save_json(config.output_dir / "index.json", payload)
    return payload


def main() -> int:
    config = _build_config(_parse_args())
    payload = export_t5_visuals(config)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
