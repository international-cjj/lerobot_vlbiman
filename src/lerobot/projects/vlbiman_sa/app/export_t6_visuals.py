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


@dataclass(slots=True)
class ExportT6Config:
    session_dir: Path
    analysis_dir: Path
    t6_dir: Path
    t5_dir: Path
    output_dir: Path


def _default_session_dir() -> Path:
    return Path("outputs/vlbiman_sa/recordings/one_shot_20260323T185847Z")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export T6 trajectory validation visuals.")
    parser.add_argument("--session-dir", type=Path, default=_default_session_dir())
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--t6-dir", type=Path, default=None)
    parser.add_argument("--t5-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> ExportT6Config:
    analysis_dir = args.analysis_dir or (args.session_dir / "analysis")
    t6_dir = args.t6_dir or (analysis_dir / "t6_trajectory")
    t5_dir = args.t5_dir or (analysis_dir / "t5_pose")
    output_dir = args.output_dir or (t6_dir / "visuals")
    return ExportT6Config(
        session_dir=args.session_dir,
        analysis_dir=analysis_dir,
        t6_dir=t6_dir,
        t5_dir=t5_dir,
        output_dir=output_dir,
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def _fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    scale = min(width / max(src_w, 1), height / max(src_h, 1))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _draw_card(title: str, lines: list[str], image: np.ndarray) -> np.ndarray:
    header_h = 148
    canvas = np.full((header_h + image.shape[0], image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(canvas, title, (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    for idx, line in enumerate(lines):
        y = 76 + idx * 28
        cv2.putText(canvas, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (45, 45, 45), 2, cv2.LINE_AA)
    canvas[header_h:, :] = image
    return canvas


def _save_montage(images: list[np.ndarray], output_path: Path, title: str, cols: int = 3) -> None:
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


def _segment_color(invariance: str) -> tuple[int, int, int]:
    return (60, 130, 255) if invariance == "var" else (255, 150, 70)


def _segment_cards(
    config: ExportT6Config,
    skill_bank: dict[str, Any],
    trajectory_points: list[dict[str, Any]],
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    points_by_frame = {int(item["frame_index"]): item for item in trajectory_points if item["frame_index"] is not None}
    overlay_dir = config.analysis_dir / "t4_vision" / "overlays"
    rep_dir = config.analysis_dir / "t3_skill_bank" / "representatives"

    cards: list[np.ndarray] = []
    payload: list[dict[str, Any]] = []
    for segment in skill_bank["segments"]:
        representative_frame = int(segment["representative_frame"])
        overlay_path = overlay_dir / f"frame_{representative_frame:06d}.png"
        image_path = overlay_path if overlay_path.exists() else rep_dir / f"{segment['segment_id']}.png"
        raw = _read_image(image_path)
        point = points_by_frame.get(representative_frame)
        color = _segment_color(str(segment["invariance"]))
        image = raw.copy()
        cv2.rectangle(image, (16, 16), (image.shape[1] - 16, image.shape[0] - 16), color, 4)
        fitted = _fit_image(image, 960, 540)
        lines = [
            f"segment={segment['segment_id']} label={segment['label']} invariance={segment['invariance']}",
            f"frames={segment['start_frame']}-{segment['end_frame']} representative={representative_frame}",
            (
                f"traj_idx={point['trajectory_index']} source={point['source']} "
                f"trans_err={point['translation_error_mm']:.2f}mm rot_err={point['rotation_error_deg']:.2f}deg"
                if point and point.get("translation_error_mm") is not None
                else f"traj_idx={point['trajectory_index']} source={point['source']}" if point else "traj mapping unavailable"
            ),
            f"green/orange border: {'adaptive var' if segment['invariance']=='var' else 'demo reuse inv'} segment",
        ]
        card = _draw_card(f"T6 Segment Keyframe | {segment['segment_id']}", lines, fitted)
        cards.append(card)
        payload.append(
            {
                "segment_id": segment["segment_id"],
                "representative_frame": representative_frame,
                "card_source_path": str(image_path),
            }
        )
    return cards, payload


def _trajectory_plot(
    trajectory_points: list[dict[str, Any]],
    joint_keys: list[str],
    skill_bank: dict[str, Any],
    summary: dict[str, Any],
) -> np.ndarray:
    width = 1800
    height = 1200
    left = 120
    right = 40
    top = 120
    bottom = 80
    band_h = 56
    plot_h = height - top - bottom - band_h
    panel_h = plot_h // len(joint_keys)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    cv2.putText(canvas, "T6 Joint Trajectory Validation", (32, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        (
            f"points={summary['point_count']}  ik_success={summary['ik_success_count']}/{summary['ik_target_frame_count']} "
            f"({summary['ik_success_rate']:.3f})  max_step={summary['max_joint_step_rad_inf']:.4f}rad"
        ),
        (32, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (50, 50, 50),
        2,
        cv2.LINE_AA,
    )

    total_points = max(len(trajectory_points), 1)
    segment_by_id = {segment["segment_id"]: segment for segment in skill_bank["segments"]}
    segment_ranges: list[tuple[int, int, dict[str, Any]]] = []
    current_id = None
    start_idx = 0
    for idx, point in enumerate(trajectory_points):
        if point["segment_id"] != current_id:
            if current_id is not None:
                segment_ranges.append((start_idx, idx - 1, segment_by_id[current_id]))
            current_id = point["segment_id"]
            start_idx = idx
    if current_id is not None:
        segment_ranges.append((start_idx, len(trajectory_points) - 1, segment_by_id[current_id]))

    for start_idx, end_idx, segment in segment_ranges:
        x0 = left + int((width - left - right) * (start_idx / max(total_points - 1, 1)))
        x1 = left + int((width - left - right) * (end_idx / max(total_points - 1, 1)))
        color = _segment_color(str(segment["invariance"]))
        fill = tuple(int(channel * 0.18 + 255 * 0.82) for channel in color)
        cv2.rectangle(canvas, (x0, top), (x1, top + band_h), fill, -1)
        cv2.rectangle(canvas, (x0, top), (x1, top + band_h), color, 2)
        cv2.putText(
            canvas,
            f"{segment['segment_id']} {segment['label']}",
            (x0 + 6, top + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

    joint_values = np.asarray([point["joint_positions"] for point in trajectory_points], dtype=float)
    colors = [
        (235, 99, 71),
        (66, 133, 244),
        (52, 168, 83),
        (251, 188, 5),
        (171, 71, 188),
        (0, 172, 193),
    ]
    representative_indices = sorted(
        point["trajectory_index"]
        for point in trajectory_points
        if point["frame_index"] is not None and point["frame_index"] in {int(seg["representative_frame"]) for seg in skill_bank["segments"]}
    )

    for joint_idx, joint_key in enumerate(joint_keys):
        y0 = top + band_h + joint_idx * panel_h
        y1 = top + band_h + (joint_idx + 1) * panel_h
        cv2.rectangle(canvas, (left, y0), (width - right, y1), (245, 245, 245), -1)
        cv2.rectangle(canvas, (left, y0), (width - right, y1), (220, 220, 220), 1)
        values = joint_values[:, joint_idx]
        vmin = float(values.min())
        vmax = float(values.max())
        if abs(vmax - vmin) < 1e-9:
            vmax = vmin + 1.0

        for tick_idx in range(5):
            y = y0 + int((panel_h - 20) * tick_idx / 4) + 10
            cv2.line(canvas, (left, y), (width - right, y), (232, 232, 232), 1)
        for rep_idx in representative_indices:
            x = left + int((width - left - right) * (rep_idx / max(total_points - 1, 1)))
            cv2.line(canvas, (x, y0), (x, y1), (180, 180, 180), 1, cv2.LINE_AA)

        points_xy = []
        for index, value in enumerate(values):
            x = left + int((width - left - right) * (index / max(total_points - 1, 1)))
            alpha = (float(value) - vmin) / (vmax - vmin)
            y = y1 - 10 - int(alpha * (panel_h - 20))
            points_xy.append((x, y))
        cv2.polylines(canvas, [np.asarray(points_xy, dtype=np.int32)], False, colors[joint_idx % len(colors)], 2, cv2.LINE_AA)
        cv2.putText(canvas, joint_key, (24, y0 + panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[joint_idx % len(colors)], 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{vmax:.2f}", (width - right + 4, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{vmin:.2f}", (width - right + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)

    cv2.putText(canvas, "trajectory index", (width // 2 - 80, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2, cv2.LINE_AA)
    return canvas


def _combine_overview(segment_montage: np.ndarray, trajectory_plot: np.ndarray, t5_projection: np.ndarray | None) -> np.ndarray:
    target_width = 1700
    top = _fit_image(segment_montage, target_width, 900)
    bottom_left = _fit_image(trajectory_plot, target_width, 850)
    if t5_projection is not None:
        bottom_right = _fit_image(t5_projection, 900, 850)
        gap = 24
        row = np.full((max(bottom_left.shape[0], bottom_right.shape[0]), bottom_left.shape[1] + gap + bottom_right.shape[1], 3), 255, dtype=np.uint8)
        row[: bottom_left.shape[0], : bottom_left.shape[1]] = bottom_left
        row[: bottom_right.shape[0], bottom_left.shape[1] + gap :] = bottom_right
    else:
        row = bottom_left
    width = max(top.shape[1], row.shape[1])
    overview = np.full((96 + top.shape[0] + 24 + row.shape[0], width, 3), 250, dtype=np.uint8)
    cv2.putText(overview, "T6 Validation Overview", (24, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (20, 20, 20), 2, cv2.LINE_AA)
    y = 96
    overview[y : y + top.shape[0], : top.shape[1]] = top
    y += top.shape[0] + 24
    overview[y : y + row.shape[0], : row.shape[1]] = row
    return overview


def export_t6_visuals(config: ExportT6Config) -> dict[str, Any]:
    skill_bank = _load_json(config.analysis_dir / "t3_skill_bank" / "skill_bank.json")
    t6_points_payload = _load_json(config.t6_dir / "trajectory_points.json")
    t6_summary = _load_json(config.t6_dir / "summary.json")
    trajectory_points = t6_points_payload["points"]

    cards, card_payload = _segment_cards(config, skill_bank, trajectory_points)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    segment_montage_path = config.output_dir / "t6_segment_keyframes.png"
    _save_montage(cards, segment_montage_path, "T6 Segment Keyframes", cols=3)
    segment_montage = _read_image(segment_montage_path)

    trajectory_plot = _trajectory_plot(trajectory_points, t6_points_payload["joint_keys"], skill_bank, t6_summary)
    trajectory_plot_path = config.output_dir / "t6_joint_trajectory.png"
    cv2.imwrite(str(trajectory_plot_path), trajectory_plot)

    t5_projection_path = config.t5_dir / "visuals" / "t5_live_target_projection.png"
    t5_projection = _read_image(t5_projection_path) if t5_projection_path.exists() else None
    overview = _combine_overview(segment_montage, trajectory_plot, t5_projection)
    overview_path = config.output_dir / "t6_validation_overview.png"
    cv2.imwrite(str(overview_path), overview)

    payload = {
        "status": "ok",
        "segment_keyframe_montage_path": str(segment_montage_path),
        "trajectory_plot_path": str(trajectory_plot_path),
        "overview_path": str(overview_path),
        "segment_cards": card_payload,
        "t6_summary_path": str(config.t6_dir / "summary.json"),
        "t6_summary": t6_summary,
        "t5_projection_path": str(t5_projection_path) if t5_projection is not None else None,
    }
    _save_json(config.output_dir / "index.json", payload)
    return payload


def main() -> int:
    config = _build_config(_parse_args())
    payload = export_t6_visuals(config)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
