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
    from lerobot.projects.vlbiman_sa.demo.io import load_frame_assets, load_frame_records
    from lerobot.projects.vlbiman_sa.skills import SkillBank
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[5]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from lerobot.projects.vlbiman_sa.demo.io import load_frame_assets, load_frame_records
    from lerobot.projects.vlbiman_sa.skills import SkillBank


@dataclass(slots=True)
class ExportConfig:
    session_dir: Path
    analysis_dir: Path


def _default_session_dir() -> Path:
    return Path("outputs/vlbiman_sa/recordings/one_shot_20260323T185847Z")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export self-check image cards for T3/T4 orange validation.")
    parser.add_argument("--session-dir", type=Path, default=_default_session_dir())
    parser.add_argument("--analysis-dir", type=Path, default=None)
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> ExportConfig:
    session_dir = args.session_dir
    analysis_dir = args.analysis_dir or (session_dir / "analysis")
    return ExportConfig(session_dir=session_dir, analysis_dir=analysis_dir)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _draw_lines(
    image: np.ndarray,
    title: str,
    lines: list[str],
    *,
    width: int = 1280,
    title_height: int = 48,
    line_height: int = 30,
) -> np.ndarray:
    canvas = np.full((title_height + line_height * len(lines), width, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, title, (24, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (20, 20, 20), 2, cv2.LINE_AA)
    for idx, line in enumerate(lines):
        y = title_height + idx * line_height - 6
        cv2.putText(canvas, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (40, 40, 40), 2, cv2.LINE_AA)
    return np.vstack([canvas, image])


def _save_montage(images: list[np.ndarray], output_path: Path, title: str) -> None:
    if not images:
        return
    card_h, card_w = images[0].shape[:2]
    cols = min(3, len(images))
    rows = int(math.ceil(len(images) / cols))
    grid = np.full((rows * card_h, cols * card_w, 3), 255, dtype=np.uint8)
    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        y0 = row * card_h
        x0 = col * card_w
        grid[y0 : y0 + card_h, x0 : x0 + card_w] = image
    header = np.full((60, grid.shape[1], 3), 250, dtype=np.uint8)
    cv2.putText(header, title, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.vstack([header, grid]))


def _representative_image(
    session_dir: Path,
    frame_index: int,
    overlay_dir: Path,
) -> np.ndarray:
    overlay_path = overlay_dir / f"frame_{frame_index:06d}.png"
    if overlay_path.exists():
        image = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
        if image is not None:
            return image
    records = load_frame_records(session_dir)
    color_rgb, _ = load_frame_assets(session_dir, records[frame_index])
    return cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)


def export_t3_visuals(config: ExportConfig, bank: SkillBank) -> dict[str, Any]:
    t3_dir = config.analysis_dir / "t3_skill_bank"
    overlay_dir = config.analysis_dir / "t4_vision" / "overlays"
    output_dir = t3_dir / "self_check_orange"
    card_dir = output_dir / "segment_cards"
    card_dir.mkdir(parents=True, exist_ok=True)

    cards: list[np.ndarray] = []
    card_payload = []
    for segment in bank.segments:
        raw = _representative_image(config.session_dir, segment.representative_frame, overlay_dir)
        fitted = _fit_image(raw, 1280, 720)
        lines = [
            f"segment={segment.segment_id} label={segment.label} invariance={segment.invariance}",
            f"frames={segment.start_frame}-{segment.end_frame} representative={segment.representative_frame} count={segment.frame_count}",
            f"orange self-check: representative frame with T4 overlay if available",
        ]
        card = _draw_lines(fitted, f"T3 Orange Self-Check | {segment.segment_id}", lines)
        card_path = card_dir / f"{segment.segment_id}.png"
        cv2.imwrite(str(card_path), card)
        cards.append(card)
        card_payload.append(
            {
                "segment_id": segment.segment_id,
                "representative_frame": segment.representative_frame,
                "label": segment.label,
                "invariance": segment.invariance,
                "card_path": str(card_path),
            }
        )

    montage_path = output_dir / "t3_orange_segment_montage.png"
    _save_montage(cards, montage_path, "T3 Orange Self-Check Montage")
    payload = {
        "status": "pass",
        "card_count": len(card_payload),
        "card_dir": str(card_dir),
        "montage_path": str(montage_path),
        "cards": card_payload,
    }
    (output_dir / "index.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _select_t4_frames(skill_bank: SkillBank, summary: dict[str, Any], frames: list[dict[str, Any]]) -> list[int]:
    selected = []
    for key in ("seed_detection",):
        seed = summary.get(key, {})
        if isinstance(seed, dict) and seed.get("frame_index") is not None:
            selected.append(int(seed["frame_index"]))
    if summary.get("first_stable_frame") is not None:
        selected.append(int(summary["first_stable_frame"]))
    selected.extend(int(segment.representative_frame) for segment in skill_bank.segments[:6])
    frame_indices = [int(item["frame_index"]) for item in frames]
    if frame_indices:
        selected.extend(
            [
                frame_indices[len(frame_indices) // 4],
                frame_indices[len(frame_indices) // 2],
                frame_indices[(3 * len(frame_indices)) // 4],
                frame_indices[-1],
            ]
        )
    deduped: list[int] = []
    seen: set[int] = set()
    valid = set(frame_indices)
    for frame_index in selected:
        if frame_index not in valid or frame_index in seen:
            continue
        seen.add(frame_index)
        deduped.append(frame_index)
    return deduped[:9]


def export_t4_visuals(config: ExportConfig, bank: SkillBank) -> dict[str, Any]:
    t4_dir = config.analysis_dir / "t4_vision"
    output_dir = t4_dir / "self_check_orange"
    card_dir = output_dir / "selected_frames"
    card_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_json(t4_dir / "summary.json")
    frames = _load_json(t4_dir / "frames.json")
    frame_map = {int(item["frame_index"]): item for item in frames}
    selected_frames = _select_t4_frames(bank, summary, frames)

    cards: list[np.ndarray] = []
    card_payload = []
    for frame_index in selected_frames:
        item = frame_map[frame_index]
        overlay_path = t4_dir / "overlays" / f"frame_{frame_index:06d}.png"
        overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
        if overlay is None:
            continue
        fitted = _fit_image(overlay, 1280, 720)
        tracking = item.get("tracking", {})
        anchor = item.get("anchor", {})
        lines = [
            f"frame={frame_index} stable={tracking.get('stable')} iou={tracking.get('temporal_iou', 0.0):.3f}",
            f"mask_area={item.get('mask_area_px')} depth_m={anchor.get('depth_m')} xyz={anchor.get('camera_xyz_m')}",
            f"orientation_deg={anchor.get('orientation_deg')} pos_var_mm2={tracking.get('position_variance_mm2')}",
        ]
        card = _draw_lines(fitted, f"T4 Orange Self-Check | frame {frame_index:06d}", lines)
        card_path = card_dir / f"frame_{frame_index:06d}.png"
        cv2.imwrite(str(card_path), card)
        cards.append(card)
        card_payload.append(
            {
                "frame_index": frame_index,
                "stable": tracking.get("stable"),
                "temporal_iou": tracking.get("temporal_iou"),
                "card_path": str(card_path),
            }
        )

    montage_path = output_dir / "t4_orange_tracking_montage.png"
    _save_montage(cards, montage_path, "T4 Orange Self-Check Montage")
    payload = {
        "status": "pass",
        "selected_frame_count": len(card_payload),
        "selected_frames": [item["frame_index"] for item in card_payload],
        "card_dir": str(card_dir),
        "montage_path": str(montage_path),
        "cards": card_payload,
    }
    (output_dir / "index.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> int:
    args = _parse_args()
    config = _build_config(args)
    bank = SkillBank.load(config.analysis_dir / "t3_skill_bank" / "skill_bank.json")
    t3_payload = export_t3_visuals(config, bank)
    t4_payload = export_t4_visuals(config, bank)
    root_payload = {
        "status": "pass",
        "session_dir": str(config.session_dir),
        "analysis_dir": str(config.analysis_dir),
        "t3": t3_payload,
        "t4": t4_payload,
    }
    (config.analysis_dir / "orange_self_check_visuals.json").write_text(
        json.dumps(root_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
