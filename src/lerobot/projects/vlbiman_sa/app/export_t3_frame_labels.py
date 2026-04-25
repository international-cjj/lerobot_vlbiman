#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
    skill_bank_path: Path
    output_dir: Path
    write_images: bool


def _default_session_dir() -> Path:
    return Path("outputs/vlbiman_sa/recordings/one_shot_full_20260327T231705")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a per-frame T3 state dataset.")
    parser.add_argument("--session-dir", type=Path, default=_default_session_dir())
    parser.add_argument("--skill-bank-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-images", action="store_true", help="Only export CSV/JSONL metadata.")
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> ExportConfig:
    session_dir = args.session_dir
    skill_bank_path = args.skill_bank_path or (session_dir / "analysis" / "t3_skill_bank" / "skill_bank.json")
    output_dir = args.output_dir or (session_dir / "analysis" / "t3_frame_dataset")
    return ExportConfig(
        session_dir=session_dir,
        skill_bank_path=skill_bank_path,
        output_dir=output_dir,
        write_images=not bool(args.no_images),
    )


def _segment_lookup(bank: SkillBank) -> dict[int, Any]:
    lookup: dict[int, Any] = {}
    for segment in bank.segments:
        for frame_index in range(int(segment.start_frame), int(segment.end_frame) + 1):
            lookup[frame_index] = segment
    return lookup


def _color_for_label(label: str) -> tuple[int, int, int]:
    palette = {
        "transfer": (51, 153, 255),
        "approach": (255, 200, 70),
        "gripper_open": (80, 200, 120),
        "gripper_close": (70, 80, 230),
        "retreat": (170, 110, 255),
        "stabilize": (160, 160, 160),
    }
    return palette.get(label, (200, 200, 200))


def _annotate_frame(image_bgr: np.ndarray, row: dict[str, Any]) -> np.ndarray:
    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]
    panel_h = 132
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (18, 18, 18), thickness=-1)
    cv2.addWeighted(overlay, 0.74, canvas, 0.26, 0.0, canvas)

    accent = _color_for_label(str(row["state_label"]))
    cv2.rectangle(canvas, (0, 0), (14, panel_h), accent, thickness=-1)

    display_state = str(row.get("display_state") or row["state_label"])
    title = f"frame {int(row['frame_index']):06d} | {row['segment_id']} | {display_state}"
    meta_1 = (
        f"time={float(row['relative_time_s']):.3f}s  canonical={row['state_label']}  "
        f"invariance={row['invariance']}  representative={row['is_representative']}"
    )
    meta_2 = (
        f"segment_frames={row['segment_start_frame']}-{row['segment_end_frame']}  "
        f"segment_time={float(row['segment_start_time_s']):.3f}-{float(row['segment_end_time_s']):.3f}s"
    )
    cv2.putText(canvas, title, (28, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(canvas, meta_1, (28, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(canvas, meta_2, (28, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (235, 235, 235), 2, cv2.LINE_AA)
    return canvas


def export_dataset(config: ExportConfig) -> dict[str, Any]:
    if not config.session_dir.exists():
        raise FileNotFoundError(f"Session dir not found: {config.session_dir}")
    if not config.skill_bank_path.exists():
        raise FileNotFoundError(f"Skill bank not found: {config.skill_bank_path}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    labeled_dir = config.output_dir / "labeled_frames"
    if config.write_images:
        labeled_dir.mkdir(parents=True, exist_ok=True)

    bank = SkillBank.load(config.skill_bank_path)
    records = load_frame_records(config.session_dir)
    lookup = _segment_lookup(bank)

    csv_path = config.output_dir / "frame_labels.csv"
    jsonl_path = config.output_dir / "frame_labels.jsonl"
    index_path = config.output_dir / "index.json"

    fieldnames = [
        "frame_index",
        "relative_time_s",
        "rgb_path",
        "annotated_rgb_path",
        "segment_id",
        "display_state",
        "state_label",
        "invariance",
        "segment_start_frame",
        "segment_end_frame",
        "segment_start_time_s",
        "segment_end_time_s",
        "segment_frame_count",
        "segment_representative_frame",
        "is_representative",
    ]

    rows: list[dict[str, Any]] = []
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file, jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            frame_index = int(record.frame_index)
            segment = lookup.get(frame_index)
            if segment is None:
                continue

            annotated_path = ""
            display_state = str(segment.metrics.get("semantic_state", segment.label))
            if config.write_images:
                color_rgb, _ = load_frame_assets(config.session_dir, record)
                image_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
                row_preview = {
                    "frame_index": frame_index,
                    "relative_time_s": float(record.relative_time_s),
                    "segment_id": str(segment.segment_id),
                    "display_state": display_state,
                    "state_label": str(segment.label),
                    "invariance": str(segment.invariance),
                    "segment_start_frame": int(segment.start_frame),
                    "segment_end_frame": int(segment.end_frame),
                    "segment_start_time_s": float(segment.start_time_s),
                    "segment_end_time_s": float(segment.end_time_s),
                    "is_representative": frame_index == int(segment.representative_frame),
                }
                annotated = _annotate_frame(image_bgr, row_preview)
                annotated_path = str(labeled_dir / f"frame_{frame_index:06d}.png")
                cv2.imwrite(annotated_path, annotated)

            row = {
                "frame_index": frame_index,
                "relative_time_s": float(record.relative_time_s),
                "rgb_path": str(config.session_dir / record.color_path),
                "annotated_rgb_path": annotated_path,
                "segment_id": str(segment.segment_id),
                "display_state": display_state,
                "state_label": str(segment.label),
                "invariance": str(segment.invariance),
                "segment_start_frame": int(segment.start_frame),
                "segment_end_frame": int(segment.end_frame),
                "segment_start_time_s": float(segment.start_time_s),
                "segment_end_time_s": float(segment.end_time_s),
                "segment_frame_count": int(segment.frame_count),
                "segment_representative_frame": int(segment.representative_frame),
                "is_representative": frame_index == int(segment.representative_frame),
            }
            writer.writerow(row)
            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)

    state_counts: dict[str, int] = {}
    for row in rows:
        key = str(row["state_label"])
        state_counts[key] = state_counts.get(key, 0) + 1

    payload = {
        "status": "pass",
        "session_dir": str(config.session_dir),
        "skill_bank_path": str(config.skill_bank_path),
        "frame_count": len(rows),
        "segment_count": len(bank.segments),
        "states": state_counts,
        "csv_path": str(csv_path),
        "jsonl_path": str(jsonl_path),
        "labeled_frame_dir": str(labeled_dir) if config.write_images else None,
    }
    index_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    args = _parse_args()
    config = _build_config(args)
    payload = export_dataset(config)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
