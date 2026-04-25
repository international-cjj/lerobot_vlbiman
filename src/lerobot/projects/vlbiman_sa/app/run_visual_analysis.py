#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
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
    for path in (repo_root / "src", repo_root):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo_root


REPO_ROOT = _bootstrap_paths()

from lerobot.projects.vlbiman_sa.runtime_env import strip_user_site_packages

strip_user_site_packages()

from lerobot.projects.vlbiman_sa.demo.io import load_frame_records
from lerobot.projects.vlbiman_sa.skills import SkillBank
from lerobot.projects.vlbiman_sa.vision import (
    AnchorEstimator,
    AnchorEstimatorConfig,
    CameraIntrinsics,
    MaskTracker,
    MaskTrackerConfig,
    OrientationMomentsEstimator,
    VLMObjectSegmentor,
    VLMObjectSegmentorConfig,
)


@dataclass(slots=True)
class VisionConfig:
    session_dir: Path
    skill_bank_path: Path
    output_dir: Path
    intrinsics_path: Path
    target_phrase: str
    task_prompt: str
    frame_stride: int = 1
    use_var_segments_only: bool = False


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "vision_analysis.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Florence-2 + SAM2 offline segmentation and anchor estimation.")
    parser.add_argument("--config", type=Path, default=_default_config_path(), help="Path to vision_analysis.yaml.")
    parser.add_argument("--session-dir", type=Path, default=None, help="Override the recording session directory.")
    parser.add_argument("--skill-bank-path", type=Path, default=None, help="Override the skill bank JSON path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override the vision output directory.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping, got {type(payload).__name__}.")
    return payload


def _build_config(payload: dict[str, Any], args: argparse.Namespace) -> VisionConfig:
    session_dir = args.session_dir or Path(payload["session_dir"])
    return VisionConfig(
        session_dir=session_dir,
        skill_bank_path=args.skill_bank_path or Path(payload["skill_bank_path"]),
        output_dir=args.output_dir or Path(payload.get("output_dir", session_dir / "analysis" / "t4_vision")),
        intrinsics_path=Path(payload["intrinsics_path"]),
        target_phrase=str(payload.get("target_phrase", "orange")),
        task_prompt=str(payload.get("task_prompt", "orange")),
        frame_stride=int(payload.get("frame_stride", 1)),
        use_var_segments_only=bool(payload.get("use_var_segments_only", False)),
    )


def _selected_frames(bank: SkillBank, frame_stride: int, use_var_segments_only: bool, frame_count: int) -> list[int]:
    if not use_var_segments_only:
        return list(range(0, frame_count, max(1, frame_stride)))

    selected: list[int] = []
    for segment in bank.segments:
        if segment.invariance != "var":
            continue
        selected.extend(range(segment.start_frame, segment.end_frame + 1, max(1, frame_stride)))
    return sorted(set(selected))


def _build_self_check(
    *,
    config: VisionConfig,
    tracker_config: MaskTrackerConfig,
    segmentor_config: VLMObjectSegmentorConfig,
    summary: dict[str, Any],
) -> dict[str, Any]:
    seed_label = str(summary["seed_detection"].get("label", ""))
    target_phrase = config.target_phrase.lower().strip()
    seed_phrase_match = target_phrase in seed_label.lower() or seed_label.lower() in target_phrase
    checks = {
        "seed_phrase_match": seed_phrase_match,
        "mask_valid_ratio_ok": float(summary["mask_valid_ratio"]) >= 0.95,
        "anchor_valid_ratio_ok": float(summary["anchor_valid_ratio"]) >= 0.95,
        "temporal_iou_ok": float(summary["mean_temporal_iou"]) >= 0.75,
        "stable_window_found": summary["first_stable_frame"] is not None,
    }
    status = "pass" if all(checks.values()) else "warn"
    return {
        "status": status,
        "validation_target": config.target_phrase,
        "pipeline": "florence2_plus_sam2",
        "models": {
            "florence_model_id": segmentor_config.florence_model_id,
            "sam2_model_id": segmentor_config.sam2_model_id,
        },
        "stability_window_size": tracker_config.stability_window_size,
        "position_variance_threshold_mm2": tracker_config.position_variance_threshold_mm2,
        "orientation_variance_threshold_deg2": tracker_config.orientation_variance_threshold_deg2,
        "checks": checks,
        "metrics": {
            "mask_valid_ratio": summary["mask_valid_ratio"],
            "anchor_valid_ratio": summary["anchor_valid_ratio"],
            "mean_temporal_iou": summary["mean_temporal_iou"],
            "stable_ratio": summary["stable_ratio"],
            "first_stable_frame": summary["first_stable_frame"],
            "min_position_variance_mm2": summary["min_position_variance_mm2"],
            "min_orientation_variance_deg2": summary["min_orientation_variance_deg2"],
        },
    }


def _overlay_mask(
    color_rgb: np.ndarray,
    mask: np.ndarray,
    bbox_xyxy: list[int],
    centroid_px: list[float] | None,
    contact_px: list[float] | None,
    text: str,
    stable: bool,
) -> np.ndarray:
    overlay = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    tint = overlay.copy()
    tint[mask > 0] = (20, 80, 240)
    overlay = cv2.addWeighted(overlay, 0.72, tint, 0.28, 0.0)
    x0, y0, x1, y1 = bbox_xyxy
    color = (0, 255, 0) if stable else (0, 140, 255)
    if x1 > x0 and y1 > y0:
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
    if centroid_px is not None:
        cv2.circle(overlay, (int(round(centroid_px[0])), int(round(centroid_px[1]))), 5, (255, 255, 0), -1)
    if contact_px is not None:
        cv2.circle(overlay, (int(round(contact_px[0])), int(round(contact_px[1]))), 6, (0, 255, 255), -1)
    cv2.putText(overlay, text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        overlay,
        f"stable={'yes' if stable else 'no'}",
        (24, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    return overlay


def run_visual_analysis_pipeline(config: VisionConfig, payload: dict[str, Any]) -> dict[str, Any]:
    records = load_frame_records(config.session_dir)
    bank = SkillBank.load(config.skill_bank_path)
    frame_indices = _selected_frames(bank, config.frame_stride, config.use_var_segments_only, len(records))
    if not frame_indices:
        raise ValueError("No frames selected for T4 visual analysis.")

    segmentor_config = VLMObjectSegmentorConfig(**dict(payload.get("segmentor", {})))
    tracker_config = MaskTrackerConfig(**dict(payload.get("tracker", {})))
    anchor_config = AnchorEstimatorConfig(**dict(payload.get("anchor", {})))

    config.output_dir.mkdir(parents=True, exist_ok=True)
    segmentor = VLMObjectSegmentor(segmentor_config)
    run_result = segmentor.segment_video(
        session_dir=config.session_dir,
        records=records,
        frame_indices=frame_indices,
        output_dir=config.output_dir,
        target_phrase=config.target_phrase,
    )

    orientation_estimator = OrientationMomentsEstimator()
    anchor_estimator = AnchorEstimator(CameraIntrinsics.from_json(config.intrinsics_path), anchor_config)

    anchor_payload = []
    anchor_by_frame: dict[int, dict[str, Any]] = {}
    anchor_payload_by_frame: dict[int, dict[str, Any]] = {}
    per_frame_assets: list[tuple[Any, np.ndarray]] = []
    for frame_result, (frame_index, mask) in zip(run_result.frame_results, run_result.masks):
        orientation = orientation_estimator.estimate(mask)
        anchor = anchor_estimator.estimate(
            frame_index=frame_index,
            mask=mask,
            depth_map=np.load(config.session_dir / records[frame_index].depth_path),
            orientation_deg=orientation.angle_deg,
            score=frame_result.score,
        )
        anchor_dict = anchor.to_dict()
        anchor_by_frame[frame_index] = anchor_dict
        payload_entry = {**anchor_dict, "orientation": orientation.to_dict()}
        anchor_payload.append(payload_entry)
        anchor_payload_by_frame[frame_index] = payload_entry
        color_rgb = cv2.cvtColor(
            cv2.imread(str(config.session_dir / records[frame_index].color_path), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )
        per_frame_assets.append((frame_result, color_rgb))

    tracker = MaskTracker(tracker_config)
    tracked_masks, tracked_meta = tracker.track(run_result.masks, anchors=anchor_by_frame)
    tracked_by_frame = {frame_index: mask for frame_index, mask in tracked_masks}
    meta_by_frame = {item.frame_index: item for item in tracked_meta}

    mask_dir = config.output_dir / "masks"
    overlay_dir = config.output_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    frame_payload = []
    for frame_result, color_rgb in per_frame_assets:
        tracked_mask = tracked_by_frame[frame_result.frame_index]
        anchor = anchor_payload_by_frame[frame_result.frame_index]
        tracking = meta_by_frame[frame_result.frame_index].to_dict()
        frame_payload.append(
            {
                **frame_result.to_dict(),
                "tracking": tracking,
                "anchor": anchor,
            }
        )
        cv2.imwrite(str(mask_dir / f"frame_{frame_result.frame_index:06d}.png"), tracked_mask)
        overlay = _overlay_mask(
            color_rgb=color_rgb,
            mask=tracked_mask,
            bbox_xyxy=anchor["bbox_xyxy"],
            centroid_px=anchor["centroid_px"],
            contact_px=anchor["contact_px"],
            text=f"frame={frame_result.frame_index} phrase={config.target_phrase}",
            stable=bool(tracking["stable"]),
        )
        cv2.imwrite(str(overlay_dir / f"frame_{frame_result.frame_index:06d}.png"), overlay)

    stable_frames = [item.frame_index for item in tracked_meta if item.stable]
    mean_temporal_iou = float(
        np.mean([item.temporal_iou for item in tracked_meta[1:]]) if len(tracked_meta) > 1 else 1.0
    )
    anchor_valid_ratio = float(np.mean([1.0 if item["camera_xyz_m"] is not None else 0.0 for item in anchor_payload]))
    mask_valid_ratio = float(np.mean([1.0 if item["mask_area_px"] > 0 else 0.0 for item in frame_payload]))
    stable_ratio = float(len(stable_frames) / max(len(tracked_meta), 1))
    orientation_std_values = [item.orientation_std_deg for item in tracked_meta if item.orientation_std_deg is not None]
    orientation_variance_values = [
        item.orientation_variance_deg2 for item in tracked_meta if item.orientation_variance_deg2 is not None
    ]
    position_std_values = [item.position_std_mm for item in tracked_meta if item.position_std_mm is not None]
    position_variance_values = [
        item.position_variance_mm2 for item in tracked_meta if item.position_variance_mm2 is not None
    ]

    summary = {
        "session_dir": str(config.session_dir),
        "skill_bank_path": str(config.skill_bank_path),
        "target_phrase": config.target_phrase,
        "task_prompt": config.task_prompt,
        "florence_model_id": segmentor_config.florence_model_id,
        "sam2_model_id": segmentor_config.sam2_model_id,
        "seed_detection": run_result.seed_detection.to_dict(),
        "selected_frames": len(frame_indices),
        "mask_valid_ratio": mask_valid_ratio,
        "anchor_valid_ratio": anchor_valid_ratio,
        "mean_temporal_iou": mean_temporal_iou,
        "stable_frame_count": len(stable_frames),
        "stable_ratio": stable_ratio,
        "min_position_std_mm": float(min(position_std_values)) if position_std_values else None,
        "min_position_variance_mm2": float(min(position_variance_values)) if position_variance_values else None,
        "min_orientation_std_deg": float(min(orientation_std_values)) if orientation_std_values else None,
        "min_orientation_variance_deg2": float(min(orientation_variance_values)) if orientation_variance_values else None,
        "first_stable_frame": stable_frames[0] if stable_frames else None,
        "status": (
            "pass"
            if mask_valid_ratio >= 0.95 and anchor_valid_ratio >= 0.95 and mean_temporal_iou >= 0.75
            else "warn"
        ),
        "output_dir": str(config.output_dir),
        "mask_dir": str(mask_dir),
        "overlay_dir": str(overlay_dir),
        "sam2_video_frame_dir": str(run_result.video_frame_dir),
        "florence_seed_detection_path": str(run_result.detection_log_path),
        "self_check_path": str(config.output_dir / "self_check.json"),
    }
    self_check = _build_self_check(
        config=config,
        tracker_config=tracker_config,
        segmentor_config=segmentor_config,
        summary=summary,
    )

    (config.output_dir / "anchors.json").write_text(json.dumps(anchor_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (config.output_dir / "frames.json").write_text(json.dumps(frame_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (config.output_dir / "self_check.json").write_text(json.dumps(self_check, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)

    payload = _load_yaml(args.config)
    config = _build_config(payload, args)
    summary = run_visual_analysis_pipeline(config, payload)
    logging.info("Vision analysis output: %s", summary["output_dir"])
    logging.info("Vision summary: %s", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
