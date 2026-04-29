from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.projects.vlbiman_sa.demo.io import load_frame_assets, load_frame_records
from lerobot.projects.vlbiman_sa.demo.schema import FrameRecord

from .invariance_classifier import InvarianceClassifier, InvarianceClassifierConfig
from .keypose_segmenter import KeyposeSegmenter, SegmenterConfig


@dataclass(slots=True)
class SkillSegment:
    segment_id: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    representative_frame: int
    label: str
    invariance: str = "unknown"
    confidence: float = 0.0
    frame_count: int = 0
    gripper_start: float = 0.0
    gripper_end: float = 0.0
    joint_keys: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    boundary_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SkillBank:
    session_dir: Path
    output_dir: Path
    frame_count: int
    joint_keys: list[str]
    segments: list[SkillSegment]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_dir": str(self.session_dir),
            "output_dir": str(self.output_dir),
            "frame_count": self.frame_count,
            "joint_keys": list(self.joint_keys),
            "segments": [segment.to_dict() for segment in self.segments],
            "summary": self.summary,
        }

    def save(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        skill_bank_path = self.output_dir / "skill_bank.json"
        skill_bank_path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return skill_bank_path

    @classmethod
    def load(cls, path: Path) -> "SkillBank":
        payload = json.loads(path.read_text(encoding="utf-8"))
        segments = [SkillSegment(**item) for item in payload.get("segments", [])]
        return cls(
            session_dir=Path(payload["session_dir"]),
            output_dir=Path(payload["output_dir"]),
            frame_count=int(payload["frame_count"]),
            joint_keys=list(payload.get("joint_keys", [])),
            segments=segments,
            summary=dict(payload.get("summary", {})),
        )


@dataclass(slots=True)
class SkillBankRunResult:
    skill_bank_path: Path
    summary_path: Path
    self_check_path: Path
    preview_dir: Path
    segment_npz_dir: Path
    bank: SkillBank


def build_skill_bank(
    session_dir: Path,
    output_dir: Path | None = None,
    segmenter_config: SegmenterConfig | None = None,
    classifier_config: InvarianceClassifierConfig | None = None,
) -> SkillBankRunResult:
    records = load_frame_records(session_dir)
    if not records:
        raise ValueError(f"No recorded frames found in {session_dir}")

    output_dir = output_dir or session_dir / "analysis" / "t3_skill_bank"
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "representatives"
    preview_dir.mkdir(parents=True, exist_ok=True)
    segment_npz_dir = output_dir / "segments"
    segment_npz_dir.mkdir(parents=True, exist_ok=True)

    segmenter = KeyposeSegmenter(segmenter_config)
    features = segmenter.extract_features(records)
    boundaries, boundary_map = segmenter.compute_boundaries(features)
    segments = _build_segments(records, features, boundaries, boundary_map)
    segments = _merge_segments(records, features, segments, segmenter.config)

    classifier = InvarianceClassifier(classifier_config)
    segments = classifier.classify(session_dir, records, segments)

    summary = _build_summary(records, segments)
    summary["session_dir"] = str(session_dir)
    bank = SkillBank(
        session_dir=session_dir,
        output_dir=output_dir,
        frame_count=len(records),
        joint_keys=features.joint_keys,
        segments=segments,
        summary=summary,
    )

    for segment in segments:
        _write_segment_npz(segment_npz_dir, segment, records, features)
        _write_preview(preview_dir, segment, session_dir, records)

    self_check_payload = _run_self_check(bank)
    summary["self_check_status"] = self_check_payload["status"]
    bank.summary = summary
    skill_bank_path = bank.save()

    self_check_path = output_dir / "self_check.json"
    self_check_path.write_text(json.dumps(self_check_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return SkillBankRunResult(
        skill_bank_path=skill_bank_path,
        summary_path=summary_path,
        self_check_path=self_check_path,
        preview_dir=preview_dir,
        segment_npz_dir=segment_npz_dir,
        bank=bank,
    )


def _build_segments(
    records: list[FrameRecord],
    features: "SegmenterFeatures",
    boundaries: list[int],
    boundary_map: dict[int, list[str]],
) -> list[SkillSegment]:
    segments: list[SkillSegment] = []
    frame_count = len(records)
    if len(boundaries) < 2:
        boundaries = [0, frame_count - 1]

    for index, start_frame in enumerate(boundaries[:-1]):
        next_start = boundaries[index + 1]
        end_frame = frame_count - 1 if index == len(boundaries) - 2 else max(start_frame, next_start - 1)
        reasons = list(dict.fromkeys(boundary_map.get(start_frame, []) + boundary_map.get(next_start, [])))
        segments.append(
            _segment_from_range(
                records=records,
                features=features,
                start_frame=int(start_frame),
                end_frame=int(end_frame),
                segment_id=f"skill_{index:03d}",
                boundary_reasons=reasons,
            )
        )
    return segments


def _merge_segments(
    records: list[FrameRecord],
    features: "SegmenterFeatures",
    segments: list[SkillSegment],
    config: SegmenterConfig,
) -> list[SkillSegment]:
    mergeable_labels = {str(label) for label in config.mergeable_labels}
    if not segments or not mergeable_labels:
        return segments

    current = list(segments)
    merged_any = True
    while merged_any:
        merged_any = False
        new_segments: list[SkillSegment] = []
        index = 0
        while index < len(current):
            left = current[index]
            if (
                bool(config.merge_short_bridge_segments)
                and index + 2 < len(current)
                and _should_merge_bridge(current[index], current[index + 1], current[index + 2], config, mergeable_labels)
            ):
                right = current[index + 2]
                new_segments.append(
                    _segment_from_range(
                        records=records,
                        features=features,
                        start_frame=left.start_frame,
                        end_frame=right.end_frame,
                        segment_id="",
                        boundary_reasons=_combine_boundary_reasons(current[index : index + 3]),
                    )
                )
                index += 3
                merged_any = True
                continue

            if (
                bool(config.merge_same_label_segments)
                and index + 1 < len(current)
                and _should_merge_pair(current[index], current[index + 1], config, mergeable_labels)
            ):
                right = current[index + 1]
                new_segments.append(
                    _segment_from_range(
                        records=records,
                        features=features,
                        start_frame=left.start_frame,
                        end_frame=right.end_frame,
                        segment_id="",
                        boundary_reasons=_combine_boundary_reasons(current[index : index + 2]),
                    )
                )
                index += 2
                merged_any = True
                continue

            new_segments.append(left)
            index += 1
        current = _renumber_segments(new_segments)
    return current


def _should_merge_pair(
    left: SkillSegment,
    right: SkillSegment,
    config: SegmenterConfig,
    mergeable_labels: set[str],
) -> bool:
    if left.label != right.label:
        return False
    if left.label not in mergeable_labels:
        return False
    if "gripper_event" in right.boundary_reasons:
        return False
    return True


def _should_merge_bridge(
    left: SkillSegment,
    middle: SkillSegment,
    right: SkillSegment,
    config: SegmenterConfig,
    mergeable_labels: set[str],
) -> bool:
    if left.label != right.label:
        return False
    if left.label not in mergeable_labels or middle.label not in mergeable_labels:
        return False
    if int(middle.frame_count) > int(config.short_segment_merge_frames):
        return False
    if "gripper_event" in middle.boundary_reasons:
        return False
    return True


def _segment_from_range(
    *,
    records: list[FrameRecord],
    features: "SegmenterFeatures",
    start_frame: int,
    end_frame: int,
    segment_id: str,
    boundary_reasons: list[str],
) -> SkillSegment:
    frame_count = len(records)
    high_velocity = float(np.quantile(features.velocity_norm, 0.75))
    medium_velocity = float(np.quantile(features.velocity_norm, 0.45))
    window = slice(start_frame, end_frame + 1)
    frame_indices = features.frame_indices[window]
    transition_window = features.transition_score[window]
    rep_offset = int(np.argmax(transition_window))
    representative_frame = int(frame_indices[rep_offset])
    velocity_window = features.velocity_norm[window]
    acceleration_window = features.acceleration_norm[window]
    gripper_window = features.gripper_positions[window]
    gripper_start = float(gripper_window[0])
    gripper_end = float(gripper_window[-1])
    gripper_span = abs(gripper_end - gripper_start)

    label = "stabilize"
    if gripper_span >= 0.08:
        label = "gripper_close" if gripper_end < gripper_start else "gripper_open"
    elif float(np.max(velocity_window)) >= high_velocity:
        label = "transfer"
    elif float(np.mean(velocity_window)) >= medium_velocity:
        label = "approach" if start_frame < frame_count // 2 else "retreat"

    metrics = {
        "mean_velocity": float(np.mean(velocity_window)),
        "peak_velocity": float(np.max(velocity_window)),
        "mean_acceleration": float(np.mean(acceleration_window)),
        "peak_acceleration": float(np.max(acceleration_window)),
        "gripper_span": float(gripper_span),
        "joint_span_mean": float(
            np.mean(np.max(features.joint_matrix[window], axis=0) - np.min(features.joint_matrix[window], axis=0))
        ),
        "transition_peak": float(np.max(transition_window)),
    }
    return SkillSegment(
        segment_id=segment_id,
        start_frame=int(start_frame),
        end_frame=int(end_frame),
        start_time_s=float(records[start_frame].relative_time_s),
        end_time_s=float(records[end_frame].relative_time_s),
        representative_frame=representative_frame,
        label=label,
        frame_count=int(end_frame - start_frame + 1),
        gripper_start=gripper_start,
        gripper_end=gripper_end,
        joint_keys=list(features.joint_keys),
        metrics=metrics,
        boundary_reasons=list(dict.fromkeys(boundary_reasons)),
    )


def _combine_boundary_reasons(segments: list[SkillSegment]) -> list[str]:
    merged: list[str] = []
    for segment in segments:
        for reason in segment.boundary_reasons:
            if reason not in merged:
                merged.append(reason)
    return merged


def _renumber_segments(segments: list[SkillSegment]) -> list[SkillSegment]:
    out: list[SkillSegment] = []
    for index, segment in enumerate(segments):
        updated = SkillSegment(**segment.to_dict())
        updated.segment_id = f"skill_{index:03d}"
        out.append(updated)
    return out


def _build_summary(records: list[FrameRecord], segments: list[SkillSegment]) -> dict[str, Any]:
    counts = {"var": 0, "inv": 0, "unknown": 0}
    for segment in segments:
        counts[segment.invariance] = counts.get(segment.invariance, 0) + 1
    return {
        "session_dir": "",
        "frame_count": len(records),
        "segment_count": len(segments),
        "var_segments": counts.get("var", 0),
        "inv_segments": counts.get("inv", 0),
        "labels": {segment.segment_id: segment.label for segment in segments},
        "representative_frames": {segment.segment_id: segment.representative_frame for segment in segments},
    }


def _run_self_check(bank: SkillBank) -> dict[str, Any]:
    gaps = []
    overlaps = []
    expected_start = 0
    for segment in bank.segments:
        if segment.start_frame > expected_start:
            gaps.append([expected_start, segment.start_frame - 1])
        if segment.start_frame < expected_start:
            overlaps.append([segment.start_frame, expected_start - 1])
        expected_start = segment.end_frame + 1
    if expected_start < bank.frame_count:
        gaps.append([expected_start, bank.frame_count - 1])

    preview_files = [bank.output_dir / "representatives" / f"{segment.segment_id}.png" for segment in bank.segments]
    segment_files = [bank.output_dir / "segments" / f"{segment.segment_id}.npz" for segment in bank.segments]
    return {
        "status": "pass" if not gaps and not overlaps else "warn",
        "frame_coverage_complete": not gaps and not overlaps,
        "gaps": gaps,
        "overlaps": overlaps,
        "segments_have_previews": all(path.exists() for path in preview_files),
        "segments_have_npz": all(path.exists() for path in segment_files),
        "segment_count": len(bank.segments),
        "var_segments": sum(1 for segment in bank.segments if segment.invariance == "var"),
        "inv_segments": sum(1 for segment in bank.segments if segment.invariance == "inv"),
    }


def _write_segment_npz(
    output_dir: Path,
    segment: SkillSegment,
    records: list[FrameRecord],
    features: "SegmenterFeatures",
) -> None:
    window = slice(segment.start_frame, segment.end_frame + 1)
    sent_action_keys = sorted(records[0].sent_action.keys())
    sent_actions = np.asarray(
        [[float(record.sent_action.get(key, 0.0)) for key in sent_action_keys] for record in records[window]],
        dtype=np.float32,
    )
    np.savez_compressed(
        output_dir / f"{segment.segment_id}.npz",
        frame_indices=features.frame_indices[window],
        relative_time_s=features.relative_time_s[window],
        joint_positions=features.joint_matrix[window].astype(np.float32),
        gripper_positions=features.gripper_positions[window].astype(np.float32),
        sent_action_keys=np.asarray(sent_action_keys),
        sent_actions=sent_actions,
    )


def _write_preview(output_dir: Path, segment: SkillSegment, session_dir: Path, records: list[FrameRecord]) -> None:
    record = records[segment.representative_frame]
    color_rgb, _ = load_frame_assets(session_dir, record)
    overlay = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(
        overlay,
        f"{segment.segment_id} {segment.label} {segment.invariance}",
        (32, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"frames {segment.start_frame}-{segment.end_frame} conf={segment.confidence:.2f}",
        (32, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_dir / f"{segment.segment_id}.png"), overlay)
