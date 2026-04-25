#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


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
    from lerobot.projects.vlbiman_sa.demo.io import load_frame_records
    from lerobot.projects.vlbiman_sa.skills import KeyposeSegmenter
    from lerobot.projects.vlbiman_sa.skills.skill_bank import (
        SkillBank,
        _build_summary,
        _run_self_check,
        _segment_from_range,
        _write_preview,
        _write_segment_npz,
    )
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[5]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from lerobot.projects.vlbiman_sa.demo.io import load_frame_records
    from lerobot.projects.vlbiman_sa.skills import KeyposeSegmenter
    from lerobot.projects.vlbiman_sa.skills.skill_bank import (
        SkillBank,
        _build_summary,
        _run_self_check,
        _segment_from_range,
        _write_preview,
        _write_segment_npz,
    )


@dataclass(slots=True)
class SegmentSpec:
    start_frame_1based: int
    end_frame_1based: int
    canonical_label: str
    semantic_state: str
    invariance: str


def _default_session_dir() -> Path:
    return Path("outputs/vlbiman_sa/recordings/one_shot_full_20260327T231705")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually override T3 segments with explicit frame ranges.")
    parser.add_argument("--session-dir", type=Path, default=_default_session_dir())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--segment",
        action="append",
        required=True,
        help="Format: start-end:canonical_label:semantic_state:invariance using 1-based inclusive frames.",
    )
    return parser.parse_args()


def _parse_segment(text: str) -> SegmentSpec:
    try:
        frame_range, canonical_label, semantic_state, invariance = text.split(":", 3)
        start_text, end_text = frame_range.split("-", 1)
        start_frame_1based = int(start_text)
        end_frame_1based = int(end_text)
    except Exception as exc:
        raise ValueError(f"Invalid --segment spec: {text}") from exc
    if start_frame_1based <= 0 or end_frame_1based < start_frame_1based:
        raise ValueError(f"Invalid frame range in --segment spec: {text}")
    if invariance not in {"var", "inv"}:
        raise ValueError(f"Invariance must be var or inv in --segment spec: {text}")
    return SegmentSpec(
        start_frame_1based=start_frame_1based,
        end_frame_1based=end_frame_1based,
        canonical_label=canonical_label,
        semantic_state=semantic_state,
        invariance=invariance,
    )


def _build_bank(session_dir: Path, output_dir: Path, specs: list[SegmentSpec]) -> dict[str, object]:
    records = load_frame_records(session_dir)
    if not records:
        raise ValueError(f"No recorded frames found in {session_dir}")
    features = KeyposeSegmenter().extract_features(records)

    frame_count = len(records)
    segments = []
    for index, spec in enumerate(specs):
        start_frame = spec.start_frame_1based - 1
        end_frame = spec.end_frame_1based - 1
        if index == len(specs) - 1 and end_frame < frame_count - 1:
            end_frame = frame_count - 1
        segment = _segment_from_range(
            records=records,
            features=features,
            start_frame=start_frame,
            end_frame=end_frame,
            segment_id=f"skill_{index:03d}",
            boundary_reasons=["manual_override"],
        )
        segment.label = spec.canonical_label
        segment.invariance = spec.invariance
        segment.confidence = 1.0
        segment.metrics["semantic_state"] = spec.semantic_state
        segment.metrics["manual_override"] = 1.0
        segments.append(segment)

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "representatives"
    preview_dir.mkdir(parents=True, exist_ok=True)
    segment_npz_dir = output_dir / "segments"
    segment_npz_dir.mkdir(parents=True, exist_ok=True)

    for segment in segments:
        _write_segment_npz(segment_npz_dir, segment, records, features)
        _write_preview(preview_dir, segment, session_dir, records)

    summary = _build_summary(records, segments)
    summary["session_dir"] = str(session_dir)
    summary["manual_override"] = True
    summary["semantic_labels"] = {segment.segment_id: segment.metrics.get("semantic_state") for segment in segments}

    bank = SkillBank(
        session_dir=session_dir,
        output_dir=output_dir,
        frame_count=len(records),
        joint_keys=list(features.joint_keys),
        segments=segments,
        summary=summary,
    )
    self_check_payload = _run_self_check(bank)
    summary["self_check_status"] = self_check_payload["status"]
    bank.summary = summary

    skill_bank_path = bank.save()
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    self_check_path = output_dir / "self_check.json"
    self_check_path.write_text(json.dumps(self_check_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "status": "pass",
        "session_dir": str(session_dir),
        "output_dir": str(output_dir),
        "frame_count": frame_count,
        "segment_count": len(segments),
        "skill_bank_path": str(skill_bank_path),
        "summary_path": str(summary_path),
        "self_check_path": str(self_check_path),
        "segments": [
            {
                "segment_id": segment.segment_id,
                "canonical_label": segment.label,
                "semantic_state": segment.metrics.get("semantic_state"),
                "invariance": segment.invariance,
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
                "start_time_s": segment.start_time_s,
                "end_time_s": segment.end_time_s,
                "representative_frame": segment.representative_frame,
            }
            for segment in segments
        ],
    }


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir or (args.session_dir / "analysis" / "t3_skill_bank")
    specs = [_parse_segment(text) for text in args.segment]
    payload = _build_bank(args.session_dir, output_dir, specs)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
