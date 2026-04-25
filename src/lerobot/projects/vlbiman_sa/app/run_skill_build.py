#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    from lerobot.projects.vlbiman_sa.skills import (
        InvarianceClassifierConfig,
        SegmenterConfig,
        build_skill_bank,
    )
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[5]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from lerobot.projects.vlbiman_sa.skills import (
        InvarianceClassifierConfig,
        SegmenterConfig,
        build_skill_bank,
    )


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "skill_build.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reusable skill bank from a recorded one-shot session.")
    parser.add_argument("--config", type=Path, default=_default_config_path(), help="Path to skill_build.yaml.")
    parser.add_argument("--session-dir", type=Path, default=None, help="Override the recording session directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override the skill bank output directory.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping, got {type(payload).__name__}.")
    return payload


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)

    payload = _load_yaml(args.config)
    session_dir = args.session_dir or Path(payload["session_dir"])
    output_dir = args.output_dir or Path(payload.get("output_dir", session_dir / "analysis" / "t3_skill_bank"))
    segmenter_config = SegmenterConfig(**dict(payload.get("segmenter", {})))
    classifier_config = InvarianceClassifierConfig(**dict(payload.get("classifier", {})))

    result = build_skill_bank(
        session_dir=session_dir,
        output_dir=output_dir,
        segmenter_config=segmenter_config,
        classifier_config=classifier_config,
    )

    logging.info("Skill bank saved to %s", result.skill_bank_path)
    logging.info("Skill summary: %s", json.dumps(result.bank.summary, ensure_ascii=False))
    logging.info("Self check: %s", result.self_check_path)
    logging.info("Representative previews: %s", result.preview_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
