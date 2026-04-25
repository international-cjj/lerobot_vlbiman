#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _default_samples_path() -> Path:
    return Path("outputs/vlbiman_sa/calib/handeye_samples.json")


def _default_transforms_path() -> Path:
    return Path("src/lerobot/projects/vlbiman_sa/configs/transforms.yaml")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve hand-eye calibration and update transforms.yaml.")
    parser.add_argument(
        "--samples",
        type=Path,
        default=_default_samples_path(),
        help="JSON file with base_from_gripper and camera_from_target sample arrays.",
    )
    parser.add_argument(
        "--transforms-config",
        type=Path,
        default=_default_transforms_path(),
        help="Destination transforms YAML file.",
    )
    parser.add_argument("--setup", choices=["eye_to_hand", "eye_in_hand"], default="eye_to_hand")
    parser.add_argument(
        "--method",
        choices=["tsai", "park", "horaud", "andreff", "daniilidis"],
        default="tsai",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--print-template",
        action="store_true",
        help="Print JSON template and exit.",
    )
    return parser.parse_args()


def _print_template() -> None:
    template = {
        "base_from_gripper": [
            [[1, 0, 0, 0.10], [0, 1, 0, 0.20], [0, 0, 1, 0.30], [0, 0, 0, 1]]
        ],
        "camera_from_target": [
            [[1, 0, 0, 0.40], [0, 1, 0, 0.10], [0, 0, 1, 0.80], [0, 0, 0, 1]]
        ],
    }
    print(json.dumps(template, indent=2))


def _load_samples(samples_path: Path) -> tuple[list[object], list[object]]:
    import numpy as np

    if not samples_path.exists():
        raise FileNotFoundError(
            f"Sample file not found: {samples_path}. "
            "Run with --print-template to generate the schema."
        )
    payload = json.loads(samples_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Sample JSON must be an object.")
    if "base_from_gripper" not in payload or "camera_from_target" not in payload:
        raise ValueError("Sample JSON requires keys: base_from_gripper, camera_from_target.")

    base_from_gripper = [np.asarray(m, dtype=float) for m in payload["base_from_gripper"]]
    camera_from_target = [np.asarray(m, dtype=float) for m in payload["camera_from_target"]]
    return base_from_gripper, camera_from_target


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)

    if args.print_template:
        _print_template()
        return 0

    try:
        import numpy as np
        from lerobot.projects.vlbiman_sa.calib.handeye_solver import solve_hand_eye
        from lerobot.projects.vlbiman_sa.geometry.frame_manager import FrameManager
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[5]
        src_root = repo_root / "src"
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))
        import numpy as np
        from lerobot.projects.vlbiman_sa.calib.handeye_solver import solve_hand_eye
        from lerobot.projects.vlbiman_sa.geometry.frame_manager import FrameManager

    base_from_gripper, camera_from_target = _load_samples(args.samples)
    result = solve_hand_eye(
        base_from_gripper,
        camera_from_target,
        setup=args.setup,
        method=args.method,
    )

    manager = FrameManager.from_yaml(args.transforms_config)
    manager.set_transform(
        target=result.target_frame,
        source=result.source_frame,
        transform=result.transform,
    )
    manager.to_yaml(args.transforms_config)

    logging.info("Solved hand-eye transform (%s, %s):", result.setup, result.method)
    logging.info("%s <- %s", result.target_frame, result.source_frame)
    logging.info("\n%s", np.array2string(result.transform, precision=6, suppress_small=True))
    logging.info("Updated transforms file: %s", args.transforms_config)
    logging.info("Samples used: %d", result.sample_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
