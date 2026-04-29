from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


CONDA_LEROBOT_ROOT = Path("/home/cjj/miniconda3/envs/lerobot")


def _maybe_reexec_in_conda_lerobot() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    conda_root = Path(os.environ.get("VLBIMAN_CONDA_LEROBOT_PREFIX", CONDA_LEROBOT_ROOT))
    conda_python = conda_root / "bin" / "python"
    already_conda = Path(sys.prefix).resolve() == conda_root.resolve() or Path(sys.executable).resolve() == conda_python.resolve()
    if already_conda and os.environ.get("PYTHONNOUSERSITE") == "1":
        return
    if not conda_python.exists():
        return
    env = os.environ.copy()
    env["VLBIMAN_CONDA_LEROBOT_REEXEC"] = "1"
    env["CONDA_PREFIX"] = str(conda_root)
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("VIRTUAL_ENV", None)
    env["PATH"] = os.pathsep.join([str(conda_root / "bin"), env.get("PATH", "")])
    pythonpath = [str(repo_root / "src"), str(repo_root)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    os.execve(str(conda_python), [str(conda_python), __file__, *sys.argv[1:]], env)


_maybe_reexec_in_conda_lerobot()


import cv2

def _bootstrap_paths() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    for path in (repo_root / "src", repo_root):
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


def _runtime_environment() -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "expected_conda_prefix": str(CONDA_LEROBOT_ROOT),
        "used_conda_lerobot": Path(sys.prefix).resolve() == CONDA_LEROBOT_ROOT.resolve(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GroundingDINO yellow-ball detection on frame 75.")
    parser.add_argument("--config", type=Path, default=default_inv_rgb_servo_config_path())
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--session-name", default="sim_one_shot")
    parser.add_argument("--frame-index", type=int, default=None)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--phrase", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    config = load_inv_rgb_servo_config(args.config)
    run_dir = args.run_dir or config.data.original_flow_dir
    frame_index = config.data.groundingdino_check_frame if args.frame_index is None else int(args.frame_index)
    phrase = args.phrase or config.detector.validation_phrase
    camera = args.camera or config.data.camera

    session_dir = run_dir / "recordings" / args.session_name
    records = load_frame_records(session_dir)
    record = records[int(frame_index)]
    camera_name, camera_asset = _resolve_camera_asset(record, camera, config.data.camera_aliases)
    image_shape: list[int] | None = None
    image_path: str | None = None
    image_rgb = None
    if camera_asset is not None:
        image_path = str(session_dir / camera_asset["color_path"])
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_shape = None if image_bgr is None else list(image_bgr.shape)
        if image_bgr is not None:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    detector_config = config.detector.to_detector_config()
    detector_config.validation_phrase = phrase
    detector = GroundingDINODetector(detector_config)
    environment = detector.check_environment()
    if camera_asset is None or image_rgb is None:
        detection_result = {
            "ok": False,
            "state": None,
            "failure_reason": "frame75_image_not_found",
        }
    else:
        detection_result = detector.detect(image_rgb, phrase=phrase, frame_index=int(frame_index)).to_dict()

    ok = bool(
        environment.ok
        and record.frame_index == int(frame_index)
        and camera_asset is not None
        and image_shape is not None
        and detection_result["ok"]
    )
    output_dir = args.output_dir or config.output.groundingdino_check_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    detection_json_path = output_dir / f"detection_{int(frame_index):06d}.json"
    overlay_path = output_dir / f"detection_{int(frame_index):06d}.png"
    compatibility_json_path = output_dir / "groundingdino_frame75_result.json"
    compatibility_overlay_path = output_dir / "groundingdino_frame75_overlay.png"

    payload: dict[str, Any] = {
        "ok": ok,
        "stage": "task5_groundingdino_frame75",
        "config_path": str(args.config),
        "frame_index": record.frame_index,
        "camera": camera_name or camera,
        "phrase": phrase,
        "image_path": image_path,
        "image_shape": image_shape,
        "groundingdino_environment": environment.to_dict(),
        "runtime_environment": _runtime_environment(),
        "detection_result": detection_result,
        "output_json_path": str(detection_json_path),
        "overlay_path": str(overlay_path),
        "compatibility_json_path": str(compatibility_json_path),
        "compatibility_overlay_path": str(compatibility_overlay_path),
    }
    detection_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    compatibility_payload = _build_compatibility_payload(
        payload=payload,
        detection_result=detection_result,
        output_path=compatibility_json_path,
        overlay_path=compatibility_overlay_path,
    )
    compatibility_json_path.write_text(json.dumps(compatibility_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if image_rgb is not None and detection_result["ok"] and detection_result["state"] is not None:
        _write_overlay(
            image_rgb=image_rgb,
            detection_state=detection_result["state"],
            phrase=phrase,
            output_path=overlay_path,
        )
        _write_overlay(
            image_rgb=image_rgb,
            detection_state=detection_result["state"],
            phrase=phrase,
            output_path=compatibility_overlay_path,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if ok:
        print("GROUNDINGDINO_YELLOW_BALL_FRAME75_OK")
    if args.strict and not ok:
        raise SystemExit(1)


def _write_overlay(*, image_rgb: Any, detection_state: dict[str, Any], phrase: str, output_path: Path) -> None:
    overlay = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    bbox = detection_state["bbox_xyxy"]
    score = float(detection_state["score"])
    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox]
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 255), 2)
    cv2.putText(
        overlay,
        f"{phrase} {score:.3f}",
        (max(0, x0), max(18, y0 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), overlay)


def _build_compatibility_payload(
    *,
    payload: dict[str, Any],
    detection_result: dict[str, Any],
    output_path: Path,
    overlay_path: Path,
) -> dict[str, Any]:
    state = detection_result.get("state") if detection_result.get("ok") else None
    return {
        "ok": bool(payload["ok"]),
        "stage": payload["stage"],
        "frame_index": payload["frame_index"],
        "camera": payload["camera"],
        "phrase": payload["phrase"],
        "image_path": payload["image_path"],
        "image_shape": payload["image_shape"],
        "runtime_environment": payload.get("runtime_environment"),
        "bbox_xyxy": None if state is None else state.get("bbox_xyxy"),
        "score": None if state is None else state.get("score"),
        "label": None if state is None else state.get("label"),
        "detection_result": detection_result,
        "source_json_path": payload["output_json_path"],
        "output_json_path": str(output_path),
        "overlay_path": str(overlay_path),
        "failure_reason": None if payload["ok"] else detection_result.get("failure_reason"),
    }


if __name__ == "__main__":
    main()
