from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from .schema import FrameRecord, RecorderConfig, RecordingSummary


def create_session_dir(output_root: Path, run_name: str | None = None, overwrite: bool = False) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_name = run_name or f"one_shot_{timestamp}"
    session_dir = output_root / session_name
    if session_dir.exists() and not overwrite:
        suffix = 1
        while (output_root / f"{session_name}_{suffix:02d}").exists():
            suffix += 1
        session_dir = output_root / f"{session_name}_{suffix:02d}"
    session_dir.mkdir(parents=True, exist_ok=overwrite)
    (session_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (session_dir / "depth").mkdir(parents=True, exist_ok=True)
    return session_dir


def save_frame_assets(session_dir: Path, frame: FrameRecord, color_rgb: np.ndarray, depth_map: np.ndarray) -> FrameRecord:
    rgb_rel = Path("rgb") / f"frame_{frame.frame_index:06d}.png"
    depth_rel = Path("depth") / f"frame_{frame.frame_index:06d}.npy"
    rgb_path = session_dir / rgb_rel
    depth_path = session_dir / depth_rel

    _write_rgb_depth(rgb_path=rgb_path, depth_path=depth_path, color_rgb=color_rgb, depth_map=depth_map)
    frame.color_path = rgb_rel.as_posix()
    frame.depth_path = depth_rel.as_posix()
    frame.color_shape = list(np.asarray(color_rgb).shape)
    frame.depth_shape = list(np.asarray(depth_map).shape)
    return frame


def save_named_camera_assets(
    session_dir: Path,
    frame: FrameRecord,
    *,
    camera_name: str,
    color_rgb: np.ndarray,
    depth_map: np.ndarray | None = None,
) -> FrameRecord:
    safe_camera_name = _safe_asset_name(camera_name)
    rgb_rel = Path("cameras") / safe_camera_name / "rgb" / f"frame_{frame.frame_index:06d}.png"
    rgb_path = session_dir / rgb_rel
    if depth_map is None:
        _write_rgb(rgb_path=rgb_path, color_rgb=color_rgb)
        depth_rel = None
    else:
        depth_rel = Path("cameras") / safe_camera_name / "depth" / f"frame_{frame.frame_index:06d}.npy"
        _write_rgb_depth(
            rgb_path=rgb_path,
            depth_path=session_dir / depth_rel,
            color_rgb=color_rgb,
            depth_map=depth_map,
        )
    frame.camera_assets[safe_camera_name] = {
        "color_path": rgb_rel.as_posix(),
        "color_shape": list(np.asarray(color_rgb).shape),
    }
    if depth_rel is not None:
        frame.camera_assets[safe_camera_name]["depth_path"] = depth_rel.as_posix()
        frame.camera_assets[safe_camera_name]["depth_shape"] = list(np.asarray(depth_map).shape)
    return frame


def _write_rgb_depth(*, rgb_path: Path, depth_path: Path, color_rgb: np.ndarray, depth_map: np.ndarray) -> None:
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_bgr = cv2.cvtColor(np.asarray(color_rgb), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", rgb_bgr)
    if not ok:
        raise RuntimeError(f"Failed to encode RGB frame to PNG: {rgb_path}")
    rgb_path.write_bytes(encoded.tobytes())
    np.save(depth_path, np.asarray(depth_map))


def _write_rgb(*, rgb_path: Path, color_rgb: np.ndarray) -> None:
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_bgr = cv2.cvtColor(np.asarray(color_rgb), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", rgb_bgr)
    if not ok:
        raise RuntimeError(f"Failed to encode RGB frame to PNG: {rgb_path}")
    rgb_path.write_bytes(encoded.tobytes())


def _safe_asset_name(value: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "_" for char in str(value).strip())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "camera"


def append_frame_metadata(metadata_path: Path, frame: FrameRecord) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(frame.to_dict(), ensure_ascii=False) + "\n")


def write_manifest(
    manifest_path: Path,
    config: RecorderConfig,
    summary: RecordingSummary,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "config": _jsonable(config.to_dict()),
        "summary": _jsonable(summary.to_dict()),
    }
    if extra:
        payload["extra"] = _jsonable(extra)
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_frame_records(session_dir: Path) -> list[FrameRecord]:
    metadata_path = session_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    records: list[FrameRecord] = []
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        payload.pop("schema_version", None)
        records.append(FrameRecord(**payload))
    return records


def load_frame_assets(session_dir: Path, frame: FrameRecord) -> tuple[np.ndarray, np.ndarray]:
    color_bgr = cv2.imread(str(session_dir / frame.color_path), cv2.IMREAD_UNCHANGED)
    if color_bgr is None:
        raise FileNotFoundError(f"RGB frame not found: {session_dir / frame.color_path}")
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    depth_map = np.load(session_dir / frame.depth_path)
    return color_rgb, depth_map


def iter_recorded_frames(session_dir: Path) -> Iterator[tuple[FrameRecord, np.ndarray, np.ndarray]]:
    for frame in load_frame_records(session_dir):
        color_rgb, depth_map = load_frame_assets(session_dir, frame)
        yield frame, color_rgb, depth_map


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
