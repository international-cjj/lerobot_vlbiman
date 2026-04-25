#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def _default_session_dir() -> Path:
    return Path("outputs/vlbiman_sa/recordings/one_shot_20260323T185847Z")


def _default_calib_dir() -> Path:
    return Path("outputs/vlbiman_sa/calib")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export hand-eye and recording image montages.")
    parser.add_argument("--session-dir", type=Path, default=_default_session_dir())
    parser.add_argument("--calib-dir", type=Path, default=_default_calib_dir())
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--record-count", type=int, default=16)
    parser.add_argument("--handeye-count", type=int, default=10)
    return parser.parse_args()


def _sample_evenly(paths: list[Path], count: int) -> list[Path]:
    if len(paths) <= count:
        return paths
    indices = np.linspace(0, len(paths) - 1, count, dtype=int)
    return [paths[int(index)] for index in indices]


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def _fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    scale = min(width / max(src_w, 1), height / max(src_h, 1))
    resized = cv2.resize(
        image,
        (max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    y0 = (height - resized.shape[0]) // 2
    x0 = (width - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def _make_card(image_path: Path, label: str, tile_width: int, tile_height: int, caption_height: int) -> np.ndarray:
    image = _fit_image(_load_image(image_path), tile_width, tile_height)
    card = np.full((tile_height + caption_height, tile_width, 3), 255, dtype=np.uint8)
    card[:tile_height] = image
    cv2.rectangle(card, (0, 0), (tile_width - 1, tile_height - 1), (210, 210, 210), 1)
    cv2.putText(card, label, (14, tile_height + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(
        card,
        image_path.name,
        (14, tile_height + 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (90, 90, 90),
        1,
        cv2.LINE_AA,
    )
    return card


def _build_montage(title: str, items: list[tuple[Path, str]], output_path: Path) -> dict[str, object]:
    if not items:
        raise ValueError(f"No images available for montage: {title}")

    cols = 4
    tile_width = 360
    tile_height = 240
    caption_height = 72
    gutter = 20
    header_height = 86
    rows = int(math.ceil(len(items) / cols))
    width = cols * tile_width + (cols + 1) * gutter
    height = header_height + rows * (tile_height + caption_height) + (rows + 1) * gutter
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)

    cv2.putText(canvas, title, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"{len(items)} images",
        (24, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (90, 90, 90),
        2,
        cv2.LINE_AA,
    )

    for idx, (path, label) in enumerate(items):
        row = idx // cols
        col = idx % cols
        x0 = gutter + col * (tile_width + gutter)
        y0 = header_height + gutter + row * (tile_height + caption_height)
        card = _make_card(path, label, tile_width, tile_height, caption_height)
        canvas[y0 : y0 + card.shape[0], x0 : x0 + card.shape[1]] = card

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return {
        "title": title,
        "count": len(items),
        "path": str(output_path),
        "images": [str(path) for path, _ in items],
    }


def main() -> int:
    args = _parse_args()
    analysis_dir = args.analysis_dir or (args.session_dir / "analysis")
    handeye_paths = sorted((args.calib_dir / "debug_frames").glob("sample_*.png"))
    recording_paths = sorted((args.session_dir / "rgb").glob("frame_*.png"))

    handeye_selected = _sample_evenly(handeye_paths, max(1, args.handeye_count))
    recording_selected = _sample_evenly(recording_paths, max(1, args.record_count))

    handeye_items = [(path, f"handeye sample {idx + 1:02d}") for idx, path in enumerate(handeye_selected)]
    recording_items = [(path, f"record frame {idx + 1:02d}") for idx, path in enumerate(recording_selected)]

    handeye_montage = _build_montage(
        title="Hand-Eye Calibration Samples",
        items=handeye_items,
        output_path=analysis_dir / "handeye_calibration_montage.png",
    )
    recording_montage = _build_montage(
        title="Recording RGB Samples",
        items=recording_items,
        output_path=analysis_dir / "recording_rgb_montage.png",
    )

    summary = {
        "handeye_montage": handeye_montage,
        "recording_montage": recording_montage,
    }
    (analysis_dir / "image_montages.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
