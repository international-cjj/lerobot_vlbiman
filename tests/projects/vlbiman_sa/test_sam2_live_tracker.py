from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from lerobot.projects.vlbiman_sa.inv_servo.sam2_live_tracker import (
    SAM2LiveTracker,
    SAM2LiveTrackerConfig,
    SAM2TrackedFrame,
)
from lerobot.projects.vlbiman_sa.inv_servo.target_state import MaskState


def _tracked_frame(frame_index: int, bbox_xyxy: tuple[int, int, int, int]) -> SAM2TrackedFrame:
    mask = np.zeros((8, 8), dtype=np.uint8)
    x0, y0, x1, y1 = bbox_xyxy
    mask[y0:y1, x0:x1] = 255
    mask_state = MaskState(
        frame_index=frame_index,
        image_size_hw=mask.shape,
        mask_area_px=int(np.count_nonzero(mask)),
        centroid_uv=((x0 + x1 - 1) * 0.5, (y0 + y1 - 1) * 0.5),
        bbox_xyxy=(float(x0), float(y0), float(x1), float(y1)),
        source="fake_sam2",
    )
    return SAM2TrackedFrame(
        frame_index=frame_index,
        local_frame_index=0,
        mask=mask,
        mask_state=mask_state,
        update_ms=1.0,
        fps=1000.0,
        obj_ids=[1],
    )


class FakeIncrementalSAM2Tracker(SAM2LiveTracker):
    def __init__(self) -> None:
        super().__init__(SAM2LiveTrackerConfig(repo_path=None))
        self.sequence_calls: list[dict[str, Any]] = []
        self.incremental_calls: list[dict[str, Any]] = []

    def track_sequence(
        self,
        frames: Iterable[np.ndarray],
        *,
        frame_indices: Iterable[int] | None = None,
        init_bbox_xyxy: Any,
        work_dir: Path | None = None,
    ) -> list[SAM2TrackedFrame]:
        frame_list = list(frames)
        indices = list(range(len(frame_list))) if frame_indices is None else list(frame_indices)
        self.sequence_calls.append({"frame_count": len(frame_list), "indices": indices, "init_bbox": init_bbox_xyxy})
        return [_tracked_frame(int(indices[index]), (1, 1, 4, 4)) for index in range(len(frame_list))]

    def _track_incremental_live_frame(
        self,
        *,
        previous_frame: np.ndarray,
        current_frame: np.ndarray,
        previous_frame_index: int,
        current_frame_index: int,
        previous_mask: np.ndarray | None,
        previous_bbox_xyxy: tuple[float, float, float, float] | None,
        work_dir: Path | None,
    ) -> SAM2TrackedFrame:
        self.incremental_calls.append(
            {
                "previous_frame_index": previous_frame_index,
                "current_frame_index": current_frame_index,
                "previous_bbox_xyxy": previous_bbox_xyxy,
                "previous_mask_area_px": 0 if previous_mask is None else int(np.count_nonzero(previous_mask)),
            }
        )
        shift = len(self.incremental_calls)
        return _tracked_frame(current_frame_index, (1 + shift, 1, 4 + shift, 4))


def test_live_update_uses_incremental_pair_and_previous_mask_bbox_seed() -> None:
    tracker = FakeIncrementalSAM2Tracker()
    frame0 = np.zeros((8, 8, 3), dtype=np.uint8)
    frame1 = np.ones((8, 8, 3), dtype=np.uint8)
    frame2 = np.full((8, 8, 3), 2, dtype=np.uint8)

    init_payload = tracker.initialize(frame0, (1, 1, 4, 4))
    update1 = tracker.update(frame1, frame_index=1)
    update2 = tracker.update(frame2, frame_index=2)

    assert init_payload["ok"]
    assert update1["ok"]
    assert update2["ok"]
    assert tracker.sequence_calls == [{"frame_count": 1, "indices": [0], "init_bbox": (1.0, 1.0, 4.0, 4.0)}]
    assert [call["current_frame_index"] for call in tracker.incremental_calls] == [1, 2]
    assert tracker.incremental_calls[0]["previous_bbox_xyxy"] == (1.0, 1.0, 4.0, 4.0)
    assert tracker.incremental_calls[0]["previous_mask_area_px"] == 9
    assert tracker.incremental_calls[1]["previous_bbox_xyxy"] == (2.0, 1.0, 5.0, 4.0)
    assert tracker.incremental_calls[1]["previous_mask_area_px"] == 9
    assert len(tracker._live_frames) == 1
    assert tracker._live_frame_indices == [2]
