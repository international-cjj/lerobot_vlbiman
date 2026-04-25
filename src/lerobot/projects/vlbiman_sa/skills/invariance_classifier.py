from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lerobot.projects.vlbiman_sa.demo.io import load_frame_assets
from lerobot.projects.vlbiman_sa.demo.schema import FrameRecord


@dataclass(slots=True)
class InvarianceClassifierConfig:
    scene_rgb_delta_threshold: float = 12.0
    scene_depth_delta_threshold_m: float = 0.01
    gripper_span_threshold: float = 0.08
    var_score_threshold: float = 0.45


class InvarianceClassifier:
    def __init__(self, config: InvarianceClassifierConfig | None = None):
        self.config = config or InvarianceClassifierConfig()

    def classify(
        self,
        session_dir: Path,
        records: list[FrameRecord],
        segments: list["SkillSegment"],
    ) -> list["SkillSegment"]:
        from .skill_bank import SkillSegment

        classified: list[SkillSegment] = []
        for segment in segments:
            start_record = records[segment.start_frame]
            end_record = records[segment.end_frame]
            start_color, start_depth = load_frame_assets(session_dir, start_record)
            end_color, end_depth = load_frame_assets(session_dir, end_record)

            rgb_delta = self._rgb_delta(start_color, end_color)
            depth_delta_m = self._depth_delta_m(start_depth, end_depth)
            gripper_span = abs(segment.gripper_end - segment.gripper_start)

            rgb_score = min(1.0, rgb_delta / max(self.config.scene_rgb_delta_threshold, 1e-6))
            depth_score = min(1.0, depth_delta_m / max(self.config.scene_depth_delta_threshold_m, 1e-6))
            gripper_score = min(1.0, gripper_span / max(self.config.gripper_span_threshold, 1e-6))

            label_prior = 0.0
            if segment.label in {"approach", "gripper_close", "gripper_open"}:
                label_prior = 1.0
            elif segment.label in {"transfer", "retreat"}:
                label_prior = 0.25

            var_score = 0.25 * rgb_score + 0.20 * depth_score + 0.35 * gripper_score + 0.20 * label_prior
            transfer_like = segment.label in {"transfer", "retreat", "stabilize"}
            if transfer_like and gripper_score < 0.25 and depth_score < 0.9:
                invariance = "inv"
            else:
                invariance = "var" if var_score >= self.config.var_score_threshold else "inv"
            segment.invariance = invariance
            segment.confidence = float(var_score)
            segment.metrics.update(
                {
                    "rgb_delta": float(rgb_delta),
                    "depth_delta_m": float(depth_delta_m),
                    "gripper_span": float(gripper_span),
                    "rgb_score": float(rgb_score),
                    "depth_score": float(depth_score),
                    "gripper_score": float(gripper_score),
                    "var_score": float(var_score),
                }
            )
            classified.append(segment)
        return classified

    @staticmethod
    def _rgb_delta(start_color: np.ndarray, end_color: np.ndarray) -> float:
        start_small = start_color[::4, ::4].astype(np.float32)
        end_small = end_color[::4, ::4].astype(np.float32)
        return float(np.mean(np.abs(end_small - start_small)))

    @staticmethod
    def _depth_delta_m(start_depth: np.ndarray, end_depth: np.ndarray) -> float:
        valid = (start_depth > 0) & (end_depth > 0)
        if not np.any(valid):
            return 0.0
        delta = np.abs(end_depth.astype(np.float32) - start_depth.astype(np.float32))
        median_depth = float(np.median(delta[valid]))
        if median_depth > 10.0:
            median_depth /= 1000.0
        return median_depth
