from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lerobot.projects.vlbiman_sa.skills import build_skill_bank

from ..app.run_pose_adaptation import build_pose_pipeline_config, run_pose_adaptation_pipeline
from ..app.run_trajectory_generation import TrajectoryPipelineConfig, run_trajectory_generation_pipeline
from ..app.run_visual_analysis import VisionConfig, run_visual_analysis_pipeline
from .contracts import TaskGraspConfig

logger = logging.getLogger(__name__)


class GraspOrchestrator:
    """Top-level runner for the VLBiMan single-arm grasp pipeline."""

    def __init__(self, config: TaskGraspConfig):
        self.config = config
        self._initialized = False

    def initialize(self) -> None:
        self.config.data_root.mkdir(parents=True, exist_ok=True)
        self._initialized = True
        logger.info("Initialized orchestrator with data root: %s", self.config.data_root)

    def build_plan(self) -> dict[str, Any]:
        session_dir = self.config.recording_session_dir
        skill_output_dir = self.config.skill_output_dir or (
            session_dir / "analysis" / "t3_skill_bank" if session_dir else None
        )
        vision_output_dir = self.config.vision_output_dir or (
            session_dir / "analysis" / "t4_vision" if session_dir else None
        )
        pose_output_dir = self.config.pose_output_dir or (
            session_dir / "analysis" / "t5_pose" if session_dir else None
        )
        trajectory_output_dir = self.config.trajectory_output_dir or (
            session_dir / "analysis" / "t6_trajectory" if session_dir else None
        )
        return {
            "task_name": self.config.task_name,
            "target_phrase": self.config.target_phrase,
            "fps": self.config.fps,
            "robot_type": self.config.robot_type,
            "camera_serial_number": self.config.camera_serial_number,
            "transforms_path": str(self.config.transforms_path),
            "handeye_result_path": str(self.config.handeye_result_path),
            "data_root": str(self.config.data_root),
            "recording_session_dir": str(session_dir) if session_dir else None,
            "skill_output_dir": str(skill_output_dir) if skill_output_dir else None,
            "vision_output_dir": str(vision_output_dir) if vision_output_dir else None,
            "pose_output_dir": str(pose_output_dir) if pose_output_dir else None,
            "trajectory_output_dir": str(trajectory_output_dir) if trajectory_output_dir else None,
            "live_result_path": str(self.config.live_result_path) if self.config.live_result_path else None,
            "pipeline": [
                "demo_acquisition",
                "task_decomposition",
                "vlm_anchor_perception",
                "pose_adaptation",
                "trajectory_composition",
                "closed_loop_execution",
                "evaluation",
            ],
        }

    def run(self) -> dict[str, Any]:
        if not self._initialized:
            self.initialize()

        plan = self.build_plan()
        if self.config.recording_session_dir is not None:
            return self._run_offline_analysis(plan)

        if self.config.dry_run:
            logger.info("Dry-run mode enabled. Skipping hardware operations.")
            return {"status": "dry_run", "plan": plan}

        logger.warning("Real execution is not enabled yet. Run with --dry-run for now.")
        return {"status": "not_implemented", "plan": plan}

    def dump_effective_config(self) -> dict[str, Any]:
        payload = asdict(self.config)
        for key, value in payload.items():
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload

    def _vision_payload(self, session_dir: Path, skill_bank_path: Path, vision_output_dir: Path) -> dict[str, Any]:
        primary_reference_phrase = (
            str(getattr(self.config, "primary_reference_phrase", "") or "").strip()
            or str(getattr(self.config, "task_prompt", "") or "").strip()
            or str(self.config.target_phrase).strip()
        )
        return {
            "session_dir": str(session_dir),
            "skill_bank_path": str(skill_bank_path),
            "output_dir": str(vision_output_dir),
            "intrinsics_path": str(self.config.intrinsics_path),
            "target_phrase": primary_reference_phrase,
            "task_prompt": primary_reference_phrase,
            "frame_stride": 1,
            "use_var_segments_only": False,
            "segmentor": {
                "florence_model_id": "microsoft/Florence-2-base",
                "sam2_model_id": "facebook/sam2-hiera-small",
                "task_prompt": "<CAPTION_TO_PHRASE_GROUNDING>",
                "local_files_only": True,
                "max_new_tokens": 256,
                "num_beams": 3,
                "seed_search_stride": 20,
                "max_seed_frames": 8,
                "mask_threshold": 0.0,
                "jpeg_quality": 95,
                "offload_video_to_cpu": True,
                "offload_state_to_cpu": True,
            },
            "tracker": {
                "stability_window_size": 5,
                "position_variance_threshold_mm2": 100.0,
                "orientation_variance_threshold_deg2": 225.0,
            },
            "anchor": {
                "depth_window": 7,
                "contact_strategy": "centroid",
            },
        }

    def _run_offline_analysis(self, plan: dict[str, Any]) -> dict[str, Any]:
        session_dir = self.config.recording_session_dir
        assert session_dir is not None

        skill_output_dir = self.config.skill_output_dir or session_dir / "analysis" / "t3_skill_bank"
        skill_result = build_skill_bank(session_dir=session_dir, output_dir=skill_output_dir)
        skill_bank_path = self.config.skill_bank_path or skill_result.skill_bank_path

        vision_output_dir = self.config.vision_output_dir or session_dir / "analysis" / "t4_vision"
        vision_payload = self._vision_payload(session_dir, skill_bank_path, vision_output_dir)
        vision_summary = run_visual_analysis_pipeline(
            VisionConfig(
                session_dir=session_dir,
                skill_bank_path=skill_bank_path,
                output_dir=vision_output_dir,
                intrinsics_path=self.config.intrinsics_path,
                target_phrase=str(vision_payload["target_phrase"]),
                task_prompt=str(vision_payload["task_prompt"]),
                frame_stride=int(vision_payload["frame_stride"]),
                use_var_segments_only=bool(vision_payload["use_var_segments_only"]),
            ),
            vision_payload,
        )

        pose_output_dir = self.config.pose_output_dir or session_dir / "analysis" / "t5_pose"
        pose_summary: dict[str, Any]
        if self.config.live_result_path is not None and self.config.live_result_path.exists():
            pose_summary = run_pose_adaptation_pipeline(
                build_pose_pipeline_config(
                    task_config=self.config,
                    session_dir=session_dir,
                    analysis_dir=session_dir / "analysis",
                    output_dir=pose_output_dir,
                    live_result_path=self.config.live_result_path,
                )
            )
        else:
            pose_summary = {
                "status": "skipped",
                "reason": "live_result_path_missing",
                "output_dir": str(pose_output_dir),
            }

        trajectory_output_dir = self.config.trajectory_output_dir or session_dir / "analysis" / "t6_trajectory"
        trajectory_summary: dict[str, Any]
        if pose_summary.get("status") == "ok":
            trajectory_summary = run_trajectory_generation_pipeline(
                TrajectoryPipelineConfig(
                    session_dir=session_dir,
                    analysis_dir=session_dir / "analysis",
                    output_dir=trajectory_output_dir,
                    skill_bank_path=skill_bank_path,
                    adapted_pose_path=pose_output_dir / "adapted_pose.json",
                )
            )
        else:
            trajectory_summary = {
                "status": "skipped",
                "reason": "pose_summary_not_ready",
                "output_dir": str(trajectory_output_dir),
            }

        summary = {
            "status": "offline_analysis_complete",
            "skill_bank_path": str(skill_bank_path),
            "skill_self_check_path": str(skill_result.self_check_path),
            "vision_output_dir": str(vision_output_dir),
            "vision_summary_path": str(vision_output_dir / "summary.json"),
            "vision_self_check_path": str(vision_output_dir / "self_check.json"),
            "vision_summary": vision_summary,
            "pose_output_dir": str(pose_output_dir),
            "pose_summary_path": str(pose_output_dir / "summary.json"),
            "pose_summary": pose_summary,
            "trajectory_output_dir": str(trajectory_output_dir),
            "trajectory_summary_path": str(trajectory_output_dir / "summary.json"),
            "trajectory_summary": trajectory_summary,
        }
        (vision_output_dir / "orchestrator_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return {"status": "offline_analysis_complete", "plan": plan, "summary": summary}
