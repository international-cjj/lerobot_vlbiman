from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from lerobot.projects.vlbiman_sa.geometry.transforms import compose_transform, invert_transform
from lerobot.projects.vlbiman_sa.skills import SkillBank
from lerobot.utils.rotation import Rotation

from .ik_solution_selector import IkSolutionSelector, IkSolutionSelectorConfig, densify_joint_transition
from .progressive_ik import (
    IKPyState,
    IKStep,
    ProgressiveIKConfig,
    ProgressiveIKPlanner,
    forward_kinematics_tool,
    full_q_from_arm_q,
)


@dataclass(slots=True)
class TrajectoryComposerConfig:
    max_dense_joint_step_rad: float = 0.15
    max_joint_step_rad: float = 0.25
    min_ee_z_m: float = 0.0


@dataclass(slots=True)
class TrajectoryPoint:
    trajectory_index: int
    frame_index: int | None
    relative_time_s: float
    segment_id: str | None
    segment_label: str | None
    invariance: str
    source: str
    joint_positions: list[float]
    gripper_position: float | None
    target_pose_6d: list[float] | None
    target_pose_matrix: list[list[float]] | None
    solved_pose_matrix: list[list[float]] | None
    translation_error_mm: float | None
    rotation_error_deg: float | None
    max_joint_step_rad: float | None
    success: bool | None
    target_object_key: str | None = None
    target_phrase: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ComposedTrajectory:
    joint_keys: list[str]
    points: list[TrajectoryPoint]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_keys": list(self.joint_keys),
            "points": [point.to_dict() for point in self.points],
            "summary": self.summary,
        }


class TrajectoryComposer:
    def __init__(
        self,
        ik_state: IKPyState,
        *,
        config: TrajectoryComposerConfig | None = None,
        ik_config: ProgressiveIKConfig | None = None,
        selector_config: IkSolutionSelectorConfig | None = None,
    ):
        self.config = config or TrajectoryComposerConfig()
        self.selector = IkSolutionSelector(selector_config)
        self.ik_planner = ProgressiveIKPlanner(ik_state, config=ik_config, selector=self.selector)
        self.ik_state = ik_state

    def compose(
        self,
        *,
        records: list[Any],
        skill_bank: SkillBank,
        adapted_pose: dict[str, Any],
        demo_pose_matrices: dict[int, np.ndarray],
        start_joint_positions: list[float] | np.ndarray | None = None,
    ) -> ComposedTrajectory:
        segment_by_frame = self._segment_lookup(skill_bank)
        fallback_adaptation = _adaptation_payload(adapted_pose)
        segment_adaptations = {
            str(segment_id): payload
            for segment_id, payload in dict(adapted_pose.get("segment_adaptations", {})).items()
            if _is_valid_adaptation_payload(payload)
        }
        records_by_segment = self._records_by_segment(records, segment_by_frame)

        joint_keys = list(skill_bank.joint_keys)
        demo_seed_q = _joint_vector(records[0], joint_keys)
        requested_start_q = _coerce_start_joint_vector(start_joint_positions, expected_dim=demo_seed_q.size)
        lower_bounds = np.asarray(self.ik_state.lower_bounds[self.ik_state.link_indices], dtype=float)
        upper_bounds = np.asarray(self.ik_state.upper_bounds[self.ik_state.link_indices], dtype=float)
        continuity_requested = requested_start_q is not None
        continuity_clipped = False
        if requested_start_q is None:
            seed_q = demo_seed_q.copy()
        else:
            seed_q, continuity_clipped = _clip_joint_vector_to_bounds(requested_start_q, lower_bounds, upper_bounds)
        current_q = seed_q.copy()
        home_pose = np.asarray(demo_pose_matrices[int(records[0].frame_index)], dtype=float)
        points: list[TrajectoryPoint] = []
        pose_frame_count = 0
        pose_success_count = 0
        segment_planning_modes: dict[str, str] = {}
        segment_fusion_diagnostics: dict[str, dict[str, Any]] = {}
        segment_target_objects_summary: dict[str, str | None] = {}
        skipped_visual_servo_segments: list[dict[str, Any]] = []

        for segment in skill_bank.segments:
            segment_records = records_by_segment.get(str(segment.segment_id), [])
            if not segment_records:
                continue

            segment_adaptation = segment_adaptations.get(segment.segment_id, fallback_adaptation)
            target_object_key = (
                str(segment_adaptation.get("object_key"))
                if isinstance(segment_adaptation, dict) and segment_adaptation.get("object_key") is not None
                else None
            )
            target_phrase = (
                str(segment_adaptation.get("target_phrase"))
                if isinstance(segment_adaptation, dict) and segment_adaptation.get("target_phrase") is not None
                else None
            )
            segment_semantic_state = _segment_semantic_state(segment)

            if _is_visual_servo_segment(segment):
                segment_planning_modes[str(segment.segment_id)] = "online_visual_servo_skipped"
                target_phrase = _segment_metric_string(segment, "target_phrase") or target_phrase
                target_frame = _segment_metric_int(segment, "target_frame")
                target_frame = int(target_frame) if target_frame is not None else int(segment.representative_frame)
                diagnostic = {
                    "planning_mode": "online_visual_servo_skipped",
                    "frame_count": len(segment_records),
                    "start_frame": int(segment.start_frame),
                    "end_frame": int(segment.end_frame),
                    "target_phrase": target_phrase,
                    "target_frame": target_frame,
                    "reason": "segment_reserved_for_real_time_visual_servo",
                }
                segment_fusion_diagnostics[str(segment.segment_id)] = diagnostic
                segment_target_objects_summary[str(segment.segment_id)] = target_object_key or target_phrase
                skipped_visual_servo_segments.append(
                    {
                        "segment_id": str(segment.segment_id),
                        "label": str(segment.label),
                        "semantic_state": segment_semantic_state,
                        "target_phrase": target_phrase,
                        "target_frame": target_frame,
                    }
                )
                continue

            if _is_locked_gripper_segment(segment):
                segment_planning_modes[str(segment.segment_id)] = "hold_arm_gripper_only"
                segment_fusion_diagnostics[str(segment.segment_id)] = {
                    "planning_mode": "hold_arm_gripper_only",
                    "frame_count": len(segment_records),
                }
                segment_target_objects_summary[str(segment.segment_id)] = target_object_key
                points.extend(
                    self._hold_arm_segment_points(
                        segment_records=segment_records,
                        segment=segment,
                        current_q=current_q,
                        trajectory_index_offset=len(points),
                    )
                )
                continue

            if _is_return_segment(segment):
                target_poses = self._build_return_segment_target_poses(
                    segment_records=segment_records,
                    current_q=current_q,
                    demo_pose_matrices=demo_pose_matrices,
                    home_pose=home_pose,
                )
                segment_planning_modes[str(segment.segment_id)] = "return_to_home_blend"
                segment_fusion_diagnostics[str(segment.segment_id)] = {
                    "planning_mode": "return_to_home_blend",
                    "frame_count": len(segment_records),
                }
                segment_target_objects_summary[str(segment.segment_id)] = "home"
                for record, target_pose in zip(segment_records, target_poses, strict=True):
                    frame_index = int(record.frame_index)
                    relative_time_s = float(record.relative_time_s)
                    gripper_position = _gripper_value(record)
                    current_q, ik_steps = self.ik_planner.solve_pose(
                        target_pose=target_pose,
                        seed_q=current_q,
                        phase=f"{segment.segment_id}:{segment.label}:{segment_semantic_state or 'return'}",
                    )
                    pose_frame_count += 1
                    pose_success_count += int(ik_steps[-1].success if ik_steps else False)
                    points.extend(
                        self._ik_steps_to_points(
                            frame_index=frame_index,
                            relative_time_s=relative_time_s,
                            segment=segment,
                            gripper_position=gripper_position,
                            steps=ik_steps,
                            trajectory_index_offset=len(points),
                            target_object_key="home",
                            target_phrase="home",
                        )
                    )
                continue

            if str(segment.invariance) == "var":
                if segment_adaptation is None:
                    raise ValueError(f"Missing pose adaptation for var segment {segment.segment_id}.")
                target_poses, fusion_diag = self._build_var_segment_target_poses(
                    segment_records=segment_records,
                    current_q=current_q,
                    demo_pose_matrices=demo_pose_matrices,
                    segment_adaptation=segment_adaptation,
                )
                segment_planning_modes[str(segment.segment_id)] = str(fusion_diag["planning_mode"])
                segment_fusion_diagnostics[str(segment.segment_id)] = fusion_diag
                segment_target_objects_summary[str(segment.segment_id)] = target_object_key
                for record, target_pose in zip(segment_records, target_poses, strict=True):
                    frame_index = int(record.frame_index)
                    relative_time_s = float(record.relative_time_s)
                    gripper_position = _gripper_value(record)
                    current_q, ik_steps = self.ik_planner.solve_pose(
                        target_pose=target_pose,
                        seed_q=current_q,
                        phase=f"{segment.segment_id}:{segment.label}",
                    )
                    pose_frame_count += 1
                    pose_success_count += int(ik_steps[-1].success if ik_steps else False)
                    points.extend(
                        self._ik_steps_to_points(
                            frame_index=frame_index,
                            relative_time_s=relative_time_s,
                            segment=segment,
                            gripper_position=gripper_position,
                            steps=ik_steps,
                            trajectory_index_offset=len(points),
                            target_object_key=target_object_key,
                            target_phrase=target_phrase,
                        )
                    )
                continue

            segment_planning_modes[str(segment.segment_id)] = "demo_joint_replay"
            segment_fusion_diagnostics[str(segment.segment_id)] = {
                "planning_mode": "demo_joint_replay",
                "frame_count": len(segment_records),
            }
            segment_target_objects_summary[str(segment.segment_id)] = target_object_key
            for record in segment_records:
                frame_index = int(record.frame_index)
                demo_q = _joint_vector(record, joint_keys)
                gripper_position = _gripper_value(record)
                relative_time_s = float(record.relative_time_s)
                demo_pose = demo_pose_matrices[frame_index]
                normalized_demo_q = self.selector.select_nearest(
                    demo_q,
                    current_q,
                    lower_bounds=self.ik_state.lower_bounds[self.ik_state.link_indices],
                    upper_bounds=self.ik_state.upper_bounds[self.ik_state.link_indices],
                )
                dense_points = densify_joint_transition(
                    current_q,
                    normalized_demo_q,
                    max_step_rad=float(self.config.max_dense_joint_step_rad),
                )
                for dense_index, dense_q in enumerate(dense_points):
                    dense_q = np.asarray(dense_q, dtype=float)
                    solved_pose = forward_kinematics_tool(self.ik_state, dense_q)
                    if self._pose_z(solved_pose) < float(self.config.min_ee_z_m):
                        continue
                    target_pose_6d = (
                        _pose6d_from_matrix(demo_pose).tolist() if dense_index == len(dense_points) - 1 else None
                    )
                    points.append(
                        TrajectoryPoint(
                            trajectory_index=len(points),
                            frame_index=frame_index if dense_index == len(dense_points) - 1 else None,
                            relative_time_s=relative_time_s,
                            segment_id=segment.segment_id,
                            segment_label=segment.label,
                            invariance=str(segment.invariance),
                            source="demo_joint" if dense_index == len(dense_points) - 1 else "joint_interp",
                            joint_positions=dense_q.astype(float).tolist(),
                            gripper_position=gripper_position,
                            target_pose_6d=target_pose_6d,
                            target_pose_matrix=demo_pose.astype(float).tolist() if dense_index == len(dense_points) - 1 else None,
                            solved_pose_matrix=solved_pose.astype(float).tolist(),
                            translation_error_mm=None,
                            rotation_error_deg=None,
                            max_joint_step_rad=float(np.max(np.abs(dense_q - current_q))),
                            success=None,
                            target_object_key=target_object_key,
                            target_phrase=target_phrase,
                        )
                    )
                    current_q = dense_q

        if continuity_requested and points:
            points[0] = _with_continuity_seed(points[0], seed_q, self.ik_state)
        points = self._densify_points(points)
        joint_path = [np.asarray(point.joint_positions, dtype=float) for point in points]
        metrics = self.selector.compute_metrics(joint_path)
        first_point_q = np.asarray(points[0].joint_positions, dtype=float) if points else None
        continuity_first_delta = (
            float(np.max(np.abs(first_point_q - seed_q))) if continuity_requested and first_point_q is not None else None
        )
        summary = {
            "status": (
                "pass"
                if pose_frame_count == 0
                or (
                    float(pose_success_count / pose_frame_count) >= 0.95
                    and metrics.max_joint_step_rad_inf < float(self.config.max_joint_step_rad)
                )
                else "warn"
            ),
            "joint_keys": list(joint_keys),
            "point_count": len(points),
            "ik_target_frame_count": int(pose_frame_count),
            "ik_success_count": int(pose_success_count),
            "ik_success_rate": float(pose_success_count / pose_frame_count) if pose_frame_count else 1.0,
            "max_joint_step_rad_inf": metrics.max_joint_step_rad_inf,
            "mean_joint_step_rad_inf": metrics.mean_joint_step_rad_inf,
            "abrupt_step_count": metrics.abrupt_step_count,
            "preferred_step_violations": metrics.preferred_step_violations,
            "thresholds": {
                "ik_success_rate_min": 0.95,
                "preferred_joint_step_rad_max": float(self.config.max_dense_joint_step_rad),
                "max_joint_step_rad_max": float(self.config.max_joint_step_rad),
            },
            "segment_target_objects": {
                segment_id: object_key
                for segment_id, object_key in segment_target_objects_summary.items()
            },
            "segment_planning_modes": segment_planning_modes,
            "segment_fusion_diagnostics": segment_fusion_diagnostics,
            "skipped_visual_servo_segments": skipped_visual_servo_segments,
            "skipped_visual_servo_segment_count": len(skipped_visual_servo_segments),
            "continuity_requested": continuity_requested,
            "continuity_start_joint_positions": seed_q.astype(float).tolist() if continuity_requested else None,
            "continuity_start_was_clipped": bool(continuity_clipped) if continuity_requested else False,
            "continuity_first_delta_rad_inf": continuity_first_delta,
            "continuity_embedded": (
                bool(continuity_first_delta is not None and continuity_first_delta <= 1e-9)
                if continuity_requested
                else False
            ),
            "first_motion_source": points[0].source if points else None,
            "first_motion_segment_id": points[0].segment_id if points else None,
        }
        return ComposedTrajectory(joint_keys=joint_keys, points=points, summary=summary)

    def _build_var_segment_target_poses(
        self,
        *,
        segment_records: list[Any],
        current_q: np.ndarray,
        demo_pose_matrices: dict[int, np.ndarray],
        segment_adaptation: dict[str, Any],
    ) -> tuple[list[np.ndarray], dict[str, Any]]:
        demo_segment_poses = [
            np.asarray(demo_pose_matrices[int(record.frame_index)], dtype=float) for record in segment_records
        ]
        if not demo_segment_poses:
            return [], {"planning_mode": "var_segment_framewise_retarget", "frame_count": 0}
        current_start_pose = self._enforce_ee_floor(forward_kinematics_tool(self.ik_state, current_q))
        reference_pose = _pose_matrix_from_pose6d(segment_adaptation["reference_gripper_pose_6d"])
        adapted_reference_pose = np.asarray(segment_adaptation["adapted_gripper_matrix"], dtype=float)
        orientation_policy = str(segment_adaptation.get("orientation_policy") or "reference")
        retarget_mode = _var_segment_retarget_mode(orientation_policy)
        framewise_target_poses = [
            self._enforce_ee_floor(
                _retarget_var_pose(
                    demo_pose,
                    reference_pose=reference_pose,
                    adapted_reference_pose=adapted_reference_pose,
                    orientation_policy=orientation_policy,
                )
            )
            for demo_pose in demo_segment_poses
        ]
        start_correction = compose_transform(current_start_pose, invert_transform(framewise_target_poses[0]))
        progress = _segment_progress(demo_segment_poses)
        target_poses = [
            self._enforce_ee_floor(_apply_decay_correction(target_pose, start_correction, alpha))
            for target_pose, alpha in zip(framewise_target_poses, progress, strict=True)
        ]
        target_poses[0] = current_start_pose
        target_poses[-1] = framewise_target_poses[-1]
        start_gap_metrics = _pose_delta_metrics(current_start_pose, framewise_target_poses[0])
        end_gap_metrics = _pose_delta_metrics(target_poses[-1], framewise_target_poses[-1])
        reference_frame_indices = segment_adaptation.get("reference_frame_indices", [])
        if not isinstance(reference_frame_indices, list):
            reference_frame_indices = []
        fusion_diag = {
            "planning_mode": f"var_segment_framewise_{retarget_mode}",
            "frame_count": len(segment_records),
            "orientation_policy": orientation_policy,
            "retarget_mode": retarget_mode,
            "reference_frame_indices": [int(frame_index) for frame_index in reference_frame_indices],
            "framewise_start_gap_mm": start_gap_metrics["translation_mm"],
            "framewise_start_gap_deg": start_gap_metrics["rotation_deg"],
            "fused_end_gap_mm": end_gap_metrics["translation_mm"],
            "fused_end_gap_deg": end_gap_metrics["rotation_deg"],
        }
        if segment_adaptation.get("object_key") is not None:
            fusion_diag["object_key"] = str(segment_adaptation["object_key"])
        if segment_adaptation.get("target_phrase") is not None:
            fusion_diag["target_phrase"] = str(segment_adaptation["target_phrase"])
        return target_poses, fusion_diag

    def _build_return_segment_target_poses(
        self,
        *,
        segment_records: list[Any],
        current_q: np.ndarray,
        demo_pose_matrices: dict[int, np.ndarray],
        home_pose: np.ndarray,
    ) -> list[np.ndarray]:
        demo_segment_poses = [
            np.asarray(demo_pose_matrices[int(record.frame_index)], dtype=float) for record in segment_records
        ]
        current_start_pose = forward_kinematics_tool(self.ik_state, current_q)
        demo_start_pose = demo_segment_poses[0]
        demo_end_pose = demo_segment_poses[-1]
        target_end_pose = self._enforce_ee_floor(np.asarray(home_pose, dtype=float))

        start_delta = compose_transform(current_start_pose, invert_transform(demo_start_pose))
        end_delta = compose_transform(target_end_pose, invert_transform(demo_end_pose))
        progress = _segment_progress(demo_segment_poses)
        target_poses = [
            self._enforce_ee_floor(_apply_segment_delta_blend(demo_pose, start_delta, end_delta, alpha))
            for demo_pose, alpha in zip(demo_segment_poses, progress, strict=True)
        ]
        if target_poses:
            target_poses[0] = self._enforce_ee_floor(current_start_pose)
            target_poses[-1] = target_end_pose
        return target_poses

    def _hold_arm_segment_points(
        self,
        *,
        segment_records: list[Any],
        segment: Any,
        current_q: np.ndarray,
        trajectory_index_offset: int,
    ) -> list[TrajectoryPoint]:
        held_pose = forward_kinematics_tool(self.ik_state, current_q)
        held_pose_6d = _pose6d_from_matrix(held_pose).tolist()
        out: list[TrajectoryPoint] = []
        for offset, record in enumerate(segment_records):
            out.append(
                TrajectoryPoint(
                    trajectory_index=trajectory_index_offset + offset,
                    frame_index=int(record.frame_index),
                    relative_time_s=float(record.relative_time_s),
                    segment_id=segment.segment_id,
                    segment_label=segment.label,
                    invariance=str(segment.invariance),
                    source="hold_arm",
                    joint_positions=np.asarray(current_q, dtype=float).tolist(),
                    gripper_position=_gripper_value(record),
                    target_pose_6d=held_pose_6d,
                    target_pose_matrix=held_pose.astype(float).tolist(),
                    solved_pose_matrix=held_pose.astype(float).tolist(),
                    translation_error_mm=None,
                    rotation_error_deg=None,
                    max_joint_step_rad=0.0,
                    success=True,
                )
            )
        return out

    def _densify_points(self, points: list[TrajectoryPoint]) -> list[TrajectoryPoint]:
        if not points:
            return []

        densified: list[TrajectoryPoint] = [points[0]]
        for point in points[1:]:
            previous_q = np.asarray(densified[-1].joint_positions, dtype=float)
            target_q = np.asarray(point.joint_positions, dtype=float)
            dense = densify_joint_transition(
                previous_q,
                target_q,
                max_step_rad=float(self.config.max_dense_joint_step_rad),
            )
            if len(dense) > 1:
                for intermediate_q in dense[:-1]:
                    intermediate_q = np.asarray(intermediate_q, dtype=float)
                    solved_pose = forward_kinematics_tool(self.ik_state, intermediate_q)
                    if self._pose_z(solved_pose) < float(self.config.min_ee_z_m):
                        continue
                    densified.append(
                        TrajectoryPoint(
                            trajectory_index=0,
                            frame_index=None,
                            relative_time_s=point.relative_time_s,
                            segment_id=point.segment_id,
                            segment_label=point.segment_label,
                            invariance=point.invariance,
                            source="joint_interp",
                            joint_positions=intermediate_q.tolist(),
                            gripper_position=point.gripper_position,
                            target_pose_6d=None,
                            target_pose_matrix=None,
                            solved_pose_matrix=solved_pose.astype(float).tolist(),
                            translation_error_mm=None,
                            rotation_error_deg=None,
                            max_joint_step_rad=float(
                                np.max(np.abs(intermediate_q - previous_q))
                            ),
                            success=None,
                            target_object_key=point.target_object_key,
                            target_phrase=point.target_phrase,
                        )
                    )
                    previous_q = intermediate_q
            point_pose = (
                np.asarray(point.solved_pose_matrix, dtype=float)
                if point.solved_pose_matrix is not None
                else forward_kinematics_tool(self.ik_state, target_q)
            )
            if self._pose_z(point_pose) < float(self.config.min_ee_z_m):
                continue
            point.solved_pose_matrix = point_pose.astype(float).tolist()
            densified.append(point)

        for index, point in enumerate(densified):
            point.trajectory_index = index
        return densified

    def _ik_steps_to_points(
        self,
        *,
        frame_index: int,
        relative_time_s: float,
        segment: Any,
        gripper_position: float | None,
        steps: list[IKStep],
        trajectory_index_offset: int,
        target_object_key: str | None,
        target_phrase: str | None,
    ) -> list[TrajectoryPoint]:
        out: list[TrajectoryPoint] = []
        for idx, step in enumerate(steps):
            solved_pose = np.asarray(step.solved_pose_matrix, dtype=float)
            out.append(
                TrajectoryPoint(
                    trajectory_index=trajectory_index_offset + idx,
                    frame_index=frame_index if idx == len(steps) - 1 else None,
                    relative_time_s=relative_time_s,
                    segment_id=segment.segment_id,
                    segment_label=segment.label,
                    invariance=segment.invariance,
                    source="ik_frame" if idx == len(steps) - 1 else "ik_substep",
                    joint_positions=list(step.joint_positions),
                    gripper_position=gripper_position,
                    target_pose_6d=_pose6d_from_matrix(np.asarray(step.target_pose_matrix, dtype=float)).tolist(),
                    target_pose_matrix=step.target_pose_matrix,
                    solved_pose_matrix=solved_pose.astype(float).tolist(),
                    translation_error_mm=step.translation_error_mm,
                    rotation_error_deg=step.rotation_error_deg,
                    max_joint_step_rad=step.max_joint_step_rad,
                    success=step.success,
                    target_object_key=target_object_key,
                    target_phrase=target_phrase,
                )
            )
        return out

    def _segment_lookup(self, skill_bank: SkillBank) -> dict[int, Any]:
        lookup: dict[int, Any] = {}
        for segment in skill_bank.segments:
            for frame_index in range(int(segment.start_frame), int(segment.end_frame) + 1):
                lookup[frame_index] = segment
        return lookup

    def _records_by_segment(self, records: list[Any], segment_by_frame: dict[int, Any]) -> dict[str, list[Any]]:
        grouped: dict[str, list[Any]] = {}
        for record in records:
            segment = segment_by_frame.get(int(record.frame_index))
            if segment is None:
                continue
            grouped.setdefault(str(segment.segment_id), []).append(record)
        return grouped

    def _enforce_ee_floor(self, pose: np.ndarray) -> np.ndarray:
        bounded = np.asarray(pose, dtype=float).copy()
        bounded[2, 3] = max(float(bounded[2, 3]), float(self.config.min_ee_z_m))
        return bounded

    @staticmethod
    def _pose_z(pose: np.ndarray) -> float:
        return float(np.asarray(pose, dtype=float)[2, 3])


def _gripper_value(record: Any) -> float | None:
    if record.gripper_state:
        if "gripper.pos" in record.gripper_state:
            return float(record.gripper_state["gripper.pos"])
        first_key = sorted(record.gripper_state.keys())[0]
        return float(record.gripper_state[first_key])
    return None


def _joint_vector(record: Any, joint_keys: list[str]) -> np.ndarray:
    return np.asarray([float(record.joint_positions[key]) for key in joint_keys], dtype=float)


def _coerce_start_joint_vector(
    joint_positions: list[float] | np.ndarray | None,
    *,
    expected_dim: int,
) -> np.ndarray | None:
    if joint_positions is None:
        return None
    vector = np.asarray(joint_positions, dtype=float).reshape(-1)
    if vector.size != expected_dim:
        raise ValueError(f"Expected {expected_dim} start-joint values, got {vector.size}.")
    return vector


def _clip_joint_vector_to_bounds(
    vector: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> tuple[np.ndarray, bool]:
    vector = np.asarray(vector, dtype=float).reshape(-1)
    clipped = np.minimum(np.maximum(vector, np.asarray(lower_bounds, dtype=float)), np.asarray(upper_bounds, dtype=float))
    was_clipped = bool(np.max(np.abs(clipped - vector)) > 1e-9)
    return clipped, was_clipped


def _with_continuity_seed(point: TrajectoryPoint, seed_q: np.ndarray, ik_state: IKPyState) -> TrajectoryPoint:
    seeded_point = TrajectoryPoint(**point.to_dict())
    seeded_q = np.asarray(seed_q, dtype=float).reshape(-1)
    seeded_pose = forward_kinematics_tool(ik_state, seeded_q)
    seeded_point.joint_positions = seeded_q.astype(float).tolist()
    seeded_point.source = "continuity_seed"
    seeded_point.solved_pose_matrix = seeded_pose.astype(float).tolist()
    seeded_point.target_pose_matrix = seeded_pose.astype(float).tolist()
    seeded_point.target_pose_6d = _pose6d_from_matrix(seeded_pose).astype(float).tolist()
    seeded_point.translation_error_mm = 0.0
    seeded_point.rotation_error_deg = 0.0
    seeded_point.max_joint_step_rad = 0.0
    seeded_point.success = True
    return seeded_point


def _full_q(joint_vector: np.ndarray, state: IKPyState) -> np.ndarray:
    return full_q_from_arm_q(state, joint_vector)


def _pose_matrix_from_pose6d(pose_6d: list[float]) -> np.ndarray:
    pose_6d = np.asarray(pose_6d, dtype=float).reshape(6)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = Rotation.from_rotvec(pose_6d[3:]).as_matrix()
    transform[:3, 3] = pose_6d[:3]
    return transform


def _pose6d_from_matrix(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=float)
    return np.concatenate([transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_rotvec()], axis=0)


def _is_valid_adaptation_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return "reference_gripper_pose_6d" in payload and "adapted_gripper_matrix" in payload


def _adaptation_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    return payload if _is_valid_adaptation_payload(payload) else None


def _transform_var_pose(demo_pose: np.ndarray, reference_pose: np.ndarray, adapted_reference_pose: np.ndarray) -> np.ndarray:
    return compose_transform(adapted_reference_pose, compose_transform(invert_transform(reference_pose), demo_pose))


def _retarget_var_pose(
    demo_pose: np.ndarray,
    *,
    reference_pose: np.ndarray,
    adapted_reference_pose: np.ndarray,
    orientation_policy: str | None,
) -> np.ndarray:
    mode = _var_segment_retarget_mode(orientation_policy)
    if mode == "pose_retarget":
        return _transform_var_pose(demo_pose, reference_pose, adapted_reference_pose)
    return _translate_var_pose(demo_pose, reference_pose, adapted_reference_pose)


def _translate_var_pose(demo_pose: np.ndarray, reference_pose: np.ndarray, adapted_reference_pose: np.ndarray) -> np.ndarray:
    target_pose = np.asarray(demo_pose, dtype=float).copy()
    target_pose[:3, 3] += np.asarray(adapted_reference_pose, dtype=float)[:3, 3] - np.asarray(reference_pose, dtype=float)[
        :3, 3
    ]
    return target_pose


def _var_segment_retarget_mode(orientation_policy: str | None) -> str:
    normalized = str(orientation_policy or "").strip().lower().replace("-", "_")
    if normalized == "target_yaw":
        return "pose_retarget"
    return "translation_retarget"


def _segment_semantic_state(segment: Any) -> str | None:
    metrics = getattr(segment, "metrics", None)
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("semantic_state")
    if value is None:
        return None
    return str(value)


def _normalized_segment_semantic_state(segment: Any) -> str:
    return str(_segment_semantic_state(segment) or "").strip().lower().replace("-", "_")


def _segment_metric_string(segment: Any, key: str) -> str | None:
    metrics = getattr(segment, "metrics", None)
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _segment_metric_int(segment: Any, key: str) -> int | None:
    metrics = getattr(segment, "metrics", None)
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_visual_servo_segment(segment: Any) -> bool:
    semantic_state = _normalized_segment_semantic_state(segment)
    if semantic_state == "visual_servo":
        return True
    return str(getattr(segment, "label", "")).strip().lower().replace("-", "_") == "visual_servo"


def _is_locked_gripper_segment(segment: Any) -> bool:
    semantic_state = _segment_semantic_state(segment)
    if semantic_state is not None:
        semantic_action = str(semantic_state).strip().lower().replace("-", "_").split("_", 1)[0]
        if semantic_action in {"grasp", "pick", "pickup", "place", "release", "drop", "insert"}:
            return True
    if semantic_state in {"grasp_orange", "place_orange"}:
        return True
    return str(getattr(segment, "label", "")) in {"gripper_close", "gripper_open"}


def _is_return_segment(segment: Any) -> bool:
    semantic_state = _segment_semantic_state(segment)
    if semantic_state == "return":
        return True
    return str(getattr(segment, "label", "")) == "retreat"


def _segment_progress(demo_segment_poses: list[np.ndarray]) -> list[float]:
    if not demo_segment_poses:
        return []
    if len(demo_segment_poses) == 1:
        return [1.0]
    translations = np.asarray([np.asarray(pose, dtype=float)[:3, 3] for pose in demo_segment_poses], dtype=float)
    step_lengths = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    cumulative = np.concatenate([np.asarray([0.0], dtype=float), np.cumsum(step_lengths, dtype=float)], axis=0)
    total = float(cumulative[-1])
    if total <= 1e-9:
        return np.linspace(0.0, 1.0, num=len(demo_segment_poses), dtype=float).tolist()
    return (cumulative / total).astype(float).tolist()


def _apply_segment_delta_blend(
    demo_pose: np.ndarray,
    start_delta: np.ndarray,
    end_delta: np.ndarray,
    alpha: float,
) -> np.ndarray:
    blended_delta = _interpolate_transform(start_delta, end_delta, alpha)
    return compose_transform(blended_delta, np.asarray(demo_pose, dtype=float))


def _apply_decay_correction(target_pose: np.ndarray, correction_delta: np.ndarray, alpha: float) -> np.ndarray:
    correction = _interpolate_transform(correction_delta, _identity_transform(), alpha)
    return compose_transform(correction, np.asarray(target_pose, dtype=float))


def _interpolate_transform(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    translation = (1.0 - alpha) * start[:3, 3] + alpha * end[:3, 3]
    start_rotvec = Rotation.from_matrix(start[:3, :3]).as_rotvec()
    end_rotvec = Rotation.from_matrix(end[:3, :3]).as_rotvec()
    blended_rotvec = (1.0 - alpha) * start_rotvec + alpha * end_rotvec
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = Rotation.from_rotvec(blended_rotvec).as_matrix()
    transform[:3, 3] = translation
    return transform


def _pose_delta_metrics(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
    delta = compose_transform(np.asarray(lhs, dtype=float), invert_transform(np.asarray(rhs, dtype=float)))
    return {
        "translation_mm": float(np.linalg.norm(delta[:3, 3]) * 1000.0),
        "rotation_deg": _rotation_angle_deg(delta[:3, :3]),
    }


def _rotation_angle_deg(rotation_matrix: np.ndarray) -> float:
    return float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(np.asarray(rotation_matrix, dtype=float)).as_rotvec())))


def _identity_transform() -> np.ndarray:
    return np.eye(4, dtype=float)
