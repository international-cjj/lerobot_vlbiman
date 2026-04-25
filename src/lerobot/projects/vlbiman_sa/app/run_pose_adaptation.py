#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


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


def _bootstrap_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[5]
    extra_paths = [
        repo_root / "src",
        repo_root,
        repo_root / "lerobot_robot_cjjarm",
    ]
    for path in extra_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo_root


_maybe_reexec_in_repo_venv()
REPO_ROOT = _bootstrap_paths()

from lerobot.projects.vlbiman_sa.demo.io import load_frame_records
from lerobot.projects.vlbiman_sa.geometry import (
    DemoPoseFrame,
    FrameManager,
    PoseAdapter,
    PoseAdapterConfig,
    TargetObservation,
    apply_transform_points,
)
from lerobot.projects.vlbiman_sa.geometry.geometry_compensator import GeometryCompensatorConfig
from lerobot.projects.vlbiman_sa.geometry.pose_adapter import pose_matrix_to_pose6d
from lerobot.projects.vlbiman_sa.trajectory.progressive_ik import (
    IKPyState,
    build_ikpy_state,
    forward_kinematics_tool,
    full_q_from_arm_q,
)
from lerobot.projects.vlbiman_sa.vision import CameraIntrinsics
from lerobot.utils.rotation import Rotation


@dataclass(slots=True)
class PosePipelineConfig:
    session_dir: Path
    analysis_dir: Path
    output_dir: Path
    live_result_path: Path
    intrinsics_path: Path
    transforms_path: Path
    handeye_result_path: Path
    reference_window_size: int = 4
    height_gain: float = 0.5
    max_height_adjustment_m: float = 0.03
    align_target_orientation: bool = False
    primary_orientation_policy: str | None = None
    primary_target_phrase: str = "orange"
    primary_reference_phrase: str | None = None
    primary_vision_dir_name: str = "t4_vision"
    secondary_target_phrase: str = "pink cup"
    secondary_reference_phrase: str | None = None
    secondary_vision_dir_name: str = "t4_vision_pink_cup"
    secondary_align_target_orientation: bool = False
    secondary_orientation_policy: str | None = None


def _default_session_dir() -> Path:
    return Path("outputs/vlbiman_sa/recordings/one_shot_20260323T185847Z")


def _default_analysis_dir(session_dir: Path) -> Path:
    return session_dir / "analysis"


def _default_output_dir(session_dir: Path) -> Path:
    return _default_analysis_dir(session_dir) / "t5_pose"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adapt demo geometry to a new scene for T5.")
    parser.add_argument("--session-dir", type=Path, default=_default_session_dir())
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--live-result-path",
        type=Path,
        default=Path("outputs/vlbiman_sa/live_orange_pose/latest_result.json"),
    )
    parser.add_argument(
        "--intrinsics-path",
        type=Path,
        default=Path("src/lerobot/projects/vlbiman_sa/configs/camera_intrinsics.json"),
    )
    parser.add_argument(
        "--transforms-path",
        type=Path,
        default=Path("src/lerobot/projects/vlbiman_sa/configs/transforms.yaml"),
    )
    parser.add_argument(
        "--handeye-result-path",
        type=Path,
        default=Path("outputs/vlbiman_sa/calib/handeye_result.json"),
    )
    parser.add_argument("--reference-window-size", type=int, default=4)
    parser.add_argument("--height-gain", type=float, default=0.5)
    parser.add_argument("--max-height-adjustment-m", type=float, default=0.03)
    parser.add_argument(
        "--align-target-orientation",
        action="store_true",
        help="Rotate the adapted pose by the target object's estimated in-plane orientation.",
    )
    parser.add_argument("--primary-orientation-policy", default=None)
    parser.add_argument("--primary-target-phrase", default="orange")
    parser.add_argument("--primary-reference-phrase", default=None)
    parser.add_argument("--primary-vision-dir-name", default="t4_vision")
    parser.add_argument("--secondary-target-phrase", default="pink cup")
    parser.add_argument("--secondary-reference-phrase", default=None)
    parser.add_argument("--secondary-vision-dir-name", default="t4_vision_pink_cup")
    parser.add_argument(
        "--secondary-align-target-orientation",
        action="store_true",
        help="Rotate the secondary target by its estimated in-plane orientation.",
    )
    parser.add_argument("--secondary-orientation-policy", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> PosePipelineConfig:
    analysis_dir = args.analysis_dir or _default_analysis_dir(args.session_dir)
    output_dir = args.output_dir or _default_output_dir(args.session_dir)
    return PosePipelineConfig(
        session_dir=args.session_dir,
        analysis_dir=analysis_dir,
        output_dir=output_dir,
        live_result_path=args.live_result_path,
        intrinsics_path=args.intrinsics_path,
        transforms_path=args.transforms_path,
        handeye_result_path=args.handeye_result_path,
        reference_window_size=max(1, int(args.reference_window_size)),
        height_gain=float(args.height_gain),
        max_height_adjustment_m=float(args.max_height_adjustment_m),
        align_target_orientation=bool(args.align_target_orientation),
        primary_orientation_policy=(
            str(args.primary_orientation_policy).strip() if args.primary_orientation_policy is not None else None
        ),
        primary_target_phrase=str(args.primary_target_phrase),
        primary_reference_phrase=(
            str(args.primary_reference_phrase).strip() if args.primary_reference_phrase is not None else None
        ),
        primary_vision_dir_name=str(args.primary_vision_dir_name),
        secondary_target_phrase=str(args.secondary_target_phrase),
        secondary_reference_phrase=(
            str(args.secondary_reference_phrase).strip() if args.secondary_reference_phrase is not None else None
        ),
        secondary_vision_dir_name=str(args.secondary_vision_dir_name),
        secondary_align_target_orientation=bool(args.secondary_align_target_orientation),
        secondary_orientation_policy=(
            str(args.secondary_orientation_policy).strip()
            if args.secondary_orientation_policy is not None
            else None
        ),
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _record_to_gripper_pose(record: Any, state: IKPyState | None) -> np.ndarray:
    if record.ee_pose is not None:
        ee_pose = np.asarray(record.ee_pose, dtype=float).reshape(-1)
        if ee_pose.shape[0] >= 6:
            return _pose6d_to_matrix(ee_pose[:6])

    if state is None:
        raise ValueError(f"Frame {record.frame_index} has no ee_pose and no IK fallback state is available.")

    arm_q: list[float] = []
    for joint_name in state.arm_joint_names:
        if joint_name in record.joint_positions:
            value = record.joint_positions[joint_name]
        else:
            value = record.joint_positions[f"{joint_name}.pos"]
        arm_q.append(float(value))
    q = full_q_from_arm_q(state, np.asarray(arm_q, dtype=float))
    return forward_kinematics_tool(state, q)


def _pose6d_to_matrix(pose_6d: np.ndarray) -> np.ndarray:
    pose_6d = np.asarray(pose_6d, dtype=float).reshape(6)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = Rotation.from_rotvec(pose_6d[3:]).as_matrix()
    transform[:3, 3] = pose_6d[:3]
    return transform


def _load_frame_manager(config: PosePipelineConfig) -> tuple[FrameManager, dict[str, Any] | None]:
    manager = FrameManager.from_yaml(config.transforms_path)
    handeye_payload: dict[str, Any] | None = None
    try:
        manager.get_transform("base", "camera")
        return manager, None
    except KeyError:
        pass

    if not config.handeye_result_path.exists():
        raise FileNotFoundError(
            "No base<-camera transform is available in transforms.yaml and handeye_result.json is missing."
        )

    handeye_payload = _load_json(config.handeye_result_path)
    base_from_camera = np.asarray(handeye_payload.get("base_from_camera"), dtype=float)
    if base_from_camera.shape != (4, 4):
        raise ValueError(f"base_from_camera in {config.handeye_result_path} must be 4x4.")
    manager.set_transform("base", "camera", base_from_camera)
    return manager, handeye_payload


def _normalize_object_key(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in str(value)).strip("_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized or "object"


def _clean_phrase(value: Any) -> str:
    return str(value or "").strip()


def _compact_object_key(value: str) -> str:
    return _normalize_object_key(value).replace("_", "")


def resolve_pose_target_phrases(
    task_config: Any,
    *,
    auxiliary_target_phrases: list[str] | None = None,
    allow_configured_secondary_fallback: bool = True,
) -> tuple[str, str]:
    primary_target_phrase = _clean_phrase(getattr(task_config, "target_phrase", "orange")) or "orange"
    primary_key = _normalize_object_key(primary_target_phrase)

    secondary_target_phrase = ""
    for phrase in auxiliary_target_phrases or []:
        cleaned_phrase = _clean_phrase(phrase)
        if cleaned_phrase and _normalize_object_key(cleaned_phrase) != primary_key:
            secondary_target_phrase = cleaned_phrase
            break

    if not secondary_target_phrase and allow_configured_secondary_fallback:
        configured_secondary = _clean_phrase(getattr(task_config, "secondary_target_phrase", ""))
        if configured_secondary and _normalize_object_key(configured_secondary) != primary_key:
            secondary_target_phrase = configured_secondary

    return primary_target_phrase, secondary_target_phrase


def resolve_pose_reference_phrases(task_config: Any) -> tuple[str, str]:
    primary_reference_phrase = _clean_phrase(getattr(task_config, "primary_reference_phrase", ""))
    if not primary_reference_phrase:
        primary_reference_phrase = (
            _clean_phrase(getattr(task_config, "task_prompt", ""))
            or _clean_phrase(getattr(task_config, "target_phrase", "orange"))
            or "orange"
        )

    secondary_reference_phrase = _clean_phrase(getattr(task_config, "secondary_reference_phrase", ""))
    if not secondary_reference_phrase:
        secondary_reference_phrase = _clean_phrase(getattr(task_config, "secondary_target_phrase", ""))

    return primary_reference_phrase, secondary_reference_phrase


def build_pose_pipeline_config(
    *,
    task_config: Any,
    session_dir: Path,
    analysis_dir: Path,
    output_dir: Path,
    live_result_path: Path,
    auxiliary_target_phrases: list[str] | None = None,
    allow_configured_secondary_fallback: bool = True,
) -> PosePipelineConfig:
    primary_target_phrase, secondary_target_phrase = resolve_pose_target_phrases(
        task_config,
        auxiliary_target_phrases=auxiliary_target_phrases,
        allow_configured_secondary_fallback=allow_configured_secondary_fallback,
    )
    primary_reference_phrase, secondary_reference_phrase = resolve_pose_reference_phrases(task_config)
    return PosePipelineConfig(
        session_dir=session_dir,
        analysis_dir=analysis_dir,
        output_dir=output_dir,
        live_result_path=live_result_path,
        intrinsics_path=Path(getattr(task_config, "intrinsics_path")),
        transforms_path=Path(getattr(task_config, "transforms_path")),
        handeye_result_path=Path(getattr(task_config, "handeye_result_path")),
        align_target_orientation=bool(getattr(task_config, "align_target_orientation", False)),
        primary_orientation_policy=_clean_phrase(getattr(task_config, "primary_orientation_policy", "")) or None,
        primary_target_phrase=primary_target_phrase,
        primary_reference_phrase=primary_reference_phrase,
        primary_vision_dir_name=str(getattr(task_config, "primary_vision_dir_name", "t4_vision")),
        secondary_target_phrase=secondary_target_phrase,
        secondary_reference_phrase=secondary_reference_phrase,
        secondary_vision_dir_name=str(getattr(task_config, "secondary_vision_dir_name", "t4_vision_pink_cup")),
        secondary_align_target_orientation=bool(getattr(task_config, "secondary_align_target_orientation", False)),
        secondary_orientation_policy=_clean_phrase(getattr(task_config, "secondary_orientation_policy", "")) or None,
    )


def _segment_lookup(analysis_dir: Path) -> dict[int, dict[str, str]]:
    skill_bank_path = analysis_dir / "t3_skill_bank" / "skill_bank.json"
    if not skill_bank_path.exists():
        return {}
    payload = _load_json(skill_bank_path)
    lookup: dict[int, dict[str, str]] = {}
    for segment in payload.get("segments", []):
        for frame_index in range(int(segment["start_frame"]), int(segment["end_frame"]) + 1):
            lookup[frame_index] = {
                "segment_id": str(segment["segment_id"]),
                "segment_label": str(segment["label"]),
            }
    return lookup


def _segment_payload(analysis_dir: Path) -> list[dict[str, Any]]:
    skill_bank_path = analysis_dir / "t3_skill_bank" / "skill_bank.json"
    if not skill_bank_path.exists():
        return []
    payload = _load_json(skill_bank_path)
    return list(payload.get("segments", []))


def _build_demo_frames(
    *,
    records: list[Any],
    anchors: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    frame_manager: FrameManager,
    ik_state: IKPyState | None,
    segment_by_frame: dict[int, dict[str, str]],
) -> list[DemoPoseFrame]:
    anchor_map = {int(item["frame_index"]): item for item in anchors}
    frame_map = {int(item["frame_index"]): item for item in frames}

    demo_frames: list[DemoPoseFrame] = []
    for record in records:
        frame_index = int(record.frame_index)
        anchor = anchor_map.get(frame_index)
        if anchor is None or anchor.get("camera_xyz_m") is None:
            continue

        gripper_matrix = _record_to_gripper_pose(record, ik_state)
        gripper_pose = pose_matrix_to_pose6d(gripper_matrix)
        camera_xyz = np.asarray(anchor["camera_xyz_m"], dtype=float).reshape(1, 3)
        anchor_base = frame_manager.transform_points("base", "camera", camera_xyz)[0]
        tracking = frame_map.get(frame_index, {}).get("tracking", {})
        segment_info = segment_by_frame.get(frame_index, {})
        orientation = anchor.get("orientation", {})
        demo_frames.append(
            DemoPoseFrame(
                frame_index=frame_index,
                stable=bool(tracking.get("stable", False)),
                segment_id=segment_info.get("segment_id"),
                segment_label=segment_info.get("segment_label"),
                gripper_pose_matrix=gripper_matrix.tolist(),
                gripper_pose_6d=gripper_pose.tolist(),
                gripper_yaw_deg=float(np.degrees(np.arctan2(gripper_matrix[1, 0], gripper_matrix[0, 0]))),
                anchor_base_xyz_m=anchor_base.astype(float).tolist(),
                object_orientation_deg=float(anchor["orientation_deg"]) if anchor.get("orientation_deg") is not None else None,
                relative_xyz_m=(anchor_base - gripper_matrix[:3, 3]).astype(float).tolist(),
                depth_m=float(anchor["depth_m"]) if anchor.get("depth_m") is not None else None,
                major_axis_px=float(orientation["major_axis_px"]) if orientation.get("major_axis_px") is not None else None,
                minor_axis_px=float(orientation["minor_axis_px"]) if orientation.get("minor_axis_px") is not None else None,
            )
        )
    return demo_frames


def _resolve_live_object_payload(
    *,
    live_result: dict[str, Any],
    object_key: str,
    target_phrase: str,
) -> dict[str, Any]:
    normalized_key = _normalize_object_key(object_key)
    normalized_phrase = _normalize_object_key(target_phrase)
    candidates: list[tuple[str, dict[str, Any]]] = []
    available_identifiers: set[str] = set()

    objects = live_result.get("objects", {})
    if isinstance(objects, dict):
        for key, payload in objects.items():
            if isinstance(payload, dict):
                candidates.append((str(key), payload))

    if isinstance(live_result, dict):
        candidates.append(("root", live_result))

    for key, payload in candidates:
        payload_phrase = str(payload.get("target_phrase", ""))
        payload_label = str(payload.get("seed_detection", {}).get("label", ""))
        if key != "root":
            available_identifiers.add(str(key))
        if payload_phrase:
            available_identifiers.add(payload_phrase)
        if payload_label:
            available_identifiers.add(payload_label)
        normalized_candidates = {
            _normalize_object_key(key),
            _normalize_object_key(payload_phrase),
            _normalize_object_key(payload_label),
        }
        if normalized_key in normalized_candidates or normalized_phrase in normalized_candidates:
            return payload

    available_objects = ", ".join(sorted(available_identifiers)) if available_identifiers else "<none>"
    raise KeyError(
        f"Live result does not contain object '{object_key}' / '{target_phrase}'. "
        f"Available objects: {available_objects}."
    )


def _target_observation(target_payload: dict[str, Any], frame_manager: FrameManager) -> TargetObservation:
    base_xyz = target_payload.get("base_xyz_m")
    if base_xyz is None:
        camera_xyz = target_payload.get("anchor", {}).get("camera_xyz_m")
        if camera_xyz is None:
            raise ValueError("Live result must contain either base_xyz_m or anchor.camera_xyz_m.")
        base_xyz = apply_transform_points(frame_manager.get_transform("base", "camera"), [camera_xyz])[0].tolist()

    orientation = target_payload.get("orientation", {})
    anchor = target_payload.get("anchor", {})
    return TargetObservation(
        anchor_base_xyz_m=[float(value) for value in base_xyz],
        object_orientation_deg=(
            float(orientation["angle_deg"])
            if orientation.get("angle_deg") is not None
            else (float(anchor["orientation_deg"]) if anchor.get("orientation_deg") is not None else None)
        ),
        depth_m=float(anchor["depth_m"]) if anchor.get("depth_m") is not None else None,
        major_axis_px=float(orientation["major_axis_px"]) if orientation.get("major_axis_px") is not None else None,
        minor_axis_px=float(orientation["minor_axis_px"]) if orientation.get("minor_axis_px") is not None else None,
    )


def _build_pose_adapter(
    *,
    intrinsics: CameraIntrinsics,
    config: PosePipelineConfig,
    align_target_orientation: bool,
    orientation_policy: str | None,
) -> PoseAdapter:
    return PoseAdapter(
        intrinsics,
        config=PoseAdapterConfig(
            reference_window_size=config.reference_window_size,
            align_target_orientation=bool(align_target_orientation),
            orientation_policy=_clean_phrase(orientation_policy) or None,
        ),
        geometry_config=GeometryCompensatorConfig(
            height_gain=config.height_gain,
            max_height_adjustment_m=config.max_height_adjustment_m,
        ),
    )


def _segment_distance_stats(demo_frames: list[DemoPoseFrame]) -> dict[str, float]:
    if not demo_frames:
        return {}
    distances = np.linalg.norm(np.asarray([frame.relative_xyz_m for frame in demo_frames], dtype=float), axis=1)
    return {
        "median_object_gripper_distance_m": float(np.median(distances)),
        "min_object_gripper_distance_m": float(np.min(distances)),
    }


def _segment_semantic_state(segment: dict[str, Any]) -> str:
    metrics = segment.get("metrics")
    if not isinstance(metrics, dict):
        return ""
    return _clean_phrase(metrics.get("semantic_state"))


def _semantic_matches_reference(semantic_state: str, reference_phrase: str) -> bool:
    semantic_key = _compact_object_key(semantic_state)
    reference_key = _compact_object_key(reference_phrase)
    return bool(reference_key) and reference_key in semantic_key


def _semantic_segment_assignment(
    *,
    segment: dict[str, Any],
    primary_object_key: str,
    secondary_object_key: str | None,
    primary_reference_phrase: str,
    secondary_reference_phrase: str,
) -> tuple[str, str] | None:
    semantic_state = _segment_semantic_state(segment)
    if not semantic_state:
        return None

    semantic_key = _normalize_object_key(semantic_state)
    semantic_action, _, _ = semantic_key.partition("_")
    matches_primary = _semantic_matches_reference(semantic_state, primary_reference_phrase)
    matches_secondary = bool(secondary_object_key) and _semantic_matches_reference(
        semantic_state,
        secondary_reference_phrase,
    )

    if semantic_action in {"grasp", "pick", "pickup", "lift", "take"}:
        return primary_object_key, "semantic_primary_action"
    if semantic_action in {"place", "release", "drop", "insert", "pour"}:
        if secondary_object_key is not None:
            return secondary_object_key, "semantic_secondary_action"
        return primary_object_key, "semantic_single_object_action"
    if matches_secondary and secondary_object_key is not None:
        return secondary_object_key, "semantic_secondary_match"
    if matches_primary:
        return primary_object_key, "semantic_primary_match"
    return None


def _assign_segment_objects(
    *,
    segments: list[dict[str, Any]],
    object_demo_frames: dict[str, list[DemoPoseFrame]],
    primary_object_key: str,
    secondary_object_key: str | None,
    primary_reference_phrase: str,
    secondary_reference_phrase: str,
) -> dict[str, dict[str, Any]]:
    grasp_index = next(
        (index for index, segment in enumerate(segments) if str(segment.get("label", "")) == "gripper_close"),
        None,
    )
    assignments: dict[str, dict[str, Any]] = {}
    for index, segment in enumerate(segments):
        segment_id = str(segment["segment_id"])
        distance_by_object = {}
        for object_key, frames in object_demo_frames.items():
            segment_frames = [frame for frame in frames if frame.segment_id == segment_id]
            distance_by_object[object_key] = _segment_distance_stats(segment_frames)

        semantic_assignment = _semantic_segment_assignment(
            segment=segment,
            primary_object_key=primary_object_key,
            secondary_object_key=secondary_object_key,
            primary_reference_phrase=primary_reference_phrase,
            secondary_reference_phrase=secondary_reference_phrase,
        )
        if semantic_assignment is not None:
            object_key, reason = semantic_assignment
        elif secondary_object_key is None or grasp_index is None:
            object_key = primary_object_key
            reason = "single_object_fallback" if secondary_object_key is None else "missing_grasp_boundary"
        elif index <= grasp_index:
            object_key = primary_object_key
            reason = "pregrasp_or_grasp"
        else:
            object_key = secondary_object_key
            reason = "postgrasp"

        assignments[segment_id] = {
            "segment_id": segment_id,
            "label": str(segment.get("label", "")),
            "invariance": str(segment.get("invariance", "")),
            "object_key": object_key,
            "reason": reason,
            "semantic_state": _segment_semantic_state(segment),
            "distance_by_object": distance_by_object,
        }
    return assignments


def _build_reference_payload(
    *,
    frames: list[DemoPoseFrame],
    reference_frame_indices: list[int],
    object_key: str,
    target_phrase: str,
    reference_scope: str,
) -> list[dict[str, Any]]:
    frame_index_set = set(int(frame_index) for frame_index in reference_frame_indices)
    payloads: list[dict[str, Any]] = []
    for frame in frames:
        if frame.frame_index not in frame_index_set:
            continue
        payload = frame.to_dict()
        payload["object_key"] = object_key
        payload["target_phrase"] = target_phrase
        payload["reference_scope"] = reference_scope
        payloads.append(payload)
    return payloads


def run_pose_adaptation_pipeline(config: PosePipelineConfig) -> dict[str, Any]:
    if not config.live_result_path.exists():
        raise FileNotFoundError(f"Live result not found: {config.live_result_path}")

    records = load_frame_records(config.session_dir)
    live_result = _load_json(config.live_result_path)
    frame_manager, handeye_payload = _load_frame_manager(config)
    ik_state = build_ikpy_state()
    segment_by_frame = _segment_lookup(config.analysis_dir)
    segments = _segment_payload(config.analysis_dir)
    object_specs = [
        {
            "object_key": _normalize_object_key(config.primary_target_phrase),
            "target_phrase": config.primary_target_phrase,
            "reference_phrase": _clean_phrase(config.primary_reference_phrase) or config.primary_target_phrase,
            "vision_dir": config.analysis_dir / config.primary_vision_dir_name,
            "align_target_orientation": bool(config.align_target_orientation),
            "orientation_policy": _clean_phrase(config.primary_orientation_policy) or None,
        }
    ]
    secondary_vision_dir = config.analysis_dir / config.secondary_vision_dir_name
    if config.secondary_target_phrase and secondary_vision_dir.exists():
        object_specs.append(
            {
                "object_key": _normalize_object_key(config.secondary_target_phrase),
                "target_phrase": config.secondary_target_phrase,
                "reference_phrase": _clean_phrase(config.secondary_reference_phrase) or config.secondary_target_phrase,
                "vision_dir": secondary_vision_dir,
                "align_target_orientation": bool(config.secondary_align_target_orientation),
                "orientation_policy": _clean_phrase(config.secondary_orientation_policy) or None,
            }
        )
    elif config.secondary_target_phrase:
        logging.warning("Secondary T4 vision directory is missing: %s", secondary_vision_dir)

    intrinsics = CameraIntrinsics.from_json(config.intrinsics_path)
    adapters = {
        str(spec["object_key"]): _build_pose_adapter(
            intrinsics=intrinsics,
            config=config,
            align_target_orientation=bool(spec.get("align_target_orientation", False)),
            orientation_policy=spec.get("orientation_policy"),
        )
        for spec in object_specs
    }

    object_demo_frames: dict[str, list[DemoPoseFrame]] = {}
    object_targets: dict[str, dict[str, Any]] = {}
    object_meta: dict[str, dict[str, Any]] = {}
    all_demo_frames_payload: list[dict[str, Any]] = []
    for spec in object_specs:
        anchors_path = spec["vision_dir"] / "anchors.json"
        frames_path = spec["vision_dir"] / "frames.json"
        if not anchors_path.exists() or not frames_path.exists():
            raise FileNotFoundError(f"Missing T4 outputs under {spec['vision_dir']}.")

        demo_frames = _build_demo_frames(
            records=records,
            anchors=_load_json(anchors_path),
            frames=_load_json(frames_path),
            frame_manager=frame_manager,
            ik_state=ik_state,
            segment_by_frame=segment_by_frame,
        )
        object_key = str(spec["object_key"])
        object_demo_frames[object_key] = demo_frames
        target_payload = _resolve_live_object_payload(
            live_result=live_result,
            object_key=object_key,
            target_phrase=str(spec["target_phrase"]),
        )
        object_targets[object_key] = target_payload
        object_meta[object_key] = {
            "target_phrase": str(spec["target_phrase"]),
            "reference_phrase": str(spec["reference_phrase"]),
            "vision_dir": str(spec["vision_dir"]),
            "demo_frame_count": len(demo_frames),
            "live_base_xyz_m": target_payload.get("base_xyz_m"),
            "align_target_orientation": bool(spec.get("align_target_orientation", False)),
            "orientation_policy": _clean_phrase(spec.get("orientation_policy")) or None,
        }
        for frame in demo_frames:
            payload = frame.to_dict()
            payload["object_key"] = object_key
            payload["target_phrase"] = str(spec["target_phrase"])
            all_demo_frames_payload.append(payload)

    secondary_object_key = (
        _normalize_object_key(config.secondary_target_phrase)
        if _normalize_object_key(config.secondary_target_phrase) in object_demo_frames
        else None
    )
    segment_assignments = _assign_segment_objects(
        segments=segments,
        object_demo_frames=object_demo_frames,
        primary_object_key=_normalize_object_key(config.primary_target_phrase),
        secondary_object_key=secondary_object_key,
        primary_reference_phrase=_clean_phrase(config.primary_reference_phrase) or config.primary_target_phrase,
        secondary_reference_phrase=_clean_phrase(config.secondary_reference_phrase) or config.secondary_target_phrase,
    )

    object_adaptations: dict[str, dict[str, Any]] = {}
    reference_frames_payload: list[dict[str, Any]] = []
    for object_key, demo_frames in object_demo_frames.items():
        adapter = adapters[object_key]
        target_phrase = str(object_meta[object_key]["target_phrase"])
        object_result = adapter.adapt(
            demo_frames,
            _target_observation(object_targets[object_key], frame_manager),
        )
        payload = object_result.to_dict()
        payload["object_key"] = object_key
        payload["target_phrase"] = target_phrase
        payload["vision_dir"] = object_meta[object_key]["vision_dir"]
        payload["frame_source"] = "object_global"
        object_adaptations[object_key] = payload
        reference_frames_payload.extend(
            _build_reference_payload(
                frames=demo_frames,
                reference_frame_indices=object_result.reference_frame_indices,
                object_key=object_key,
                target_phrase=target_phrase,
                reference_scope="object_global",
            )
        )

    segment_adaptations: dict[str, dict[str, Any]] = {}
    for segment in segments:
        if str(segment.get("invariance", "")) != "var":
            continue
        segment_id = str(segment["segment_id"])
        assignment = segment_assignments[segment_id]
        object_key = str(assignment["object_key"])
        adapter = adapters[object_key]
        target_phrase = str(object_meta[object_key]["target_phrase"])
        demo_frames = [frame for frame in object_demo_frames[object_key] if frame.segment_id == segment_id]
        frame_source = "segment"
        if len(demo_frames) < max(1, int(adapter.config.min_reference_frames)):
            demo_frames = object_demo_frames[object_key]
            frame_source = "object_fallback"

        result = adapter.adapt(
            demo_frames,
            _target_observation(object_targets[object_key], frame_manager),
        )
        payload = result.to_dict()
        payload["object_key"] = object_key
        payload["target_phrase"] = target_phrase
        payload["vision_dir"] = object_meta[object_key]["vision_dir"]
        payload["frame_source"] = frame_source
        payload["segment_id"] = segment_id
        payload["segment_label"] = str(segment.get("label", ""))
        segment_adaptations[segment_id] = payload

        for reference_payload in _build_reference_payload(
            frames=demo_frames,
            reference_frame_indices=result.reference_frame_indices,
            object_key=object_key,
            target_phrase=target_phrase,
            reference_scope=f"segment:{segment_id}",
        ):
            reference_payload["reference_segment_id"] = segment_id
            reference_frames_payload.append(reference_payload)

    primary_result = next(
        (
            segment_adaptations[str(segment["segment_id"])]
            for segment in segments
            if str(segment.get("segment_id")) in segment_adaptations
            and segment_adaptations[str(segment["segment_id"])]["object_key"]
            == _normalize_object_key(config.primary_target_phrase)
        ),
        next(iter(segment_adaptations.values()), None),
    )
    if primary_result is None:
        raise ValueError("Failed to produce any segment-level pose adaptation results.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    demo_frames_path = config.output_dir / "demo_frames.json"
    reference_frames_path = config.output_dir / "reference_frames.json"
    adapted_pose_path = config.output_dir / "adapted_pose.json"
    summary_path = config.output_dir / "summary.json"
    adapted_pose = {
        **primary_result,
        "status": "ok",
        "mode": "segment_multi_object",
        "object_meta": object_meta,
        "object_adaptations": object_adaptations,
        "segment_assignments": segment_assignments,
        "segment_adaptations": segment_adaptations,
    }
    _save_json(demo_frames_path, all_demo_frames_payload)
    _save_json(reference_frames_path, reference_frames_payload)
    _save_json(adapted_pose_path, adapted_pose)

    handeye_metrics = dict(handeye_payload.get("metrics", {})) if isinstance(handeye_payload, dict) else {}
    summary = {
        "status": "ok",
        "session_dir": str(config.session_dir),
        "analysis_dir": str(config.analysis_dir),
        "output_dir": str(config.output_dir),
        "live_result_path": str(config.live_result_path),
        "mode": "segment_multi_object",
        "demo_frame_count": len(all_demo_frames_payload),
        "reference_frame_indices": primary_result["reference_frame_indices"],
        "reference_selection": primary_result["reference_selection"],
        "alignment_mode": primary_result["alignment_mode"],
        "orientation_policy": primary_result["orientation_policy"],
        "delta_x_m": primary_result["delta_x_m"],
        "delta_theta_deg": primary_result["delta_theta_deg"],
        "reference_gripper_yaw_deg": primary_result["reference_gripper_yaw_deg"],
        "adapted_gripper_yaw_deg": primary_result["adapted_gripper_yaw_deg"],
        "applied_yaw_delta_deg": primary_result["applied_yaw_delta_deg"],
        "delta_h_m": primary_result["delta_h_m"],
        "adapted_gripper_pose_6d": primary_result["adapted_gripper_pose_6d"],
        "adapted_pose_path": str(adapted_pose_path),
        "demo_frames_path": str(demo_frames_path),
        "reference_frames_path": str(reference_frames_path),
        "summary_path": str(summary_path),
        "primary_target_phrase": config.primary_target_phrase,
        "secondary_target_phrase": config.secondary_target_phrase,
        "primary_reference_phrase": _clean_phrase(config.primary_reference_phrase) or config.primary_target_phrase,
        "secondary_reference_phrase": _clean_phrase(config.secondary_reference_phrase) or config.secondary_target_phrase,
        "object_meta": object_meta,
        "object_adaptations": object_adaptations,
        "object_adaptation_count": len(object_adaptations),
        "segment_assignments": segment_assignments,
        "segment_adaptation_count": len(segment_adaptations),
        "handeye_status": {
            "path": str(config.handeye_result_path),
            "passed": handeye_payload.get("passed") if isinstance(handeye_payload, dict) else None,
            "accepted_without_passing_thresholds": (
                handeye_payload.get("accepted_without_passing_thresholds") if isinstance(handeye_payload, dict) else None
            ),
            "translation_mean_mm": handeye_metrics.get("translation_mean_mm"),
            "rotation_mean_deg": handeye_metrics.get("rotation_mean_deg"),
        },
    }
    _save_json(summary_path, summary)
    return summary


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)

    config = _build_config(args)
    summary = run_pose_adaptation_pipeline(config)
    logging.info("T5 pose adaptation output: %s", summary["output_dir"])
    logging.info("T5 summary: %s", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
