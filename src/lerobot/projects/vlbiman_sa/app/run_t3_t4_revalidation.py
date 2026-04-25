#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


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
from lerobot.projects.vlbiman_sa.geometry.transforms import apply_transform_points, rotation_error_deg, translation_error_m
from lerobot.projects.vlbiman_sa.skills import KeyposeSegmenter, SegmenterConfig, SkillBank
from lerobot.projects.vlbiman_sa.trajectory.progressive_ik import (
    IKPyState,
    build_ikpy_state,
    forward_kinematics_tool,
    full_q_from_arm_q,
    tool_pose_to_tip_pose,
)


@dataclass(slots=True)
class RevalidationConfig:
    session_dir: Path
    analysis_dir: Path
    output_dir: Path
    skill_build_config: Path
    handeye_result_path: Path
    audit_frame_count: int = 30


def _default_session_dir() -> Path:
    return Path("outputs/vlbiman_sa/recordings/one_shot_20260323T185847Z")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Revalidate T3/T4 against task-plan criteria.")
    parser.add_argument("--session-dir", type=Path, default=_default_session_dir())
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--skill-build-config",
        type=Path,
        default=Path("src/lerobot/projects/vlbiman_sa/configs/skill_build.yaml"),
    )
    parser.add_argument(
        "--handeye-result-path",
        type=Path,
        default=Path("outputs/vlbiman_sa/calib/handeye_result.json"),
    )
    parser.add_argument("--audit-frame-count", type=int, default=30)
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> RevalidationConfig:
    session_dir = args.session_dir
    analysis_dir = args.analysis_dir or (session_dir / "analysis")
    output_dir = args.output_dir or (analysis_dir / "revalidation")
    return RevalidationConfig(
        session_dir=session_dir,
        analysis_dir=analysis_dir,
        output_dir=output_dir,
        skill_build_config=args.skill_build_config,
        handeye_result_path=args.handeye_result_path,
        audit_frame_count=max(1, int(args.audit_frame_count)),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config, got {type(payload).__name__}.")
    return payload


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _full_q(record: Any, state: IKPyState) -> np.ndarray:
    arm_q: list[float] = []
    for joint_name, idx in zip(state.arm_joint_names, state.link_indices, strict=True):
        if joint_name in record.joint_positions:
            value = record.joint_positions[joint_name]
        else:
            value = record.joint_positions[f"{joint_name}.pos"]
        arm_q.append(float(value))
    return full_q_from_arm_q(state, np.asarray(arm_q, dtype=float))


def _select_min_delta_solution(candidate: np.ndarray, seed: np.ndarray, state: IKPyState) -> np.ndarray:
    normalized = np.asarray(candidate, dtype=float).copy()
    for idx in range(normalized.shape[0]):
        low = float(state.lower_bounds[idx])
        high = float(state.upper_bounds[idx])
        value = float(normalized[idx])
        best = value
        best_delta = abs(value - float(seed[idx]))
        for sign in (-1, 1):
            alt = value + sign * 2.0 * np.pi
            if alt < low or alt > high:
                continue
            delta = abs(alt - float(seed[idx]))
            if delta < best_delta:
                best = alt
                best_delta = delta
        normalized[idx] = best
    return np.minimum(np.maximum(normalized, state.lower_bounds), state.upper_bounds)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    scale = min(width / max(src_w, 1), height / max(src_h, 1))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _save_montage(images: list[np.ndarray], output_path: Path, title: str, cols: int = 5) -> None:
    if not images:
        return
    h, w = images[0].shape[:2]
    cols = min(cols, len(images))
    rows = int(math.ceil(len(images) / cols))
    grid = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)
    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = image
    header = np.full((60, grid.shape[1], 3), 250, dtype=np.uint8)
    cv2.putText(header, title, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.vstack([header, grid]))


def _frame_indices_evenly(items: list[int], count: int) -> list[int]:
    if len(items) <= count:
        return items
    positions = np.linspace(0, len(items) - 1, count)
    return sorted({items[int(round(pos))] for pos in positions})


def _t3_boundary_validation(records: list[Any], bank: SkillBank, segmenter_config: SegmenterConfig) -> dict[str, Any]:
    features = KeyposeSegmenter(segmenter_config).extract_features(records)
    velocity_threshold = float(np.quantile(features.velocity_norm, segmenter_config.velocity_quantile))
    acceleration_threshold = float(np.quantile(features.acceleration_norm, segmenter_config.acceleration_quantile))
    gripper_threshold = float(segmenter_config.gripper_delta_threshold)

    velocity_events = np.flatnonzero(features.velocity_norm >= velocity_threshold)
    acceleration_events = np.flatnonzero(features.acceleration_norm >= acceleration_threshold)
    gripper_events = np.flatnonzero(features.gripper_delta >= gripper_threshold)

    def _nearest_distance(events: np.ndarray, boundary: int) -> int | None:
        if events.size == 0:
            return None
        return int(np.min(np.abs(events - boundary)))

    boundary_checks = []
    aligned_count = 0
    for previous, current in zip(bank.segments[:-1], bank.segments[1:], strict=True):
        boundary = int(current.start_frame)
        nearest_velocity = _nearest_distance(velocity_events, boundary)
        nearest_acceleration = _nearest_distance(acceleration_events, boundary)
        nearest_gripper = _nearest_distance(gripper_events, boundary)
        nearest = min(
            [item for item in (nearest_velocity, nearest_acceleration, nearest_gripper) if item is not None],
            default=None,
        )
        aligned = nearest is not None and nearest <= 2
        aligned_count += int(aligned)
        boundary_checks.append(
            {
                "boundary_frame": boundary,
                "left_segment": previous.segment_id,
                "right_segment": current.segment_id,
                "nearest_velocity_event_frames": nearest_velocity,
                "nearest_acceleration_event_frames": nearest_acceleration,
                "nearest_gripper_event_frames": nearest_gripper,
                "aligned_with_physical_change": aligned,
            }
        )

    return {
        "status": "pass" if aligned_count == len(boundary_checks) else "warn",
        "boundary_count": len(boundary_checks),
        "aligned_count": aligned_count,
        "aligned_ratio": float(aligned_count / max(len(boundary_checks), 1)),
        "thresholds": {
            "velocity_quantile": segmenter_config.velocity_quantile,
            "acceleration_quantile": segmenter_config.acceleration_quantile,
            "gripper_delta_threshold": segmenter_config.gripper_delta_threshold,
            "alignment_tolerance_frames": 2,
        },
        "checks": boundary_checks,
    }


def _load_base_from_camera(config: RevalidationConfig) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    if not config.handeye_result_path.exists():
        return None, None
    payload = _load_json(config.handeye_result_path)
    matrix = np.asarray(payload.get("base_from_camera"), dtype=float)
    if matrix.shape != (4, 4):
        return None, payload
    return matrix, payload


def _segment_frame_range(segment: Any) -> range:
    return range(int(segment.start_frame), int(segment.end_frame) + 1)


def _compute_object_gripper_state(
    records: list[Any],
    anchors: list[dict[str, Any]],
    state: IKPyState,
    base_from_camera: np.ndarray | None,
) -> dict[int, dict[str, Any]]:
    anchor_map = {int(item["frame_index"]): item for item in anchors}
    frame_state: dict[int, dict[str, Any]] = {}
    for record in records:
        frame_index = int(record.frame_index)
        q = _full_q(record, state)
        gripper_tf = forward_kinematics_tool(state, q)
        gripper_xyz = gripper_tf[:3, 3].tolist()
        payload = {
            "gripper_xyz_m": gripper_xyz,
            "anchor_base_xyz_m": None,
            "relative_xyz_m": None,
        }
        anchor = anchor_map.get(frame_index)
        if anchor is not None and anchor.get("camera_xyz_m") is not None and base_from_camera is not None:
            camera_xyz = np.asarray(anchor["camera_xyz_m"], dtype=float).reshape(1, 3)
            base_xyz = apply_transform_points(base_from_camera, camera_xyz)[0]
            payload["anchor_base_xyz_m"] = base_xyz.tolist()
            payload["relative_xyz_m"] = (base_xyz - gripper_tf[:3, 3]).tolist()
        frame_state[frame_index] = payload
    return frame_state


def _t3_semantic_validation(bank: SkillBank, frame_state: dict[int, dict[str, Any]]) -> dict[str, Any]:
    checks = []
    comparable = 0
    matches = 0
    for segment in bank.segments:
        relative_vectors = []
        for frame_index in _segment_frame_range(segment):
            relative_xyz = frame_state.get(frame_index, {}).get("relative_xyz_m")
            if relative_xyz is not None:
                relative_vectors.append(np.asarray(relative_xyz, dtype=float))

        expected = "var"
        relative_span_mm = None
        median_distance_mm = None
        if relative_vectors:
            stacked = np.stack(relative_vectors, axis=0)
            relative_span_mm = float(np.max(np.ptp(stacked, axis=0)) * 1000.0)
            median_distance_mm = float(np.median(np.linalg.norm(stacked, axis=1)) * 1000.0)
            rigid_proxy = relative_span_mm <= 30.0 and median_distance_mm <= 120.0
            if segment.label not in {"gripper_open", "gripper_close"} and rigid_proxy:
                expected = "inv"
            comparable = comparable + 1
            matches = matches + int(segment.invariance == expected)

        checks.append(
            {
                "segment_id": segment.segment_id,
                "label": segment.label,
                "predicted_invariance": segment.invariance,
                "expected_invariance_proxy": expected,
                "matches_proxy": segment.invariance == expected if relative_vectors else None,
                "relative_span_mm": relative_span_mm,
                "median_object_gripper_distance_mm": median_distance_mm,
                "valid_frame_count": len(relative_vectors),
            }
        )

    status = "pass" if comparable > 0 and matches == comparable else "warn"
    if comparable == 0:
        status = "blocked"
    return {
        "status": status,
        "comparable_segments": comparable,
        "matched_segments": matches,
        "matched_ratio": float(matches / comparable) if comparable else None,
        "proxy_definition": {
            "rigid_relative_span_mm_max": 30.0,
            "median_object_gripper_distance_mm_max": 120.0,
            "note": "Proxy bind check using object-center/base and gripper/base relative-vector stability.",
        },
        "checks": checks,
    }


def _t3_ik_validation(records: list[Any], bank: SkillBank, state: IKPyState) -> dict[str, Any]:
    checks = []
    pass_count = 0
    for previous, current in zip(bank.segments[:-1], bank.segments[1:], strict=True):
        seed_record = records[int(previous.end_frame)]
        target_record = records[int(current.start_frame)]
        q_seed = _full_q(seed_record, state)
        q_target = _full_q(target_record, state)
        target_tf = forward_kinematics_tool(state, q_target)
        raw_solution = np.asarray(
            state.chain.inverse_kinematics_frame(
                tool_pose_to_tip_pose(state, target_tf),
                initial_position=q_seed,
                orientation_mode="all",
                regularization_parameter=1e-4,
                optimizer="least_squares",
            ),
            dtype=float,
        )
        solution = _select_min_delta_solution(raw_solution, q_seed, state)
        solved_tf = forward_kinematics_tool(state, solution)
        trans_error_mm = translation_error_m(target_tf, solved_tf) * 1000.0
        rot_error = rotation_error_deg(target_tf, solved_tf)
        max_joint_step = float(np.max(np.abs(solution - q_seed)))
        solved = trans_error_mm <= 5.0 and rot_error <= 5.0 and max_joint_step <= 0.35
        pass_count += int(solved)
        checks.append(
            {
                "from_segment": previous.segment_id,
                "to_segment": current.segment_id,
                "translation_error_mm": float(trans_error_mm),
                "rotation_error_deg": float(rot_error),
                "max_joint_step_rad": max_joint_step,
                "reachable": solved,
            }
        )
    return {
        "status": "pass" if pass_count == len(checks) else "warn",
        "transition_count": len(checks),
        "reachable_count": pass_count,
        "reachable_ratio": float(pass_count / max(len(checks), 1)),
        "thresholds": {
            "translation_error_mm_max": 5.0,
            "rotation_error_deg_max": 5.0,
            "max_joint_step_rad_max": 0.35,
        },
        "checks": checks,
    }


def _sample_t4_frames(frames: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered = sorted(frames, key=lambda item: int(item["frame_index"]))
    selected_indices = _frame_indices_evenly([int(item["frame_index"]) for item in ordered], count)
    selected_set = set(selected_indices)
    return [item for item in ordered if int(item["frame_index"]) in selected_set]


def _export_t4_audit_images(config: RevalidationConfig, sampled_frames: list[dict[str, Any]]) -> dict[str, Any]:
    overlay_dir = config.analysis_dir / "t4_vision" / "overlays"
    output_dir = config.output_dir / "t4_mask_audit" / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    montage_images = []
    cards = []
    for item in sampled_frames:
        frame_index = int(item["frame_index"])
        overlay_path = overlay_dir / f"frame_{frame_index:06d}.png"
        image = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = _fit_image(image, 640, 360)
        header = np.full((90, image.shape[1], 3), 250, dtype=np.uint8)
        tracking = item.get("tracking", {})
        cv2.putText(header, f"frame={frame_index}", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(
            header,
            f"stable={tracking.get('stable')} iou={tracking.get('temporal_iou', 0.0):.3f}",
            (18, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        card = np.vstack([header, image])
        card_path = output_dir / f"frame_{frame_index:06d}.png"
        cv2.imwrite(str(card_path), card)
        montage_images.append(card)
        cards.append({"frame_index": frame_index, "card_path": str(card_path)})

    montage_path = config.output_dir / "t4_mask_audit" / "t4_mask_audit_montage.png"
    _save_montage(montage_images, montage_path, "T4 Mask Audit Frames", cols=5)
    return {
        "frame_count": len(cards),
        "frame_dir": str(output_dir),
        "montage_path": str(montage_path),
        "cards": cards,
    }


def _t4_mask_validation(config: RevalidationConfig, frames: list[dict[str, Any]]) -> dict[str, Any]:
    sampled_frames = _sample_t4_frames(frames, config.audit_frame_count)
    audit_visuals = _export_t4_audit_images(config, sampled_frames)
    component_ratio = float(
        np.mean([1.0 if int(item.get("component_count", 0)) == 1 else 0.0 for item in sampled_frames])
    )
    mean_temporal_iou = float(
        np.mean([float(item.get("tracking", {}).get("temporal_iou", 0.0)) for item in sampled_frames])
    )
    return {
        "status": "blocked",
        "strict_iou_evaluation": {
            "status": "blocked",
            "reason": "No 30-frame manual ground-truth mask set is present, so true Mask IoU > 0.75 cannot be computed.",
        },
        "proxy_metrics": {
            "sample_frame_count": len(sampled_frames),
            "single_component_ratio": component_ratio,
            "mean_temporal_iou_sample": mean_temporal_iou,
        },
        "manual_audit": audit_visuals,
    }


def _t4_base_validation(
    config: RevalidationConfig,
    anchors: list[dict[str, Any]],
    frame_state: dict[int, dict[str, Any]],
    handeye_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    output_path = config.output_dir / "t4_base_anchors.json"
    base_payload = []
    relative_vectors = []
    for anchor in anchors:
        frame_index = int(anchor["frame_index"])
        frame_info = frame_state.get(frame_index, {})
        base_xyz = frame_info.get("anchor_base_xyz_m")
        relative_xyz = frame_info.get("relative_xyz_m")
        base_payload.append(
            {
                "frame_index": frame_index,
                "base_xyz_m": base_xyz,
                "relative_to_gripper_xyz_m": relative_xyz,
            }
        )
        if relative_xyz is not None:
            relative_vectors.append(np.asarray(relative_xyz, dtype=float))
    _save_json(output_path, base_payload)

    if handeye_payload is None:
        return {
            "status": "blocked",
            "reason": "No base_from_camera transform is available.",
            "base_anchor_path": str(output_path),
        }

    relative_span_mm = None
    if relative_vectors:
        stacked = np.stack(relative_vectors, axis=0)
        relative_span_mm = float(np.max(np.ptp(stacked, axis=0)) * 1000.0)

    return {
        "status": "warn",
        "absolute_error_evaluation": {
            "status": "blocked",
            "reason": "No robot-base ground-truth object position is available, so absolute 3D localization error cannot be computed.",
        },
        "base_transform_status": {
            "handeye_passed": bool(handeye_payload.get("passed")),
            "accepted_without_passing_thresholds": bool(handeye_payload.get("accepted_without_passing_thresholds")),
            "translation_mean_mm": handeye_payload.get("metrics", {}).get("translation_mean_mm"),
            "rotation_mean_deg": handeye_payload.get("metrics", {}).get("rotation_mean_deg"),
        },
        "proxy_metrics": {
            "relative_object_gripper_span_mm": relative_span_mm,
            "valid_base_anchor_count": int(sum(1 for item in base_payload if item["base_xyz_m"] is not None)),
        },
        "base_anchor_path": str(output_path),
    }


def main() -> int:
    args = _parse_args()
    config = _build_config(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_frame_records(config.session_dir)
    bank = SkillBank.load(config.analysis_dir / "t3_skill_bank" / "skill_bank.json")
    anchors = _load_json(config.analysis_dir / "t4_vision" / "anchors.json")
    frames = _load_json(config.analysis_dir / "t4_vision" / "frames.json")

    skill_cfg_payload = _load_yaml(config.skill_build_config)
    segmenter_config = SegmenterConfig(**dict(skill_cfg_payload.get("segmenter", {})))
    ik_state = build_ikpy_state()
    base_from_camera, handeye_payload = _load_base_from_camera(config)
    frame_state = _compute_object_gripper_state(records, anchors, ik_state, base_from_camera)

    t3_boundary = _t3_boundary_validation(records, bank, segmenter_config)
    t3_semantic = _t3_semantic_validation(bank, frame_state)
    t3_ik = _t3_ik_validation(records, bank, ik_state)
    t4_mask = _t4_mask_validation(config, frames)
    t4_base = _t4_base_validation(config, anchors, frame_state, handeye_payload)

    summary = {
        "status": "completed",
        "session_dir": str(config.session_dir),
        "analysis_dir": str(config.analysis_dir),
        "output_dir": str(config.output_dir),
        "t3": {
            "boundary": t3_boundary["status"],
            "semantic": t3_semantic["status"],
            "ik": t3_ik["status"],
        },
        "t4": {
            "mask": t4_mask["status"],
            "base_3d": t4_base["status"],
        },
    }

    _save_json(config.output_dir / "t3_boundary_validation.json", t3_boundary)
    _save_json(config.output_dir / "t3_semantic_validation.json", t3_semantic)
    _save_json(config.output_dir / "t3_ik_validation.json", t3_ik)
    _save_json(config.output_dir / "t4_mask_validation.json", t4_mask)
    _save_json(config.output_dir / "t4_base_validation.json", t4_base)
    _save_json(config.output_dir / "summary.json", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
