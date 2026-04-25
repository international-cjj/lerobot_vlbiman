#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


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


def _bootstrap_pythonpath() -> Path:
    repo_root = Path(__file__).resolve().parents[5]
    for candidate in (repo_root / "src", repo_root, repo_root / "lerobot_robot_cjjarm"):
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    return repo_root


_maybe_reexec_in_repo_venv()
REPO_ROOT = _bootstrap_pythonpath()

from lerobot.projects.vlbiman_sa.grasp.closed_loop_controller import FRRGClosedLoopController
from lerobot.projects.vlbiman_sa.grasp.contracts import GraspState, Pose6D, load_frrg_config
from lerobot.projects.vlbiman_sa.grasp.frame_math import pose6d_to_matrix
from lerobot.utils.rotation import Rotation
from lerobot_robot_cjjarm import CjjArmSim, CjjArmSimConfig
from lerobot_robot_cjjarm.kinematics import compose_pose_delta
import mujoco


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "frrg_grasp.yaml"


def _default_output_root() -> Path:
    return REPO_ROOT / "outputs" / "vlbiman_sa" / "frrg" / "mujoco_validate"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay FRRG safe actions inside CJJ arm MuJoCo sim and record metrics.")
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument("--mock-state", type=Path, required=True, help="FRRG mock state fixture to run.")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument(
        "--trajectory-points",
        type=Path,
        default=None,
        help="Optional T6 trajectory_points.json used to anchor FRRG replay from a reachable coarse pose.",
    )
    parser.add_argument(
        "--start-point-index",
        type=int,
        default=-1,
        help="Point index inside trajectory_points.json. Default -1 means the last point.",
    )
    parser.add_argument("--output-root", type=Path, default=_default_output_root())
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-fps", type=float, default=8.0)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(payload).__name__}.")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _make_run_id(explicit: str | None, mock_state_path: Path) -> str:
    if explicit:
        return explicit
    return mock_state_path.stem


def _pose_payload_to_rotvec(pose_payload: dict[str, Any]) -> np.ndarray:
    transform = pose6d_to_matrix(Pose6D(xyz=tuple(pose_payload["xyz"]), rpy=tuple(pose_payload["rpy"])))
    return np.concatenate([transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_rotvec()], axis=0)


def _matrix_to_rotvec_pose(matrix_payload: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(matrix_payload, dtype=float)
    return np.concatenate([matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_rotvec()], axis=0)


def _object_delta_to_base(delta_pose_object: list[float], object_pose_payload: dict[str, Any]) -> np.ndarray:
    object_transform = pose6d_to_matrix(Pose6D(xyz=tuple(object_pose_payload["xyz"]), rpy=tuple(object_pose_payload["rpy"])))
    object_rotation = object_transform[:3, :3]
    delta = np.asarray(delta_pose_object, dtype=float)
    base_delta = np.zeros(6, dtype=float)
    base_delta[:3] = object_rotation @ delta[:3]
    base_delta[3:] = object_rotation @ delta[3:]
    return base_delta


def _joint_positions_from_observation(obs: dict[str, Any]) -> np.ndarray:
    return np.asarray([obs[f"joint_{idx}.pos"] for idx in range(1, 7)], dtype=float)


def _translation_error_mm(target_pose: np.ndarray, achieved_pose: np.ndarray) -> float:
    return float(np.linalg.norm(target_pose[:3] - achieved_pose[:3]) * 1000.0)


def _rotation_error_deg(target_pose: np.ndarray, achieved_pose: np.ndarray) -> float:
    target_rot = Rotation.from_rotvec(target_pose[3:]).as_matrix()
    achieved_rot = Rotation.from_rotvec(achieved_pose[3:]).as_matrix()
    delta = Rotation.from_matrix(target_rot @ achieved_rot.T).as_rotvec()
    return float(np.rad2deg(np.linalg.norm(delta)))


def _gripper_command_value(gripper_width_m: float, handoff_open_width_m: float) -> float:
    return -1.0 if float(gripper_width_m) <= float(handoff_open_width_m) * 0.5 else 0.0


def _gripper_target_for_width(sim: CjjArmSim, gripper_width_m: float, handoff_open_width_m: float) -> float:
    return (
        float(sim.config.gripper_closed_pos)
        if _gripper_command_value(gripper_width_m, handoff_open_width_m) < 0.0
        else float(sim.config.gripper_open_pos)
    )


def _build_renderer(
    sim: CjjArmSim,
    *,
    width: int,
    height: int,
) -> tuple[mujoco.Renderer, mujoco.MjvCamera]:
    render_width = min(int(width), int(sim.model.vis.global_.offwidth))
    render_height = min(int(height), int(sim.model.vis.global_.offheight))
    renderer = mujoco.Renderer(sim.model, render_height, render_width)
    camera = mujoco.MjvCamera()
    camera.azimuth = 148.0
    camera.elevation = -20.0
    camera.distance = 1.75
    camera.lookat[:] = np.asarray([0.0, 0.0, 0.24], dtype=float)
    return renderer, camera


def _render_frame(sim: CjjArmSim, renderer: mujoco.Renderer, camera: mujoco.MjvCamera) -> np.ndarray:
    mujoco.mj_forward(sim.model, sim.data)
    renderer.update_scene(sim.data, camera=camera)
    return np.asarray(renderer.render(), dtype=np.uint8)


def _solve_and_apply_pose_target(
    sim: CjjArmSim,
    *,
    target_pose: np.ndarray,
    seed_joints: np.ndarray,
    gripper_width_m: float,
    handoff_open_width_m: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    ik_solution = sim.kinematics.compute_ik(
        target_pose,
        seed_joints,
        position_weight=1.0,
        orientation_weight=0.05,
        keep_pointing_only=True,
    )
    sim._apply_joint_targets(
        ik_solution,
        _gripper_target_for_width(sim, gripper_width_m, handoff_open_width_m),
    )
    obs = sim.get_observation()
    joint_positions = _joint_positions_from_observation(obs)
    achieved_pose = sim.kinematics.compute_fk(joint_positions)
    return obs, joint_positions, achieved_pose


def _overlay_text(
    frame_bgr: np.ndarray,
    *,
    step_index: int,
    phase: str,
    controller_final_phase: str,
    translation_error_mm: float,
    rotation_error_deg: float,
    gripper_width_m: float,
) -> np.ndarray:
    overlay = frame_bgr.copy()
    lines = [
        f"step={step_index} phase={phase}",
        f"controller_final={controller_final_phase}",
        f"track_err={translation_error_mm:.2f}mm rot_err={rotation_error_deg:.2f}deg",
        f"gripper_width={gripper_width_m:.4f}m",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(
            overlay,
            line,
            (18, 32 + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


def _run_controller(config_path: Path, mock_state_path: Path, max_steps: int) -> tuple[dict[str, Any], Any]:
    config = load_frrg_config(config_path)
    payload = _load_json(mock_state_path)
    controller = FRRGClosedLoopController(config, payload)
    result = controller.run(max_steps=int(max_steps))
    summary = result.summary_dict(
        config_path=str(config_path.resolve()),
        mock_state_path=str(mock_state_path.resolve()),
        output_dir="",
        max_steps=int(max_steps),
        input_mode=config.runtime.default_input_mode,
        input_summary={"fixture": mock_state_path.name},
    )
    return summary, result


def _load_anchor_point(trajectory_points_path: Path, point_index: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload = _load_json(trajectory_points_path)
    points = payload.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError(f"trajectory_points payload at {trajectory_points_path} has no points.")
    resolved_index = int(point_index)
    if resolved_index < 0:
        resolved_index = len(points) + resolved_index
    if resolved_index < 0 or resolved_index >= len(points):
        raise IndexError(f"start-point-index {point_index} out of range for {len(points)} points.")
    point = dict(points[resolved_index])
    return (
        np.asarray(point["joint_positions"], dtype=float),
        _matrix_to_rotvec_pose(point["target_pose_matrix"]),
        {
            "anchor_mode": "trajectory_point",
            "trajectory_points_path": str(trajectory_points_path.resolve()),
            "point_index": resolved_index,
            "segment_id": point.get("segment_id"),
            "source": point.get("source"),
        },
    )


def run_mujoco_validation(
    *,
    config_path: Path,
    mock_state_path: Path,
    max_steps: int,
    trajectory_points_path: Path | None,
    start_point_index: int,
    output_root: Path,
    run_id: str | None = None,
    video_width: int = 640,
    video_height: int = 480,
    video_fps: float = 8.0,
) -> tuple[int, Path, dict[str, Any]]:
    controller_summary, controller_result = _run_controller(config_path, mock_state_path, max_steps)
    config = load_frrg_config(config_path)
    resolved_output_root = output_root.resolve()
    run_dir = resolved_output_root / _make_run_id(run_id, mock_state_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    sim = CjjArmSim(
        CjjArmSimConfig(
            render_width=int(video_width),
            render_height=int(video_height),
            use_viewer=False,
            max_delta_per_step=10.0,
            qpos_interp_step=0.05,
        )
    )
    sim.connect()
    writer = None
    renderer = None
    frame_rows: list[dict[str, Any]] = []
    try:
        states = list(controller_result.states)
        actions = list(controller_result.actions)
        if not states or not actions:
            raise RuntimeError("Controller produced no states/actions for MuJoCo validation.")

        initial_state = dict(states[0]["state"])
        initial_gripper_width = float(initial_state["gripper_width"])
        anchor_meta: dict[str, Any]
        if trajectory_points_path is not None:
            initial_joint_positions, initial_target_pose, anchor_meta = _load_anchor_point(
                trajectory_points_path,
                start_point_index,
            )
            sim._apply_joint_targets(
                initial_joint_positions,
                _gripper_target_for_width(sim, initial_gripper_width, config.handoff.handoff_open_width_m),
            )
            initial_obs = sim.get_observation()
            initial_joint_positions = _joint_positions_from_observation(initial_obs)
            initial_achieved_pose = sim.kinematics.compute_fk(initial_joint_positions)
        else:
            home_obs = sim.get_observation()
            home_joint_positions = _joint_positions_from_observation(home_obs)
            home_pose = sim.kinematics.compute_fk(home_joint_positions)
            initial_target_pose = home_pose.copy()
            initial_target_pose[:3] = _pose_payload_to_rotvec(dict(initial_state["ee_pose_base"]))[:3]
            _, initial_joint_positions, initial_achieved_pose = _solve_and_apply_pose_target(
                sim,
                target_pose=initial_target_pose,
                seed_joints=home_joint_positions,
                gripper_width_m=initial_gripper_width,
                handoff_open_width_m=config.handoff.handoff_open_width_m,
            )
            anchor_meta = {
                "anchor_mode": "fixture_position_with_home_orientation",
                "fixture_ee_pose_base": dict(initial_state["ee_pose_base"]),
            }
        init_translation_error_mm = _translation_error_mm(initial_target_pose, initial_achieved_pose)
        init_rotation_error_deg = _rotation_error_deg(initial_target_pose, initial_achieved_pose)

        video_path = run_dir / "validation.mp4"
        renderer, camera = _build_renderer(sim, width=int(video_width), height=int(video_height))
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(video_fps),
            (int(min(int(video_width), int(sim.model.vis.global_.offwidth))), int(min(int(video_height), int(sim.model.vis.global_.offheight)))),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {video_path}")

        target_pose = initial_target_pose.copy()
        current_gripper_width = initial_gripper_width
        previous_joint_positions = initial_joint_positions
        max_translation_error_mm = init_translation_error_mm
        max_rotation_error_deg = init_rotation_error_deg
        max_joint_step_rad = 0.0
        total_translation_error_mm = 0.0
        total_rotation_error_deg = 0.0

        for step_index, (state_record, action_record) in enumerate(zip(states, actions, strict=True)):
            state = dict(state_record["state"])
            safe_action = dict(action_record["safe_action"])
            phase = str(action_record["phase"])
            base_delta = _object_delta_to_base(list(safe_action["delta_pose_object"]), dict(state["object_pose_base"]))
            target_pose = compose_pose_delta(target_pose, base_delta, rotation_frame="world")
            current_gripper_width = max(0.0, float(current_gripper_width) + float(safe_action["delta_gripper"]))
            _, joint_positions, achieved_pose = _solve_and_apply_pose_target(
                sim,
                target_pose=target_pose,
                seed_joints=previous_joint_positions,
                gripper_width_m=current_gripper_width,
                handoff_open_width_m=config.handoff.handoff_open_width_m,
            )
            translation_error_mm = _translation_error_mm(target_pose, achieved_pose)
            rotation_error_deg = _rotation_error_deg(target_pose, achieved_pose)
            joint_step_rad = float(np.max(np.abs(joint_positions - previous_joint_positions)))
            previous_joint_positions = joint_positions

            max_translation_error_mm = max(max_translation_error_mm, translation_error_mm)
            max_rotation_error_deg = max(max_rotation_error_deg, rotation_error_deg)
            max_joint_step_rad = max(max_joint_step_rad, joint_step_rad)
            total_translation_error_mm += translation_error_mm
            total_rotation_error_deg += rotation_error_deg

            frame_rgb = _render_frame(sim, renderer, camera)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_bgr = _overlay_text(
                frame_bgr,
                step_index=step_index,
                phase=phase,
                controller_final_phase=str(controller_result.final_phase),
                translation_error_mm=translation_error_mm,
                rotation_error_deg=rotation_error_deg,
                gripper_width_m=current_gripper_width,
            )
            writer.write(frame_bgr)
            if step_index == 0:
                cv2.imwrite(str(run_dir / "frame_000.png"), frame_bgr)
            if step_index == len(actions) - 1:
                cv2.imwrite(str(run_dir / "frame_last.png"), frame_bgr)

            frame_rows.append(
                {
                    "step": int(step_index),
                    "phase": phase,
                    "controller_stop": bool(action_record["stop"]),
                    "controller_reason": action_record["reason"],
                    "safe_action_norm": float(action_record["safe_action_norm"]),
                    "target_pose_rotvec": [float(value) for value in target_pose],
                    "achieved_pose_rotvec": [float(value) for value in achieved_pose],
                    "translation_error_mm": translation_error_mm,
                    "rotation_error_deg": rotation_error_deg,
                    "joint_step_rad": joint_step_rad,
                    "gripper_width_m": float(current_gripper_width),
                }
            )
    finally:
        if writer is not None:
            writer.release()
        if renderer is not None:
            renderer.close()
        sim.disconnect()

    steps_count = max(len(frame_rows), 1)
    summary = {
        "status": "ok",
        "config_path": str(config_path.resolve()),
        "mock_state_path": str(mock_state_path.resolve()),
        "output_dir": str(run_dir),
        "video_path": str((run_dir / "validation.mp4").resolve()),
        "frame0_path": str((run_dir / "frame_000.png").resolve()),
        "frame_last_path": str((run_dir / "frame_last.png").resolve()),
        "mujoco_available": True,
        "pinocchio_available": True,
        "controller_status": controller_result.status,
        "controller_final_phase": controller_result.final_phase,
        "controller_failure_reason": controller_result.failure_reason,
        "controller_steps_run": controller_result.steps_run,
        "controller_max_safe_action_norm": controller_result.max_safe_action_norm,
        "controller_max_residual_norm": controller_result.max_residual_norm,
        "sim_steps_replayed": len(frame_rows),
        "init_translation_error_mm": init_translation_error_mm,
        "init_rotation_error_deg": init_rotation_error_deg,
        "max_tracking_translation_error_mm": max_translation_error_mm,
        "max_tracking_rotation_error_deg": max_rotation_error_deg,
        "mean_tracking_translation_error_mm": float(total_translation_error_mm / steps_count),
        "mean_tracking_rotation_error_deg": float(total_rotation_error_deg / steps_count),
        "max_joint_step_rad": max_joint_step_rad,
        "anchor": anchor_meta,
        "controller_summary": controller_summary,
    }
    _write_json(run_dir / "summary.json", summary)
    _write_jsonl(run_dir / "step_metrics.jsonl", frame_rows)
    return 0, run_dir / "summary.json", summary


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)
    exit_code, summary_path, _ = run_mujoco_validation(
        config_path=args.config,
        mock_state_path=args.mock_state,
        max_steps=int(args.max_steps),
        trajectory_points_path=args.trajectory_points,
        start_point_index=int(args.start_point_index),
        output_root=args.output_root,
        run_id=args.run_id,
        video_width=int(args.video_width),
        video_height=int(args.video_height),
        video_fps=float(args.video_fps),
    )
    logging.info("MuJoCo validation summary written to %s", summary_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
