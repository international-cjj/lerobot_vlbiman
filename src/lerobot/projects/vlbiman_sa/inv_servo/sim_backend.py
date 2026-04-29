from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from .execution_backend import ExecutionBackend
from .pose_delta_ik import compose_target_pose, pose_from_body, pose_from_camera, pose_to_plain, solve_ik
from .target_state import InvServoResult, ServoCommand


@dataclass(slots=True)
class SimBackendConfig:
    dry_run: bool = True
    start_frame: int = 75
    target_frame: int = 100
    data_dir: Path | None = None
    session_name: str = "sim_one_shot"
    camera: str = "wrist"
    camera_aliases: tuple[str, ...] = ("wrist", "hand", "hand_camera", "wrist_camera")
    replay_end_frame: int | None = None


class SimExecutionBackend(ExecutionBackend):
    name = "sim"

    def __init__(self, config: SimBackendConfig | None = None):
        self.config = config or SimBackendConfig()
        self.current_frame = int(self.config.start_frame)
        self.commands: list[dict[str, object]] = []
        self._records_by_index: dict[int, Any] | None = None
        self._session_dir: Path | None = None

    def reset_to_frame(self, frame_index: int) -> InvServoResult:
        self.current_frame = int(frame_index)
        self.commands.clear()
        if self.config.data_dir is not None:
            load_result = self._ensure_records_loaded()
            if not load_result.ok:
                return load_result
            if self.current_frame not in (self._records_by_index or {}):
                return InvServoResult.failure(
                    "sim_frame_not_found",
                    {"backend": self.name, "frame_index": self.current_frame, "session_dir": str(self._session_dir)},
                )
        return InvServoResult.success({"backend": self.name, "frame_index": self.current_frame})

    def get_observation(self) -> InvServoResult:
        if self.config.data_dir is not None:
            return self.get_rgb_frame(self.config.camera)
        return InvServoResult.success(
            {
                "backend": self.name,
                "frame_index": self.current_frame,
                "dry_run": self.config.dry_run,
            }
        )

    def get_rgb_frame(self, camera: str | None = None) -> InvServoResult:
        if self.config.data_dir is None:
            return InvServoResult.failure("sim_rgb_frame_unavailable", {"backend": self.name, "frame_index": self.current_frame})
        load_result = self._ensure_records_loaded()
        if not load_result.ok:
            return load_result

        records = self._records_by_index or {}
        frame_index = self.current_frame
        if self.config.replay_end_frame is not None and frame_index > int(self.config.replay_end_frame):
            frame_index = int(self.config.replay_end_frame)
        record = records.get(frame_index)
        if record is None:
            return InvServoResult.failure(
                "sim_rgb_frame_unavailable",
                {"backend": self.name, "frame_index": frame_index, "session_dir": str(self._session_dir)},
            )

        camera_name, camera_asset = self._resolve_camera_asset(record, camera or self.config.camera)
        if camera_name is None or camera_asset is None:
            return InvServoResult.failure(
                "sim_camera_asset_not_found",
                {"backend": self.name, "frame_index": frame_index, "camera": camera or self.config.camera},
            )
        image_path = Path(self._session_dir or Path()) / camera_asset["color_path"]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            return InvServoResult.failure(
                "sim_rgb_frame_unavailable",
                {"backend": self.name, "frame_index": frame_index, "image_path": str(image_path)},
            )
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return InvServoResult.success(
            {
                "backend": self.name,
                "frame_index": frame_index,
                "camera": camera_name,
                "image_path": str(image_path),
                "image_shape": list(image_rgb.shape),
                "image_rgb": image_rgb,
                "robot_state": {
                    "joint_positions": dict(getattr(record, "joint_positions", {})),
                    "gripper_state": dict(getattr(record, "gripper_state", {})),
                    "ee_pose": getattr(record, "ee_pose", None),
                },
            }
        )

    def execute_servo_command(self, command: ServoCommand) -> InvServoResult:
        record = {
            "frame_index": self.current_frame,
            "command": command.to_dict(),
            "dry_run": self.config.dry_run,
        }
        self.commands.append(record)
        replay_end_frame = self.config.replay_end_frame
        if replay_end_frame is None:
            replay_end_frame = self.config.target_frame
        if not command.stop and self.current_frame < int(replay_end_frame):
            self.current_frame += 1
        return InvServoResult.success({"backend": self.name, "accepted": True, "record": record})

    def apply_delta_cam(self, delta_cam: list[float] | tuple[float, float, float]) -> InvServoResult:
        command = ServoCommand(delta_xyz_m=tuple(float(value) for value in delta_cam))
        return self.execute_servo_command(command)

    def close_gripper(self) -> InvServoResult:
        return InvServoResult.success({"backend": self.name, "closed": True, "dry_run": self.config.dry_run})

    def stop(self) -> InvServoResult:
        return InvServoResult.success({"backend": self.name, "stopped": True, "frame_index": self.current_frame})

    def _ensure_records_loaded(self) -> InvServoResult:
        if self._records_by_index is not None:
            return InvServoResult.success({"session_dir": str(self._session_dir), "record_count": len(self._records_by_index)})
        if self.config.data_dir is None:
            return InvServoResult.failure("sim_data_dir_missing", {"backend": self.name})

        session_dir = Path(self.config.data_dir) / "recordings" / self.config.session_name
        try:
            from lerobot.projects.vlbiman_sa.demo.io import load_frame_records

            records = load_frame_records(session_dir)
        except Exception as exc:
            return InvServoResult.failure(
                "sim_records_load_failed",
                {"backend": self.name, "session_dir": str(session_dir), "error": f"{type(exc).__name__}: {exc}"},
            )
        self._session_dir = session_dir
        self._records_by_index = {int(record.frame_index): record for record in records}
        return InvServoResult.success({"session_dir": str(session_dir), "record_count": len(self._records_by_index)})

    def _resolve_camera_asset(self, record: Any, preferred: str) -> tuple[str | None, dict[str, Any] | None]:
        assets = dict(getattr(record, "camera_assets", {}))
        candidates = [preferred, *self.config.camera_aliases]
        for value in list(candidates):
            if value.endswith("_camera"):
                candidates.append(value.removesuffix("_camera"))
            else:
                candidates.append(f"{value}_camera")
        for candidate in dict.fromkeys(candidates):
            if candidate in assets:
                return candidate, assets[candidate]
        return None, None


@dataclass(slots=True)
class MujocoSimBackendConfig:
    start_frame: int = 75
    target_frame: int = 100
    data_dir: Path | None = None
    session_name: str = "sim_one_shot"
    camera: str = "wrist"
    camera_aliases: tuple[str, ...] = ("wrist", "hand", "hand_camera", "wrist_camera")
    mujoco_model_path: Path | None = None
    render_width: int = 640
    render_height: int = 480
    scene_settle_steps: int = 0
    action_substeps: int = 160
    max_delta_per_step: float = 0.08
    use_viewer: bool = False
    motion_smoothing_segments: int = 1
    motion_smoothing_profile: str = "linear"
    motion_smoothing_segment_delay_s: float = 0.0
    arm_position_kp: float | None = None


class MujocoSimExecutionBackend(ExecutionBackend):
    """Independent MuJoCo execution backend initialized from a recorded frame state.

    Unlike ``SimExecutionBackend``, this backend does not replay recorded RGB frames.
    It restores the robot state from one recorded frame, then all observations come
    from MuJoCo rendering after simulator actions.
    """

    name = "real_mujoco"

    def __init__(self, config: MujocoSimBackendConfig):
        self.config = config
        self.robot: Any | None = None
        self.current_frame = int(config.start_frame)
        self.commands: list[dict[str, Any]] = []
        self._records_by_index: dict[int, Any] | None = None
        self._session_dir: Path | None = None
        self._start_record: Any | None = None

    def connect(self) -> InvServoResult:
        if self.robot is not None:
            return InvServoResult.success({"backend": self.name, "connected": True})
        if self.config.mujoco_model_path is None:
            return InvServoResult.failure("mujoco_model_path_missing", {"backend": self.name})
        try:
            from lerobot_robot_cjjarm import CjjArmSim, CjjArmSimConfig

            self.robot = CjjArmSim(
                CjjArmSimConfig(
                    id="inv_rgb_servo_real_mujoco",
                    mujoco_model_path=str(Path(self.config.mujoco_model_path).resolve()),
                    render_width=int(self.config.render_width),
                    render_height=int(self.config.render_height),
                    scene_settle_steps=int(self.config.scene_settle_steps),
                    action_substeps=int(self.config.action_substeps),
                    max_delta_per_step=float(self.config.max_delta_per_step),
                    use_viewer=bool(self.config.use_viewer),
                )
            )
            self.robot.connect()
            self._configure_arm_position_actuators()
        except Exception as exc:
            self.robot = None
            return InvServoResult.failure(
                "mujoco_backend_connect_failed",
                {
                    "backend": self.name,
                    "mujoco_model_path": str(self.config.mujoco_model_path),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        return InvServoResult.success(
            {
                "backend": self.name,
                "connected": True,
                "mujoco_model_path": str(self.config.mujoco_model_path),
                "arm_position_kp": self.config.arm_position_kp,
            }
        )

    def _configure_arm_position_actuators(self) -> None:
        if self.robot is None or self.config.arm_position_kp is None:
            return
        kp = float(self.config.arm_position_kp)
        if kp <= 0.0:
            raise ValueError("arm_position_kp must be positive when provided.")
        for actuator_id in self.robot._arm_actuator_ids:
            self.robot.model.actuator_gainprm[actuator_id, 0] = kp
            self.robot.model.actuator_biasprm[actuator_id, 1] = -kp

    def disconnect(self) -> InvServoResult:
        if self.robot is not None:
            try:
                self.robot.disconnect()
            finally:
                self.robot = None
        return InvServoResult.success({"backend": self.name, "connected": False})

    def reset_to_frame(self, frame_index: int) -> InvServoResult:
        connect_result = self.connect()
        if not connect_result.ok:
            return connect_result
        load_result = self._ensure_records_loaded()
        if not load_result.ok:
            return load_result
        records = self._records_by_index or {}
        record = records.get(int(frame_index))
        if record is None:
            return InvServoResult.failure(
                "mujoco_start_frame_not_found",
                {"backend": self.name, "frame_index": int(frame_index), "session_dir": str(self._session_dir)},
            )
        try:
            q = self.joint_vector_from_record(record)
            gripper = float(getattr(record, "gripper_state", {}).get("gripper.pos", 0.0))
            self._set_joint_state(q, gripper_position=gripper)
        except Exception as exc:
            return InvServoResult.failure(
                "mujoco_reset_to_frame_failed",
                {"backend": self.name, "frame_index": int(frame_index), "error": f"{type(exc).__name__}: {exc}"},
            )
        self.current_frame = int(frame_index)
        self.commands.clear()
        self._start_record = record
        return InvServoResult.success(
            {
                "backend": self.name,
                "frame_index": self.current_frame,
                "joint_positions": self.current_joint_positions().tolist(),
                "used_recorded_observation_sequence": False,
                "used_target_frame_robot_state": False,
            }
        )

    def get_observation(self) -> InvServoResult:
        return self.get_rgb_frame(self.config.camera)

    def get_rgb_frame(self, camera: str | None = None) -> InvServoResult:
        if self.robot is None:
            return InvServoResult.failure("mujoco_backend_not_connected", {"backend": self.name})
        render_camera = self._render_camera_name(camera or self.config.camera)
        try:
            image_rgb = self.robot._render_camera(render_camera)
        except Exception as exc:
            return InvServoResult.failure(
                "mujoco_rgb_render_failed",
                {"backend": self.name, "camera": render_camera, "error": f"{type(exc).__name__}: {exc}"},
            )
        return InvServoResult.success(
            {
                "backend": self.name,
                "frame_index": self.current_frame,
                "camera": render_camera,
                "image_shape": list(image_rgb.shape),
                "image_rgb": image_rgb,
                "robot_state": self.robot_state(),
            }
        )

    def current_joint_positions(self) -> np.ndarray:
        if self.robot is None:
            raise RuntimeError("MuJoCo backend is not connected.")
        return self.robot._get_arm_qpos_logical()

    def get_current_ee_pose(self, ee_body_name: str) -> InvServoResult:
        if self.robot is None:
            return InvServoResult.failure("mujoco_backend_not_connected", {"backend": self.name})
        try:
            pose = pose_from_body(self.robot.model, self.robot.data, ee_body_name)
        except Exception as exc:
            return InvServoResult.failure(
                "mujoco_ee_pose_failed",
                {"backend": self.name, "ee_body_name": ee_body_name, "error": f"{type(exc).__name__}: {exc}"},
            )
        return InvServoResult.success(pose_to_plain(pose))

    def get_camera_pose(self, camera: str | None = None) -> InvServoResult:
        if self.robot is None:
            return InvServoResult.failure("mujoco_backend_not_connected", {"backend": self.name})
        camera_name = self._render_camera_name(camera or self.config.camera)
        try:
            pose = pose_from_camera(self.robot.model, self.robot.data, camera_name)
        except Exception as exc:
            return InvServoResult.failure(
                "mujoco_camera_pose_failed",
                {"backend": self.name, "camera": camera_name, "error": f"{type(exc).__name__}: {exc}"},
            )
        return InvServoResult.success(pose_to_plain(pose))

    def compose_target_ee_pose(
        self,
        current_ee_pose: dict[str, Any],
        delta_pose: dict[str, Any],
        *,
        translation_frame: str,
        camera: str | None = None,
    ) -> InvServoResult:
        camera_pose = None
        if str(translation_frame or "").lower() in {"camera", "camera_image", "image_camera"}:
            camera_pose_result = self.get_camera_pose(camera)
            if not camera_pose_result.ok or camera_pose_result.state is None:
                return camera_pose_result
            camera_pose = camera_pose_result.state
        try:
            target_pose = compose_target_pose(
                current_ee_pose,
                delta_pose,
                translation_frame=translation_frame,
                camera_pose=camera_pose,
            )
        except Exception as exc:
            return InvServoResult.failure(
                "compose_target_ee_pose_failed",
                {
                    "backend": self.name,
                    "translation_frame": translation_frame,
                    "delta_pose": delta_pose,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        return InvServoResult.success(pose_to_plain(target_pose))

    def solve_ik_to_pose(
        self,
        target_pose: dict[str, Any],
        *,
        ee_body_name: str,
        q_init: np.ndarray | None = None,
        max_iters: int = 80,
        pos_tol: float = 0.002,
        rot_tol: float = 0.08,
        damping: float = 1e-4,
        step_scale: float = 0.65,
        max_step_rad: float = 0.08,
        position_weight: float = 1.0,
        rotation_weight: float = 0.05,
    ) -> InvServoResult:
        if self.robot is None:
            return InvServoResult.failure("mujoco_backend_not_connected", {"backend": self.name})
        directions = np.asarray(
            [self.robot._joint_directions[name] for name in self.robot._arm_joint_names],
            dtype=float,
        )
        result = solve_ik(
            model=self.robot.model,
            data=self.robot.data,
            target_pose=target_pose,
            ee_body_name=ee_body_name,
            q_init=self.current_joint_positions() if q_init is None else np.asarray(q_init, dtype=float),
            arm_qpos_addr=list(self.robot._arm_qpos_addr),
            arm_dof_addr=list(self.robot._arm_qvel_addr),
            joint_limits=np.asarray(self.robot._arm_joint_limits_logical, dtype=float),
            joint_directions=directions,
            max_iters=max_iters,
            pos_tol=pos_tol,
            rot_tol=rot_tol,
            damping=damping,
            step_scale=step_scale,
            max_step_rad=max_step_rad,
            position_weight=position_weight,
            rotation_weight=rotation_weight,
        )
        if not result.get("ok", False):
            return InvServoResult.failure(str(result.get("failure_reason") or "ik_failed"), result)
        return InvServoResult.success(result)

    def joint_vector_from_record(self, record: Any) -> np.ndarray:
        return np.asarray([float(record.joint_positions[f"joint_{index}.pos"]) for index in range(1, 7)], dtype=float)

    def control_targets_for_joints(self, joint_positions: np.ndarray, *, gripper_position: float = 0.0) -> np.ndarray:
        if self.robot is None:
            raise RuntimeError("MuJoCo backend is not connected.")
        gripper = np.asarray([float(gripper_position), float(gripper_position)], dtype=float)
        return self.robot._build_ctrl_targets(np.asarray(joint_positions, dtype=float), gripper)

    def execute_joint_target(
        self,
        joint_positions: np.ndarray,
        *,
        gripper_position: float = 0.0,
        substeps: int | None = None,
        debug: dict[str, Any] | None = None,
    ) -> InvServoResult:
        if self.robot is None:
            return InvServoResult.failure("mujoco_backend_not_connected", {"backend": self.name})
        q = np.asarray(joint_positions, dtype=float).reshape(6)
        try:
            applied, smoothing_state = self._apply_joint_target(q, gripper_position=gripper_position, substeps=substeps)
        except Exception as exc:
            return InvServoResult.failure(
                "mujoco_joint_target_failed",
                {"backend": self.name, "error": f"{type(exc).__name__}: {exc}", "joint_target": q.tolist()},
            )
        self.current_frame += 1
        record = {
            "frame_index": self.current_frame,
            "joint_target": q.tolist(),
            "applied_ctrl": np.asarray(applied, dtype=float).tolist(),
            "robot_state": self.robot_state(),
            "debug": {**(debug or {}), "motion_smoothing": smoothing_state},
        }
        self.commands.append(record)
        return InvServoResult.success({"backend": self.name, "accepted": True, "record": record})

    def execute_servo_command(self, command: ServoCommand) -> InvServoResult:
        return InvServoResult.failure(
            "mujoco_delta_cam_not_directly_supported",
            {"backend": self.name, "command": command.to_dict()},
        )

    def apply_delta_cam(self, delta_cam: list[float] | tuple[float, float, float]) -> InvServoResult:
        return self.execute_servo_command(ServoCommand(delta_xyz_m=tuple(float(value) for value in delta_cam)))

    def close_gripper(self) -> InvServoResult:
        if self.robot is None:
            return InvServoResult.failure("mujoco_backend_not_connected", {"backend": self.name})
        try:
            q = self.current_joint_positions()
            closed = float(getattr(self.robot.config, "gripper_closed_pos", -0.045))
            applied, smoothing_state = self._apply_joint_target(q, gripper_position=closed, substeps=self.config.action_substeps)
        except Exception as exc:
            return InvServoResult.failure(
                "mujoco_close_gripper_failed",
                {"backend": self.name, "error": f"{type(exc).__name__}: {exc}"},
            )
        return InvServoResult.success(
            {
                "backend": self.name,
                "closed": True,
                "gripper_position": closed,
                "applied_ctrl": np.asarray(applied, dtype=float).tolist(),
                "motion_smoothing": smoothing_state,
            }
        )

    def _apply_joint_target(
        self,
        joint_positions: np.ndarray,
        *,
        gripper_position: float,
        substeps: int | None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.robot is None:
            raise RuntimeError("MuJoCo backend is not connected.")
        q = np.asarray(joint_positions, dtype=float).reshape(6)
        total_substeps = max(int(self.config.action_substeps if substeps is None else substeps), 1)
        segments = max(int(self.config.motion_smoothing_segments), 1)
        profile = str(self.config.motion_smoothing_profile or "linear").lower()
        delay_s = max(float(self.config.motion_smoothing_segment_delay_s), 0.0)
        if segments <= 1:
            targets = self.control_targets_for_joints(q, gripper_position=gripper_position)
            applied = self.robot.set_control_targets(targets, substeps=total_substeps)
            return applied, {
                "segments": 1,
                "profile": "none",
                "substeps_per_segment": [total_substeps],
                "segment_delay_s": 0.0,
            }

        start_q = self.current_joint_positions().copy()
        start_gripper = self._current_gripper_position(default=gripper_position)
        substeps_per_segment = self._split_substeps(total_substeps, segments)
        applied: np.ndarray | None = None
        for index, segment_substeps in enumerate(substeps_per_segment, start=1):
            alpha = self._motion_alpha(index / float(segments), profile)
            segment_q = start_q + alpha * (q - start_q)
            segment_gripper = start_gripper + alpha * (float(gripper_position) - start_gripper)
            targets = self.control_targets_for_joints(segment_q, gripper_position=segment_gripper)
            applied = self.robot.set_control_targets(targets, substeps=segment_substeps)
            if delay_s > 0.0:
                time.sleep(delay_s)
        if applied is None:
            raise RuntimeError("motion smoothing produced no control target")
        return applied, {
            "segments": segments,
            "profile": profile,
            "substeps_per_segment": substeps_per_segment,
            "segment_delay_s": delay_s,
        }

    def _current_gripper_position(self, *, default: float) -> float:
        if self.robot is None or not getattr(self.robot.config, "use_gripper", False):
            return float(default)
        values = np.asarray(self.robot.data.qpos[self.robot._gripper_qpos_addr], dtype=float)
        if values.size == 0:
            return float(default)
        return float(np.mean(values))

    @staticmethod
    def _split_substeps(total_substeps: int, segments: int) -> list[int]:
        segments = max(int(segments), 1)
        total_substeps = max(int(total_substeps), 1)
        base = total_substeps // segments
        remainder = total_substeps % segments
        return [max(base + (1 if index < remainder else 0), 1) for index in range(segments)]

    @staticmethod
    def _motion_alpha(alpha: float, profile: str) -> float:
        value = float(np.clip(alpha, 0.0, 1.0))
        if profile == "smoothstep":
            return float(value * value * (3.0 - 2.0 * value))
        if profile == "smootherstep":
            return float(value * value * value * (value * (value * 6.0 - 15.0) + 10.0))
        if profile in {"linear", "none"}:
            return value
        raise ValueError(f"Unsupported motion smoothing profile: {profile!r}")

    def stop(self) -> InvServoResult:
        return InvServoResult.success({"backend": self.name, "stopped": True, "frame_index": self.current_frame})

    def snapshot_state(self) -> dict[str, Any]:
        if self.robot is None:
            raise RuntimeError("MuJoCo backend is not connected.")
        return {
            "qpos": self.robot.data.qpos.copy(),
            "qvel": self.robot.data.qvel.copy(),
            "ctrl": self.robot.data.ctrl.copy(),
            "last_joint_positions": None
            if self.robot._last_joint_positions is None
            else self.robot._last_joint_positions.copy(),
            "current_frame": int(self.current_frame),
        }

    def restore_state(self, snapshot: dict[str, Any]) -> None:
        if self.robot is None:
            raise RuntimeError("MuJoCo backend is not connected.")
        import mujoco

        self.robot.data.qpos[:] = snapshot["qpos"]
        self.robot.data.qvel[:] = snapshot["qvel"]
        self.robot.data.ctrl[:] = snapshot["ctrl"]
        mujoco.mj_forward(self.robot.model, self.robot.data)
        self.robot._last_joint_positions = snapshot["last_joint_positions"]
        self.current_frame = int(snapshot["current_frame"])

    def robot_state(self) -> dict[str, Any]:
        if self.robot is None:
            return {}
        return {
            "joint_positions": {
                f"joint_{index + 1}.pos": float(value)
                for index, value in enumerate(self.current_joint_positions())
            },
            "control_targets": self.robot.get_control_targets().tolist(),
        }

    def _set_joint_state(self, joint_positions: np.ndarray, *, gripper_position: float = 0.0) -> None:
        if self.robot is None:
            raise RuntimeError("MuJoCo backend is not connected.")
        import mujoco

        q = np.asarray(joint_positions, dtype=float).reshape(6)
        actual = np.asarray(
            [q[index] * self.robot._joint_directions[name] for index, name in enumerate(self.robot._arm_joint_names)],
            dtype=float,
        )
        self.robot.data.qpos[self.robot._arm_qpos_addr] = actual
        self.robot.data.qvel[self.robot._arm_qvel_addr] = 0.0
        if self.robot.config.use_gripper:
            self.robot.data.qpos[self.robot._gripper_qpos_addr] = float(gripper_position)
            self.robot.data.qvel[self.robot._gripper_qvel_addr] = 0.0
        self.robot.data.ctrl[:] = self.control_targets_for_joints(q, gripper_position=gripper_position)
        mujoco.mj_forward(self.robot.model, self.robot.data)
        self.robot._last_joint_positions = q.copy()

    def _ensure_records_loaded(self) -> InvServoResult:
        if self._records_by_index is not None:
            return InvServoResult.success({"session_dir": str(self._session_dir), "record_count": len(self._records_by_index)})
        if self.config.data_dir is None:
            return InvServoResult.failure("mujoco_data_dir_missing", {"backend": self.name})

        session_dir = Path(self.config.data_dir) / "recordings" / self.config.session_name
        try:
            from lerobot.projects.vlbiman_sa.demo.io import load_frame_records

            records = load_frame_records(session_dir)
        except Exception as exc:
            return InvServoResult.failure(
                "mujoco_records_load_failed",
                {"backend": self.name, "session_dir": str(session_dir), "error": f"{type(exc).__name__}: {exc}"},
            )
        self._session_dir = session_dir
        self._records_by_index = {int(record.frame_index): record for record in records}
        return InvServoResult.success({"session_dir": str(session_dir), "record_count": len(self._records_by_index)})

    def _render_camera_name(self, camera: str) -> str:
        camera_value = str(camera or "").strip()
        aliases = set(self.config.camera_aliases) | {"wrist", "hand", "hand_camera", "wrist_camera"}
        if camera_value in aliases:
            return "wrist_camera"
        if camera_value == "front":
            return "front_camera"
        return camera_value
