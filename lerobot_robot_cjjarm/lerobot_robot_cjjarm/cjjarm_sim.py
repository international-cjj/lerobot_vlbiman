import os
import logging
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.projects.vlbiman_sa.runtime_env import (
    configure_mujoco_gl_backend,
    require_mujoco_viewer_backend,
)

from .config_cjjarm_sim import CjjArmSimConfig
from .kinematics import CjjArmKinematics, compose_pose_delta

configure_mujoco_gl_backend()

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - exercised when mujoco is missing
    raise ImportError(
        "mujoco is required for CjjArmSim. Install it with: pip install mujoco"
    ) from exc

logger = logging.getLogger(__name__)


class CjjArmSim(Robot):
    config_class = CjjArmSimConfig
    name = "cjjarm_sim"

    _ARM_ACTUATOR_NAMES = {
        "joint_1": "act_joint1",
        "joint_2": "act_joint2",
        "joint_3": "act_joint3",
        "joint_4": "act_joint4",
        "joint_5": "act_joint5",
        "joint_6": "act_joint6",
    }

    def __init__(self, config: CjjArmSimConfig):
        super().__init__(config)
        self.config = config
        self.cameras: dict[str, Any] = {}
        self._viewer = None
        self._renderer_failed = False
        self._render_camera_missing_warned: set[str] = set()
        self._generated_scene_path: Path | None = None
        self._load_source_path: Path | None = None
        self._load_chain_description = ""

        self.kinematics = None
        try:
            self.kinematics = CjjArmKinematics(
                urdf_path=self.config.urdf_path,
                end_effector_frame=self.config.end_effector_frame,
                joint_names=[
                    self.config.urdf_joint_map[name]
                    for name in self.config.joint_action_order
                ],
            )
        except ImportError as exc:
            logger.warning("Pinocchio unavailable, pose actions will be disabled: %s", exc)

        self._arm_joint_names = list(self.config.joint_action_order)
        missing = set(self._arm_joint_names) - set(self.config.urdf_joint_map)
        if missing:
            raise ValueError(f"joint_action_order contains unknown joints: {sorted(missing)}")
        self._urdf_joint_names = [self.config.urdf_joint_map[name] for name in self._arm_joint_names]
        self._joint_directions = {
            name: float(self.config.joint_directions.get(name, 1.0))
            for name in self._arm_joint_names
        }
        self._gripper_direction = float(self.config.joint_directions.get("gripper", 1.0))
        self._joint_names = list(self._arm_joint_names)
        if self.config.use_gripper:
            self._joint_names.append("gripper")
        self._camera_name_by_observation_key = {
            self.config.front_camera_observation_key: "front_camera",
            self.config.wrist_camera_observation_key: "wrist_camera",
        }

        self.model = self._load_model_from_config()
        self.model.opt.gravity[:] = np.asarray(self.config.gravity, dtype=float)
        self.data = mujoco.MjData(self.model)
        self.renderer = None

        self._arm_qpos_addr = [self._get_qpos_addr(name) for name in self._urdf_joint_names]
        self._arm_qvel_addr = [self._get_qvel_addr(name) for name in self._urdf_joint_names]
        self._arm_joint_id_by_name = {
            name: int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name))
            for name in self._urdf_joint_names
        }
        self._arm_actuator_ids = [self._get_actuator_id(self._ARM_ACTUATOR_NAMES[name]) for name in self._arm_joint_names]
        self._arm_joint_limits_logical = np.asarray(
            [
                self._logical_range_from_joint(self._urdf_joint_names[i], self._joint_directions[name])
                for i, name in enumerate(self._arm_joint_names)
            ],
            dtype=float,
        )
        self._arm_ctrl_limits_actual = np.asarray(
            [self.model.actuator_ctrlrange[actuator_id] for actuator_id in self._arm_actuator_ids],
            dtype=float,
        )

        self._gripper_joint_names = ("gripper_left", "gripper_right")
        self._gripper_actuator_names = ("act_gripper_left", "act_gripper_right")
        self._gripper_qpos_addr = [self._get_qpos_addr(name) for name in self._gripper_joint_names]
        self._gripper_qvel_addr = [self._get_qvel_addr(name) for name in self._gripper_joint_names]
        self._gripper_actuator_ids = [self._get_actuator_id(name) for name in self._gripper_actuator_names]
        self._gripper_ctrl_limits_actual = np.asarray(
            [self.model.actuator_ctrlrange[actuator_id] for actuator_id in self._gripper_actuator_ids],
            dtype=float,
        )

        self._control_names = [self.model.actuator(i).name or f"actuator_{i}" for i in range(int(self.model.nu))]
        self._control_limits = np.asarray(self.model.actuator_ctrlrange, dtype=float).copy() if self.model.nu > 0 else np.zeros((0, 2), dtype=float)
        self._home_qpos = self._read_home_qpos()
        self._home_ctrl = self._read_home_ctrl()

        self._virtual_target_qpos_addr, self._virtual_target_qvel_addr = self._get_freejoint_addresses("virtual_target_body_free")
        self._target_cube_qpos_addr, self._target_cube_qvel_addr = self._get_freejoint_addresses("target_cube_free")

        self._finger_body_ids: set[int] = set()
        for body_name in ("finger_left", "finger_right"):
            body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name))
            if body_id >= 0:
                self._finger_body_ids.add(body_id)

        self._scene_object_geom_ids: set[int] = set()
        for geom_index in range(int(self.model.ngeom)):
            geom_name = self.model.geom(geom_index).name or ""
            if geom_name == "target_cube_geom" or geom_name.startswith("scene_object_"):
                self._scene_object_geom_ids.add(int(geom_index))

        self._last_joint_positions: np.ndarray | None = None
        self._target_pose: np.ndarray | None = None
        self._is_connected = False

        self._log_model_summary()

    def _instance_name(self) -> str:
        return str(self.config.id or "default")

    def _write_generated_scene(self, xml_text: str) -> Path:
        output_dir = Path(self.config.scene_generated_mjcf_dir).resolve() / self._instance_name()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "generated_dual_camera_target_scene.mjcf"
        output_path.write_text(xml_text, encoding="utf-8")
        return output_path

    def _load_model_from_config(self) -> mujoco.MjModel:
        explicit_model_path = str(getattr(self.config, "mujoco_model_path", "") or "").strip()
        if explicit_model_path:
            resolved_path = Path(explicit_model_path).resolve()
            self._load_source_path = resolved_path
            self._generated_scene_path = None
            self._load_chain_description = f"{resolved_path} -> MjModel.from_xml_path(...)"
            return mujoco.MjModel.from_xml_path(str(resolved_path))

        scene_profile = str(getattr(self.config, "scene_profile", "") or "").strip().lower()
        if scene_profile in {"dual_camera_target", "canonical"}:
            from lerobot.projects.vlbiman_sa.sim import (
                DualCameraSceneConfig,
                ScenePrimitiveObjectConfig,
                TargetSphereConfig,
                build_dual_camera_scene,
                load_base_from_camera_transform,
                load_wrist_camera_mount_pose,
                scene_preset_objects,
            )

            base_mjcf_path = Path(str(self.config.scene_base_mjcf_path)).resolve()
            handeye_result_path = Path(str(self.config.scene_handeye_result_path)).resolve()
            urdf_path = Path(str(self.config.urdf_path)).resolve()
            if not base_mjcf_path.exists():
                raise FileNotFoundError(f"Missing base MJCF: {base_mjcf_path}")
            if not handeye_result_path.exists():
                raise FileNotFoundError(f"Missing handeye result: {handeye_result_path}")
            if not urdf_path.exists():
                raise FileNotFoundError(f"Missing URDF: {urdf_path}")

            base_from_camera = load_base_from_camera_transform(handeye_result_path)
            wrist_xyz, wrist_rpy = load_wrist_camera_mount_pose(urdf_path)
            scene_config = DualCameraSceneConfig(
                camera_fovy_deg=float(self.config.scene_camera_fovy_deg),
                include_target_cube=str(self.config.scene_preset).strip().lower() == "default",
                target=TargetSphereConfig(
                    position_xyz_m=(
                        float(self.config.scene_target_x),
                        float(self.config.scene_target_y),
                        float(self.config.scene_target_z),
                    ),
                    radius_m=float(self.config.scene_target_radius_m),
                    mass_kg=float(self.config.scene_target_mass_kg),
                    rgba=(
                        (1.0, 1.0, 0.0, 0.0)
                        if str(self.config.scene_preset).strip().lower() != "default"
                        else (0.93, 0.36, 0.08, 0.35)
                    ),
                ),
                target_cube=ScenePrimitiveObjectConfig(
                    object_key="target_cube",
                    shape="box",
                    position_xyz_m=(
                        float(self.config.scene_target_x),
                        float(self.config.scene_target_y),
                        float(max(self.config.scene_target_z, 0.0605)),
                    ),
                    size_xyz_m=(0.020, 0.020, 0.020),
                    mass_kg=0.08,
                    rgba=(0.21, 0.62, 0.92, 1.0),
                    friction=(1.6, 0.08, 0.003),
                    solref=(0.003, 1.0),
                    solimp=(0.96, 0.995, 0.001),
                    body_name="target_cube",
                    geom_name="target_cube_geom",
                ),
                objects=scene_preset_objects(
                    str(self.config.scene_preset),
                    center_xy_m=(float(self.config.scene_target_x), float(self.config.scene_target_y)),
                ),
            )
            artifacts = build_dual_camera_scene(
                base_mjcf_path=base_mjcf_path,
                base_from_external_camera=base_from_camera,
                wrist_camera_xyz_m=wrist_xyz,
                wrist_camera_rpy_rad=wrist_rpy,
                config=scene_config,
            )
            self._load_source_path = base_mjcf_path
            self._generated_scene_path = self._write_generated_scene(artifacts.xml_text)
            self._load_chain_description = (
                f"{base_mjcf_path} -> build_dual_camera_scene(...) -> "
                f"{self._generated_scene_path} -> MjModel.from_xml_string(...)"
            )
            return mujoco.MjModel.from_xml_string(artifacts.xml_text)

        if scene_profile == "legacy":
            if not bool(self.config.legacy_raw_urdf_enabled):
                raise RuntimeError(
                    "Legacy/raw URDF mode is disabled by default. "
                    "Set legacy_raw_urdf_enabled=True only when you intentionally need it."
                )
            resolved_urdf_path = Path(self.config.urdf_path).resolve()
            self._load_source_path = resolved_urdf_path
            self._generated_scene_path = None
            self._load_chain_description = f"{resolved_urdf_path} -> MjModel.from_xml_path(...)"
            return mujoco.MjModel.from_xml_path(str(resolved_urdf_path))

        raise ValueError(f"Unsupported CjjArmSim scene_profile: {scene_profile}")

    def _log_model_summary(self) -> None:
        logger.info("CjjArmSim canonical load chain: %s", self._load_chain_description)
        logger.info(
            "CjjArmSim model stats: source=%s generated=%s nbody=%d njnt=%d ngeom=%d nu=%d ncam=%d",
            self._load_source_path,
            self._generated_scene_path,
            int(self.model.nbody),
            int(self.model.njnt),
            int(self.model.ngeom),
            int(self.model.nu),
            int(self.model.ncam),
        )

    def _sync_viewer(self) -> None:
        if self._viewer is None:
            return
        try:
            if hasattr(self._viewer, "is_running") and not self._viewer.is_running():
                self._viewer = None
                return
            self._viewer.sync()
        except Exception as exc:
            logger.warning("MuJoCo viewer sync failed: %s", exc)
            self._viewer = None

    def _get_qpos_addr(self, joint_name: str) -> int:
        joint_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
        if joint_id < 0:
            raise ValueError(f"Joint '{joint_name}' not found in MuJoCo model.")
        return int(self.model.jnt_qposadr[joint_id])

    def _get_qvel_addr(self, joint_name: str) -> int:
        joint_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
        if joint_id < 0:
            raise ValueError(f"Joint '{joint_name}' not found in MuJoCo model.")
        return int(self.model.jnt_dofadr[joint_id])

    def _get_actuator_id(self, actuator_name: str) -> int:
        actuator_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name))
        if actuator_id < 0:
            raise ValueError(f"Actuator '{actuator_name}' not found in MuJoCo model.")
        return actuator_id

    def _logical_range_from_joint(self, joint_name: str, direction: float) -> tuple[float, float]:
        joint_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
        if joint_id < 0:
            raise ValueError(f"Joint '{joint_name}' not found in MuJoCo model.")
        lower, upper = [float(v) for v in self.model.jnt_range[joint_id]]
        logical_bounds = np.asarray([lower, upper], dtype=float) * float(direction)
        return float(np.min(logical_bounds)), float(np.max(logical_bounds))

    def _read_home_qpos(self) -> np.ndarray:
        if self.model.nkey > 0:
            qpos = np.asarray(self.model.key_qpos, dtype=float).reshape(int(self.model.nkey), int(self.model.nq))[0].copy()
        else:
            qpos = np.asarray(self.model.qpos0, dtype=float).copy()

        qpos0 = np.asarray(self.model.qpos0, dtype=float)
        for joint_id in range(int(self.model.njnt)):
            if int(self.model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            qpos_addr = int(self.model.jnt_qposadr[joint_id])
            key_position = qpos[qpos_addr : qpos_addr + 3]
            authored_position = qpos0[qpos_addr : qpos_addr + 3]
            # MuJoCo pads keyframes that only define robot joints with zeroed
            # freejoint poses; keep authored scene object poses from qpos0.
            if np.allclose(key_position, 0.0) and not np.allclose(authored_position, 0.0):
                qpos[qpos_addr : qpos_addr + 7] = qpos0[qpos_addr : qpos_addr + 7]
        return qpos

    def _read_home_ctrl(self) -> np.ndarray:
        if self.model.nu <= 0:
            return np.zeros(0, dtype=float)
        if self.model.nkey > 0 and self.model.key_ctrl.size >= self.model.nu:
            return np.asarray(self.model.key_ctrl, dtype=float).reshape(int(self.model.nkey), int(self.model.nu))[0].copy()
        lower = np.asarray(self.model.actuator_ctrlrange[:, 0], dtype=float)
        upper = np.asarray(self.model.actuator_ctrlrange[:, 1], dtype=float)
        return 0.5 * (lower + upper)

    def _get_freejoint_addresses(self, joint_name: str) -> tuple[int | None, int | None]:
        joint_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
        if joint_id < 0:
            return None, None
        return int(self.model.jnt_qposadr[joint_id]), int(self.model.jnt_dofadr[joint_id])

    def _reset_sim(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._home_qpos
        self.data.qvel[:] = 0.0
        if self.model.nu > 0:
            self.data.ctrl[:] = self._home_ctrl
        self._set_freejoint_position(
            self._virtual_target_qpos_addr,
            self._virtual_target_qvel_addr,
            np.asarray(
                [
                    float(self.config.scene_target_x),
                    float(self.config.scene_target_y),
                    float(self.config.scene_target_z),
                ],
                dtype=float,
            ),
        )
        self._set_freejoint_position(
            self._target_cube_qpos_addr,
            self._target_cube_qvel_addr,
            np.asarray(
                [
                    float(self.config.scene_target_x),
                    float(self.config.scene_target_y),
                    float(max(self.config.scene_target_z, 0.0605)),
                ],
                dtype=float,
            ),
        )
        mujoco.mj_forward(self.model, self.data)
        for _ in range(max(int(self.config.scene_settle_steps), 0)):
            mujoco.mj_step(self.model, self.data)
        self._last_joint_positions = self._get_arm_qpos_logical()
        self._target_pose = None
        self._sync_viewer()

    def _set_freejoint_position(
        self,
        qpos_addr: int | None,
        qvel_addr: int | None,
        position_xyz_m: np.ndarray,
    ) -> None:
        if qpos_addr is None:
            return
        position = np.asarray(position_xyz_m, dtype=float).reshape(3)
        self.data.qpos[qpos_addr : qpos_addr + 3] = position
        self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
        if qvel_addr is not None:
            self.data.qvel[qvel_addr : qvel_addr + 6] = 0.0

    def set_target_cube_position(self, position_xyz_m: np.ndarray, *, settle_steps: int = 0) -> None:
        self._set_freejoint_position(self._target_cube_qpos_addr, self._target_cube_qvel_addr, position_xyz_m)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(max(int(settle_steps), 0)):
            mujoco.mj_step(self.model, self.data)
        self._sync_viewer()

    def _get_arm_qpos_actual(self) -> np.ndarray:
        return np.asarray(self.data.qpos[self._arm_qpos_addr], dtype=float).copy()

    def _get_arm_qpos_logical(self) -> np.ndarray:
        actual = self._get_arm_qpos_actual()
        return np.asarray(
            [actual[i] * self._joint_directions[name] for i, name in enumerate(self._arm_joint_names)],
            dtype=float,
        )

    def _get_arm_qvel_logical(self) -> np.ndarray:
        actual = np.asarray(self.data.qvel[self._arm_qvel_addr], dtype=float).copy()
        return np.asarray(
            [actual[i] * self._joint_directions[name] for i, name in enumerate(self._arm_joint_names)],
            dtype=float,
        )

    def get_actuator_names(self) -> list[str]:
        return list(self._control_names)

    def get_camera_names(self) -> list[str]:
        return [self._camera_name_by_observation_key[key] for key in self._camera_name_by_observation_key]

    def get_control_limits(self) -> np.ndarray:
        return np.asarray(self._control_limits, dtype=float).copy()

    def get_proprio_state(self) -> np.ndarray:
        positions = np.concatenate(
            [
                self._get_arm_qpos_logical(),
                np.asarray(self.data.qpos[self._gripper_qpos_addr], dtype=float).copy(),
            ]
        )
        velocities = np.concatenate(
            [
                self._get_arm_qvel_logical(),
                np.asarray(self.data.qvel[self._gripper_qvel_addr], dtype=float).copy(),
            ]
        )
        return np.concatenate([positions, velocities], dtype=float)

    def get_control_targets(self) -> np.ndarray:
        if self.model.nu <= 0:
            return np.zeros(0, dtype=float)
        return np.asarray(self.data.ctrl, dtype=float).copy()

    def set_control_targets(self, ctrl_targets: np.ndarray, *, substeps: int | None = None) -> np.ndarray:
        if self.model.nu <= 0:
            raise RuntimeError("MuJoCo model does not expose any actuators.")
        ctrl = np.asarray(ctrl_targets, dtype=float).reshape(self.model.nu)
        clipped = np.clip(ctrl, self._control_limits[:, 0], self._control_limits[:, 1])
        self.data.ctrl[:] = clipped
        for _ in range(max(int(self.config.action_substeps if substeps is None else substeps), 1)):
            mujoco.mj_step(self.model, self.data)
        self._last_joint_positions = self._get_arm_qpos_logical()
        self._sync_viewer()
        return clipped

    def _render_camera(self, camera_name: str) -> np.ndarray:
        if self.renderer is None and not self._renderer_failed:
            try:
                self.renderer = mujoco.Renderer(self.model, self.config.render_height, self.config.render_width)
            except Exception as exc:
                logger.warning("MuJoCo renderer unavailable: %s", exc)
                self._renderer_failed = True
        if self.renderer is None:
            return np.zeros((self.config.render_height, self.config.render_width, 3), dtype=np.uint8)

        camera_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name))
        if camera_id < 0:
            if camera_name not in self._render_camera_missing_warned:
                logger.warning("Render camera '%s' not found in MuJoCo model; returning blank frame.", camera_name)
                self._render_camera_missing_warned.add(camera_name)
            return np.zeros((self.config.render_height, self.config.render_width, 3), dtype=np.uint8)

        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = camera_id
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    def render_cameras(self) -> dict[str, np.ndarray]:
        return {
            obs_key: self._render_camera(camera_name)
            for obs_key, camera_name in self._camera_name_by_observation_key.items()
        }

    def _extract_pose_action(self, action: dict[str, Any]) -> np.ndarray | None:
        pos = {}
        for key, axis in (("ee.x", "x"), ("x", "x"), ("ee.y", "y"), ("y", "y"), ("ee.z", "z"), ("z", "z")):
            if key in action:
                pos[axis] = action[key]

        rot = {}
        for key, axis in (
            ("ee.rx", "rx"),
            ("rx", "rx"),
            ("ee.ry", "ry"),
            ("ry", "ry"),
            ("ee.rz", "rz"),
            ("rz", "rz"),
            ("ee.wx", "rx"),
            ("wx", "rx"),
            ("ee.wy", "ry"),
            ("wy", "ry"),
            ("ee.wz", "rz"),
            ("wz", "rz"),
        ):
            if key in action:
                rot[axis] = action[key]

        if {"x", "y", "z"} <= pos.keys() and {"rx", "ry", "rz"} <= rot.keys():
            return np.asarray([pos["x"], pos["y"], pos["z"], rot["rx"], rot["ry"], rot["rz"]], dtype=float)
        return None

    def _extract_pose_delta(self, action: dict[str, Any]) -> np.ndarray | None:
        pos = {}
        for key, axis in (
            ("delta_x", "x"),
            ("dx", "x"),
            ("delta_y", "y"),
            ("dy", "y"),
            ("delta_z", "z"),
            ("dz", "z"),
        ):
            if key in action:
                pos[axis] = float(action[key])

        rot = {}
        for key, axis in (
            ("delta_rx", "rx"),
            ("drx", "rx"),
            ("delta_ry", "ry"),
            ("dry", "ry"),
            ("delta_rz", "rz"),
            ("drz", "rz"),
        ):
            if key in action:
                rot[axis] = float(action[key])

        if not pos and not rot:
            return None

        return np.asarray(
            [
                pos.get("x", 0.0),
                pos.get("y", 0.0),
                pos.get("z", 0.0),
                rot.get("rx", 0.0),
                rot.get("ry", 0.0),
                rot.get("rz", 0.0),
            ],
            dtype=float,
        )

    def _extract_delta_frame(self, action: dict[str, Any]) -> str:
        frame = str(
            action.get(
                "delta_frame",
                action.get("pose_delta_frame", action.get("reference_frame", "world")),
            )
        ).strip().lower()
        if frame in {"world", "base", "base_link"}:
            return "world"
        if frame in {"tool", "tool0", "ee", "end_effector", "end-effector"}:
            return "tool"
        raise ValueError(f"Unsupported delta_frame '{frame}'. Expected 'world' or 'tool'.")

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        self._reset_sim()
        if self.config.use_viewer:
            try:
                require_mujoco_viewer_backend()
                import mujoco.viewer as mj_viewer

                self._viewer = mj_viewer.launch_passive(self.model, self.data)
            except Exception as exc:
                logger.warning("Failed to launch MuJoCo viewer: %s", exc)
                self._viewer = None
        self._is_connected = True
        logger.info("%s connected (sim).", self)

    def disconnect(self) -> None:
        self._is_connected = False
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
            self.renderer = None
        logger.info("%s disconnected (sim).", self)

    def _build_ctrl_targets(
        self,
        arm_targets_logical: np.ndarray,
        gripper_targets_actual: np.ndarray | None,
    ) -> np.ndarray:
        ctrl_targets = self.get_control_targets()
        if ctrl_targets.size == 0:
            raise RuntimeError("MuJoCo model does not expose any actuators.")

        arm_actual = np.asarray(
            [
                arm_targets_logical[i] * self._joint_directions[name]
                for i, name in enumerate(self._arm_joint_names)
            ],
            dtype=float,
        )
        arm_actual = np.clip(arm_actual, self._arm_ctrl_limits_actual[:, 0], self._arm_ctrl_limits_actual[:, 1])
        ctrl_targets[self._arm_actuator_ids] = arm_actual

        if gripper_targets_actual is not None:
            clipped_gripper = np.clip(
                np.asarray(gripper_targets_actual, dtype=float),
                self._gripper_ctrl_limits_actual[:, 0],
                self._gripper_ctrl_limits_actual[:, 1],
            )
            ctrl_targets[self._gripper_actuator_ids] = clipped_gripper

        return ctrl_targets

    def _send_joint_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        arm_targets = self._get_arm_qpos_logical()
        for index, logical_name in enumerate(self._arm_joint_names):
            urdf_name = self._urdf_joint_names[index]
            for key in (f"{logical_name}.pos", f"{urdf_name}.pos"):
                if key in action:
                    arm_targets[index] = float(action[key])
                    break
        arm_targets = np.clip(arm_targets, self._arm_joint_limits_logical[:, 0], self._arm_joint_limits_logical[:, 1])

        max_step = float(self.config.max_delta_per_step)
        if max_step > 0:
            current = self._get_arm_qpos_logical()
            arm_targets = current + np.clip(arm_targets - current, -max_step, max_step)

        gripper_targets_actual = None
        if self.config.use_gripper:
            current_gripper = np.asarray(self.data.qpos[self._gripper_qpos_addr], dtype=float).copy()
            left_target = current_gripper[0]
            right_target = current_gripper[1]
            if "gripper.pos" in action:
                joint_target = float(np.clip(action["gripper.pos"], self.config.gripper_min_pos, self.config.gripper_max_pos))
                left_target = joint_target
                right_target = joint_target
            if "gripper_left.pos" in action:
                left_target = float(np.clip(action["gripper_left.pos"], self.config.gripper_min_pos, self.config.gripper_max_pos))
            if "gripper_right.pos" in action:
                right_target = float(np.clip(action["gripper_right.pos"], self.config.gripper_min_pos, self.config.gripper_max_pos))
            gripper_targets_actual = np.asarray([left_target, right_target], dtype=float)
            max_gripper_step = float(getattr(self.config, "gripper_max_delta_per_step", 0.0))
            if max_gripper_step > 0.0:
                gripper_targets_actual = current_gripper + np.clip(
                    gripper_targets_actual - current_gripper,
                    -max_gripper_step,
                    max_gripper_step,
                )

        ctrl_targets = self._build_ctrl_targets(arm_targets, gripper_targets_actual)
        applied_ctrl = self.set_control_targets(ctrl_targets)

        sent_action = dict(action)
        for index, logical_name in enumerate(self._arm_joint_names):
            sent_action[f"{logical_name}.pos"] = float(arm_targets[index])
        if gripper_targets_actual is not None:
            sent_action["gripper_left.pos"] = float(applied_ctrl[self._gripper_actuator_ids[0]])
            sent_action["gripper_right.pos"] = float(applied_ctrl[self._gripper_actuator_ids[1]])
            sent_action["gripper.pos"] = float(np.mean(gripper_targets_actual))
        return sent_action

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if isinstance(action, (list, np.ndarray)):
            action = np.asarray(action, dtype=float).reshape(-1)
            if action.shape[0] == 8:
                action = {
                    "joint_1.pos": float(action[0]),
                    "joint_2.pos": float(action[1]),
                    "joint_3.pos": float(action[2]),
                    "joint_4.pos": float(action[3]),
                    "joint_5.pos": float(action[4]),
                    "joint_6.pos": float(action[5]),
                    "gripper_left.pos": float(action[6]),
                    "gripper_right.pos": float(action[7]),
                }
            elif action.shape[0] == 7:
                action = {
                    "joint_1.pos": float(action[0]),
                    "joint_2.pos": float(action[1]),
                    "joint_3.pos": float(action[2]),
                    "joint_4.pos": float(action[3]),
                    "joint_5.pos": float(action[4]),
                    "joint_6.pos": float(action[5]),
                    "gripper.pos": float(action[6]),
                }
            elif action.shape[0] == 6:
                action = {
                    "ee.x": float(action[0]),
                    "ee.y": float(action[1]),
                    "ee.z": float(action[2]),
                    "ee.rx": float(action[3]),
                    "ee.ry": float(action[4]),
                    "ee.rz": float(action[5]),
                }
            else:
                raise ValueError(f"Unsupported array action shape: {action.shape}")

        delta_action = None
        base_target_pose = None

        pose_action = self._extract_pose_action(action)
        if pose_action is not None:
            self._target_pose = np.asarray(pose_action, dtype=float)

        if pose_action is None:
            delta_action = self._extract_pose_delta(action)
            if delta_action is not None:
                if self.kinematics is None:
                    raise RuntimeError("Pose actions require pinocchio. Install it with: pip install pinocchio")
                seed = self._last_joint_positions
                if seed is None:
                    seed = self._get_arm_qpos_logical()
                if self._target_pose is None:
                    self._target_pose = self.kinematics.compute_fk(seed)
                base_target_pose = np.asarray(self._target_pose, dtype=float)
                delta_frame = self._extract_delta_frame(action)
                pose_action = compose_pose_delta(
                    base_target_pose,
                    delta_action,
                    rotation_frame=delta_frame,
                    translation_frame=delta_frame,
                )

        if pose_action is not None:
            if self.kinematics is None:
                raise RuntimeError("Pose actions require pinocchio. Install it with: pip install pinocchio")
            seed = self._last_joint_positions
            if seed is None:
                seed = self._get_arm_qpos_logical()
            translation_only = delta_action is not None and np.linalg.norm(delta_action[3:]) < 1e-9
            ik_solution = None

            if delta_action is not None and base_target_pose is not None and translation_only:
                best_solution = None
                best_target = None
                best_rot_err = float("inf")
                for scale in (1.0, 0.5, 0.25, 0.1):
                    trial_target = base_target_pose.copy()
                    trial_target[:3] = base_target_pose[:3] + delta_action[:3] * scale
                    trial_target[3:] = base_target_pose[3:]
                    trial_solution = self.kinematics.compute_ik(
                        trial_target,
                        seed,
                        position_weight=1.0,
                        orientation_weight=20.0,
                        keep_pointing_only=True,
                    )
                    achieved = self.kinematics.compute_fk(trial_solution)
                    rot_err = float(np.linalg.norm(achieved[3:] - trial_target[3:]))
                    if rot_err < best_rot_err:
                        best_rot_err = rot_err
                        best_solution = trial_solution
                        best_target = trial_target
                    if rot_err <= 0.05:
                        ik_solution = trial_solution
                        pose_action = trial_target
                        break
                if ik_solution is None and best_solution is not None and best_target is not None:
                    ik_solution = best_solution
                    pose_action = best_target

            if ik_solution is None:
                ik_solution = self.kinematics.compute_ik(
                    pose_action,
                    seed,
                    position_weight=1.0,
                    orientation_weight=20.0,
                    keep_pointing_only=translation_only,
                )

            self._target_pose = np.asarray(pose_action, dtype=float)
            joint_action = {f"{name}.pos": float(ik_solution[i]) for i, name in enumerate(self._arm_joint_names)}
            if "gripper.pos" in action:
                joint_action["gripper.pos"] = action["gripper.pos"]
            if "gripper_left.pos" in action:
                joint_action["gripper_left.pos"] = action["gripper_left.pos"]
            if "gripper_right.pos" in action:
                joint_action["gripper_right.pos"] = action["gripper_right.pos"]
            self._last_joint_positions = ik_solution
            return self._send_joint_action(joint_action)

        self._target_pose = None
        return self._send_joint_action(action)

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs: dict[str, Any] = {}
        actuator_force = np.asarray(self.data.actuator_force, dtype=float).copy() if self.model.nu > 0 else np.zeros(0, dtype=float)

        for index, (logical_name, qpos_addr, qvel_addr) in enumerate(
            zip(self._arm_joint_names, self._arm_qpos_addr, self._arm_qvel_addr, strict=True)
        ):
            direction = self._joint_directions[logical_name]
            obs[f"{logical_name}.pos"] = float(self.data.qpos[qpos_addr]) * direction
            obs[f"{logical_name}.vel"] = float(self.data.qvel[qvel_addr]) * direction
            obs[f"{logical_name}.torque"] = float(actuator_force[self._arm_actuator_ids[index]]) if actuator_force.size else 0.0

        if self.config.use_gripper:
            gripper_qpos = np.asarray(self.data.qpos[self._gripper_qpos_addr], dtype=float)
            gripper_qvel = np.asarray(self.data.qvel[self._gripper_qvel_addr], dtype=float)
            obs["gripper.pos"] = float(np.mean(gripper_qpos)) * self._gripper_direction
            obs["gripper.vel"] = float(np.mean(gripper_qvel)) * self._gripper_direction
            obs["gripper.torque"] = float(np.mean(actuator_force[self._gripper_actuator_ids])) if actuator_force.size else 0.0

        if self._arm_joint_names:
            self._last_joint_positions = np.asarray(
                [obs[f"{name}.pos"] for name in self._arm_joint_names],
                dtype=float,
            )

        obs.update(self.render_cameras())
        return obs

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in self._joint_names}

    @property
    def observation_features(self) -> dict[str, Any]:
        features = {f"{name}.pos": float for name in self._joint_names}
        features.update({f"{name}.vel": float for name in self._joint_names})
        features.update({f"{name}.torque": float for name in self._joint_names})
        features[self.config.front_camera_observation_key] = (
            self.config.render_height,
            self.config.render_width,
            3,
        )
        features[self.config.wrist_camera_observation_key] = (
            self.config.render_height,
            self.config.render_width,
            3,
        )
        return features

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._is_connected

    @property
    def generated_scene_path(self) -> Path | None:
        return self._generated_scene_path

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass
