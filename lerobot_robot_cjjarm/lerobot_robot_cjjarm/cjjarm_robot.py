# uploaded: lerobot_robot_cjjarm/cjjarm_robot.py

import logging
import time
import serial
import numpy as np
from typing import Any

from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.cameras import make_cameras_from_configs

from .DM_CAN import MotorControl, Motor, DM_Motor_Type, Control_Type, DM_variable
from .config_cjjarm import CjjArmConfig
from .kinematics import CjjArmKinematics, compose_pose_delta

logger = logging.getLogger(__name__)

class CjjArm(Robot):
    config_class = CjjArmConfig
    name = "cjjarm"

    def __init__(self, config: CjjArmConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)

        self.kinematics = None
        try:
            self.kinematics = CjjArmKinematics(
                urdf_path=self.config.urdf_path,
                end_effector_frame=self.config.end_effector_frame,
                joint_names=list(self.config.urdf_joint_map.values()),
                damping=self.config.ik_damping,
            )
            logger.info("CjjArm IK damping set to %.2e", self.config.ik_damping)
        except ImportError as exc:
            logger.warning(
                "Pinocchio unavailable, pose actions will be disabled: %s",
                exc,
            )
        self._arm_joint_names = list(self.config.urdf_joint_map.keys())
        self._joint_order = list(self.config.joint_map.keys())
        self._last_joint_positions = None

        self._is_connected = False
        self.serial_device = None
        self.motor_control = None
        
        self.motors = {} 
        self.motor_directions = {}
        self._prev_targets = {}
        
        self._gripper_open_pos = float(self.config.gripper_open_pos)
        self._gripper_closed_pos = float(self.config.gripper_closed_pos)
        self._gripper_trigger_threshold = float(self.config.gripper_trigger_threshold)
        self._gripper_max_current = int(self.config.gripper_max_current)
        self._gripper_speed = int(self.config.gripper_speed)

    def _configure_motor(self, joint_name: str, motor: Motor) -> None:
        if joint_name == "gripper":
            self.motor_control.switchControlMode(motor, Control_Type.Torque_Pos)
        else:
            self.motor_control.switchControlMode(motor, Control_Type.POS_VEL)

        if joint_name in ["joint_1", "joint_2", "joint_3"]:
            self.motor_control.change_motor_param(motor, DM_variable.ACC, 10.0)
            self.motor_control.change_motor_param(motor, DM_variable.DEC, -10.0)
            self.motor_control.change_motor_param(motor, DM_variable.KP_APR, 200)
            self.motor_control.change_motor_param(motor, DM_variable.KI_APR, 10)
        elif joint_name == "gripper":
            self.motor_control.change_motor_param(motor, DM_variable.KP_APR, 100)

    def connect(self):
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            if not self.config.serial_port:
                raise DeviceNotConnectedError(
                    "serial_port is empty. Set --robot.serial_port to connect to the arm."
                )
            logger.info(f"Connecting to {self.config.serial_port}...")
            self.serial_device = serial.Serial(
                self.config.serial_port, 
                baudrate=self.config.baud_rate, 
                timeout=self.config.timeout
            )
            self.motor_control = MotorControl(self.serial_device)
            
            for joint_name, params in self.config.joint_map.items():
                slave_id, master_id, motor_type_int, direction = params
                
                motor = Motor(DM_Motor_Type(motor_type_int), slave_id, master_id)
                self.motor_control.addMotor(motor)
                self.motors[joint_name] = motor
                
                self.motor_directions[joint_name] = direction
                
                self.motor_control.refresh_motor_status(motor)
                time.sleep(0.02)

                #logger.info(f"Setting zero position for {joint_name}...")
                self.motor_control.set_zero_position(motor)

                self._configure_motor(joint_name, motor)
                
                self.motor_control.enable(motor)
                
                current_pos = motor.getPosition() * direction
                self._prev_targets[joint_name] = current_pos
                
            for cam in self.cameras.values():
                cam.connect()

            self._is_connected = True
            self._last_joint_positions = np.array(
                [self._prev_targets[name] for name in self._arm_joint_names], dtype=float
            )
            logger.info(f"{self} connected successfully.")
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.disconnect()
            raise DeviceNotConnectedError(f"Connection failed: {e}")

    def disconnect(self):
        if not self.is_connected: return
        if self.motor_control:
            for motor in self.motors.values():
                try:
                    self.motor_control.disable(motor)
                except Exception: pass
            if self.serial_device and self.serial_device.is_open:
                self.serial_device.close()
        for cam in self.cameras.values():
            cam.disconnect()
        self._is_connected = False
        logger.info(f"{self} disconnected.")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        self.motor_control.recv()
        
        obs_dict = {}
        for joint_name, motor in self.motors.items():
            sign = self.motor_directions[joint_name]
            obs_dict[f"{joint_name}.pos"] = motor.getPosition() * sign
            obs_dict[f"{joint_name}.vel"] = motor.getVelocity() * sign
            obs_dict[f"{joint_name}.torque"] = motor.getTorque() * sign

        if self._arm_joint_names:
            self._last_joint_positions = np.array(
                [obs_dict[f"{name}.pos"] for name in self._arm_joint_names], dtype=float
            )
        
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()
            
        return obs_dict

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
            return np.array(
                [pos["x"], pos["y"], pos["z"], rot["rx"], rot["ry"], rot["rz"]], dtype=float
            )
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

        return np.array(
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

    def _send_joint_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        for joint_name, motor in self.motors.items():
            target_key = f"{joint_name}.pos"
            if target_key not in action:
                continue
            
            raw_input = action[target_key] # 主手传来的原始数据
            sign = self.motor_directions[joint_name]
            
            # === [修改] 爪子阈值控制逻辑 ===
            if joint_name == "gripper":
                # 逻辑：如果输入值小于阈值(按下)，则闭合；否则保持张开
                # 注意：raw_input 是主手的值，sign是1
                
                if raw_input < self._gripper_trigger_threshold:
                    # 触发闭合
                    physical_target = self._gripper_closed_pos
                else:
                    # 保持张开
                    physical_target = self._gripper_open_pos
                
                # 发送力位混合指令
                self.motor_control.control_pos_force(
                    motor, 
                    physical_target, 
                    self._gripper_speed, 
                    self._gripper_max_current
                )
            
            # === 手臂关节控制 (保持平滑逻辑) ===
            else:
                smooth_factor = self.config.per_joint_smooth_factor.get(
                    joint_name, self.config.default_smooth_factor
                )
                prev_target = self._prev_targets.get(joint_name, motor.getPosition() * sign)
                
                final_target = (1 - smooth_factor) * prev_target + smooth_factor * raw_input
                self._prev_targets[joint_name] = final_target
                
                self.motor_control.control_Pos_Vel(
                    motor, 
                    final_target * sign, 
                    5.0 
                )

        return action

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if isinstance(action, (list, np.ndarray)):
            action = np.asarray(action, dtype=float)
            if action.shape[0] == 6:
                action = {
                    "ee.x": action[0],
                    "ee.y": action[1],
                    "ee.z": action[2],
                    "ee.rx": action[3],
                    "ee.ry": action[4],
                    "ee.rz": action[5],
                }
            else:
                action = {
                    f"{name}.pos": float(action[i])
                    for i, name in enumerate(self._joint_order)
                    if i < action.shape[0]
                }

        pose_action = self._extract_pose_action(action)
        if pose_action is None:
            delta_action = self._extract_pose_delta(action)
            if delta_action is not None:
                if self.kinematics is None:
                    raise RuntimeError(
                        "Pose actions require pinocchio. Install it with: pip install pinocchio"
                    )
                seed = self._last_joint_positions
                if seed is None:
                    seed = np.array(
                        [
                            self.motors[name].getPosition() * self.motor_directions[name]
                            for name in self._arm_joint_names
                        ],
                        dtype=float,
                    )
                current_pose = self.kinematics.compute_fk(seed)
                pose_action = compose_pose_delta(current_pose, delta_action, rotation_frame="world")

        if pose_action is not None:
            if self.kinematics is None:
                raise RuntimeError(
                    "Pose actions require pinocchio. Install it with: pip install pinocchio"
                )
            seed = self._last_joint_positions
            if seed is None:
                seed = np.array(
                    [self.motors[name].getPosition() * self.motor_directions[name] for name in self._arm_joint_names],
                    dtype=float,
                )
            ik_solution = self.kinematics.compute_ik(pose_action, seed)
            joint_action = {
                f"{name}.pos": float(ik_solution[i]) for i, name in enumerate(self._arm_joint_names)
            }
            if "gripper.pos" in action:
                joint_action["gripper.pos"] = action["gripper.pos"]
            self._last_joint_positions = ik_solution
            return self._send_joint_action(joint_action)

        return self._send_joint_action(action)

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in self.config.joint_map}

    @property
    def observation_features(self) -> dict[str, Any]:
        features = {f"{name}.pos": float for name in self.config.joint_map}
        features.update({f"{name}.vel": float for name in self.config.joint_map})
        features.update({f"{name}.torque": float for name in self.config.joint_map})
        for name, cam_conf in self.config.cameras.items():
            features[name] = (cam_conf.height, cam_conf.width, 3)
        return features
        
    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._is_connected

    def calibrate(self):
        pass
    
    def configure(self):
        pass
