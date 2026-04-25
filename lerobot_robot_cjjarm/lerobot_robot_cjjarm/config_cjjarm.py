# uploaded: lerobot_robot_cjjarm/config_cjjarm.py

from dataclasses import dataclass, field
import glob
from pathlib import Path
from lerobot.robots.config import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from .DM_CAN import DM_Motor_Type

from lerobot.cameras.configs import ColorMode


def _auto_detect_serial_port(pattern: str, device_name: str) -> str:
    """
    Auto-detects a serial port based on a glob pattern.

    Args:
        pattern: The glob pattern to search for (e.g., "/dev/ttyACM*").
        device_name: The human-readable name of the device for error/warning messages.

    Returns:
        The path to the detected serial port.

    Raises:
        RuntimeError: If no port matching the pattern is found.
    """
    ports = sorted(glob.glob(pattern))
    if not ports:
        print(
            f"WARNING: {device_name} not found. "
            f"No ports matching '{pattern}'. Set the port explicitly to connect."
        )
        return ""
    if len(ports) > 1:
        # Use print instead of logging to ensure visibility without special config
        print(f"WARNING: Multiple '{device_name}' devices found: {ports}. Using the first one: '{ports[0]}'.")
    return ports[0]


def _default_urdf_path() -> str:
    base_dir = Path(__file__).resolve().parent
    return str(base_dir / "cjjarm_urdf" / "TRLC-DK1-Follower.urdf")


@RobotConfig.register_subclass("cjjarm")
@dataclass
class CjjArmConfig(RobotConfig):
    # Auto-detect serial port for the cjjarm (mechanical arm)
    serial_port: str = field(default_factory=lambda: _auto_detect_serial_port("/dev/ttyACM*", "cjjarm"))
    # Auto-detect serial port for the Zhongling controller
    zhongling_serial_port: str = field(default_factory=lambda: _auto_detect_serial_port("/dev/ttyUSB*", "Zhongling Controller"))

    baud_rate: int = 921600
    timeout: float = 0.5
    
    joint_map: dict[str, tuple[int, int, int, int]] = field(
        default_factory=lambda: {
            "joint_1": (0x01, 0x11, DM_Motor_Type.DM4340, 1),
            "joint_2": (0x02, 0x12, DM_Motor_Type.DM4340, -1),
            "joint_3": (0x03, 0x13, DM_Motor_Type.DM4340, 1),
            "joint_4": (0x04, 0x14, DM_Motor_Type.DM4310, 1),
            "joint_5": (0x05, 0x15, DM_Motor_Type.DM4310, 1),
            "joint_6": (0x06, 0x16, DM_Motor_Type.DM4310, 1),
            # [修正]: 方向设为 1。
            # 逻辑：主手负值(按下) -> 乘以1 -> 电机负值(逆时针) -> 爪子闭合
            "gripper": (0x07, 0x17, DM_Motor_Type.DM4310, 1), 
        }
    )

    gripper_open_pos: float = 0.0
    gripper_closed_pos: float = -2.8
    # Trigger close when the teleop input drops below this threshold.
    gripper_trigger_threshold: float = -0.15
    # Gripper force control parameters.
    gripper_max_current: int = 1000
    gripper_speed: int = 800

    urdf_path: str = field(default_factory=_default_urdf_path)
    end_effector_frame: str = "tool0"
    # Damped least squares regularization for IK near singularities.
    # Larger values improve stability but reduce tracking sharpness.
    ik_damping: float = 1e-4
    urdf_joint_map: dict[str, str] = field(
        default_factory=lambda: {
            "joint_1": "joint1",
            "joint_2": "joint2",
            "joint_3": "joint3",
            "joint_4": "joint4",
            "joint_5": "joint5",
            "joint_6": "joint6",
        }
    )
    
    default_max_step: float = 1
    default_smooth_factor: float = 1

    per_joint_smooth_factor: dict[str, float] = field(
        default_factory=lambda: {
            "joint_1": 1,
            "joint_2": 1,
            "gripper": 1.0, 
        }
    )

    use_effort: bool = False
    use_velocity: bool = True
    use_acceleration: bool = False
    
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            # 上帝视角相机 (例如笔记本自带或顶部USB相机)
            "laptop": OpenCVCameraConfig(
                index_or_path="/dev/video0",  # 替换为你实际的相机ID
                fps=30,
                width=640,
                height=480,
                color_mode=ColorMode.RGB
            ),
            # 机械臂手腕相机
            "wrist": OpenCVCameraConfig(
                index_or_path="/dev/video2",  # 替换为你实际的相机ID
                fps=30,
                width=640,
                height=480,
                color_mode=ColorMode.RGB
            )
        }
    )
