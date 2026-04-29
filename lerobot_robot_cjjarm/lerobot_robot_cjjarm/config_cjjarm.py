# uploaded: lerobot_robot_cjjarm/config_cjjarm.py

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.config import RobotConfig

from .DM_CAN import DM_Motor_Type


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


def _coerce_camera_index_or_path(value: str | int | Path) -> str | int | Path:
    if isinstance(value, int | Path):
        return value
    stripped = str(value).strip()
    if stripped.isdecimal():
        return int(stripped)
    return stripped


def _auto_detect_dabaidcw_rgb_path() -> str | None:
    by_id_dir = Path("/dev/v4l/by-id")
    if not by_id_dir.exists():
        return None

    tokens = ("dabai", "da-bai", "da_bai", "dcw")
    for entry in sorted(by_id_dir.iterdir(), key=lambda item: item.name):
        name = entry.name.lower()
        if any(token in name for token in tokens):
            return str(entry)
    return None


def _default_dabaidcw_index_or_path() -> str | int | Path:
    for env_name in (
        "LEROBOT_DABAIDCW_RGB_INDEX_OR_PATH",
        "LEROBOT_DABAIDCW_RGB",
        "LEROBOT_WRIST_RGB_INDEX_OR_PATH",
    ):
        value = os.environ.get(env_name)
        if value:
            return _coerce_camera_index_or_path(value)

    detected_path = _auto_detect_dabaidcw_rgb_path()
    if detected_path is not None:
        return detected_path

    return 2


def dabaidcw_rgb_camera_config(
    *,
    index_or_path: str | int | Path | None = None,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    fourcc: str | None = "MJPG",
) -> OpenCVCameraConfig:
    """DaBai DCW RGB-only wrist camera over OpenCV/V4L2 UVC, without Orbbec SDK."""

    if index_or_path is None:
        resolved_index_or_path = _default_dabaidcw_index_or_path()
    else:
        resolved_index_or_path = _coerce_camera_index_or_path(index_or_path)
    return OpenCVCameraConfig(
        index_or_path=resolved_index_or_path,
        fps=fps,
        width=width,
        height=height,
        color_mode=ColorMode.RGB,
        fourcc=fourcc,
    )


def wrist_rgb_camera_config(
    *,
    index_or_path: str | int | Path | None = None,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    fourcc: str | None = "MJPG",
) -> OpenCVCameraConfig:
    return dabaidcw_rgb_camera_config(
        index_or_path=index_or_path,
        width=width,
        height=height,
        fps=fps,
        fourcc=fourcc,
    )


def default_cjjarm_cameras_config() -> dict[str, CameraConfig]:
    return {
        # 上帝视角相机 (例如笔记本自带或顶部USB相机)
        "laptop": OpenCVCameraConfig(
            index_or_path="/dev/video0",  # 替换为你实际的相机ID
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.RGB,
        ),
        # DaBai DCW 手腕 RGB 相机：普通 UVC/OpenCV 输入，只读 RGB，不启用 depth/点云/对齐。
        "wrist": wrist_rgb_camera_config(),
    }


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
    default_arm_velocity: float = 5.0

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
    
    cameras: dict[str, CameraConfig] = field(default_factory=default_cjjarm_cameras_config)
