# lerobot_teleoperator_zhonglin/config_zhonglin.py
from dataclasses import dataclass, field
import glob
from lerobot.teleoperators.config import TeleoperatorConfig

def _auto_detect_serial_port(pattern: str, device_name: str) -> str:
    ports = sorted(glob.glob(pattern))
    if not ports:
        raise RuntimeError(
            f"{device_name} not found. "
            f"Please check hardware connection. No ports matching '{pattern}'."
        )
    if len(ports) > 1:
        print(
            f"WARNING: Multiple '{device_name}' devices found: {ports}. "
            f"Using the first one: '{ports[0]}'."
        )
    return ports[0]

@TeleoperatorConfig.register_subclass("zhonglin_teleop")
@dataclass
class ZhonglinTeleopConfig(TeleoperatorConfig):
    # 自动检测唯一的 Zhongling 控制器串口（/dev/ttyUSB*）
    port: str = field(default_factory=lambda: _auto_detect_serial_port("/dev/ttyUSB*", "Zhongling Controller"))
    baudrate: int = 115200
    timeout: float = 0.1
    command_delay_s: float = 0.01
    # Output sign for each joint name (after zeroing, before sending action).
    # Use -1.0 to invert a joint.
    output_signs: dict[str, float] = field(
        default_factory=lambda: {
            "joint_1.pos": 1.0,
            "joint_2.pos": -1.0,
            "joint_3.pos": -1.0,
            "joint_4.pos": -1.0,
            "joint_5.pos": -1.0,
            "joint_6.pos": 1.0,
            "gripper.pos": 1.0,
        }
    )
    # Ignore small servo readback jitter around the current commanded value.
    # joint_1 is most sensitive because it is the base yaw joint with low static friction in sim.
    joint_deadband_rad: dict[str, float] = field(
        default_factory=lambda: {
            "joint_1.pos": 0.015,
            "joint_2.pos": 0.01,
            "joint_3.pos": 0.01,
            "joint_4.pos": 0.01,
            "joint_5.pos": 0.01,
            "joint_6.pos": 0.01,
            "gripper.pos": 0.015,
        }
    )
