# lerobot_teleoperator_zhonglin/zhonglin_teleop.py
import logging
import time
import re
import serial
import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from .config_zhonglin import ZhonglinTeleopConfig

logger = logging.getLogger(__name__)

class ZhonglinTeleop(Teleoperator):
    config_class = ZhonglinTeleopConfig
    name = "zhonglin_teleop"

    def __init__(self, config: ZhonglinTeleopConfig):
        super().__init__(config)
        self.config = config
        
        self.ser = None
        self._connected = False
        self._calibrated = False
        self._command_delay_s = float(self.config.command_delay_s)
        
        # 1. 轴配置：使用 7 个轴 (舵机ID通常从 0 到 6)
        # 修改点：添加 6 到列表中
        self.valid_servos = [0, 1, 2, 3, 4, 5, 6]
        # 2. 映射配置：舵机ID -> LeRobot关节名称
        # 修改为与 cjjarm config 一致的命名 (加下划线，最后改为 gripper)
        self.servo_id_to_name = {
            0: "joint_1.pos",
            1: "joint_2.pos",
            2: "joint_3.pos",
            3: "joint_4.pos",
            4: "joint_6.pos",
            5: "joint_5.pos",
            6: "gripper.pos", # 注意：ID 6 对应 config 中的 "gripper"
        }
        
        # 存储零位角度（单位：度）
        self.zero_angles = {}
        # 存储当前角度（单位：弧度），用于读取失败时保持上一帧
        self.current_angles = {name: 0.0 for name in self.servo_id_to_name.values()}

    @property
    def action_features(self) -> dict:
        return {name: float for name in self.servo_id_to_name.values()}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected and self.ser is not None and self.ser.is_open

    @property
    def is_calibrated(self) -> bool:
        return self.is_connected and self._calibrated

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info(f"Connecting to ZhonglinTeleop on {self.config.port}...")
        
        try:
            self.ser = serial.Serial(
                self.config.port, 
                self.config.baudrate, 
                timeout=self.config.timeout
            )
            self._connected = True
            self._calibrated = False
            logger.info("Serial port opened")
            
            if calibrate:
                self.calibrate()
                
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise ConnectionError(f"Could not open serial port: {e}")

    def calibrate(self) -> None:
        """
        读取当前位置作为零点。
        注意：零点仍然以【度】为单位存储，方便计算相对值。
        """
        logger.info("Initializing servos and setting zero positions...")
        
        # 发送版本查询指令作为握手（可选）
        self._send_command('#000PVER!')
        
        for i in self.valid_servos:
            # 清除之前的状态（如果需要）
            self._send_command("#000PCSK!")
            # 某些型号可能需要先解锁或查询
            self._send_command(f'#{i:03d}PULK!')
            
            angle_deg = self._read_servo_angle_deg(i)
            
            if angle_deg is not None:
                self.zero_angles[i] = angle_deg
            else:
                self.zero_angles[i] = 0.0
                logger.warning(f"Could not read initial position for servo {i}, assuming 0.0")
        
        logger.info(f"Calibration completed. Zero angles (deg): {self.zero_angles}")
        self._calibrated = True

    def get_action(self) -> dict:
        """
        读取舵机角度(度) -> 减去零点 -> 转换为弧度 -> 发送
        """
        if not self.is_connected:
            raise DeviceNotConnectedError("Teleoperator is not connected")

        actions = {}
        
        for i in self.valid_servos:
            joint_name = self.servo_id_to_name.get(i)
            if not joint_name:
                continue

            # 读取指令
            angle_deg = self._read_servo_angle_deg(i)
            
            if angle_deg is not None:
                # 1. 计算相对角度（度）
                relative_deg = angle_deg - self.zero_angles.get(i, 0.0)
                
                # 2. 转换为弧度
                relative_rad = np.deg2rad(relative_deg)
                # 3. 按配置翻转输出方向
                sign = float(self.config.output_signs.get(joint_name, 1.0))
                relative_rad *= sign

                previous_value = float(self.current_angles.get(joint_name, 0.0))
                deadband_rad = float(self.config.joint_deadband_rad.get(joint_name, 0.0))
                if deadband_rad > 0.0 and abs(relative_rad - previous_value) < deadband_rad:
                    relative_rad = previous_value

                # 更新缓存和输出
                self.current_angles[joint_name] = relative_rad
                actions[joint_name] = relative_rad
            else:
                # 读取失败时使用上一次的弧度值
                actions[joint_name] = self.current_angles.get(joint_name, 0.0)

        return actions
    def send_feedback(self, feedback: dict) -> None:
        pass

    def disconnect(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()
        self._connected = False
        self._calibrated = False
        logger.info("Disconnected.")

    def configure(self) -> None:
        pass

    # --- 私有辅助方法 ---

    def _send_command(self, cmd: str) -> str:
        if not self.ser:
            return ""
        try:
            self.ser.write(cmd.encode('ascii'))
            # 这里的延时取决于下位机处理速度，0.008秒通常足够
            time.sleep(self._command_delay_s)
            return self.ser.read_all().decode('ascii', errors='ignore')
        except Exception as e:
            logger.error(f"Serial write error: {e}")
            return ""

    def _read_servo_angle_deg(self, servo_id: int) -> float | None:
        response = self._send_command(f'#{servo_id:03d}PRAD!')
        return self._pwm_to_angle(response.strip())

    def _pwm_to_angle(self, response_str: str, pwm_min=500, pwm_max=2500, angle_range=270) -> float | None:
        # 假设返回格式类似于 "#001P1500" 或 "P1500"
        match = re.search(r'P(\d{4})', response_str)
        if not match:
            return None
        pwm_val = int(match.group(1))
        
        # 简单的线性映射：PWM值 -> 0~270度
        pwm_span = pwm_max - pwm_min
        angle = (pwm_val - pwm_min) / pwm_span * angle_range
        return angle
