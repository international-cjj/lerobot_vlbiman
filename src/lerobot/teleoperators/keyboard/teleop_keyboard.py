#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import time
from queue import Queue
from typing import Any

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_keyboard import (
    KeyboardEndEffectorTeleopConfig,
    KeyboardJointTeleopConfig,
    KeyboardPoseTeleopConfig,
    KeyboardTeleopConfig,
)

PYNPUT_AVAILABLE = True
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logging.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")

    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as e:
    keyboard = None
    PYNPUT_AVAILABLE = False
    logging.info(f"Could not import pynput: {e}")


def _normalize_key_event(key: Any) -> Any:
    """Convert pynput KeyCode objects into comparable lowercase chars."""
    key_char = getattr(key, "char", None)
    if isinstance(key_char, str):
        return key_char.lower()
    return key


class KeyboardTeleop(Teleoperator):
    """
    Teleop class to use keyboard inputs for control.
    """

    config_class = KeyboardTeleopConfig
    name = "keyboard"

    def __init__(self, config: KeyboardTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None
        self.logs = {}

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.arm),),
            "names": {"motors": list(self.arm.motors)},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        pass

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "Keyboard is already connected. Do not run `robot.connect()` twice."
            )

        if PYNPUT_AVAILABLE:
            logging.info("pynput is available - enabling local keyboard listener.")
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start()
        else:
            logging.info("pynput not available - skipping local keyboard listener.")
            self.listener = None

    def calibrate(self) -> None:
        pass

    def _on_press(self, key):
        normalized_key = _normalize_key_event(key)
        if normalized_key is not None:
            self.event_queue.put((normalized_key, True))

    def _on_release(self, key):
        normalized_key = _normalize_key_event(key)
        if normalized_key is not None:
            self.event_queue.put((normalized_key, False))
        if keyboard is not None and key == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()

    def _drain_pressed_keys(self):
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            self.current_pressed[key_char] = is_pressed

    def configure(self):
        pass

    def get_action(self) -> dict[str, Any]:
        before_read_t = time.perf_counter()

        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()

        # Generate action based on current key states
        action = {key for key, val in self.current_pressed.items() if val}
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return dict.fromkeys(action, None)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `robot.connect()` before `disconnect()`."
            )
        if self.listener is not None:
            self.listener.stop()


class KeyboardEndEffectorTeleop(KeyboardTeleop):
    """
    Teleop class to use keyboard inputs for end effector control.
    Designed to be used with the `So100FollowerEndEffector` robot.
    """

    config_class = KeyboardEndEffectorTeleopConfig
    name = "keyboard_ee"

    def __init__(self, config: KeyboardEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self.misc_keys_queue = Queue()

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()
        delta_x = 0.0
        delta_y = 0.0
        delta_z = 0.0
        gripper_action = 1.0

        # Generate action based on current key states
        for key, val in self.current_pressed.items():
            if key == keyboard.Key.up:
                delta_y = -int(val)
            elif key == keyboard.Key.down:
                delta_y = int(val)
            elif key == keyboard.Key.left:
                delta_x = int(val)
            elif key == keyboard.Key.right:
                delta_x = -int(val)
            elif key == keyboard.Key.shift:
                delta_z = -int(val)
            elif key == keyboard.Key.shift_r:
                delta_z = int(val)
            elif key == keyboard.Key.ctrl_r:
                # Gripper actions are expected to be between 0 (close), 1 (stay), 2 (open)
                gripper_action = int(val) + 1
            elif key == keyboard.Key.ctrl_l:
                gripper_action = int(val) - 1
            elif val:
                # If the key is pressed, add it to the misc_keys_queue
                # this will record key presses that are not part of the delta_x, delta_y, delta_z
                # this is useful for retrieving other events like interventions for RL, episode success, etc.
                self.misc_keys_queue.put(key)

        self.current_pressed.clear()

        action_dict = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
        }

        if self.config.use_gripper:
            action_dict["gripper"] = gripper_action

        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the keyboard such as intervention status,
        episode termination, success indicators, etc.

        Keyboard mappings:
        - Any movement keys pressed = intervention active
        - 's' key = success (terminate episode successfully)
        - 'r' key = rerecord episode (terminate and rerecord)
        - 'q' key = quit episode (terminate without success)

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        if not self.is_connected:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Check if any movement keys are currently pressed (indicates intervention)
        movement_keys = [
            keyboard.Key.up,
            keyboard.Key.down,
            keyboard.Key.left,
            keyboard.Key.right,
            keyboard.Key.shift,
            keyboard.Key.shift_r,
            keyboard.Key.ctrl_r,
            keyboard.Key.ctrl_l,
        ]
        is_intervention = any(self.current_pressed.get(key, False) for key in movement_keys)

        # Check for episode control commands from misc_keys_queue
        terminate_episode = False
        success = False
        rerecord_episode = False

        # Process any pending misc keys
        while not self.misc_keys_queue.empty():
            key = self.misc_keys_queue.get_nowait()
            if key == "s":
                success = True
            elif key == "r":
                terminate_episode = True
                rerecord_episode = True
            elif key == "q":
                terminate_episode = True
                success = False

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }
# [在文件末尾添加这个类]
class KeyboardJointTeleop(KeyboardTeleop):
    """
    Teleop class to control robot joints directly using keyboard inputs.
    Maintains virtual joint positions that are updated by key presses.
    """
    config_class = KeyboardJointTeleopConfig
    name = "keyboard_joint"

    def __init__(self, config: KeyboardJointTeleopConfig):
        super().__init__(config)
        self.config = config
        
        # 初始化所有关节的目标位置
        self.joint_positions = {
            joint: self.config.init_pos 
            for joint in self.config.joints
        }
        
        # 定义按键映射 (Key -> (Joint Name, Direction))
        # 1-6 增加角度, Q-Y 减少角度
        self.key_map = {
            '1': (self.config.joints[0], 1),  'q': (self.config.joints[0], -1),
            '2': (self.config.joints[1], 1),  'w': (self.config.joints[1], -1),
            '3': (self.config.joints[2], 1),  'e': (self.config.joints[2], -1),
            '4': (self.config.joints[3], 1),  'r': (self.config.joints[3], -1),
            '5': (self.config.joints[4], 1),  't': (self.config.joints[4], -1),
            '6': (self.config.joints[5], 1),  'y': (self.config.joints[5], -1),
        }

    @property
    def action_features(self) -> dict:
        # 告诉 LeRobot 我们输出的是 float 类型的关节数据
        return {f"{joint}.pos": float for joint in self.config.joints}

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardJointTeleop is not connected. Run `connect()` first."
            )

        # 1. 读取按键
        self._drain_pressed_keys()
        
        # 2. 根据按键更新虚拟关节角度
        for key_char, is_pressed in self.current_pressed.items():
            if is_pressed and key_char in self.key_map:
                joint_name, direction = self.key_map[key_char]
                self.joint_positions[joint_name] += direction * self.config.step_size

        # 3. 构造标准的 Action 字典
        # 例如: {'joint1.pos': 0.15, 'joint2.pos': -0.5, ...}
        action_dict = {
            f"{joint}.pos": pos 
            for joint, pos in self.joint_positions.items()
        }
        
        return action_dict


class KeyboardPoseTeleop(KeyboardTeleop):
    """
    Teleop class to control end-effector pose (delta position + delta rotation) using keyboard inputs.
    """

    config_class = KeyboardPoseTeleopConfig
    name = "keyboard_pose"

    def __init__(self, config: KeyboardPoseTeleopConfig):
        super().__init__(config)
        self.config = config
        self._gripper_value = 0.0

    @property
    def action_features(self) -> dict:
        features = {
            "delta_x": float,
            "delta_y": float,
            "delta_z": float,
            "delta_rx": float,
            "delta_ry": float,
            "delta_rz": float,
        }
        if self.config.use_gripper:
            features["gripper.pos"] = float
        return features

    def _on_press(self, key):
        normalized_key = _normalize_key_event(key)
        if normalized_key is not None:
            self.event_queue.put((normalized_key, True))

    def _on_release(self, key):
        normalized_key = _normalize_key_event(key)
        if normalized_key is not None:
            self.event_queue.put((normalized_key, False))
        if keyboard is not None and key == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardPoseTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()

        delta_x = 0.0
        delta_y = 0.0
        delta_z = 0.0
        delta_rx = 0.0
        delta_ry = 0.0
        delta_rz = 0.0
        z_up_pressed = False
        z_down_pressed = False

        for key, val in self.current_pressed.items():
            if not val:
                continue
            if keyboard is not None:
                if key == keyboard.Key.up:
                    # Forward (arm pointing direction) is defined as +X.
                    delta_x = 1.0
                elif key == keyboard.Key.down:
                    delta_x = -1.0
                elif key == keyboard.Key.left:
                    # Left/right strafing is defined on Y axis.
                    delta_y = 1.0
                elif key == keyboard.Key.right:
                    delta_y = -1.0
                elif key == keyboard.Key.shift:
                    z_up_pressed = True
                elif key == keyboard.Key.shift_r:
                    z_down_pressed = True
                elif key == keyboard.Key.ctrl_l:
                    self._gripper_value = -1.0
                elif key == keyboard.Key.ctrl_r:
                    self._gripper_value = 1.0

            if isinstance(key, str):
                if key == "q":
                    delta_rx = 1.0
                elif key == "e":
                    delta_rx = -1.0
                elif key == "w":
                    delta_ry = 1.0
                elif key == "s":
                    delta_ry = -1.0
                elif key == "a":
                    delta_rz = 1.0
                elif key == "d":
                    delta_rz = -1.0
        if z_up_pressed and not z_down_pressed:
            delta_z = 1.0
        elif z_down_pressed and not z_up_pressed:
            delta_z = -1.0

        self.current_pressed.clear()

        action = {
            "delta_x": delta_x * self.config.pos_step,
            "delta_y": delta_y * self.config.pos_step,
            "delta_z": delta_z * self.config.pos_step,
            "delta_rx": delta_rx * self.config.rot_step,
            "delta_ry": delta_ry * self.config.rot_step,
            "delta_rz": delta_rz * self.config.rot_step,
        }
        if self.config.use_gripper:
            action["gripper.pos"] = float(self._gripper_value)
        return action
