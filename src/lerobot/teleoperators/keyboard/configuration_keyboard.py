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

from dataclasses import dataclass, field  # <--- 关键修复：导入 field

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("keyboard")
@dataclass
class KeyboardTeleopConfig(TeleoperatorConfig):
    """KeyboardTeleopConfig"""
    pass


@TeleoperatorConfig.register_subclass("keyboard_ee")
@dataclass
class KeyboardEndEffectorTeleopConfig(KeyboardTeleopConfig):
    use_gripper: bool = True


@TeleoperatorConfig.register_subclass("keyboard_pose")
@dataclass
class KeyboardPoseTeleopConfig(KeyboardTeleopConfig):
    pos_step: float = 0.01
    rot_step: float = 0.05
    use_gripper: bool = True


# [新增] 关节空间键盘控制配置
@TeleoperatorConfig.register_subclass("keyboard_joint")
@dataclass
class KeyboardJointTeleopConfig(KeyboardTeleopConfig):
    # 默认控制的关节列表 (适配你的达妙机械臂)
    joints: list[str] = field(default_factory=lambda: [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"
    ])
    # 每次按键增加/减少的角度 (弧度)
    step_size: float = 0.05
    # 初始位置
    init_pos: float = 0.0
