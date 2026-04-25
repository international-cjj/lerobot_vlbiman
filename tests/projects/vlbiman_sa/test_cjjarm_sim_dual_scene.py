from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from lerobot.projects.vlbiman_sa.sim import (
    DEFAULT_EXTERNAL_CAMERA_NAME,
    PROJECT_SCENE_PRESET_NAME,
    PROJECT_YELLOW_BALL_BODY_NAME,
    PROJECT_YELLOW_BALL_RIGHT_OFFSET_M,
)
from lerobot_robot_cjjarm import CjjArmSim, CjjArmSimConfig


def test_cjjarm_sim_default_scene_uses_canonical_dual_camera_chain() -> None:
    sim = CjjArmSim(
        CjjArmSimConfig(
            id="test_default_scene",
            use_viewer=False,
            render_width=160,
            render_height=120,
            scene_settle_steps=0,
        )
    )
    target_joint_id = int(mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "virtual_target_body_free"))
    camera_id = int(mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, DEFAULT_EXTERNAL_CAMERA_NAME))

    assert sim.generated_scene_path is not None
    assert "build_dual_camera_scene" in sim._load_chain_description
    assert target_joint_id >= 0
    assert camera_id >= 0
    assert sim.model.nu == 8
    assert sim.model.ncam == 2

    sim.connect()
    try:
        obs = sim.get_observation()
        assert "observation.images.front" in obs
        assert "observation.images.wrist" in obs
        assert obs["observation.images.front"].shape == (120, 160, 3)
        assert obs["observation.images.wrist"].shape == (120, 160, 3)
    finally:
        sim.disconnect()


def test_cjjarm_sim_project_scene_preserves_freejoint_object_pose() -> None:
    sim = CjjArmSim(
        CjjArmSimConfig(
            id="test_project_scene_freejoint_pose",
            use_viewer=False,
            render_width=160,
            render_height=120,
            scene_preset=PROJECT_SCENE_PRESET_NAME,
            scene_settle_steps=0,
        )
    )
    yellow_ball_body_id = int(
        mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, PROJECT_YELLOW_BALL_BODY_NAME)
    )

    sim.connect()
    try:
        yellow_ball_position = np.asarray(sim.data.xpos[yellow_ball_body_id], dtype=float)
        assert np.allclose(yellow_ball_position[:2], [0.45, -PROJECT_YELLOW_BALL_RIGHT_OFFSET_M])
        assert yellow_ball_position[2] > 0.05
    finally:
        sim.disconnect()


def test_cjjarm_sim_gripper_ctrl_is_continuous() -> None:
    sim = CjjArmSim(
        CjjArmSimConfig(
            id="test_gripper_continuous",
            use_viewer=False,
            render_width=160,
            render_height=120,
            scene_settle_steps=0,
        )
    )
    sim.connect()
    try:
        initial = sim.get_proprio_state()[:8]
        sim.send_action({"gripper.pos": -0.015})
        mid = sim.get_proprio_state()[:8]
        sim.send_action({"gripper.pos": -0.03})
        final = sim.get_proprio_state()[:8]

        left_joint_id = int(mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_left"))
        left_qpos = float(sim.data.qpos[sim.model.jnt_qposadr[left_joint_id]])
        assert mid[6] < initial[6]
        assert final[6] < mid[6]
        assert -0.045 <= left_qpos <= 0.001
    finally:
        sim.disconnect()


def test_cjjarm_sim_gripper_can_reopen_after_closing() -> None:
    sim = CjjArmSim(
        CjjArmSimConfig(
            id="test_gripper_reopen",
            use_viewer=False,
            render_width=160,
            render_height=120,
            scene_settle_steps=0,
        )
    )
    sim.connect()
    try:
        for _ in range(12):
            sim.send_action({"gripper.pos": -0.04})
        closed = float(sim.data.qpos[sim._gripper_qpos_addr[0]])
        for _ in range(12):
            sim.send_action({"gripper.pos": 0.001})
        reopened = float(sim.data.qpos[sim._gripper_qpos_addr[0]])

        assert closed < -0.02
        assert reopened > -0.005
    finally:
        sim.disconnect()
