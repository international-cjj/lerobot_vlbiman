from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image

from lerobot_robot_cjjarm import CjjArmSim, CjjArmSimConfig


def _make_sim(test_id: str) -> CjjArmSim:
    return CjjArmSim(
        CjjArmSimConfig(
            id=test_id,
            use_viewer=False,
            render_width=160,
            render_height=120,
            scene_settle_steps=0,
        )
    )


def test_default_scene_actuator_names_and_counts() -> None:
    sim = _make_sim("test_actuator_names")
    assert sim.model.nu == 8
    assert sim.get_actuator_names() == [
        "act_joint1",
        "act_joint2",
        "act_joint3",
        "act_joint4",
        "act_joint5",
        "act_joint6",
        "act_gripper_left",
        "act_gripper_right",
    ]


def test_ctrl_targets_move_joint_positions_toward_goal() -> None:
    sim = _make_sim("test_ctrl_motion")
    sim.connect()
    try:
        initial = sim.get_proprio_state()[:8]
        target = np.asarray([0.2, -0.7, 1.0, -0.25, 0.15, 0.1, -0.02, -0.02], dtype=float)
        low_high = sim.get_control_limits()
        target = np.clip(target, low_high[:, 0], low_high[:, 1])
        sim.set_control_targets(target, substeps=200)
        final = sim.get_proprio_state()[:8]

        assert np.linalg.norm(final - target) < np.linalg.norm(initial - target)
    finally:
        sim.disconnect()


def test_joint1_actuator_is_not_pinned_by_home_pose_self_collision() -> None:
    sim = _make_sim("test_joint1_free_motion")
    sim.connect()
    try:
        initial_contacts = {
            tuple(sorted((sim.model.geom(int(contact.geom1)).name or "", sim.model.geom(int(contact.geom2)).name or "")))
            for contact in (sim.data.contact[i] for i in range(int(sim.data.ncon)))
        }
        assert ("base_link_visual", "link1-2_visual") not in initial_contacts

        control = sim.get_control_targets()
        control[0] = 0.5
        sim.set_control_targets(control, substeps=300)
        final = sim.get_proprio_state()[:8]

        assert final[0] > 0.15
    finally:
        sim.disconnect()


def test_reset_holds_home_pose_for_two_seconds() -> None:
    sim = _make_sim("test_home_stability")
    sim.connect()
    try:
        initial = sim.get_proprio_state()[:8]
        sim.set_control_targets(sim.get_control_targets(), substeps=1000)
        final = sim.get_proprio_state()[:8]
        qvel = sim.get_proprio_state()[8:]

        assert abs(final[1] - initial[1]) < 0.12
        assert np.max(np.abs(qvel)) < 3.0
    finally:
        sim.disconnect()


def test_gripper_open_close_cycle_tracks_targets_without_sticking() -> None:
    sim = _make_sim("test_gripper_cycle")
    sim.connect()
    try:
        for _ in range(15):
            sim.send_action({"gripper.pos": -0.04})
        closed = float(sim.data.qpos[sim._gripper_qpos_addr[0]])

        for _ in range(20):
            sim.send_action({"gripper.pos": 0.001})
        reopened = float(sim.data.qpos[sim._gripper_qpos_addr[0]])

        finger_contacts = {
            tuple(sorted((sim.model.geom(int(contact.geom1)).name or "", sim.model.geom(int(contact.geom2)).name or "")))
            for contact in (sim.data.contact[i] for i in range(int(sim.data.ncon)))
        }

        assert closed < -0.02
        assert reopened > -0.005
        assert ("finger_left_inner_pad", "finger_right_inner_pad") not in finger_contacts
    finally:
        sim.disconnect()


def test_render_outputs_front_and_wrist_images() -> None:
    sim = _make_sim("test_render_outputs")
    sim.connect()
    try:
        images = sim.render_cameras()
        output_dir = Path("outputs/debug_camera")
        output_dir.mkdir(parents=True, exist_ok=True)
        front_path = output_dir / "front_camera.png"
        wrist_path = output_dir / "wrist_camera.png"
        Image.fromarray(images["observation.images.front"]).save(front_path)
        Image.fromarray(images["observation.images.wrist"]).save(wrist_path)

        assert images["observation.images.front"].shape == (120, 160, 3)
        assert images["observation.images.wrist"].shape == (120, 160, 3)
        assert front_path.is_file()
        assert wrist_path.is_file()
    finally:
        sim.disconnect()
