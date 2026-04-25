from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

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


def _contact_pairs(sim: CjjArmSim) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for contact_index in range(int(sim.data.ncon)):
        contact = sim.data.contact[contact_index]
        geom1 = sim.model.geom(int(contact.geom1)).name or f"geom_{int(contact.geom1)}"
        geom2 = sim.model.geom(int(contact.geom2)).name or f"geom_{int(contact.geom2)}"
        pairs.append((geom1, geom2))
    return pairs


def test_gripper_pad_contacts_target_cube() -> None:
    sim = _make_sim("test_pad_contact")
    sim.connect()
    try:
        sim.set_target_cube_position(np.asarray([0.200, 0.0, 0.160], dtype=float), settle_steps=0)
        for _ in range(8):
            sim.send_action({"gripper.pos": -0.025})
        pairs = _contact_pairs(sim)
        contact_names = {name for pair in pairs for name in pair}

        assert "target_cube_geom" in contact_names
        assert (
            "finger_left_inner_pad" in contact_names
            or "finger_right_inner_pad" in contact_names
        )
    finally:
        sim.disconnect()


def test_cube_lifts_when_gripper_closes_and_arm_moves_up() -> None:
    sim = _make_sim("test_cube_lift")
    sim.connect()
    try:
        sim.set_target_cube_position(np.asarray([0.200, 0.0, 0.160], dtype=float), settle_steps=0)
        for _ in range(10):
            sim.send_action({"gripper.pos": -0.03})
        cube_before = float(sim.data.xpos[int(sim.model.body("target_cube").id)][2])

        for _ in range(5):
            sim.send_action({"delta_z": 0.01, "gripper.pos": -0.03})

        cube_after = float(sim.data.xpos[int(sim.model.body("target_cube").id)][2])
        assert cube_after > cube_before + 0.01
    finally:
        sim.disconnect()
