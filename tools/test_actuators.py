from __future__ import annotations

import sys
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from lerobot_robot_cjjarm import CjjArmSim, CjjArmSimConfig


def main() -> int:
    sim = CjjArmSim(
        CjjArmSimConfig(
            id="test_actuators",
            use_viewer=False,
            scene_settle_steps=0,
            render_width=160,
            render_height=120,
        )
    )
    print(f"load_chain={sim._load_chain_description}")
    print(f"generated_mjcf={sim.generated_scene_path}")
    print(f"nu={sim.model.nu}")
    print(f"actuator_names={sim.get_actuator_names()}")
    if sim.model.nu != 8:
        print("ERROR: expected model.nu == 8")
        return 1

    sim.connect()
    try:
        initial_qpos = sim.get_proprio_state()[:8]
        limits = sim.get_control_limits()
        target = np.asarray(
            [
                0.25,
                -0.8,
                1.1,
                -0.35,
                0.2,
                0.1,
                -0.015,
                -0.015,
            ],
            dtype=float,
        )
        target = np.clip(target, limits[:, 0], limits[:, 1])
        sim.set_control_targets(target, substeps=180)
        final_qpos = sim.get_proprio_state()[:8]
        print(f"initial_qpos={initial_qpos.tolist()}")
        print(f"target_ctrl={target.tolist()}")
        print(f"final_qpos={final_qpos.tolist()}")

        if np.linalg.norm(final_qpos - target) >= np.linalg.norm(initial_qpos - target):
            print("ERROR: qpos did not move closer to control target")
            return 1

        print("PASS")
        return 0
    finally:
        sim.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
