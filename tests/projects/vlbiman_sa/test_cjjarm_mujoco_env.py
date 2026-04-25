from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from lerobot.projects.vlbiman_sa.sim.cjjarm_mujoco_env import CjjArmMujocoEnv
from lerobot_robot_cjjarm import CjjArmSimConfig


def test_cjjarm_mujoco_env_smoke() -> None:
    env = CjjArmMujocoEnv(
        CjjArmSimConfig(
            id="test_env_smoke",
            use_viewer=False,
            render_width=160,
            render_height=120,
            scene_settle_steps=0,
        ),
        max_episode_steps=120,
    )
    try:
        obs, info = env.reset()
        assert obs["observation.state"].shape == (16,)
        assert obs["observation.images.front"].shape == (120, 160, 3)
        assert obs["observation.images.wrist"].shape == (120, 160, 3)
        assert len(info["action_names"]) == 8

        zero_action = np.zeros(8, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(zero_action)
        assert np.isfinite(reward)
        assert not np.isnan(obs["observation.state"]).any()
        assert not terminated
        assert not truncated

        rng = np.random.default_rng(0)
        for _ in range(100):
            action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(reward)
            assert not np.isnan(obs["observation.state"]).any()
            if terminated or truncated:
                break
    finally:
        env.close()
