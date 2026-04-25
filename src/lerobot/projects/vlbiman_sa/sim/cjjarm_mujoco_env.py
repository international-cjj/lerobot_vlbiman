from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from lerobot_robot_cjjarm import CjjArmSim, CjjArmSimConfig


class CjjArmMujocoEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        config: CjjArmSimConfig | None = None,
        *,
        max_episode_steps: int = 200,
    ) -> None:
        super().__init__()
        self.sim = CjjArmSim(config or CjjArmSimConfig(use_viewer=False))
        self.max_episode_steps = int(max_episode_steps)
        self._step_count = 0

        control_limits = self.sim.get_control_limits()
        if control_limits.shape != (8, 2):
            raise RuntimeError(
                f"CjjArmMujocoEnv expects exactly 8 actuators; got ctrl limits with shape {control_limits.shape}."
            )

        self.action_names = self.sim.get_actuator_names()
        self.action_space = gym.spaces.Box(
            low=control_limits[:, 0].astype(np.float32),
            high=control_limits[:, 1].astype(np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "observation.state": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(16,),
                    dtype=np.float32,
                ),
                "observation.images.front": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.sim.config.render_height, self.sim.config.render_width, 3),
                    dtype=np.uint8,
                ),
                "observation.images.wrist": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.sim.config.render_height, self.sim.config.render_width, 3),
                    dtype=np.uint8,
                ),
            }
        )

    def _table_top_z(self) -> float:
        geom_id = int(self.sim.model.geom("table_geom").id)
        return float(self.sim.model.geom_pos[geom_id, 2] + self.sim.model.geom_size[geom_id, 2])

    def _cube_position(self) -> np.ndarray:
        body_id = int(self.sim.model.body("target_cube").id)
        return np.asarray(self.sim.data.xpos[body_id], dtype=float).copy()

    def _finger_center(self) -> np.ndarray:
        left_body = int(self.sim.model.body("finger_left").id)
        right_body = int(self.sim.model.body("finger_right").id)
        return 0.5 * (
            np.asarray(self.sim.data.xpos[left_body], dtype=float)
            + np.asarray(self.sim.data.xpos[right_body], dtype=float)
        )

    def _build_observation(self) -> dict[str, Any]:
        images = self.sim.render_cameras()
        return {
            "observation.state": self.sim.get_proprio_state().astype(np.float32),
            "observation.images.front": images["observation.images.front"],
            "observation.images.wrist": images["observation.images.wrist"],
        }

    def _reward_and_success(self) -> tuple[float, bool, dict[str, float]]:
        cube_position = self._cube_position()
        finger_center = self._finger_center()
        distance = float(np.linalg.norm(cube_position - finger_center))
        table_top_z = self._table_top_z()
        cube_height = float(cube_position[2] - table_top_z)
        success = cube_height > 0.04
        reward = -distance + 5.0 * max(cube_height, 0.0) + (2.0 if success else 0.0)
        return reward, success, {
            "cube_height_m": cube_height,
            "cube_distance_m": distance,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        if not self.sim.is_connected:
            self.sim.connect()
        else:
            self.sim._reset_sim()
        self._step_count = 0
        obs = self._build_observation()
        info = {"action_names": list(self.action_names)}
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        control = np.asarray(action, dtype=float).reshape(8)
        self.sim.set_control_targets(control)
        self._step_count += 1

        obs = self._build_observation()
        reward, terminated, info = self._reward_and_success()
        truncated = self._step_count >= self.max_episode_steps
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> np.ndarray:
        return self.sim.render_cameras()["observation.images.front"]

    def close(self) -> None:
        if self.sim.is_connected:
            self.sim.disconnect()
