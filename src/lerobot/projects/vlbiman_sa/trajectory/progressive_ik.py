from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from ikpy.chain import Chain

from lerobot.projects.vlbiman_sa.geometry.transforms import (
    compose_transform,
    invert_transform,
    make_transform,
    rotation_error_deg,
    translation_error_m,
)
from lerobot.utils.rotation import Rotation
from lerobot_robot_cjjarm.config_cjjarm import CjjArmConfig

from .ik_solution_selector import IkSolutionSelector


@dataclass(slots=True)
class IKPyState:
    chain: Chain
    arm_joint_names: list[str]
    link_indices: list[int]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    end_effector_frame: str
    tip_from_tool: np.ndarray
    tool_from_tip: np.ndarray


@dataclass(slots=True)
class ProgressiveIKConfig:
    translation_step_m: float = 0.015
    rotation_step_deg: float = 8.0
    regularization_parameter: float = 1e-4
    optimizer: str = "least_squares"
    orientation_mode: str = "all"
    translation_tolerance_mm: float = 25.0
    rotation_tolerance_deg: float = 5.0
    min_ee_z_m: float = 0.0


@dataclass(slots=True)
class IKStep:
    index: int
    phase: str
    joint_positions: list[float]
    target_pose_matrix: list[list[float]]
    solved_pose_matrix: list[list[float]]
    translation_error_mm: float
    rotation_error_deg: float
    max_joint_step_rad: float
    success: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_ikpy_state(robot_cfg: CjjArmConfig | None = None) -> IKPyState:
    robot_cfg = robot_cfg or CjjArmConfig()
    arm_joint_names = list(robot_cfg.urdf_joint_map.keys())
    urdf_joint_names = [robot_cfg.urdf_joint_map[name] for name in arm_joint_names]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        probe_chain = Chain.from_urdf_file(str(robot_cfg.urdf_path))
    index_by_name = {link.name: idx for idx, link in enumerate(probe_chain.links)}
    link_indices = [index_by_name[name] for name in urdf_joint_names]
    active_mask = [False] * len(probe_chain.links)
    for idx in link_indices:
        active_mask[idx] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        chain = Chain.from_urdf_file(str(robot_cfg.urdf_path), active_links_mask=active_mask)

    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    for link in chain.links:
        low, high = link.bounds
        lower_bounds.append(float(-np.inf if low is None else low))
        upper_bounds.append(float(np.inf if high is None else high))
    tip_from_tool = _load_tip_from_tool_transform(Path(str(robot_cfg.urdf_path)), str(robot_cfg.end_effector_frame))
    return IKPyState(
        chain=chain,
        arm_joint_names=arm_joint_names,
        link_indices=link_indices,
        lower_bounds=np.asarray(lower_bounds, dtype=float),
        upper_bounds=np.asarray(upper_bounds, dtype=float),
        end_effector_frame=str(robot_cfg.end_effector_frame),
        tip_from_tool=tip_from_tool,
        tool_from_tip=invert_transform(tip_from_tool),
    )


class ProgressiveIKPlanner:
    def __init__(
        self,
        state: IKPyState,
        config: ProgressiveIKConfig | None = None,
        selector: IkSolutionSelector | None = None,
    ):
        self.state = state
        self.config = config or ProgressiveIKConfig()
        self.selector = selector or IkSolutionSelector()

    def solve_pose(
        self,
        *,
        target_pose: np.ndarray,
        seed_q: np.ndarray,
        phase: str,
    ) -> tuple[np.ndarray, list[IKStep]]:
        seed_q = np.asarray(seed_q, dtype=float)
        current_q = seed_q.copy()
        current_full_q = self._clip_full_q(self._arm_to_full_q(current_q))
        current_q = self._full_to_arm_q(current_full_q)
        current_pose = forward_kinematics_tool(self.state, current_full_q)
        target_pose = self._enforce_ee_floor(np.asarray(target_pose, dtype=float))
        subtargets = self._interpolate_targets(current_pose, target_pose)
        steps: list[IKStep] = []
        for index, subtarget in enumerate(subtargets):
            tip_target = tool_pose_to_tip_pose(self.state, subtarget)
            raw_solution = np.asarray(
                self.state.chain.inverse_kinematics_frame(
                    tip_target,
                    initial_position=current_full_q,
                    orientation_mode=self.config.orientation_mode,
                    regularization_parameter=float(self.config.regularization_parameter),
                    optimizer=self.config.optimizer,
                ),
                dtype=float,
            )
            solution = self.selector.select_nearest(
                raw_solution,
                current_full_q,
                lower_bounds=self.state.lower_bounds,
                upper_bounds=self.state.upper_bounds,
            )
            solved_pose = forward_kinematics_tool(self.state, solution)
            if self._pose_z(solved_pose) < float(self.config.min_ee_z_m):
                solution = current_full_q.copy()
                solved_pose = current_pose.copy()
            translation_error_mm = translation_error_m(subtarget, solved_pose) * 1000.0
            rotation_err = rotation_error_deg(subtarget, solved_pose)
            arm_solution = self._full_to_arm_q(solution)
            max_joint_step = float(np.max(np.abs(arm_solution - current_q)))
            success = (
                translation_error_mm <= float(self.config.translation_tolerance_mm)
                and rotation_err <= float(self.config.rotation_tolerance_deg)
            )
            steps.append(
                IKStep(
                    index=index,
                    phase=phase,
                    joint_positions=arm_solution.astype(float).tolist(),
                    target_pose_matrix=subtarget.astype(float).tolist(),
                    solved_pose_matrix=solved_pose.astype(float).tolist(),
                    translation_error_mm=float(translation_error_mm),
                    rotation_error_deg=float(rotation_err),
                    max_joint_step_rad=max_joint_step,
                    success=bool(success),
                )
            )
            current_q = arm_solution
            current_full_q = self._clip_full_q(solution)
            current_pose = solved_pose
        return current_q, steps

    def _interpolate_targets(self, start_pose: np.ndarray, target_pose: np.ndarray) -> list[np.ndarray]:
        translation_distance = float(np.linalg.norm(target_pose[:3, 3] - start_pose[:3, 3]))
        delta_rotation = target_pose[:3, :3] @ start_pose[:3, :3].T
        rotation_distance_deg = float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(delta_rotation).as_rotvec())))
        step_count = max(
            1,
            int(
                np.ceil(
                    max(
                        translation_distance / max(float(self.config.translation_step_m), 1e-9),
                        rotation_distance_deg / max(float(self.config.rotation_step_deg), 1e-9),
                    )
                )
            ),
        )

        start_rot = Rotation.from_matrix(start_pose[:3, :3])
        delta_rot = Rotation.from_matrix(delta_rotation)
        targets: list[np.ndarray] = []
        for step_index in range(1, step_count + 1):
            alpha = step_index / step_count
            transform = np.eye(4, dtype=float)
            transform[:3, 3] = start_pose[:3, 3] + (target_pose[:3, 3] - start_pose[:3, 3]) * alpha
            interp_rot = Rotation.from_rotvec(delta_rot.as_rotvec() * alpha) * start_rot
            transform[:3, :3] = interp_rot.as_matrix()
            targets.append(self._enforce_ee_floor(transform))
        return targets

    def _arm_to_full_q(self, arm_q: np.ndarray) -> np.ndarray:
        full_q = np.zeros(len(self.state.chain.links), dtype=float)
        for value, idx in zip(np.asarray(arm_q, dtype=float), self.state.link_indices, strict=True):
            full_q[idx] = float(value)
        return full_q

    def _full_to_arm_q(self, full_q: np.ndarray) -> np.ndarray:
        return np.asarray([float(full_q[idx]) for idx in self.state.link_indices], dtype=float)

    def _clip_full_q(self, full_q: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(np.asarray(full_q, dtype=float), self.state.lower_bounds), self.state.upper_bounds)

    def _enforce_ee_floor(self, pose: np.ndarray) -> np.ndarray:
        bounded = np.asarray(pose, dtype=float).copy()
        bounded[2, 3] = max(float(bounded[2, 3]), float(self.config.min_ee_z_m))
        return bounded

    @staticmethod
    def _pose_z(pose: np.ndarray) -> float:
        return float(np.asarray(pose, dtype=float)[2, 3])


def full_q_from_arm_q(state: IKPyState, arm_q: np.ndarray) -> np.ndarray:
    full_q = np.zeros(len(state.chain.links), dtype=float)
    for value, idx in zip(np.asarray(arm_q, dtype=float), state.link_indices, strict=True):
        full_q[idx] = float(value)
    return full_q


def forward_kinematics_tip(state: IKPyState, q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape[0] == len(state.link_indices):
        q = full_q_from_arm_q(state, q)
    return np.asarray(state.chain.forward_kinematics(q), dtype=float)


def forward_kinematics_tool(state: IKPyState, q: np.ndarray) -> np.ndarray:
    return compose_transform(forward_kinematics_tip(state, q), state.tip_from_tool)


def tool_pose_to_tip_pose(state: IKPyState, tool_pose: np.ndarray) -> np.ndarray:
    return compose_transform(np.asarray(tool_pose, dtype=float), state.tool_from_tip)


def _load_tip_from_tool_transform(urdf_path: Path, end_effector_frame: str) -> np.ndarray:
    if not urdf_path.exists():
        return np.eye(4, dtype=float)

    root = ET.fromstring(urdf_path.read_text(encoding="utf-8"))
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is None or child.attrib.get("link") != end_effector_frame:
            continue
        joint_type = joint.attrib.get("type", "")
        if joint_type != "fixed":
            continue
        origin = joint.find("origin")
        xyz = np.zeros(3, dtype=float)
        rpy = np.zeros(3, dtype=float)
        if origin is not None:
            if origin.attrib.get("xyz"):
                xyz = np.asarray([float(value) for value in origin.attrib["xyz"].split()], dtype=float)
            if origin.attrib.get("rpy"):
                rpy = np.asarray([float(value) for value in origin.attrib["rpy"].split()], dtype=float)
        rotation = _rotation_from_rpy(rpy)
        return make_transform(rotation, xyz)
    return np.eye(4, dtype=float)


def _rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(value) for value in np.asarray(rpy, dtype=float).reshape(3)]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )
