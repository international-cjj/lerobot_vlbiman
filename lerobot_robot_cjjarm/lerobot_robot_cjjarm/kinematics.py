from __future__ import annotations

from pathlib import Path

import numpy as np
from lerobot.utils.rotation import Rotation


def _import_pinocchio():
    try:
        import pinocchio as pin
    except ImportError as exc:  # pragma: no cover - exercised when pinocchio is missing
        raise ImportError(
            "pinocchio is required for CjjArmKinematics. "
            "Install it with: pip install pinocchio"
        ) from exc
    return pin


def compose_pose_delta(
    base_pose: np.ndarray | list[float],
    delta_pose: np.ndarray | list[float],
    *,
    rotation_frame: str = "world",
) -> np.ndarray:
    """Apply a Cartesian pose delta to a pose represented as [x, y, z, rx, ry, rz].

    The translational part is applied additively. The rotational part is interpreted as a
    small rotation-vector increment and composed on SO(3), instead of being added component-wise.

    Args:
        base_pose: The starting pose as [x, y, z, rx, ry, rz], where rotation is a rotvec.
        delta_pose: The pose delta as [dx, dy, dz, drx, dry, drz].
        rotation_frame: ``"world"`` for fixed/world-frame increments, or ``"tool"`` for body-frame increments.

    Returns:
        The composed pose in the same representation as ``base_pose``.
    """
    base = np.asarray(base_pose, dtype=float)
    delta = np.asarray(delta_pose, dtype=float)
    if base.shape != (6,) or delta.shape != (6,):
        raise ValueError("base_pose and delta_pose must both be 1D arrays with 6 elements.")

    composed = base.copy()
    composed[:3] += delta[:3]

    if np.linalg.norm(delta[3:]) < 1e-12:
        return composed

    base_rot = Rotation.from_rotvec(base[3:])
    delta_rot = Rotation.from_rotvec(delta[3:])

    if rotation_frame == "world":
        composed_rot = delta_rot * base_rot
    elif rotation_frame == "tool":
        composed_rot = base_rot * delta_rot
    else:
        raise ValueError(f"Unsupported rotation_frame '{rotation_frame}'. Expected 'world' or 'tool'.")

    composed[3:] = composed_rot.as_rotvec()
    return composed


class CjjArmKinematics:
    """Pinocchio-based FK/IK helper for the CJJ arm."""

    def __init__(
        self,
        urdf_path: str | Path,
        end_effector_frame: str = "tool0",
        joint_names: list[str] | None = None,
        *,
        max_iters: int = 100,
        tol: float = 1e-4,
        damping: float = 1e-6,
        step_scale: float = 1.0,
    ) -> None:
        pin = _import_pinocchio()
        self._pin = pin
        self.urdf_path = str(urdf_path)
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()

        if joint_names is None:
            joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.joint_names = list(joint_names)

        frame_names = {frame.name for frame in self.model.frames}
        if end_effector_frame not in frame_names:
            raise ValueError(
                f"End-effector frame '{end_effector_frame}' not found in URDF. "
                f"Available frames: {sorted(frame_names)}"
            )
        self.frame_id = self.model.getFrameId(end_effector_frame)

        self.joint_ids = []
        self.joint_q_indices = []
        for name in self.joint_names:
            joint_id = self.model.getJointId(name)
            if joint_id == 0:
                raise ValueError(f"Joint '{name}' not found in URDF.")
            joint = self.model.joints[joint_id]
            if joint.nq != 1:
                raise ValueError(f"Joint '{name}' has nq={joint.nq}; only 1-DoF joints are supported.")
            self.joint_ids.append(joint_id)
            self.joint_q_indices.append(joint.idx_q)

        self._q_lower = self.model.lowerPositionLimit.copy()
        self._q_upper = self.model.upperPositionLimit.copy()

        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.damping = float(damping)
        self.step_scale = float(step_scale)

    def _pack_q(self, joint_angles: np.ndarray) -> np.ndarray:
        pin = self._pin
        q = pin.neutral(self.model).copy()
        for idx, angle in zip(self.joint_q_indices, joint_angles, strict=True):
            q[idx] = angle
        return q

    def _unpack_q(self, q: np.ndarray) -> np.ndarray:
        return np.array([q[idx] for idx in self.joint_q_indices], dtype=float)

    def _clamp_q(self, q: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(q, self._q_lower), self._q_upper)

    def compute_fk(self, joint_angles: np.ndarray | list[float]) -> np.ndarray:
        """Compute forward kinematics and return pose as [x, y, z, rx, ry, rz]."""
        pin = self._pin
        angles = np.asarray(joint_angles, dtype=float)
        if angles.ndim != 1:
            raise ValueError("joint_angles must be a 1D array.")

        if angles.shape[0] == self.model.nq:
            q = angles
        elif angles.shape[0] == len(self.joint_names):
            q = self._pack_q(angles)
        else:
            raise ValueError(
                f"Expected {len(self.joint_names)} joints (or full nq={self.model.nq}), got {angles.shape[0]}."
            )

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        oMf = self.data.oMf[self.frame_id]

        pos = oMf.translation
        rotvec = pin.log3(oMf.rotation)
        return np.concatenate([pos, rotvec]).astype(float)

    def compute_ik(
        self,
        target_pose: np.ndarray | list[float],
        seed_joints: np.ndarray | list[float],
        *,
        position_weight: float = 1.0,
        orientation_weight: float = 0.05,
        keep_pointing_only: bool = False,
    ) -> np.ndarray:
        """Compute IK using damped least squares; returns joint angles in the configured order."""
        pin = self._pin
        target = np.asarray(target_pose, dtype=float)
        if target.ndim != 1 or target.shape[0] != 6:
            raise ValueError("target_pose must be a 1D array with 6 elements [x, y, z, rx, ry, rz].")

        seed = np.asarray(seed_joints, dtype=float)
        if seed.ndim != 1:
            raise ValueError("seed_joints must be a 1D array.")

        if seed.shape[0] == self.model.nq:
            q = seed
        elif seed.shape[0] == len(self.joint_names):
            q = self._pack_q(seed)
        else:
            raise ValueError(
                f"Expected seed length {len(self.joint_names)} (or full nq={self.model.nq}), got {seed.shape[0]}."
            )

        q = self._clamp_q(q)
        oMdes = pin.SE3(pin.exp3(target[3:6]), target[0:3])

        for _ in range(self.max_iters):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            oMf = self.data.oMf[self.frame_id]
            pos_err = oMdes.translation - oMf.translation
            # Use world-frame orientation error to match LOCAL_WORLD_ALIGNED Jacobian.
            rot_err = pin.log3(oMdes.rotation @ oMf.rotation.T)
            proj = None
            if keep_pointing_only:
                # Lock pointing direction while allowing twist around tool Z axis.
                tool_axis = oMf.rotation[:, 2]
                proj = np.eye(3) - np.outer(tool_axis, tool_axis)
                rot_err = proj @ rot_err
            err = np.concatenate(
                [pos_err * position_weight, rot_err * orientation_weight]
            )
            if np.linalg.norm(err) < self.tol:
                break

            J = pin.computeFrameJacobian(
                self.model, self.data, q, self.frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            J[:3, :] *= position_weight
            if proj is not None:
                J[3:, :] = proj @ J[3:, :]
            J[3:, :] *= orientation_weight
            jj_t = J @ J.T
            dq = J.T @ np.linalg.solve(jj_t + self.damping * np.eye(6), err)
            q = pin.integrate(self.model, q, dq * self.step_scale)
            q = self._clamp_q(q)

        return self._unpack_q(q)
