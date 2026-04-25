from __future__ import annotations

import pytest

from lerobot.projects.vlbiman_sa.grasp.contracts import GraspAction, Pose6D, StepReport


def test_pose6d_rejects_wrong_vector_length():
    with pytest.raises(ValueError):
        Pose6D(xyz=(0.0, 0.0), rpy=(0.0, 0.0, 0.0))


def test_grasp_action_defaults_are_zero_and_serializable():
    action = GraspAction()

    assert action.delta_pose_object == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert action.delta_gripper == 0.0
    assert action.stop is False
    assert action.reason is None


def test_step_report_to_dict_keeps_required_flags():
    report = StepReport(step="step_01", status="passed", next_step_allowed=True)

    payload = report.to_dict()

    assert payload["hardware_called"] is False
    assert payload["camera_opened"] is False
    assert payload["mujoco_available"] is False
    assert payload["next_step_allowed"] is True
