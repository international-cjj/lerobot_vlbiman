from __future__ import annotations

import math

import numpy as np

from lerobot.projects.vlbiman_sa.trajectory.ik_solution_selector import (
    IkSolutionSelector,
    IkSolutionSelectorConfig,
    densify_joint_transition,
)


def test_select_nearest_prefers_wrapped_solution_close_to_seed():
    selector = IkSolutionSelector()
    seed = np.array([math.pi - 0.05, 0.0], dtype=float)
    candidate = np.array([-math.pi + 0.04, 0.0], dtype=float)
    lower = np.array([-2.0 * math.pi, -math.pi], dtype=float)
    upper = np.array([2.0 * math.pi, math.pi], dtype=float)

    selected = selector.select_nearest(candidate, seed, lower_bounds=lower, upper_bounds=upper)

    assert np.allclose(selected, np.array([math.pi + 0.04, 0.0], dtype=float))


def test_densify_joint_transition_limits_max_step():
    dense = densify_joint_transition(
        np.array([0.0, 0.0], dtype=float),
        np.array([0.31, -0.29], dtype=float),
        max_step_rad=0.1,
    )

    assert len(dense) == 4
    prev = np.array([0.0, 0.0], dtype=float)
    for point in dense:
        assert float(np.max(np.abs(point - prev))) <= 0.1 + 1e-9
        prev = point


def test_compute_metrics_counts_preferred_and_hard_violations():
    selector = IkSolutionSelector(IkSolutionSelectorConfig(preferred_joint_step_rad=0.15, max_joint_step_rad=0.25))
    joint_path = [
        np.array([0.0, 0.0], dtype=float),
        np.array([0.1, 0.0], dtype=float),
        np.array([0.28, 0.0], dtype=float),
    ]

    metrics = selector.compute_metrics(joint_path)

    assert metrics.point_count == 3
    assert metrics.preferred_step_violations == 1
    assert metrics.abrupt_step_count == 0
    assert abs(metrics.max_joint_step_rad_inf - 0.18) < 1e-9
