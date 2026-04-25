from __future__ import annotations

from pathlib import Path

import pytest

from lerobot.projects.vlbiman_sa.grasp.contracts import ConfigError, load_frrg_config


def test_load_frrg_config_reads_default_yaml():
    config_path = Path("src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml")

    config = load_frrg_config(config_path)

    assert config.runtime.control_hz == 20
    assert config.runtime.default_input_mode == "mock"
    assert config.safety.max_step_xyz_m == (0.003, 0.003, 0.003)
    assert config.residual.enabled is False
    assert config.residual.checkpoint_path is None
    assert config.source_path == config_path.resolve()


def test_load_frrg_config_rejects_missing_required_section(tmp_path: Path):
    broken = tmp_path / "broken.yaml"
    broken.write_text(
        """
runtime:
  control_hz: 20
  max_steps: 100
  stable_window_frames: 5
  max_retry_count: 1
  default_input_mode: mock
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Missing required section: feature_geometry"):
        load_frrg_config(broken)


def test_load_frrg_config_rejects_invalid_safety_value(tmp_path: Path):
    broken = tmp_path / "broken.yaml"
    broken.write_text(
        """
runtime:
  control_hz: 20
  max_steps: 100
  stable_window_frames: 5
  max_retry_count: 1
  default_input_mode: mock
feature_geometry:
  lateral_unit: m
  occ_definition: object_ratio
  angle_symmetry_weight_default: 1.0
handoff:
  handoff_pos_tol_m: 0.015
  handoff_yaw_tol_rad: 0.18
  handoff_vis_min: 0.45
  handoff_open_width_m: 0.06
capture_build:
  mode: enveloping
  solve_score_threshold: 0.55
  close_score_threshold: 0.72
  forward_enable_lat_tol_m: 0.004
  forward_enable_ang_tol_rad: 0.12
  forward_enable_occ_min: 0.55
  target_depth_goal_m: 0.012
  target_depth_max_m: 0.028
  lat_gain: 0.8
  sym_gain: 0.4
  dep_gain: 0.5
  vert_gain: 0.0
  yaw_gain: 0.6
  capture_timeout_s: 8.0
close_hold:
  preclose_distance_m: 0.006
  preclose_pause_s: 0.15
  close_speed_raw_per_s: 0.25
  settle_time_s: 0.25
  close_width_target_m: 0.0
  contact_current_min: 0.12
  contact_current_max: 0.85
  hold_drift_max_m: 0.003
  hold_score_threshold: 0.70
lift_test:
  lift_height_m: 0.015
  lift_speed_mps: 0.01
  lift_hold_s: 0.25
  slip_threshold_m: 0.004
  lift_score_threshold: 0.74
safety:
  max_step_xyz_m: [0.003, -0.003, 0.003]
  max_step_rpy_rad: [0.0, 0.0, 0.06]
  max_gripper_delta_m: 0.002
  vision_hardstop_min: 0.15
  obj_jump_stop_m: 0.02
residual:
  enabled: false
  policy: zero
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match=r"safety.max_step_xyz_m\[1\] must be > 0"):
        load_frrg_config(broken)


def test_load_frrg_config_allows_bc_policy_with_checkpoint_path(tmp_path: Path):
    config_path = tmp_path / "bc.yaml"
    config_path.write_text(
        """
runtime:
  control_hz: 20
  max_steps: 100
  stable_window_frames: 5
  max_retry_count: 1
  default_input_mode: mock
feature_geometry:
  lateral_unit: m
  occ_definition: object_ratio
  angle_symmetry_weight_default: 1.0
handoff:
  handoff_pos_tol_m: 0.015
  handoff_yaw_tol_rad: 0.18
  handoff_vis_min: 0.45
  handoff_open_width_m: 0.06
capture_build:
  mode: enveloping
  solve_score_threshold: 0.55
  close_score_threshold: 0.72
  forward_enable_lat_tol_m: 0.004
  forward_enable_ang_tol_rad: 0.12
  forward_enable_occ_min: 0.55
  target_depth_goal_m: 0.012
  target_depth_max_m: 0.028
  lat_gain: 0.8
  sym_gain: 0.4
  dep_gain: 0.5
  vert_gain: 0.0
  yaw_gain: 0.6
  capture_timeout_s: 8.0
close_hold:
  preclose_distance_m: 0.006
  preclose_pause_s: 0.15
  close_speed_raw_per_s: 0.25
  settle_time_s: 0.25
  close_width_target_m: 0.0
  contact_current_min: 0.12
  contact_current_max: 0.85
  hold_drift_max_m: 0.003
  hold_score_threshold: 0.70
lift_test:
  lift_height_m: 0.015
  lift_speed_mps: 0.01
  lift_hold_s: 0.25
  slip_threshold_m: 0.004
  lift_score_threshold: 0.74
safety:
  max_step_xyz_m: [0.003, 0.003, 0.003]
  max_step_rpy_rad: [0.0, 0.0, 0.06]
  max_gripper_delta_m: 0.002
  vision_hardstop_min: 0.15
  obj_jump_stop_m: 0.02
residual:
  enabled: true
  policy: bc
  checkpoint_path: outputs/vlbiman_sa/frrg/demo_checkpoint.json
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_frrg_config(config_path)

    assert config.residual.enabled is True
    assert config.residual.policy == "bc"
    assert config.residual.checkpoint_path == Path("outputs/vlbiman_sa/frrg/demo_checkpoint.json")
