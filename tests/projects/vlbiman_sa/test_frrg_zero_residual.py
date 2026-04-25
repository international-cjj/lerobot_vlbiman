from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from lerobot.projects.vlbiman_sa.grasp.closed_loop_controller import FRRGClosedLoopController
from lerobot.projects.vlbiman_sa.grasp.contracts import GraspAction, load_frrg_config
from lerobot.projects.vlbiman_sa.grasp.observer import build_grasp_state
from lerobot.projects.vlbiman_sa.grasp.primitives.nominal_capture import nominal_capture_action
from lerobot.projects.vlbiman_sa.grasp.residual import (
    BCResidualPolicy,
    ZeroResidualPolicy,
    build_residual_policy,
    compute_demo_residual_label,
    residual_clip_limits_from_config,
    save_bc_checkpoint,
)
from lerobot.projects.vlbiman_sa.grasp.safety_limits import apply_safety_limits
from lerobot.projects.vlbiman_sa.grasp.state_machine import PHASE_SUCCESS


FIXTURE_DIR = Path("tests/fixtures/vlbiman_sa")
CONFIG_PATH = Path("src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml")


def _load_fixture(name: str) -> tuple[object, dict[str, object]]:
    payload = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    feature_debug_terms = dict(payload.get("feature_debug_terms", {}))
    state = build_grasp_state(payload).state
    return state, feature_debug_terms


def _load_payload(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_zero_residual_policy_outputs_zero_action_and_zero_norm():
    config = load_frrg_config(CONFIG_PATH)
    state, feature_debug_terms = _load_fixture("frrg_capture_ready.json")
    nominal_action = nominal_capture_action(state, config, feature_debug_terms=feature_debug_terms)

    result = ZeroResidualPolicy().compute(state, nominal_action)

    assert result.action.delta_pose_object == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert result.action.delta_gripper == 0.0
    assert result.norm == 0.0
    assert result.debug_terms["policy"] == "zero"


def test_zero_residual_does_not_change_safety_output_norms():
    config = load_frrg_config(CONFIG_PATH)
    state, feature_debug_terms = _load_fixture("frrg_capture_ready.json")
    nominal_action = nominal_capture_action(state, config, feature_debug_terms=feature_debug_terms)
    residual = ZeroResidualPolicy().compute(state, nominal_action)

    result = apply_safety_limits(state, config, nominal_action, residual_action=residual.action)

    assert result.residual_norm == 0.0
    assert result.raw_action_norm > 0.0
    assert result.safe_action_norm >= 0.0
    assert result.to_dict()["residual_action"]["delta_pose_object"] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_compute_demo_residual_label_clips_against_safety_limits():
    config = load_frrg_config(CONFIG_PATH)
    clip_pose_limits, clip_gripper_limit = residual_clip_limits_from_config(config)
    nominal_action = GraspAction()
    demo_action = GraspAction(
        delta_pose_object=(0.01, -0.01, 0.02, 0.2, -0.2, 0.3),
        delta_gripper=0.01,
    )

    label = compute_demo_residual_label(
        demo_action,
        nominal_action,
        clip_pose_limits=clip_pose_limits,
        clip_gripper_limit=clip_gripper_limit,
    )

    assert label.clip_applied is True
    assert label.action.delta_pose_object == (
        config.safety.max_step_xyz_m[0],
        -config.safety.max_step_xyz_m[1],
        config.safety.max_step_xyz_m[2],
        0.0,
        -0.0,
        config.safety.max_step_rpy_rad[2],
    )
    assert label.action.delta_gripper == config.safety.max_gripper_delta_m


def test_build_residual_policy_skips_loader_when_disabled():
    config = load_frrg_config(CONFIG_PATH)
    disabled_config = replace(
        config,
        residual=replace(
            config.residual,
            checkpoint_path=Path("/tmp/this_checkpoint_should_not_be_loaded.json"),
        ),
    )
    loader_called = False

    def _loader(path: Path, config_obj: object) -> object:
        nonlocal loader_called
        loader_called = True
        raise AssertionError(f"loader should not be called when residual is disabled: {path} {config_obj}")

    policy = build_residual_policy(disabled_config, checkpoint_loader=_loader)
    state, feature_debug_terms = _load_fixture("frrg_capture_ready.json")
    nominal_action = nominal_capture_action(state, config, feature_debug_terms=feature_debug_terms)
    result = policy.compute(state, nominal_action)

    assert isinstance(policy, ZeroResidualPolicy)
    assert loader_called is False
    assert result.debug_terms["requested_policy"] == "zero"
    assert result.debug_terms["residual_enabled"] is False


def test_build_residual_policy_falls_back_to_zero_on_loader_error():
    config = load_frrg_config(CONFIG_PATH)
    state, feature_debug_terms = _load_fixture("frrg_capture_ready.json")
    nominal_action = nominal_capture_action(state, config, feature_debug_terms=feature_debug_terms)
    bc_config = replace(
        config,
        residual=replace(
            config.residual,
            enabled=True,
            policy="bc",
            checkpoint_path=Path("broken_bc_checkpoint.json"),
        ),
    )

    def _loader(path: Path, config_obj: object) -> object:
        raise RuntimeError(f"cannot load checkpoint: {path} {config_obj}")

    policy = build_residual_policy(bc_config, checkpoint_loader=_loader)
    result = policy.compute(state, nominal_action)

    assert isinstance(policy, ZeroResidualPolicy)
    assert result.action.delta_pose_object == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert result.action.delta_gripper == 0.0
    assert result.debug_terms["fallback_reason"] == "checkpoint_load_failed"
    assert "RuntimeError" in result.debug_terms["fallback_detail"]


def test_bc_residual_policy_output_still_goes_through_safety(tmp_path: Path):
    config = load_frrg_config(CONFIG_PATH)
    clip_pose_limits, clip_gripper_limit = residual_clip_limits_from_config(config)
    checkpoint_path = save_bc_checkpoint(
        tmp_path / "bc_checkpoint.json",
        weights=np.zeros((7, 18), dtype=float),
        bias=np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.3, 0.01], dtype=float),
        clip_pose_limits=clip_pose_limits,
        clip_gripper_limit=clip_gripper_limit,
    )
    bc_config = replace(
        config,
        residual=replace(
            config.residual,
            enabled=True,
            policy="bc",
            checkpoint_path=checkpoint_path,
        ),
    )
    policy = build_residual_policy(bc_config)
    state, _ = _load_fixture("frrg_capture_ready.json")
    nominal_action = GraspAction(
        delta_pose_object=(0.001, 0.0, 0.0, 0.0, 0.0, 0.02),
        delta_gripper=0.001,
    )
    residual = policy.compute(state, nominal_action)
    safety_result = apply_safety_limits(state, bc_config, nominal_action, residual_action=residual.action)

    assert isinstance(policy, BCResidualPolicy)
    assert residual.norm > 0.0
    assert safety_result.residual_norm > 0.0
    assert safety_result.limited is True
    assert safety_result.safe_action.delta_pose_object[0] == bc_config.safety.max_step_xyz_m[0]
    assert safety_result.safe_action.delta_pose_object[5] == bc_config.safety.max_step_rpy_rad[2]
    assert safety_result.safe_action.delta_gripper == bc_config.safety.max_gripper_delta_m


def test_controller_default_residual_factory_keeps_nominal_success():
    config = load_frrg_config(CONFIG_PATH)
    payload = _load_payload("frrg_nominal_success.json")

    result = FRRGClosedLoopController(config, payload).run(max_steps=10)

    assert result.status == "success"
    assert result.final_phase == PHASE_SUCCESS
    assert result.max_residual_norm == 0.0
