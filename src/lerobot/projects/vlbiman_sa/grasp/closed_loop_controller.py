from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .contracts import FRRGGraspConfig, GraspAction, GraspState, GuardResult, Pose6D
from .feature_geometry import FeatureGeometryResult, apply_feature_geometry
from .frame_math import compose_transform, matrix_to_pose6d, pose6d_to_matrix
from .observer import ObserverResult, build_grasp_state
from .phase_guards import (
    capture_timeout_guard,
    capture_to_close_guard,
    close_to_lift_guard,
    handoff_guard,
    lift_to_success_guard,
    slip_detected_guard,
    vision_lost_guard,
)
from .primitives import nominal_capture_action, nominal_close_action, nominal_lift_action
from .recovery_policy import RecoveryPolicy, RecoveryPolicyResult
from .residual import ResidualPolicy, ResidualResult, build_residual_policy
from .safety_limits import SafetyLimitResult, apply_safety_limits
from .scores import ScoreResult, apply_scores
from .state_machine import (
    PHASE_CAPTURE_BUILD,
    PHASE_CLOSE_HOLD,
    PHASE_FAILURE,
    PHASE_HANDOFF,
    PHASE_LIFT_TEST,
    PHASE_SUCCESS,
    PhaseTraceEntry,
    next_phase,
)


FEATURE_OVERRIDE_FIELDS = (
    "e_dep",
    "e_lat",
    "e_vert",
    "e_ang",
    "e_sym",
    "occ_corridor",
    "drift_obj",
)


@dataclass
class ControllerStepResult:
    step_index: int
    observation: ObserverResult
    geometry: FeatureGeometryResult
    scored: ScoreResult
    feature_debug_terms: dict[str, Any]
    guard_results: dict[str, GuardResult]
    phase_transition: PhaseTraceEntry
    raw_action: GraspAction
    recovery_result: RecoveryPolicyResult | None
    residual_result: ResidualResult
    safety_result: SafetyLimitResult
    next_payload: dict[str, Any]

    def action_record(self) -> dict[str, Any]:
        safety_dict = self.safety_result.to_dict()
        return {
            "step": self.step_index,
            "phase": self.scored.state.phase,
            "raw_action": safety_dict["raw_action"],
            "recovery_proposal": None
            if self.recovery_result is None
            else {
                "proposal_type": self.recovery_result.proposal_type,
                "reason": self.recovery_result.reason,
                "action": {
                    "delta_pose_object": list(self.recovery_result.action.delta_pose_object),
                    "delta_gripper": float(self.recovery_result.action.delta_gripper),
                    "stop": bool(self.recovery_result.action.stop),
                    "reason": self.recovery_result.action.reason,
                    "debug_terms": dict(self.recovery_result.action.debug_terms),
                },
                "debug_terms": dict(self.recovery_result.debug_terms),
            },
            "residual_action": safety_dict["residual_action"],
            "safe_action": safety_dict["safe_action"],
            "limited": self.safety_result.limited,
            "stop": self.safety_result.stop,
            "reason": self.safety_result.reason,
            "raw_action_norm": self.safety_result.raw_action_norm,
            "residual_norm": self.safety_result.residual_norm,
            "safe_action_norm": self.safety_result.safe_action_norm,
        }

    def guards_record(self) -> dict[str, Any]:
        return {
            "step": self.step_index,
            "phase": self.scored.state.phase,
            "guards": {
                name: {
                    "passed": result.passed,
                    "score": result.score,
                    "reason": result.reason,
                    "debug_terms": dict(result.debug_terms),
                }
                for name, result in self.guard_results.items()
            },
        }

    def phase_trace_record(self) -> dict[str, Any]:
        record = self.phase_transition.to_dict()
        record["step"] = self.step_index
        record["phase"] = self.scored.state.phase
        record["capture_score"] = self.scored.state.capture_score
        record["hold_score"] = self.scored.state.hold_score
        record["lift_score"] = self.scored.state.lift_score
        return record

    def state_record(self) -> dict[str, Any]:
        return {
            "step": self.step_index,
            "phase": self.scored.state.phase,
            "state": asdict(self.scored.state),
            "missing_fields": list(self.observation.missing_fields),
            "feature_debug_terms": dict(self.feature_debug_terms),
            "score_debug_terms": dict(self.scored.debug_terms),
        }


@dataclass
class ControllerRunResult:
    status: str
    final_phase: str
    failure_reason: str | None
    steps_run: int
    phase_trace: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    states: list[dict[str, Any]] = field(default_factory=list)
    guards: list[dict[str, Any]] = field(default_factory=list)
    max_raw_action_norm: float = 0.0
    max_safe_action_norm: float = 0.0
    max_residual_norm: float = 0.0
    all_actions_limited: bool = True
    last_state_summary: dict[str, Any] | None = None
    failure_report: dict[str, Any] | None = None

    def summary_dict(
        self,
        *,
        config_path: str,
        mock_state_path: str,
        output_dir: str,
        max_steps: int,
        input_mode: str,
        input_summary: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "status": self.status,
            "final_phase": self.final_phase,
            "failure_reason": self.failure_reason,
            "config_path": config_path,
            "mock_state_path": mock_state_path,
            "output_dir": output_dir,
            "max_steps": max_steps,
            "steps_run": self.steps_run,
            "phase_trace": self.phase_trace,
            "max_raw_action_norm": self.max_raw_action_norm,
            "max_safe_action_norm": self.max_safe_action_norm,
            "max_residual_norm": self.max_residual_norm,
            "all_actions_limited": self.all_actions_limited,
            "input_mode": input_mode,
            "input_summary": input_summary,
            "last_state_summary": self.last_state_summary,
            "hardware_called": False,
            "camera_opened": False,
            "mujoco_available": False,
            "mujoco_validation_status": "not_run",
            "failure_report": self.failure_report,
        }


class FRRGClosedLoopController:
    def __init__(
        self,
        config: FRRGGraspConfig,
        input_payload: dict[str, Any],
        *,
        residual_policy: ResidualPolicy | None = None,
        recovery_policy: RecoveryPolicy | None = None,
        input_mode: str | None = None,
    ) -> None:
        self.config = config
        self.input_mode = input_mode or config.runtime.default_input_mode
        self.current_payload = deepcopy(input_payload)
        self.previous_state: GraspState | None = None
        self.residual_policy = residual_policy or build_residual_policy(config)
        self.recovery_policy = recovery_policy or RecoveryPolicy()

    def step(self, step_index: int, state_payload: dict[str, Any] | None = None) -> ControllerStepResult:
        payload = deepcopy(state_payload or self.current_payload)
        observation = build_grasp_state(payload, default_mode=self.input_mode)
        geometry = apply_feature_geometry(
            observation.state,
            self.config,
            previous_state=self.previous_state,
        )
        feature_state, feature_debug_terms = self._apply_mock_feature_overrides(payload, geometry)
        scored = apply_scores(
            feature_state,
            self.config,
            feature_debug_terms=feature_debug_terms,
            contact_current_available=self._contact_current_available(payload),
        )
        guard_results = self._evaluate_guards(
            scored.state,
            feature_debug_terms=feature_debug_terms,
            contact_current_available=self._contact_current_available(payload),
        )
        phase_transition = next_phase(
            scored.state.phase,
            guard_results,
            retry_count=scored.state.retry_count,
            max_retry_count=self.config.runtime.max_retry_count,
            recovery_target=str(payload.get("recovery_target", PHASE_HANDOFF)),
        )
        raw_action = self._select_nominal_action(scored.state, feature_debug_terms=feature_debug_terms)
        recovery_result = self._select_recovery_action(scored.state, phase_transition)
        if recovery_result is not None and phase_transition.transition_kind == "recover":
            raw_action = recovery_result.action
        residual_result = self.residual_policy.compute(scored.state, raw_action)
        safety_result = apply_safety_limits(
            scored.state,
            self.config,
            raw_action,
            residual_action=residual_result.action,
            object_jump_m=float(payload.get("object_jump_m", 0.0)),
            invalid_phase=phase_transition.invalid_phase,
        )
        effective_transition = self._apply_safety_stop_to_transition(phase_transition, safety_result)
        next_payload = self._advance_payload(payload, scored.state, effective_transition, safety_result.safe_action)
        self.previous_state = scored.state
        self.current_payload = deepcopy(next_payload)
        return ControllerStepResult(
            step_index=step_index,
            observation=observation,
            geometry=geometry,
            scored=scored,
            feature_debug_terms=feature_debug_terms,
            guard_results=guard_results,
            phase_transition=effective_transition,
            raw_action=raw_action,
            recovery_result=recovery_result,
            residual_result=residual_result,
            safety_result=safety_result,
            next_payload=next_payload,
        )

    def run(self, max_steps: int) -> ControllerRunResult:
        phase_trace: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        states: list[dict[str, Any]] = []
        guards: list[dict[str, Any]] = []
        max_raw_action_norm = 0.0
        max_safe_action_norm = 0.0
        max_residual_norm = 0.0
        failure_reason: str | None = None
        failure_report: dict[str, Any] | None = None
        final_phase = str(self.current_payload.get("phase", PHASE_HANDOFF))

        for step_index in range(max(1, int(max_steps))):
            step_result = self.step(step_index)
            phase_trace.append(step_result.phase_trace_record())
            actions.append(step_result.action_record())
            states.append(step_result.state_record())
            guards.append(step_result.guards_record())
            max_raw_action_norm = max(max_raw_action_norm, step_result.safety_result.raw_action_norm)
            max_safe_action_norm = max(max_safe_action_norm, step_result.safety_result.safe_action_norm)
            max_residual_norm = max(max_residual_norm, step_result.safety_result.residual_norm)
            final_phase = step_result.phase_transition.next_phase
            if final_phase == PHASE_FAILURE:
                failure_reason = step_result.phase_transition.reason or step_result.safety_result.reason
                failure_report = self._build_failure_report(step_result, phase_trace)
                break
            if final_phase == PHASE_SUCCESS:
                break

        status = "success" if final_phase == PHASE_SUCCESS else "failure"
        return ControllerRunResult(
            status=status,
            final_phase=final_phase,
            failure_reason=failure_reason,
            steps_run=len(actions),
            phase_trace=phase_trace,
            actions=actions,
            states=states,
            guards=guards,
            max_raw_action_norm=max_raw_action_norm,
            max_safe_action_norm=max_safe_action_norm,
            max_residual_norm=max_residual_norm,
            all_actions_limited=all("safe_action" in action for action in actions),
            last_state_summary=states[-1] if states else None,
            failure_report=failure_report,
        )

    def _apply_mock_feature_overrides(
        self,
        payload: dict[str, Any],
        geometry: FeatureGeometryResult,
    ) -> tuple[GraspState, dict[str, Any]]:
        debug_terms = dict(geometry.debug_terms)
        payload_feature_debug_terms = payload.get("feature_debug_terms")
        if isinstance(payload_feature_debug_terms, dict):
            debug_terms.update(payload_feature_debug_terms)
        overrides = {name: payload[name] for name in FEATURE_OVERRIDE_FIELDS if name in payload}
        if overrides:
            return replace(geometry.state, **overrides), debug_terms
        return geometry.state, debug_terms

    def _contact_current_available(self, payload: dict[str, Any]) -> bool:
        return bool(payload.get("contact_current_available", True))

    def _evaluate_guards(
        self,
        state: GraspState,
        *,
        feature_debug_terms: dict[str, Any],
        contact_current_available: bool,
    ) -> dict[str, GuardResult]:
        guards: dict[str, GuardResult] = {"vision_lost": vision_lost_guard(state, self.config)}
        if state.phase == PHASE_HANDOFF:
            guards["handoff"] = handoff_guard(state, self.config, feature_debug_terms=feature_debug_terms)
        elif state.phase == PHASE_CAPTURE_BUILD:
            guards["capture_timeout"] = capture_timeout_guard(state, self.config)
            guards["capture_to_close"] = capture_to_close_guard(
                state,
                self.config,
                feature_debug_terms=feature_debug_terms,
            )
        elif state.phase == PHASE_CLOSE_HOLD:
            guards["close_to_lift"] = close_to_lift_guard(
                state,
                self.config,
                feature_debug_terms=feature_debug_terms,
                contact_current_available=contact_current_available,
            )
        elif state.phase == PHASE_LIFT_TEST:
            guards["slip_detected"] = slip_detected_guard(
                state,
                self.config,
                feature_debug_terms=feature_debug_terms,
            )
            guards["lift_to_success"] = lift_to_success_guard(
                state,
                self.config,
                feature_debug_terms=feature_debug_terms,
                contact_current_available=contact_current_available,
            )
        return guards

    def _select_recovery_action(
        self,
        state: GraspState,
        phase_transition: PhaseTraceEntry,
    ) -> RecoveryPolicyResult | None:
        reason = phase_transition.reason
        if not reason:
            return None
        if phase_transition.transition_kind == "recover":
            return self.recovery_policy.propose(state, self.config, reason)
        if phase_transition.next_phase == PHASE_FAILURE:
            return self.recovery_policy.propose(state, self.config, reason)
        return None

    def _select_nominal_action(
        self,
        state: GraspState,
        *,
        feature_debug_terms: dict[str, Any],
    ) -> GraspAction:
        if state.phase == PHASE_CAPTURE_BUILD:
            return nominal_capture_action(state, self.config, feature_debug_terms=feature_debug_terms)
        if state.phase == PHASE_CLOSE_HOLD:
            return nominal_close_action(state, self.config)
        if state.phase == PHASE_LIFT_TEST:
            return nominal_lift_action(state, self.config)
        return GraspAction(
            debug_terms={
                "primitive": "idle",
                "is_raw_action": True,
            }
        )

    def _apply_safety_stop_to_transition(
        self,
        phase_transition: PhaseTraceEntry,
        safety_result: SafetyLimitResult,
    ) -> PhaseTraceEntry:
        if not safety_result.stop or phase_transition.next_phase in (PHASE_FAILURE, PHASE_SUCCESS):
            return phase_transition
        return PhaseTraceEntry(
            current_phase=phase_transition.current_phase,
            next_phase=PHASE_FAILURE,
            transition_kind="fail",
            allowed=False,
            reason=safety_result.reason,
            guard_name=phase_transition.guard_name,
            retry_count=phase_transition.retry_count,
            invalid_phase=phase_transition.invalid_phase or safety_result.reason == "invalid_phase",
            debug_terms={
                **phase_transition.debug_terms,
                "safety_stop": True,
                "safety_reason": safety_result.reason,
            },
        )

    def _advance_payload(
        self,
        payload: dict[str, Any],
        state: GraspState,
        phase_transition: PhaseTraceEntry,
        safe_action: GraspAction,
    ) -> dict[str, Any]:
        next_payload = deepcopy(payload)
        dt = 1.0 / float(self.config.runtime.control_hz)
        same_phase = phase_transition.next_phase == state.phase
        retry_count = int(state.retry_count)
        if phase_transition.transition_kind == "recover":
            retry_count += 1

        next_payload["timestamp"] = float(state.timestamp) + dt
        next_payload["phase"] = phase_transition.next_phase
        next_payload["retry_count"] = retry_count
        next_payload["phase_elapsed_s"] = float(state.phase_elapsed_s) + dt if same_phase else 0.0
        next_payload["stable_count"] = int(state.stable_count) + 1 if same_phase else 0
        next_payload["gripper_width"] = max(0.0, float(state.gripper_width) + float(safe_action.delta_gripper))
        next_payload["gripper_cmd"] = max(0.0, float(state.gripper_cmd) + float(safe_action.delta_gripper))

        ee_pose_object = Pose6D(
            xyz=(
                float(state.ee_pose_object.xyz[0]) + float(safe_action.delta_pose_object[0]),
                float(state.ee_pose_object.xyz[1]) + float(safe_action.delta_pose_object[1]),
                float(state.ee_pose_object.xyz[2]) + float(safe_action.delta_pose_object[2]),
            ),
            rpy=(
                float(state.ee_pose_object.rpy[0]) + float(safe_action.delta_pose_object[3]),
                float(state.ee_pose_object.rpy[1]) + float(safe_action.delta_pose_object[4]),
                float(state.ee_pose_object.rpy[2]) + float(safe_action.delta_pose_object[5]),
            ),
        )
        ee_pose_base = matrix_to_pose6d(
            compose_transform(
                pose6d_to_matrix(state.object_pose_base),
                pose6d_to_matrix(ee_pose_object),
            )
        )
        next_payload["ee_pose_base"] = {
            "xyz": list(ee_pose_base.xyz),
            "rpy": list(ee_pose_base.rpy),
        }
        return next_payload

    def _build_failure_report(
        self,
        step_result: ControllerStepResult,
        phase_trace: list[dict[str, Any]],
    ) -> dict[str, Any]:
        failure_reason = step_result.phase_transition.reason or step_result.safety_result.reason
        recovery_result = step_result.recovery_result
        if recovery_result is None and failure_reason is not None:
            recovery_result = self.recovery_policy.propose(step_result.scored.state, self.config, failure_reason)
        return {
            "status": "failure",
            "failure_reason": failure_reason,
            "final_phase": step_result.phase_transition.next_phase,
            "step": step_result.step_index,
            "last_state": step_result.state_record(),
            "safe_action": step_result.action_record()["safe_action"],
            "raw_action": step_result.action_record()["raw_action"],
            "recovery_proposal": None
            if recovery_result is None
            else {
                "proposal_type": recovery_result.proposal_type,
                "reason": recovery_result.reason,
                "action": {
                    "delta_pose_object": list(recovery_result.action.delta_pose_object),
                    "delta_gripper": float(recovery_result.action.delta_gripper),
                    "stop": bool(recovery_result.action.stop),
                    "reason": recovery_result.action.reason,
                    "debug_terms": dict(recovery_result.action.debug_terms),
                },
                "debug_terms": dict(recovery_result.debug_terms),
            },
            "phase_trace": list(phase_trace),
        }


__all__ = [
    "ControllerRunResult",
    "ControllerStepResult",
    "FRRGClosedLoopController",
]
