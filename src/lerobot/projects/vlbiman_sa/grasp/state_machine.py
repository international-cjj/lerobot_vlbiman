from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from .contracts import GuardResult


PHASE_HANDOFF = "HANDOFF"
PHASE_CAPTURE_BUILD = "CAPTURE_BUILD"
PHASE_CLOSE_HOLD = "CLOSE_HOLD"
PHASE_LIFT_TEST = "LIFT_TEST"
PHASE_RECOVERY = "RECOVERY"
PHASE_SUCCESS = "SUCCESS"
PHASE_FAILURE = "FAILURE"

VALID_PHASES = (
    PHASE_HANDOFF,
    PHASE_CAPTURE_BUILD,
    PHASE_CLOSE_HOLD,
    PHASE_LIFT_TEST,
    PHASE_RECOVERY,
    PHASE_SUCCESS,
    PHASE_FAILURE,
)

TRANSITION_GUARDS: dict[str, tuple[str, str]] = {
    PHASE_HANDOFF: ("handoff", PHASE_CAPTURE_BUILD),
    PHASE_CAPTURE_BUILD: ("capture_to_close", PHASE_CLOSE_HOLD),
    PHASE_CLOSE_HOLD: ("close_to_lift", PHASE_LIFT_TEST),
    PHASE_LIFT_TEST: ("lift_to_success", PHASE_SUCCESS),
}

FAILURE_GUARD_PRIORITY = (
    "vision_lost",
    "capture_timeout",
    "slip_detected",
)

DEFAULT_RECOVERABLE_FAILURE_REASONS = frozenset({"capture_timeout"})

LEGAL_TRANSITIONS: dict[str, frozenset[str]] = {
    PHASE_HANDOFF: frozenset({PHASE_HANDOFF, PHASE_CAPTURE_BUILD, PHASE_RECOVERY, PHASE_FAILURE}),
    PHASE_CAPTURE_BUILD: frozenset({PHASE_CAPTURE_BUILD, PHASE_CLOSE_HOLD, PHASE_RECOVERY, PHASE_FAILURE}),
    PHASE_CLOSE_HOLD: frozenset({PHASE_CLOSE_HOLD, PHASE_LIFT_TEST, PHASE_RECOVERY, PHASE_FAILURE}),
    PHASE_LIFT_TEST: frozenset({PHASE_LIFT_TEST, PHASE_SUCCESS, PHASE_RECOVERY, PHASE_FAILURE}),
    PHASE_RECOVERY: frozenset(
        {
            PHASE_HANDOFF,
            PHASE_CAPTURE_BUILD,
            PHASE_CLOSE_HOLD,
            PHASE_LIFT_TEST,
            PHASE_FAILURE,
        }
    ),
    PHASE_SUCCESS: frozenset({PHASE_SUCCESS}),
    PHASE_FAILURE: frozenset({PHASE_FAILURE}),
}


@dataclass
class PhaseTraceEntry:
    current_phase: str
    next_phase: str
    transition_kind: str
    allowed: bool
    reason: str | None = None
    guard_name: str | None = None
    retry_count: int = 0
    invalid_phase: bool = False
    debug_terms: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ordered_failure_guard_names(guard_results: Mapping[str, GuardResult], failure_guard_names: Iterable[str]) -> list[str]:
    configured = list(dict.fromkeys(failure_guard_names))
    prioritized = [name for name in FAILURE_GUARD_PRIORITY if name in configured]
    remainder = [name for name in configured if name not in prioritized]
    discovered = [
        name
        for name, result in guard_results.items()
        if name not in prioritized and name not in remainder and result.reason is not None
    ]
    return prioritized + remainder + discovered


def _trace_entry(
    current_phase: str,
    next_phase: str,
    *,
    transition_kind: str,
    allowed: bool,
    reason: str | None = None,
    guard_name: str | None = None,
    retry_count: int = 0,
    invalid_phase: bool = False,
    debug_terms: Mapping[str, Any] | None = None,
) -> PhaseTraceEntry:
    return PhaseTraceEntry(
        current_phase=current_phase,
        next_phase=next_phase,
        transition_kind=transition_kind,
        allowed=allowed,
        reason=reason,
        guard_name=guard_name,
        retry_count=int(retry_count),
        invalid_phase=invalid_phase,
        debug_terms=dict(debug_terms or {}),
    )


def next_phase(
    current_phase: str,
    guard_results: Mapping[str, GuardResult],
    *,
    retry_count: int = 0,
    max_retry_count: int = 0,
    recovery_target: str = PHASE_HANDOFF,
    failure_guard_names: Iterable[str] = FAILURE_GUARD_PRIORITY,
    recoverable_failure_reasons: Iterable[str] = DEFAULT_RECOVERABLE_FAILURE_REASONS,
) -> PhaseTraceEntry:
    if current_phase not in VALID_PHASES:
        return _trace_entry(
            current_phase,
            PHASE_FAILURE,
            transition_kind="fail",
            allowed=False,
            reason="unknown_state",
            retry_count=retry_count,
            invalid_phase=True,
            debug_terms={"valid_phases": list(VALID_PHASES)},
        )

    if current_phase in (PHASE_SUCCESS, PHASE_FAILURE):
        return _trace_entry(
            current_phase,
            current_phase,
            transition_kind="terminal",
            allowed=True,
            retry_count=retry_count,
        )

    if current_phase == PHASE_RECOVERY:
        if int(retry_count) > int(max_retry_count):
            return _trace_entry(
                current_phase,
                PHASE_FAILURE,
                transition_kind="fail",
                allowed=True,
                reason="max_retry_exceeded",
                retry_count=retry_count,
                debug_terms={"max_retry_count": int(max_retry_count)},
            )
        if recovery_target not in LEGAL_TRANSITIONS[PHASE_RECOVERY]:
            return _trace_entry(
                current_phase,
                PHASE_FAILURE,
                transition_kind="fail",
                allowed=False,
                reason="invalid_phase",
                retry_count=retry_count,
                invalid_phase=True,
                debug_terms={
                    "recovery_target": recovery_target,
                    "legal_recovery_targets": sorted(LEGAL_TRANSITIONS[PHASE_RECOVERY]),
                },
            )
        return _trace_entry(
            current_phase,
            recovery_target,
            transition_kind="recover_exit",
            allowed=True,
            retry_count=retry_count,
            debug_terms={"recovery_target": recovery_target},
        )

    recoverable_reasons = set(recoverable_failure_reasons)
    for guard_name in _ordered_failure_guard_names(guard_results, failure_guard_names):
        guard_result = guard_results.get(guard_name)
        if guard_result is None or not guard_result.passed:
            continue

        failure_reason = guard_result.reason or guard_name
        debug_terms = {
            "triggered_failure_reason": failure_reason,
            "max_retry_count": int(max_retry_count),
            **guard_result.debug_terms,
        }
        if failure_reason in recoverable_reasons:
            if int(retry_count) >= int(max_retry_count):
                return _trace_entry(
                    current_phase,
                    PHASE_FAILURE,
                    transition_kind="fail",
                    allowed=True,
                    reason="max_retry_exceeded",
                    guard_name=guard_name,
                    retry_count=retry_count,
                    debug_terms=debug_terms,
                )
            return _trace_entry(
                current_phase,
                PHASE_RECOVERY,
                transition_kind="recover",
                allowed=True,
                reason=failure_reason,
                guard_name=guard_name,
                retry_count=retry_count,
                debug_terms=debug_terms,
            )

        return _trace_entry(
            current_phase,
            PHASE_FAILURE,
            transition_kind="fail",
            allowed=True,
            reason=failure_reason,
            guard_name=guard_name,
            retry_count=retry_count,
            debug_terms=debug_terms,
        )

    expected_guard_name, target_phase = TRANSITION_GUARDS[current_phase]
    illegal_transition_guards = [
        guard_name
        for guard_name, _ in TRANSITION_GUARDS.values()
        if guard_name != expected_guard_name and guard_results.get(guard_name, GuardResult(False, 0.0)).passed
    ]
    if illegal_transition_guards:
        return _trace_entry(
            current_phase,
            current_phase,
            transition_kind="hold",
            allowed=False,
            reason="invalid_phase",
            guard_name=illegal_transition_guards[0],
            retry_count=retry_count,
            invalid_phase=True,
            debug_terms={
                "expected_transition_guard": expected_guard_name,
                "illegal_transition_guards": illegal_transition_guards,
            },
        )

    transition_guard = guard_results.get(expected_guard_name)
    if transition_guard is None:
        return _trace_entry(
            current_phase,
            current_phase,
            transition_kind="hold",
            allowed=False,
            reason="invalid_phase",
            guard_name=expected_guard_name,
            retry_count=retry_count,
            invalid_phase=True,
            debug_terms={"missing_transition_guard": expected_guard_name},
        )

    if transition_guard.passed:
        return _trace_entry(
            current_phase,
            target_phase,
            transition_kind="advance",
            allowed=True,
            guard_name=expected_guard_name,
            retry_count=retry_count,
            debug_terms=transition_guard.debug_terms,
        )

    return _trace_entry(
        current_phase,
        current_phase,
        transition_kind="hold",
        allowed=True,
        reason=transition_guard.reason,
        guard_name=expected_guard_name,
        retry_count=retry_count,
        debug_terms=transition_guard.debug_terms,
    )


__all__ = [
    "DEFAULT_RECOVERABLE_FAILURE_REASONS",
    "FAILURE_GUARD_PRIORITY",
    "LEGAL_TRANSITIONS",
    "PHASE_CAPTURE_BUILD",
    "PHASE_CLOSE_HOLD",
    "PHASE_FAILURE",
    "PHASE_HANDOFF",
    "PHASE_LIFT_TEST",
    "PHASE_RECOVERY",
    "PHASE_SUCCESS",
    "PhaseTraceEntry",
    "TRANSITION_GUARDS",
    "VALID_PHASES",
    "next_phase",
]
