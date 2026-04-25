#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _bootstrap_pythonpath() -> None:
    repo_root = _repo_root()
    for candidate in (repo_root / "src", repo_root):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


_bootstrap_pythonpath()

from lerobot.projects.vlbiman_sa.grasp.contracts import GraspAction, load_frrg_config
from lerobot.projects.vlbiman_sa.grasp.observer import build_grasp_state
from lerobot.projects.vlbiman_sa.grasp.residual.bc_policy import (
    BC_FEATURE_NAMES,
    compute_demo_residual_label,
    extract_bc_features,
    residual_clip_limits_from_config,
    save_bc_checkpoint,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a placeholder BC residual model from offline JSONL data.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the FRRG YAML config.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to the JSONL training dataset.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the checkpoint JSON to write.")
    parser.add_argument("--ridge-lambda", type=float, default=1e-6, help="Ridge regularization coefficient.")
    return parser.parse_args(argv)


def _require_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object, got {type(value).__name__}.")
    return value


def _action_from_payload(payload: Any, *, context: str) -> GraspAction:
    action_payload = _require_mapping(payload, context=context)
    return GraspAction(
        delta_pose_object=tuple(action_payload.get("delta_pose_object", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))),
        delta_gripper=float(action_payload.get("delta_gripper", 0.0)),
        stop=bool(action_payload.get("stop", False)),
        reason=action_payload.get("reason"),
        debug_terms=dict(action_payload.get("debug_terms", {})),
    )


def _load_dataset(dataset_path: Path, config_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    config = load_frrg_config(config_path)
    clip_pose_limits, clip_gripper_limit = residual_clip_limits_from_config(config)
    feature_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    clip_count = 0

    for line_index, raw_line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        record = json.loads(line)
        record_payload = _require_mapping(record, context=f"dataset line {line_index}")
        state_payload = _require_mapping(record_payload.get("state"), context=f"dataset line {line_index}.state")
        state = build_grasp_state(state_payload).state
        nominal_action = _action_from_payload(record_payload.get("nominal_action"), context=f"dataset line {line_index}.nominal_action")
        demo_action = _action_from_payload(record_payload.get("demo_action"), context=f"dataset line {line_index}.demo_action")

        label = compute_demo_residual_label(
            demo_action,
            nominal_action,
            clip_pose_limits=clip_pose_limits,
            clip_gripper_limit=clip_gripper_limit,
        )
        feature_rows.append(extract_bc_features(state, nominal_action, feature_names=BC_FEATURE_NAMES))
        label_rows.append(np.asarray(label.clipped_vector, dtype=float))
        if label.clip_applied:
            clip_count += 1

    if not feature_rows:
        raise ValueError(f"Dataset contains no usable training records: {dataset_path}")

    return (
        np.vstack(feature_rows),
        np.vstack(label_rows),
        {
            "sample_count": len(feature_rows),
            "clipped_label_count": clip_count,
            "feature_dim": len(BC_FEATURE_NAMES),
        },
    )


def _fit_ridge_regression(features: np.ndarray, labels: np.ndarray, ridge_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    sample_count, feature_dim = features.shape
    if labels.shape != (sample_count, 7):
        raise ValueError(f"labels must have shape ({sample_count}, 7), got {labels.shape}")
    if ridge_lambda < 0.0:
        raise ValueError(f"ridge_lambda must be >= 0, got {ridge_lambda}")

    design = np.concatenate([features, np.ones((sample_count, 1), dtype=float)], axis=1)
    regularizer = np.eye(design.shape[1], dtype=float)
    regularizer[-1, -1] = 0.0
    solution = np.linalg.solve(
        design.T @ design + float(ridge_lambda) * regularizer,
        design.T @ labels,
    )
    weights = solution[:-1, :].T
    bias = solution[-1, :]
    return weights, bias


def train_bc_residual(
    *,
    config_path: Path,
    dataset_path: Path,
    output_path: Path,
    ridge_lambda: float = 1e-6,
) -> dict[str, Any]:
    config = load_frrg_config(config_path)
    clip_pose_limits, clip_gripper_limit = residual_clip_limits_from_config(config)
    features, labels, dataset_summary = _load_dataset(dataset_path, config_path)
    weights, bias = _fit_ridge_regression(features, labels, ridge_lambda=float(ridge_lambda))
    checkpoint_path = save_bc_checkpoint(
        output_path,
        weights=weights,
        bias=bias,
        clip_pose_limits=clip_pose_limits,
        clip_gripper_limit=clip_gripper_limit,
    )
    return {
        "config_path": str(config_path.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "output_path": str(checkpoint_path.resolve()),
        "sample_count": dataset_summary["sample_count"],
        "clipped_label_count": dataset_summary["clipped_label_count"],
        "feature_dim": dataset_summary["feature_dim"],
        "ridge_lambda": float(ridge_lambda),
        "policy": "bc",
        "model_family": "linear",
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = train_bc_residual(
        config_path=args.config,
        dataset_path=args.dataset,
        output_path=args.output,
        ridge_lambda=float(args.ridge_lambda),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
