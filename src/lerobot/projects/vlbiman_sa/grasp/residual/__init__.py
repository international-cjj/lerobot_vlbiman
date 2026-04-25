"""Residual policy interfaces for FRRG dry-run."""

from .bc_policy import (
    BC_FEATURE_NAMES,
    BCResidualPolicy,
    ResidualLabel,
    build_residual_policy,
    clip_residual_vector,
    compute_demo_residual_label,
    create_bc_checkpoint_payload,
    extract_bc_feature_dict,
    extract_bc_features,
    residual_action_to_vector,
    residual_clip_limits_from_config,
    residual_vector_to_action,
    save_bc_checkpoint,
)
from .policy import ResidualPolicy, ResidualResult, action_l2_norm
from .zero_policy import ZeroResidualPolicy

__all__ = [
    "BC_FEATURE_NAMES",
    "BCResidualPolicy",
    "ResidualLabel",
    "ResidualPolicy",
    "ResidualResult",
    "ZeroResidualPolicy",
    "action_l2_norm",
    "build_residual_policy",
    "clip_residual_vector",
    "compute_demo_residual_label",
    "create_bc_checkpoint_payload",
    "extract_bc_feature_dict",
    "extract_bc_features",
    "residual_action_to_vector",
    "residual_clip_limits_from_config",
    "residual_vector_to_action",
    "save_bc_checkpoint",
]
