#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif [[ -x "$HOME/miniconda3/envs/lerobot/bin/python" ]]; then
  PYTHON_BIN="$HOME/miniconda3/envs/lerobot/bin/python"
else
  echo "Error: conda env 'lerobot' not found. Run: conda activate lerobot" >&2
  exit 1
fi

TASK_CONFIG="${TASK_CONFIG:-src/lerobot/projects/vlbiman_sa/configs/task_grasp_one_shot_full_20260411T061326.yaml}"
FRRG_CONFIG="${FRRG_CONFIG:-src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml}"
LIVE_RESULT="${LIVE_RESULT:-outputs/vlbiman_sa/live_orange_pose/live_scene_20260420T133513Z.json}"
TARGET_PHRASE="${TARGET_PHRASE:-orange}"
AUX_TARGET_PHRASE="${AUX_TARGET_PHRASE:-pink cup}"
START_INDEX="${START_INDEX:-47}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

exec env \
  MUJOCO_GL="${MUJOCO_GL:-egl}" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT" \
  "$PYTHON_BIN" src/lerobot/projects/vlbiman_sa/app/run_visual_pickorange_branch_compare.py \
    --task-config "$TASK_CONFIG" \
    --frrg-config "$FRRG_CONFIG" \
    --reuse-live-result "$LIVE_RESULT" \
    --target-phrase "$TARGET_PHRASE" \
    --aux-target-phrase "$AUX_TARGET_PHRASE" \
    --start-index "$START_INDEX" \
    --branch original_replay \
    --headless \
    --log-level "$LOG_LEVEL" \
    "$@"
