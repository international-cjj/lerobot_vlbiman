#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

resolve_config_path() {
  local raw_path="$1"
  if [[ -d "$raw_path" ]]; then
    printf '%s\n' "${raw_path%/}/handeye_auto.yaml"
    return
  fi
  printf '%s\n' "$raw_path"
}

CONFIG_PATH_RAW="${1:-src/lerobot/projects/vlbiman_sa/configs/handeye_auto.yaml}"
CONFIG_PATH="$(resolve_config_path "$CONFIG_PATH_RAW")"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Error: calibration config not found: $CONFIG_PATH" >&2
  exit 1
fi

INTRINSICS_PATH="${INTRINSICS_PATH:-src/lerobot/projects/vlbiman_sa/configs/camera_intrinsics.json}"
RUN_SETUP="${RUN_SETUP:-1}"
DUMP_INTRINSICS="${DUMP_INTRINSICS:-1}"

if [[ "$RUN_SETUP" == "1" ]]; then
  "$REPO_ROOT/scripts/vlbiman_sa/setup_handeye_env.sh"
fi

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

if [[ "$DUMP_INTRINSICS" == "1" ]]; then
  "$PYTHON_BIN" src/lerobot/projects/vlbiman_sa/calib/dump_gemini_intrinsics.py \
    --config "$CONFIG_PATH" \
    --output "$INTRINSICS_PATH" \
    --stream rgb
fi

exec "$PYTHON_BIN" src/lerobot/projects/vlbiman_sa/calib/run_handeye_auto.py --config "$CONFIG_PATH"
