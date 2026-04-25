#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
	SELECTED_PYTHON="${PYTHON_BIN}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
	SELECTED_PYTHON="${CONDA_PREFIX}/bin/python"
elif [[ -x "${HOME}/miniconda3/envs/lerobot/bin/python" ]]; then
  SELECTED_PYTHON="${HOME}/miniconda3/envs/lerobot/bin/python"
else
	echo "Error: conda env 'lerobot' not found. Run: conda activate lerobot, or set PYTHON_BIN." >&2
	exit 1
fi

cd "${REPO_ROOT}"
PYTHONPATH=src:. "${SELECTED_PYTHON}" -m lerobot_camera_gemini335l.diagnose_depth_startup "$@"
