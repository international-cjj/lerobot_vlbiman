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

SERIAL="${1:-}"
cd "${REPO_ROOT}"

echo "[1/4] Import/driver smoke check"
PYTHONPATH=src:. "${SELECTED_PYTHON}" -m lerobot_camera_gemini335l.smoke_test --list

if [[ -z "${SERIAL}" ]]; then
  echo "\nNo serial provided. Usage:" >&2
  echo "  ./scripts/gemini335l/quick_check.sh <SERIAL_OR_NAME>" >&2
  echo "You can get serial from the list above." >&2
  exit 2
fi

echo "[2/4] Startup diagnostics"
PYTHONPATH=src:. "${SELECTED_PYTHON}" -m lerobot_camera_gemini335l.diagnose_depth_startup \
  --serial-number-or-name "${SERIAL}" \
  --width 640 --height 400 --fps 30 \
  --color-stream-format MJPG --depth-stream-format Y16 \
  --align-mode sw --align-depth-to-color

echo "[3/4] Read frames headlessly (for algorithm callability)"
PYTHONPATH=src:. "${SELECTED_PYTHON}" -m lerobot_camera_gemini335l.stream_viewer \
  --serial-number-or-name "${SERIAL}" \
  --stream rgbd \
  --width 640 --height 400 --fps 15 \
  --align-mode sw --align-depth-to-color \
  --profile-selection-strategy closest \
  --headless --frame-limit 60

echo "[4/4] Done. For real-time display run:"
echo "  ./scripts/gemini335l/run_stream_viewer.sh --serial-number-or-name ${SERIAL} --stream rgbd --width 640 --height 400 --fps 15 --align-mode sw --align-depth-to-color --profile-selection-strategy closest"
