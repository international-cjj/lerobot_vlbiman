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

PIP_BIN="${PIP_BIN:-$PYTHON_BIN -m pip}"

echo "[1/4] Installing base deps..."
$PIP_BIN install --upgrade pip wheel "setuptools>=71,<81"
$PIP_BIN uninstall -y pin pinocchio cmeel cmeel-boost cmeel-assimp cmeel-octomap eigenpy coal hppfcl >/dev/null 2>&1 || true
$PIP_BIN install "numpy<2.3" pyyaml "opencv-python-headless>=4.9,<4.13" "opencv-python>=4.9,<4.13" pyserial

echo "[2/4] Installing IKPy fallback deps..."
$PIP_BIN install ikpy scipy sympy mujoco

echo "[3/4] Verifying runtime modules..."
$PYTHON_BIN - <<'PY'
import importlib
required = ["numpy", "yaml", "cv2", "serial", "ikpy", "mujoco"]
for name in required:
    try:
        importlib.import_module(name)
        print(name, "OK")
    except Exception as exc:
        print(name, "MISSING", exc)
        raise

for optional in ["pyorbbecsdk", "pinocchio"]:
    try:
        importlib.import_module(optional)
        print(optional, "OK")
    except Exception as exc:
        print(optional, "OPTIONAL_MISSING", exc)

import cv2
for line in cv2.getBuildInformation().splitlines():
    if line.strip().startswith("GUI:"):
        print("cv2_build", line.strip())
        break
PY

echo "[4/4] Environment setup complete. Use PYTHON_BIN=$PYTHON_BIN"
