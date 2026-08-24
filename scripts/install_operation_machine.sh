#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/mzm/code/Lingbot_RECAP}
LEROBOT_ENV=${LEROBOT_ENV:-/home/mzm/miniconda3/envs/lerobot}

cd "$PROJECT_DIR"
"$LEROBOT_ENV/bin/pip" install -e . --no-deps
"$LEROBOT_ENV/bin/python" -m compileall -q lingbot_recap
"$LEROBOT_ENV/bin/python" -m lingbot_recap.cli --help

if ! id -nG | tr ' ' '\n' | grep -qx input; then
  echo "WARNING: current user is not in the input group; two-button evdev reads will fail."
  echo 'Run: sudo usermod -aG input "$USER"  (then log out and back in)'
fi
