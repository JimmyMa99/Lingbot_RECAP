#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/mzm/code/Lingbot_RECAP}
LEROBOT_ENV=${LEROBOT_ENV:-/home/mzm/miniconda3/envs/lerobot}

cd "$PROJECT_DIR"
"$LEROBOT_ENV/bin/pip" install -e . --no-deps
"$LEROBOT_ENV/bin/python" -m compileall -q lingbot_recap
"$LEROBOT_ENV/bin/python" -m lingbot_recap.cli --help
