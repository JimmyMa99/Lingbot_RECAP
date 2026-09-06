#!/usr/bin/env bash
set -euo pipefail

WS=/mnt/workspace/users/zb/lingbot_ws
CODE="$WS/code/Lingbot_RECAP-rlt-test"
PY="$WS/.venv/bin/python"
OUTPUT="$WS/experiments/lingbot_recap_straw_cup_intervention_residual_v3_gripper_20260906"
LOG="$WS/logs/recap_intervention_residual_v3_gripper_20260906.log"

export PYTHONPATH="$CODE:${PYTHONPATH:-}"
mkdir -p "$OUTPUT" "$(dirname "$LOG")"

CUDA_VISIBLE_DEVICES=1 "$PY" "$CODE/tools/train_intervention_residual.py" \
  --encoded-root "$WS/experiments/lingbot_recap_straw_cup_encoded_20260906" \
  --auto-root "$WS/experiments/lingbot_recap_straw_cup_auto_context_20260906" \
  --output-root "$OUTPUT" \
  --epochs 200 \
  --batch-size 128 \
  --lr 3e-4 \
  --residual-limit .20 \
  --gripper-residual-limit .60 \
  --held-out \
    episode_20260906_142957_6cd26719.complete \
    episode_20260906_143028_2658daed.complete \
    episode_20260906_143103_cb5eda43.complete \
  2>&1 | tee "$LOG"
