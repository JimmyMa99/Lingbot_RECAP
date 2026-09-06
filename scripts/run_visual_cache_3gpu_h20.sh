#!/usr/bin/env bash
set -uo pipefail

WORKSPACE=/mnt/workspace/users/zb/lingbot_ws
RECAP_ROOT="$WORKSPACE/code/Lingbot_RECAP-rlt-test"
LINGBOT_ROOT="$WORKSPACE/code/lingbot"
PYTHON="$WORKSPACE/.venv/bin/python"
OUTPUT_ROOT="$WORKSPACE/experiments/lingbot_recap_visual_cache_all776_16000_3h20_20260906"
LOG_ROOT="$OUTPUT_ROOT/logs"

mkdir -p "$LOG_ROOT"
export QWEN3VL_PATH="$WORKSPACE/models/Qwen3-VL-4B-Instruct"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$RECAP_ROOT:$LINGBOT_ROOT"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/blas:/usr/lib/x86_64-linux-gnu/lapack:${LD_LIBRARY_PATH:-}"

pids=()
for rank in 0 1 2; do
  gpu=$((rank + 1))
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$RECAP_ROOT/tools/cache_lingbot_visual_tokens.py" \
    --lingbot-root "$LINGBOT_ROOT" \
    --model-path "$WORKSPACE/checkpoints/newdata_yellow_beside_milk_teacher_cumulative_epoch90_merged_20260902" \
    --training-config "$WORKSPACE/experiments/newdata_yellow_beside_milk_teacher_continue_ep61_to90_lr1e4_4h20_20260902_retry/lingbotvla_cli.yaml" \
    --norm-stats "$WORKSPACE/norm_stats/all_data_train776_selectorfix_20260829.json" \
    --data-manifest "$WORKSPACE/configs/all_data_train776.txt" \
    --output-root "$OUTPUT_ROOT" \
    --rank "$rank" \
    --world-size 3 \
    --max-samples 16000 \
    >"$LOG_ROOT/rank_${rank}.log" 2>&1 &
  pids+=("$!")
  echo "$!" >"$LOG_ROOT/rank_${rank}.pid"
done

failed=0
for rank in 0 1 2; do
  if ! wait "${pids[$rank]}"; then
    echo "rank $rank failed; see $LOG_ROOT/rank_${rank}.log" >&2
    failed=1
  fi
done

if [[ "$failed" -eq 0 ]]; then
  date --iso-8601=seconds >"$OUTPUT_ROOT/COMPLETE"
  echo "All three visual-token cache shards completed."
else
  date --iso-8601=seconds >"$OUTPUT_ROOT/FAILED"
fi
exit "$failed"
