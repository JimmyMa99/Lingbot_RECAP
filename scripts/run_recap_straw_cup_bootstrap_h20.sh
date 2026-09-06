#!/usr/bin/env bash
set -euo pipefail

WS=/mnt/workspace/users/zb/lingbot_ws
CODE="$WS/code/Lingbot_RECAP-rlt-test"
PY="$WS/.venv/bin/python"
ENCODED="$WS/experiments/lingbot_recap_straw_cup_encoded_20260906"
RESIDUAL="$WS/experiments/lingbot_recap_straw_cup_residual_20260906"
LOGS="$WS/logs"
mkdir -p "$ENCODED" "$RESIDUAL" "$LOGS"
export PYTHONPATH="$CODE:${PYTHONPATH:-}"

COMMON=(
  --experience-root "$WS/data/recap_straw_into_cup"
  --output-root "$ENCODED"
  --lingbot-root "$WS/code/lingbot"
  --model-path "$WS/checkpoints/newdata_yellow_beside_milk_teacher_cumulative_epoch90_merged_20260902"
  --training-config "$WS/experiments/newdata_yellow_beside_milk_teacher_continue_ep61_to90_lr1e4_4h20_20260902_retry/lingbotvla_cli.yaml"
  --norm-stats "$WS/norm_stats/all_data_train776_selectorfix_20260829.json"
  --bottleneck "$WS/experiments/lingbot_recap_visual_bottleneck_512d_3h20_20260906/best.pt"
  --world-size 3
  --stride 4
)

pids=()
for rank in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$((rank + 1)) "$PY" "$CODE/tools/encode_recap_experience.py" \
    "${COMMON[@]}" --rank "$rank" \
    >"$LOGS/recap_encode_rank${rank}_20260906.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

CUDA_VISIBLE_DEVICES=1 "$PY" "$CODE/tools/bootstrap_recap_residual.py" \
  --encoded-root "$ENCODED" \
  --output-root "$RESIDUAL" \
  --held-out \
    episode_20260906_142957_6cd26719.complete \
    episode_20260906_143028_2658daed.complete \
    episode_20260906_143103_cb5eda43.complete \
  2>&1 | tee "$LOGS/recap_bootstrap_residual_20260906.log"

touch "$RESIDUAL/COMPLETE"
