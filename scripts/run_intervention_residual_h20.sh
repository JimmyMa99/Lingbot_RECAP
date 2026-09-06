#!/usr/bin/env bash
set -euo pipefail
WS=/mnt/workspace/users/zb/lingbot_ws
CODE="$WS/code/Lingbot_RECAP-rlt-test"
PY="$WS/.venv/bin/python"
AUTO="$WS/experiments/lingbot_recap_straw_cup_auto_context_20260906"
OUTPUT="$WS/experiments/lingbot_recap_straw_cup_intervention_residual_v2_20260906"
LOGS="$WS/logs"
export PYTHONPATH="$CODE:${PYTHONPATH:-}"
mkdir -p "$AUTO" "$OUTPUT" "$LOGS"
COMMON=(
 --experience-root "$WS/data/recap_straw_into_cup" --output-root "$AUTO"
 --lingbot-root "$WS/code/lingbot"
 --model-path "$WS/checkpoints/newdata_yellow_beside_milk_teacher_cumulative_epoch90_merged_20260902"
 --training-config "$WS/experiments/newdata_yellow_beside_milk_teacher_continue_ep61_to90_lr1e4_4h20_20260902_retry/lingbotvla_cli.yaml"
 --norm-stats "$WS/norm_stats/all_data_train776_selectorfix_20260829.json"
 --bottleneck "$WS/experiments/lingbot_recap_visual_bottleneck_512d_3h20_20260906/best.pt"
 --world-size 3 --stride 4
)
pids=()
for rank in 0 1 2; do
 CUDA_VISIBLE_DEVICES=$((rank+1)) "$PY" "$CODE/tools/encode_recap_auto_context.py" \
  "${COMMON[@]}" --rank "$rank" >"$LOGS/recap_auto_encode_rank${rank}_20260906.log" 2>&1 &
 pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
CUDA_VISIBLE_DEVICES=1 "$PY" "$CODE/tools/train_intervention_residual.py" \
 --encoded-root "$WS/experiments/lingbot_recap_straw_cup_encoded_20260906" \
 --auto-root "$AUTO" --output-root "$OUTPUT" --epochs 200 --batch-size 128 \
 --lr 3e-4 --residual-limit .20 \
 --held-out episode_20260906_142957_6cd26719.complete \
  episode_20260906_143028_2658daed.complete \
  episode_20260906_143103_cb5eda43.complete \
 2>&1 | tee "$LOGS/recap_intervention_residual_v2_20260906.log"
