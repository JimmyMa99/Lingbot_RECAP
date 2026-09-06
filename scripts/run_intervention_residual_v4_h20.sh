#!/usr/bin/env bash
set -euo pipefail

WS=/mnt/workspace/users/zb/lingbot_ws
CODE="$WS/code/Lingbot_RECAP-rlt-test"
PY="$WS/.venv/bin/python"
EXPERIENCE="$WS/data/recap_straw_into_cup"
ENCODED="$WS/experiments/lingbot_recap_straw_cup_encoded_20260906"
AUTO="$WS/experiments/lingbot_recap_straw_cup_auto_context_20260906"
OUTPUT="$WS/experiments/lingbot_recap_straw_cup_intervention_residual_v4_20260906"
LOGS="$WS/logs"

export PYTHONPATH="$CODE:${PYTHONPATH:-}"
mkdir -p "$ENCODED" "$AUTO" "$OUTPUT" "$LOGS"

COMMON=(
  --experience-root "$EXPERIENCE"
  --lingbot-root "$WS/code/lingbot"
  --model-path "$WS/checkpoints/newdata_yellow_beside_milk_teacher_cumulative_epoch90_merged_20260902"
  --training-config "$WS/experiments/newdata_yellow_beside_milk_teacher_continue_ep61_to90_lr1e4_4h20_20260902_retry/lingbotvla_cli.yaml"
  --norm-stats "$WS/norm_stats/all_data_train776_selectorfix_20260829.json"
  --bottleneck "$WS/experiments/lingbot_recap_visual_bottleneck_512d_3h20_20260906/best.pt"
  --world-size 3
  --stride 4
)

run_distributed_stage() {
  local tool="$1"
  local output="$2"
  local log_prefix="$3"
  local pids=()
  for rank in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$((rank + 1)) "$PY" "$CODE/tools/$tool" \
      "${COMMON[@]}" --output-root "$output" --rank "$rank" \
      >"$LOGS/${log_prefix}_rank${rank}_20260906.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

run_distributed_stage encode_recap_experience.py "$ENCODED" recap_human_encode_v4
run_distributed_stage encode_recap_auto_context.py "$AUTO" recap_auto_encode_v4

CUDA_VISIBLE_DEVICES=1 "$PY" "$CODE/tools/train_intervention_residual.py" \
  --encoded-root "$ENCODED" \
  --auto-root "$AUTO" \
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
    episode_20260906_213358_e6d2aa9f.complete \
    episode_20260906_214831_55aef2eb.complete \
    episode_20260906_215652_c78f3cdb.complete \
  2>&1 | tee "$LOGS/recap_intervention_residual_v4_20260906.log"
