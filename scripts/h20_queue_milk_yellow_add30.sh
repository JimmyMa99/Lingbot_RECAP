#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/mnt/workspace/users/zb/lingbot_ws}"
GPUS="${GPUS:-1,2,4,7}"
BOX_SESSION="${BOX_SESSION:-box_teacher_continue_61to90_4h20}"
BOX_FINAL="${BASE}/experiments/newdata_ducks_into_box_teacher_continue_ep61_to90_lr1e4_4h20_20260901/epoch_adapters/epoch_30_step_8970.pt"
SOURCE_CONFIG="${BASE}/configs/box_teacher_continue_ep61_to90_4h20_20260901.yaml"
QUEUE_LOG="${BASE}/logs/milk_yellow_add30_queue_20260901.log"

export PATH="${BASE}/.venv/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/blas:/usr/lib/x86_64-linux-gnu/lapack:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

log() {
  echo "$(date -Is) $*" | tee -a "${QUEUE_LOG}"
}

while tmux has-session -t "${BOX_SESSION}" 2>/dev/null; do
  sleep 60
done
if [[ ! -s "${BOX_FINAL}" ]]; then
  log "box training did not produce ${BOX_FINAL}; refusing to consume its GPUs"
  exit 1
fi

make_config() {
  local model_path="$1" train_path="$2" eval_path="$3" output_dir="$4" target="$5"
  sed \
    -e "s|^  model_path: .*|  model_path: ${model_path}|" \
    -e "s|^  train_path: .*|  train_path: ${train_path}|" \
    -e "s|^  eval_path: .*|  eval_path: ${eval_path}|" \
    -e "s|^  output_dir: .*|  output_dir: ${output_dir}|" \
    "${SOURCE_CONFIG}" >"${target}"
}

MILK_CONFIG="${BASE}/configs/milk_teacher_continue_ep31_to60_4h20_20260901.yaml"
MILK_OUTPUT="${BASE}/experiments/newdata_yellow_beside_milk_teacher_continue_ep31_to60_lr1e4_4h20_20260901"
MILK_LOG="${BASE}/logs/milk_teacher_continue_ep31_to60_lr1e4_4h20_20260901.log"
make_config \
  "${BASE}/checkpoints/newdata_yellow_beside_milk_from_good_ep8_aug_lr1e4_epoch30_merged_20260831" \
  "${BASE}/configs/newdata_yellow_beside_milk_train_20260829.txt" \
  "${BASE}/configs/newdata_yellow_beside_milk_eval_20260829.txt" \
  "${MILK_OUTPUT}" "${MILK_CONFIG}"

log "starting milk teacher +30ep on GPUs ${GPUS}"
cd "${BASE}/code/lingbot"
CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT=62664 \
  bash train.sh tasks/vla/train_lingbotvla_eval.py "${MILK_CONFIG}" \
  2>&1 | tee "${MILK_LOG}"
log "milk teacher +30ep complete"

YELLOW_CONFIG="${BASE}/configs/yellow_teacher_continue_ep31_to60_4h20_20260901.yaml"
YELLOW_OUTPUT="${BASE}/experiments/newdata_pick_yellow_teacher_continue_ep31_to60_lr1e4_4h20_20260901"
YELLOW_LOG="${BASE}/logs/yellow_teacher_continue_ep31_to60_lr1e4_4h20_20260901.log"
make_config \
  "${BASE}/checkpoints/newdata_pick_yellow_from_good_ep8_aug_lr1e4_epoch30_merged_20260830" \
  "${BASE}/configs/newdata_pick_yellow_train_20260829.txt" \
  "${BASE}/configs/newdata_pick_yellow_eval_20260829.txt" \
  "${YELLOW_OUTPUT}" "${YELLOW_CONFIG}"

log "starting yellow-pick teacher +30ep on GPUs ${GPUS}"
CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT=62665 \
  bash train.sh tasks/vla/train_lingbotvla_eval.py "${YELLOW_CONFIG}" \
  2>&1 | tee "${YELLOW_LOG}"
log "yellow-pick teacher +30ep complete"
