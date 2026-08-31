#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/mnt/workspace/users/zb/lingbot_ws}"
RUN="${RUN:-${BASE}/experiments/mopd_offline_bootstrap_single_gpu_20260831}"
REGISTRY="${REGISTRY:-${BASE}/code/Lingbot_RECAP/configs/multi_policy_teachers.h20.local.json}"
IMPORT_SESSION="${IMPORT_SESSION:-mopd_offline_import_20260831}"
export PATH="${BASE}/.venv/bin:${PATH}"
export PYTHONPATH="${BASE}/code/Lingbot_RECAP"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/blas:/usr/lib/x86_64-linux-gnu/lapack:${LD_LIBRARY_PATH:-}"
export QWEN3VL_PATH="${BASE}/models/Qwen3-VL-4B-Instruct"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

mkdir -p "${RUN}/logs"

while tmux has-session -t "${IMPORT_SESSION}" 2>/dev/null; do
  sleep 10
done
if grep -Eq "Traceback|Error|Exception" "${RUN}/import.log"; then
  echo "offline import failed; see ${RUN}/import.log" >&2
  exit 1
fi

wait_health() {
  local port="$1"
  for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${port}/healthz" | grep -q '"model_loaded":true'; then
      return 0
    fi
    sleep 2
  done
  return 1
}

temporary_teacher_pids=()
cleanup_teachers() {
  local pid
  for pid in "${temporary_teacher_pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${temporary_teacher_pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  temporary_teacher_pids=()
}
trap cleanup_teachers EXIT

if ! wait_health 18007; then
  echo "yellow teacher on port 18007 is unavailable" >&2
  exit 1
fi
start_teacher() {
  local checkpoint="$1" port="$2" log="$3"
  cd "${BASE}/code/lingbot"
  CUDA_VISIBLE_DEVICES=7 python -u server_http.py \
    --model_path "${checkpoint}" --host 127.0.0.1 --port "${port}" --use_length 16 \
    >"${log}" 2>&1 &
  temporary_teacher_pids+=("$!")
  wait_health "${port}"
}

start_teacher \
  "${BASE}/checkpoints/newdata_yellow_beside_milk_from_good_ep8_aug_lr1e4_epoch30_merged_20260831" \
  18008 "${RUN}/logs/teacher_milk.log"
start_teacher \
  "${BASE}/checkpoints/newdata_ducks_into_box_from_good_ep8_aug_lr1e4_epoch30_merged_20260831" \
  18009 "${RUN}/logs/teacher_box.log"

python -m lingbot_recap.cli relabel \
  --teacher-registry "${REGISTRY}" \
  --experience-root "${RUN}/experience/yellow" \
  --include-offline-demonstrations &
yellow_label_pid="$!"
python -m lingbot_recap.cli relabel \
  --teacher-registry "${REGISTRY}" \
  --experience-root "${RUN}/experience/milk" \
  --include-offline-demonstrations &
milk_label_pid="$!"
python -m lingbot_recap.cli relabel \
  --teacher-registry "${REGISTRY}" \
  --experience-root "${RUN}/experience/box" \
  --include-offline-demonstrations &
box_label_pid="$!"
wait "${yellow_label_pid}"
wait "${milk_label_pid}"
wait "${box_label_pid}"
cleanup_teachers

mkdir -p "${RUN}/experience/all"
for task_root in yellow milk box; do
  for episode in "${RUN}/experience/${task_root}"/episode_*.complete; do
    ln -sfn "${episode}" "${RUN}/experience/all/$(basename "${episode}")"
  done
done

python -m lingbot_recap.cli export-distill \
  --experience-root "${RUN}/experience/all" \
  --output-root "${RUN}/distilled_lerobot" \
  --repo-id mzm/mopd_offline_bootstrap_20260831

: >"${RUN}/train_manifest.txt"
for _ in $(seq 1 10); do
  echo "so_arm101 ${RUN}/distilled_lerobot" >>"${RUN}/train_manifest.txt"
done
cat "${BASE}/configs/all_data_train776.txt" >>"${RUN}/train_manifest.txt"
cat "${BASE}/configs/newdata_pick_yellow_eval_20260829.txt" \
    "${BASE}/configs/newdata_yellow_beside_milk_eval_20260829.txt" \
    "${BASE}/configs/newdata_ducks_into_box_eval_20260829.txt" \
    >"${RUN}/eval_manifest.txt"

cd "${BASE}/code/lingbot"
export CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=62641
bash train.sh tasks/vla/train_lingbotvla_eval.py \
  "${BASE}/code/Lingbot_RECAP/configs/mopd_offline_bootstrap_h20_single_gpu.yaml" \
  2>&1 | tee "${RUN}/logs/student_train_gpu0.log"
