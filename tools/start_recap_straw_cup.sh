#!/usr/bin/env bash
set -euo pipefail

# Operation-machine launcher for one RECAP rollout episode.
# The experience root is reusable: the collector creates a new timestamped
# episode directory for every invocation.

RECAP_BIN="/home/mzm/miniconda3/envs/lerobot/bin/lingbot-recap"
RECAP_ROOT="/home/mzm/code/Lingbot_RECAP_rlt"
SERVER="${RECAP_SERVER:-http://127.0.0.1:8010}"
TASK="${RECAP_TASK:-把吸管放进杯子里}"
EXPERIENCE_ROOT="${RECAP_EXPERIENCE_ROOT:-/home/mzm/lerobot_data/recap_straw_into_cup}"
POLICY_CHECKPOINT="${RECAP_POLICY_CHECKPOINT:-/mnt/workspace/users/zb/lingbot_ws/checkpoints/newdata_yellow_beside_milk_teacher_cumulative_epoch90_merged_20260902}"

FOLLOWER_PORT="/dev/ttyACM1"
LEADER_PORT="/dev/ttyACM0"
TOP_CAMERA="/dev/video2"
WRIST_CAMERA="/dev/video1"
KEYPAD_CONFIG="${RECAP_ROOT}/configs/y03_recap.local.json"

for path in \
  "${RECAP_BIN}" \
  "${FOLLOWER_PORT}" \
  "${LEADER_PORT}" \
  "${TOP_CAMERA}" \
  "${WRIST_CAMERA}" \
  "${KEYPAD_CONFIG}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[RECAP] 缺少设备或文件: ${path}" >&2
    exit 1
  fi
done

if [[ ! -r /dev/input/by-id/usb-SDCINNOVATION_Y03_313504190974-event-kbd ]]; then
  echo "[RECAP] 当前用户无权读取三键键盘；请重新 SSH 登录后再试。" >&2
  exit 1
fi

if fuser "${FOLLOWER_PORT}" "${LEADER_PORT}" >/dev/null 2>&1; then
  echo "[RECAP] 机械臂串口正被其他进程占用：" >&2
  fuser -v "${FOLLOWER_PORT}" "${LEADER_PORT}" >&2 || true
  exit 1
fi

echo "[RECAP] 检查 LingBot 服务: ${SERVER}"
curl --fail --silent --show-error "${SERVER}/healthz"
echo
echo "[RECAP] 任务: ${TASK}"
echo "[RECAP] top=${TOP_CAMERA}, wrist=${WRIST_CAMERA}"
echo "[RECAP] 数据目录: ${EXPERIENCE_ROOT}"
echo "[RECAP] 启动后策略会控制 follower；请确认工作区清空、急停/电源可触达。"

cd "${RECAP_ROOT}"
exec "${RECAP_BIN}" collect \
  --server "${SERVER}" \
  --task "${TASK}" \
  --policy-checkpoint "${POLICY_CHECKPOINT}" \
  --experience-root "${EXPERIENCE_ROOT}" \
  --follower-port "${FOLLOWER_PORT}" \
  --leader-port "${LEADER_PORT}" \
  --top-camera "${TOP_CAMERA}" \
  --wrist-camera "${WRIST_CAMERA}" \
  --three-button-config "${KEYPAD_CONFIG}"
