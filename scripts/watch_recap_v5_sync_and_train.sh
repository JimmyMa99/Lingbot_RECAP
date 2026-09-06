#!/usr/bin/env bash
set -euo pipefail

SYNC_PID_FILE=/home/mzm/recap_sync_v5_20260906.pid
WATCH_LOG=/home/mzm/recap_v5_watcher_20260906.log
SOURCE=/home/mzm/lerobot_data/recap_straw_into_cup/
DEST=root@47.93.14.206:/mnt/workspace/users/zb/lingbot_ws/data/recap_straw_into_cup/
H20_SCRIPT=/mnt/workspace/users/zb/lingbot_ws/code/Lingbot_RECAP-rlt-test/scripts/run_intervention_residual_v5_h20.sh

exec >>"$WATCH_LOG" 2>&1
echo "$(date -Is) waiting for initial rsync"
sync_pid="$(cat "$SYNC_PID_FILE")"
while kill -0 "$sync_pid" 2>/dev/null; do
  sleep 10
done

echo "$(date -Is) verifying data with final rsync pass"
rsync -a --partial -e "ssh -p 1020" "$SOURCE" "$DEST"

echo "$(date -Is) starting H20 v5 pipeline"
ssh -p 1020 root@47.93.14.206 "
  set -e
  count=\$(find /mnt/workspace/users/zb/lingbot_ws/data/recap_straw_into_cup \\
    -maxdepth 1 -type d -name 'episode_*.complete' | wc -l)
  test \"\$count\" -eq 99
  tmux kill-session -t recap_intervention_v5_20260906 2>/dev/null || true
  tmux new-session -d -s recap_intervention_v5_20260906 \\
    'bash $H20_SCRIPT'
  tmux has-session -t recap_intervention_v5_20260906
"
echo "$(date -Is) H20 v5 pipeline dispatched"
