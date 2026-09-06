#!/usr/bin/env bash
set -euo pipefail

WS=/mnt/workspace/users/zb/lingbot_ws
CODE="$WS/code/Lingbot_RECAP-rlt-test"
PY="$WS/.venv/bin/python"
OUTPUT="$WS/experiments/lingbot_recap_straw_cup_intervention_residual_v4_20260906"
WATCH_LOG="$WS/logs/recap_v4_deploy_watcher_20260906.log"
SERVER_LOG="$WS/logs/recap_residual_v4_server_gpu1_20260906.log"

exec >>"$WATCH_LOG" 2>&1
echo "$(date -Is) waiting for v4 COMPLETE"
while [[ ! -f "$OUTPUT/COMPLETE" ]]; do
  if ! tmux has-session -t recap_intervention_v4_20260906 2>/dev/null; then
    echo "$(date -Is) training session ended without COMPLETE"
    exit 1
  fi
  sleep 10
done

echo "$(date -Is) validating v4 report"
"$PY" - "$OUTPUT/training_report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
for phase in ("grasp", "place"):
    metrics = report["phases"][phase]
    assert metrics["all_finite"], f"{phase}: non-finite metrics"
    assert metrics["val_action_mae"] < metrics["val_teacher_action_mae"], (
        f"{phase}: residual actor does not beat frozen teacher on held-out data"
    )
print(json.dumps(report, ensure_ascii=False))
PY

echo "$(date -Is) deploying v4 on GPU1 port 8011"
tmux kill-session -t recap_residual_server_gpu1 2>/dev/null || true
tmux new-session -d -s recap_residual_server_gpu1 "
  cd '$CODE' &&
  export PYTHONPATH='$CODE' &&
  export CUDA_VISIBLE_DEVICES=1 &&
  exec '$PY' tools/serve_recap_residual.py \\
    --lingbot-root '$WS/code/lingbot' \\
    --model-path '$WS/checkpoints/newdata_yellow_beside_milk_teacher_cumulative_epoch90_merged_20260902' \\
    --training-config '$WS/experiments/newdata_yellow_beside_milk_teacher_continue_ep61_to90_lr1e4_4h20_20260902_retry/lingbotvla_cli.yaml' \\
    --norm-stats '$WS/norm_stats/all_data_train776_selectorfix_20260829.json' \\
    --bottleneck '$WS/experiments/lingbot_recap_visual_bottleneck_512d_3h20_20260906/best.pt' \\
    --residual-root '$OUTPUT' \\
    --residual-scale .25 \\
    --gripper-scale 1.0 \\
    --host 127.0.0.1 \\
    --port 8011 2>&1 | tee '$SERVER_LOG'
"

for _ in $(seq 1 30); do
  if curl -fsS --max-time 2 http://127.0.0.1:8011/healthz; then
    echo
    echo "$(date -Is) v4 deployment ready"
    exit 0
  fi
  sleep 5
done
echo "$(date -Is) v4 server failed readiness timeout"
exit 1
