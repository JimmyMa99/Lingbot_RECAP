#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/mnt/workspace/users/zb/lingbot_ws
RECAP_ROOT="$WORKSPACE/code/Lingbot_RECAP-rlt-test"
PYTHON="$WORKSPACE/.venv/bin/python"
CACHE_ROOT="$WORKSPACE/experiments/lingbot_recap_visual_cache_all776_16000_3h20_20260906"
OUTPUT_ROOT="$WORKSPACE/experiments/lingbot_recap_visual_bottleneck_512d_3h20_20260906"

mkdir -p "$OUTPUT_ROOT"
export CUDA_VISIBLE_DEVICES=1,2,3
export PYTHONPATH="$RECAP_ROOT"
export OMP_NUM_THREADS=8

exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=3 \
  "$RECAP_ROOT/tools/train_visual_bottleneck.py" \
  --cache-root "$CACHE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --epochs 10 \
  --batch-size 32 \
  --workers 6 \
  --lr 2e-4
