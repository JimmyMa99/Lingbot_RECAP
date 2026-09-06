#!/usr/bin/env python3
"""多进程分片缓存冻结 LingBot 的 prefix 视觉 token。"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lingbot-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--max-samples", type=int, default=16000)
    parser.add_argument("--robot", default="so_arm101")
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size:
        raise SystemExit("rank 必须位于 [0, world_size)")
    if args.max_samples <= 0:
        raise SystemExit("max_samples 必须为正数")

    os.environ["LINGBOT_TRAINING_CONFIG"] = str(args.training_config.resolve())
    sys.path.insert(0, str(args.lingbot_root.resolve()))
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    from lingbotvla.data.dataset import build_vla_dataset
    from lingbot_recap.lingbot_features import extract_visual_tokens_from_transformed

    server = LingbotVLAv2Server(
        str(args.model_path),
        robot_norm_path=str(args.norm_stats),
        use_length=16,
        chunk_ret=True,
        use_bf16=True,
        use_fp32=False,
        use_compile=False,
    )
    # 官方 reset() 仍以当前工作目录解析 configs/robot_configs。
    os.chdir(args.lingbot_root)
    server.reset(args.robot)
    data_config = SimpleNamespace(**vars(server.data_config))
    data_config.train_path = str(args.data_manifest)
    data_config.chunk_size = int(server.config.chunk_size)
    data_config.image_augment = False
    data_config.use_future_image = False
    model_config = SimpleNamespace(tokenizer_path=server.config.tokenizer_path)
    dataset = build_vla_dataset(
        dataset_config=data_config,
        model_config=model_config,
        config=server.config,
        processor=server.processor,
        use_depth_align=False,
    )
    total = len(dataset)
    sample_count = min(total, args.max_samples)
    if sample_count == total:
        indices = np.arange(total, dtype=np.int64)
    else:
        indices = np.linspace(0, total - 1, sample_count, dtype=np.int64)
        indices = np.unique(indices)
    indices = indices[args.rank :: args.world_size]

    shard = args.output_root / f"shard_{args.rank:02d}"
    shard.mkdir(parents=True, exist_ok=True)
    records = shard / "manifest.jsonl"
    started = time.time()
    completed = 0
    with records.open("a", encoding="utf-8") as manifest_stream:
        for ordinal, dataset_index in enumerate(indices):
            target = shard / f"sample_{int(dataset_index):08d}.npz"
            if target.exists():
                completed += 1
                continue
            item = dataset[int(dataset_index)]
            if isinstance(item, list):
                if len(item) != 1:
                    raise RuntimeError("缓存器只支持单样本 item")
                item = item[0]
            tokens = extract_visual_tokens_from_transformed(server.vla.model, item)
            atomic_npz(target, visual_tokens=tokens.astype(np.float16))
            record = {
                "dataset_index": int(dataset_index),
                "file": target.name,
                "shape": list(tokens.shape),
                "finite": bool(np.isfinite(tokens).all()),
                "std": float(tokens.std()),
                "rank": args.rank,
            }
            manifest_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest_stream.flush()
            completed += 1
            if completed == 1 or completed % 25 == 0:
                rate = completed / max(time.time() - started, 1e-6)
                eta = (len(indices) - completed) / max(rate, 1e-9)
                print(
                    f"rank={args.rank} {completed}/{len(indices)} "
                    f"rate={rate:.3f}/s eta={math.ceil(eta)}s shape={tokens.shape}",
                    flush=True,
                )
    print(f"rank={args.rank} COMPLETE {completed}/{len(indices)}", flush=True)


if __name__ == "__main__":
    main()
