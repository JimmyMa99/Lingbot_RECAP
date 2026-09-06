#!/usr/bin/env python3
"""Train and validate the 512D LingBot visual-token bottleneck with DDP."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from lingbot_recap.visual_token import VisualTokenBottleneck, VisualTokenConfig


class TokenDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as value:
            tokens = np.asarray(value["visual_tokens"], dtype=np.float32)
        if tokens.ndim == 3 and tokens.shape[0] == 1:
            tokens = tokens[0]
        if tokens.ndim != 2 or tokens.shape != (128, 2560):
            raise RuntimeError(f"token shape mismatch: {path}: {tokens.shape}")
        if not np.isfinite(tokens).all():
            raise FloatingPointError(f"non-finite visual token: {path}")
        return torch.from_numpy(tokens)


def atomic_save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def reduce_stats(values: list[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor)
    return tensor.cpu().tolist()


def evaluate(model, loader, device, dtype):
    model.eval()
    loss_sum = ratio_sum = token_rms_sum = count = 0.0
    with torch.inference_mode():
        for tokens in loader:
            tokens = tokens.to(device, non_blocking=True)
            valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=device)
            with torch.autocast("cuda", dtype=dtype):
                result = model(tokens, valid)
                ratio = model.module.zero_token_loss_ratio(tokens, valid) if hasattr(model, "module") else model.zero_token_loss_ratio(tokens, valid)
            batch = float(tokens.shape[0])
            loss_sum += float(result["loss"]) * batch
            ratio_sum += float(ratio) * batch
            token_rms_sum += float(result["token_rms"]) * batch
            count += batch
    loss_sum, ratio_sum, token_rms_sum, count = reduce_stats(
        [loss_sum, ratio_sum, token_rms_sum, count], device
    )
    return {
        "val_loss": loss_sum / count,
        "zero_token_loss_ratio": ratio_sum / count,
        "val_token_rms": token_rms_sum / count,
        "val_samples": int(count),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32, help="per GPU")
    parser.add_argument("--workers", type=int, default=6, help="per GPU")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank() if dist.is_initialized() else 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    paths = sorted(args.cache_root.glob("shard_*/sample_*.npz"))
    if len(paths) != 16000:
        raise RuntimeError(f"expected 16000 cached samples, found {len(paths)}")
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    val_count = max(1, round(len(paths) * 0.05))
    val_paths, train_paths = paths[:val_count], paths[val_count:]
    train_set, val_set = TokenDataset(train_paths), TokenDataset(val_paths)
    train_sampler = DistributedSampler(train_set, shuffle=True, seed=args.seed) if world_size > 1 else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if world_size > 1 else None
    common = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                  persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_set, sampler=train_sampler, shuffle=train_sampler is None,
                              drop_last=True, **common)
    val_loader = DataLoader(val_set, sampler=val_sampler, shuffle=False, drop_last=False, **common)

    config = VisualTokenConfig(input_dim=2560, token_dim=512, max_tokens=128)
    model = VisualTokenBottleneck(config).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.1
    )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    args.output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_root / "metrics.jsonl"
    start_epoch = 0
    best_val = float("inf")
    latest = args.output_root / "latest.pt"
    if latest.exists():
        value = torch.load(latest, map_location="cpu", weights_only=False)
        target = model.module if hasattr(model, "module") else model
        target.load_state_dict(value["model"], strict=True)
        optimizer.load_state_dict(value["optimizer"])
        scheduler.load_state_dict(value["scheduler"])
        start_epoch = int(value["epoch"])
        best_val = float(value["best_val"])

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        loss_sum = count = 0.0
        started = time.time()
        for step, tokens in enumerate(train_loader, 1):
            tokens = tokens.to(device, non_blocking=True)
            valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype):
                result = model(tokens, valid)
            loss = result["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite bottleneck loss at epoch={epoch + 1} step={step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite grad norm at epoch={epoch + 1} step={step}")
            optimizer.step()
            loss_sum += float(loss.detach()) * tokens.shape[0]
            count += tokens.shape[0]
            if rank == 0 and (step == 1 or step % 25 == 0):
                print(f"epoch={epoch + 1}/{args.epochs} step={step}/{len(train_loader)} loss={float(loss):.6f} grad={float(grad_norm):.4f}", flush=True)
        loss_sum, count = reduce_stats([loss_sum, count], device)
        metrics = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / count,
            **evaluate(model, val_loader, device, dtype),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.time() - started,
            "world_size": world_size,
            "global_batch_size": args.batch_size * world_size,
        }
        scheduler.step()
        if rank == 0:
            target = model.module if hasattr(model, "module") else model
            improved = metrics["val_loss"] < best_val
            best_val = min(best_val, metrics["val_loss"])
            checkpoint = {
                "schema_version": 1,
                "epoch": epoch + 1,
                "config": config.to_dict(),
                "model": target.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val": best_val,
                "metrics": metrics,
                "seed": args.seed,
            }
            atomic_save(args.output_root / f"epoch_{epoch + 1:02d}.pt", checkpoint)
            atomic_save(latest, checkpoint)
            if improved:
                atomic_save(args.output_root / "best.pt", checkpoint)
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
        if dist.is_initialized():
            dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
