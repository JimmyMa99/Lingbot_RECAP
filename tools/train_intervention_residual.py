#!/usr/bin/env python3
"""Supervised warm-start of bounded residual actors from human intervention deltas."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from lingbot_recap.online_rl import OnlineRLConfig, ResidualChunkActor


class ResidualDataset(Dataset):
    def __init__(self, state, reference, action, intervention):
        self.state = torch.from_numpy(np.asarray(state, dtype=np.float32))
        self.reference = torch.from_numpy(np.asarray(reference, dtype=np.float32))
        self.action = torch.from_numpy(np.asarray(action, dtype=np.float32))
        self.intervention = torch.from_numpy(np.asarray(intervention, dtype=np.float32))

    def __len__(self):
        return len(self.state)

    def __getitem__(self, index):
        return self.state[index], self.reference[index], self.action[index], self.intervention[index]


def load_phase(encoded_root: Path, auto_root: Path, phase: str, held_out: set[str]):
    splits = {"train": [[], [], [], []], "val": [[], [], [], []]}
    successful_episodes = set()
    for metadata_path in sorted(encoded_root.glob("shard_*/episode_*.complete.json")):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("outcome") != "success":
            continue
        successful_episodes.add(metadata["episode"])
        split = "val" if metadata["episode"] in held_out else "train"
        with np.load(metadata_path.parent / metadata["phases"][phase]["file"], allow_pickle=False) as value:
            count = len(value["state"])
            for bucket, key in zip(splits[split][:3], ("state", "reference", "action"), strict=True):
                bucket.append(value[key])
            splits[split][3].append(np.ones(count, dtype=np.float32))
    if phase == "grasp":
        for metadata_path in sorted(auto_root.glob("shard_*/episode_*.complete.auto.json")):
            metadata = json.loads(metadata_path.read_text())
            if metadata["episode"] not in successful_episodes:
                continue
            split = "val" if metadata["episode"] in held_out else "train"
            with np.load(metadata_path.parent / metadata["file"], allow_pickle=False) as value:
                count = len(value["state"])
                for bucket, key in zip(splits[split][:3], ("state", "reference", "action"), strict=True):
                    bucket.append(value[key])
                splits[split][3].append(np.zeros(count, dtype=np.float32))
    return {
        split: tuple(np.concatenate(items) for items in values)
        for split, values in splits.items()
    }


@torch.inference_mode()
def evaluate(model, dataset, device, limits):
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    sums = {"loss": 0.0, "pred_mae": 0.0, "teacher_mae": 0.0, "count": 0}
    auto_residual, human_cosine = [], []
    model.eval()
    for state, reference, action, intervention in loader:
        state, reference, action = state.to(device), reference.to(device), action.to(device)
        predicted = model.mean(state, reference)
        delta = action - reference
        target = torch.clamp(reference + torch.maximum(torch.minimum(delta, limits), -limits), -1, 1)
        count = len(state)
        sums["loss"] += float(F.smooth_l1_loss(predicted, target, reduction="sum"))
        sums["pred_mae"] += float((predicted - action).abs().sum())
        sums["teacher_mae"] += float((reference - action).abs().sum())
        sums["count"] += count * action.shape[1] * action.shape[2]
        delta = (predicted - reference).flatten(1)
        target_delta = (action - reference).flatten(1)
        auto = intervention < 0.5
        human = ~auto
        if auto.any():
            auto_residual.append(delta[auto].abs().amax(1).cpu())
        if human.any():
            human_cosine.append(F.cosine_similarity(delta[human], target_delta[human], dim=1).cpu())
    return {
        "loss": sums["loss"] / sums["count"],
        "action_mae": sums["pred_mae"] / sums["count"],
        "teacher_action_mae": sums["teacher_mae"] / sums["count"],
        "auto_residual_abs_max_p95": float(torch.cat(auto_residual).quantile(.95)) if auto_residual else None,
        "human_delta_cosine_mean": float(torch.cat(human_cosine).mean()) if human_cosine else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoded-root", type=Path, required=True)
    parser.add_argument("--auto-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--held-out", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--residual-limit", type=float, default=.20)
    parser.add_argument("--gripper-residual-limit", type=float, default=.60)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda")
    held_out = set(args.held_out)
    args.output_root.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "held_out": sorted(held_out), "phases": {}}
    for phase in ("grasp", "place"):
        data = load_phase(args.encoded_root, args.auto_root, phase, held_out)
        train_set = ResidualDataset(*data["train"])
        val_set = ResidualDataset(*data["val"])
        intervention = data["train"][3]
        # Give pre-takeover zero-residual context and human corrections equal total mass.
        if np.any(intervention == 0) and np.any(intervention == 1):
            weights = np.where(
                intervention > .5, .5 / max(1, (intervention > .5).sum()),
                .5 / max(1, (intervention < .5).sum()),
            )
            sampler = WeightedRandomSampler(torch.from_numpy(weights).double(), len(train_set), replacement=True)
            loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler)
        else:
            loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        config = OnlineRLConfig(
            residual_limit=args.residual_limit,
            gripper_residual_limit=args.gripper_residual_limit,
            reference_dropout=.25,
        )
        model = ResidualChunkActor(config).to(device)
        limits = torch.tensor(
            [args.residual_limit] * 5 + [args.gripper_residual_limit],
            dtype=torch.float32,
            device=device,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * .1)
        phase_root = args.output_root / phase
        phase_root.mkdir(parents=True, exist_ok=True)
        metrics_path = phase_root / "metrics.jsonl"
        best_loss = float("inf")
        for epoch in range(1, args.epochs + 1):
            model.train(); loss_sum = 0.0; count = 0
            for state, reference, action, _intervention in loader:
                state, reference, action = state.to(device), reference.to(device), action.to(device)
                delta = action - reference
                target = torch.clamp(
                    reference + torch.maximum(torch.minimum(delta, limits), -limits), -1, 1
                )
                predicted = model.mean(state, reference, apply_reference_dropout=True)
                loss = F.smooth_l1_loss(predicted, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{phase} epoch {epoch}: non-finite loss")
                optimizer.zero_grad(set_to_none=True); loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(grad):
                    raise FloatingPointError(f"{phase} epoch {epoch}: non-finite grad")
                optimizer.step(); loss_sum += float(loss) * len(state); count += len(state)
            scheduler.step()
            metrics = {
                "epoch": epoch, "train_loss": loss_sum / count,
                **{f"val_{k}": v for k, v in evaluate(model, val_set, device, limits).items()},
                "lr": optimizer.param_groups[0]["lr"], "train_samples": len(train_set),
                "val_samples": len(val_set), "all_finite": True,
            }
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            checkpoint = {
                "schema_version": 2, "kind": "supervised_intervention_residual",
                "phase": phase, "epoch": epoch, "config": config.to_dict(),
                "actor": model.state_dict(), "optimizer": optimizer.state_dict(),
                "metrics": metrics,
            }
            torch.save(checkpoint, phase_root / "latest.pt")
            if metrics["val_loss"] < best_loss:
                best_loss = metrics["val_loss"]
                torch.save(checkpoint, phase_root / "best.pt")
            if epoch == 1 or epoch % 20 == 0:
                print(json.dumps({"phase": phase, **metrics}), flush=True)
        report["phases"][phase] = torch.load(phase_root / "best.pt", map_location="cpu", weights_only=False)["metrics"]
    (args.output_root / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    (args.output_root / "COMPLETE").touch()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
