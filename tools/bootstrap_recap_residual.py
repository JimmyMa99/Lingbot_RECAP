#!/usr/bin/env python3
"""Commit encoded experience and bootstrap the two phase residual learners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lingbot_recap.experience_replay import replay_batch
from lingbot_recap.online_rl import OnlineRLConfig
from lingbot_recap.rl_learner import LearnerConfig, PhaseLearner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoded-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--held-out", nargs="*", default=[])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    metadata_paths = sorted(args.encoded_root.glob("shard_*/episode_*.complete.json"))
    if len(metadata_paths) != 26:
        raise RuntimeError(f"expected 26 encoded episodes, found {len(metadata_paths)}")
    held_out = set(args.held_out)
    config = OnlineRLConfig()
    learner_config = LearnerConfig()
    learners = {
        phase: PhaseLearner(phase, args.output_root, config, learner_config, device=args.device)
        for phase in ("grasp", "place")
    }
    held_batches: dict[str, list[dict[str, np.ndarray]]] = {"grasp": [], "place": []}
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text())
        episode = metadata["episode"]
        success = metadata["outcome"] == "success"
        for phase in ("grasp", "place"):
            path = metadata_path.parent / metadata["phases"][phase]["file"]
            with np.load(path, allow_pickle=False) as value:
                partial = {key: value[key] for key in ("state", "action", "reference")}
            batch = replay_batch(**partial, success=success)
            if episode in held_out:
                held_batches[phase].append(batch)
            else:
                learners[phase].commit(
                    commit_id=f"{episode}:{phase}", collection_mode="warmup",
                    success=success, batch=batch, operator_note="sealed RECAP intervention",
                )

    report = {"schema_version": 1, "held_out": sorted(held_out), "phases": {}}
    for phase, learner in learners.items():
        before = learner.summary()
        result = learner.bootstrap()
        states = np.concatenate([item["state"] for item in held_batches[phase]])
        refs = np.concatenate([item["reference"] for item in held_batches[phase]])
        action, mean = [], []
        for start in range(0, len(states), 256):
            state_tensor = torch.from_numpy(states[start : start + 256]).to(learner.agent.device)
            ref_tensor = torch.from_numpy(refs[start : start + 256]).to(learner.agent.device)
            with torch.inference_mode():
                pred = learner.agent.actor.mean(state_tensor, ref_tensor)
                action.append(pred.cpu().numpy())
                mean.append(torch.minimum(learner.agent.q1(state_tensor, pred), learner.agent.q2(state_tensor, pred)).cpu().numpy())
        predicted = np.concatenate(action)
        q_values = np.concatenate(mean)
        delta = predicted - refs
        metrics = {
            "train_before": before,
            "train_after": result["status"],
            "held_out_transitions": int(len(states)),
            "held_out_residual_abs_mean": float(np.abs(delta).mean()),
            "held_out_residual_abs_max": float(np.abs(delta).max()),
            "held_out_q_mean": float(q_values.mean()),
            "all_finite": bool(np.isfinite(predicted).all() and np.isfinite(q_values).all()),
        }
        if not metrics["all_finite"] or metrics["held_out_residual_abs_max"] > config.residual_limit + 1e-5:
            raise FloatingPointError(f"{phase}: held-out safety validation failed: {metrics}")
        report["phases"][phase] = metrics
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "bootstrap_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
