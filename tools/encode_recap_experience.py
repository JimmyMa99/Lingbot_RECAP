#!/usr/bin/env python3
"""Encode sealed RECAP episodes with frozen LingBot and the trained bottleneck."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lingbot_recap.experience_replay import (
    chunk_actions,
    decision_indices,
    find_grasp_boundary,
    joint_vector,
    phase_slices,
)
from lingbot_recap.lingbot_features import LingBotVisualTokenCapture
from lingbot_recap.online_rl import make_state_feature
from lingbot_recap.visual_token import VisualTokenBottleneck, VisualTokenConfig


def atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def load_frames(episode: Path) -> list[dict]:
    frames = [json.loads(line) for line in (episode / "frames.jsonl").read_text().splitlines()]
    frames = [frame for frame in frames if frame.get("control_mode") == "human"]
    if len(frames) < 24:
        raise RuntimeError(f"{episode.name}: too few human intervention frames")
    for expected, frame in enumerate(frames):
        if not frame.get("executed_action") or not frame.get("images"):
            raise RuntimeError(f"{episode.name}: incomplete human frame {expected}")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experience-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lingbot-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--bottleneck", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--task", default="把吸管放进杯子里")
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size:
        raise SystemExit("rank must be in [0, world_size)")

    os.environ["LINGBOT_TRAINING_CONFIG"] = str(args.training_config.resolve())
    sys.path.insert(0, str(args.lingbot_root.resolve()))
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

    server = LingbotVLAv2Server(
        str(args.model_path), robot_norm_path=str(args.norm_stats), use_length=16,
        chunk_ret=True, use_bf16=True, use_fp32=False, use_compile=False,
    )
    os.chdir(args.lingbot_root)
    server.reset("so_arm101")
    capture = LingBotVisualTokenCapture(server)
    checkpoint = torch.load(args.bottleneck, map_location="cpu", weights_only=False)
    config = VisualTokenConfig(**checkpoint["config"])
    bottleneck = VisualTokenBottleneck(config).cuda().eval()
    bottleneck.load_state_dict(checkpoint["model"], strict=True)

    episodes = sorted(args.experience_root.glob("episode_*.complete"))[args.rank :: args.world_size]
    shard = args.output_root / f"shard_{args.rank:02d}"
    shard.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for ordinal, episode in enumerate(episodes, 1):
        marker = shard / f"{episode.name}.json"
        if marker.exists():
            print(f"rank={args.rank} skip {episode.name}", flush=True)
            continue
        result = json.loads((episode / "result.json").read_text())
        outcome = result["outcome"]
        if outcome not in {"success", "failure"}:
            raise RuntimeError(f"{episode.name}: unsupported outcome {outcome}")
        frames = load_frames(episode)
        positions = np.stack([joint_vector(frame["observation"]["state"]) for frame in frames])
        actions = np.stack([joint_vector(frame["executed_action"]) for frame in frames])
        timestamps = np.asarray([frame["timestamp"] for frame in frames], dtype=np.float64)
        velocity = np.zeros_like(positions)
        dt = np.maximum(np.diff(timestamps), 1e-3)
        velocity[1:] = np.diff(positions, axis=0) / dt[:, None]
        boundary = find_grasp_boundary(actions[:, -1])
        phase_metadata = {}
        for phase in phase_slices(len(frames), boundary):
            indices = decision_indices(phase, args.stride)
            states, references = [], []
            for index in indices:
                frame = frames[int(index)]
                observation = {
                    "observation.images.top": np.asarray(
                        Image.open(episode / frame["images"]["top"]).convert("RGB"), dtype=np.uint8
                    ),
                    "observation.images.wrist": np.asarray(
                        Image.open(episode / frame["images"]["wrist"]).convert("RGB"), dtype=np.uint8
                    ),
                    "observation.state": positions[index].astype(np.float32),
                    "task": args.task,
                }
                features = capture.infer(observation)
                raw = torch.from_numpy(features.visual_tokens).cuda()
                valid = torch.ones(raw.shape[:2], dtype=torch.bool, device="cuda")
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    visual = bottleneck.encode(raw, valid)[0].float().cpu().numpy()
                states.append(make_state_feature(visual, positions[index], velocity[index], feature_dim=512))
                reference = features.normalized_action
                if len(reference) < 16:
                    reference = np.concatenate((reference, np.repeat(reference[-1:], 16 - len(reference), axis=0)))
                references.append(reference[:16])
            payload = {
                "state": np.stack(states).astype(np.float32),
                "action": chunk_actions(actions, indices, phase, 16),
                "reference": np.stack(references).astype(np.float32),
                "frame_indices": indices.astype(np.int64),
            }
            target = shard / f"{episode.name}.{phase.name}.npz"
            atomic_npz(target, **payload)
            phase_metadata[phase.name] = {"file": target.name, "transitions": len(indices)}
        metadata = {
            "schema_version": 1, "episode": episode.name, "outcome": outcome,
            "human_frames": len(frames), "grasp_boundary": boundary,
            "stride": args.stride, "phases": phase_metadata,
        }
        temporary = marker.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, marker)
        elapsed = time.time() - started
        print(
            f"rank={args.rank} {ordinal}/{len(episodes)} {episode.name} "
            f"frames={len(frames)} split={boundary} elapsed={elapsed:.1f}s",
            flush=True,
        )
    print(f"rank={args.rank} COMPLETE episodes={len(episodes)}", flush=True)


if __name__ == "__main__":
    main()
