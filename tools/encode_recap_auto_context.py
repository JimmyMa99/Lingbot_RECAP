#!/usr/bin/env python3
"""Encode pre-takeover frames as zero-residual RECAP context."""

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

from lingbot_recap.experience_replay import joint_vector
from lingbot_recap.lingbot_features import LingBotVisualTokenCapture
from lingbot_recap.online_rl import SO101ActionCodec, make_state_feature
from lingbot_recap.visual_token import VisualTokenBottleneck, VisualTokenConfig


def atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


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
    parser.add_argument("--anchor-decisions", type=int, default=4)
    parser.add_argument("--task", default="把吸管放进杯子里")
    args = parser.parse_args()

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
        target = shard / f"{episode.name}.auto.npz"
        marker = shard / f"{episode.name}.auto.json"
        if target.exists() and marker.exists():
            continue
        all_frames = [json.loads(line) for line in (episode / "frames.jsonl").read_text().splitlines()]
        frames = [frame for frame in all_frames if frame.get("control_mode") == "auto"]
        human_frames = [frame for frame in all_frames if frame.get("control_mode") == "human"]
        if not frames:
            raise RuntimeError(f"{episode.name}: no pre-takeover auto frames")
        if len(human_frames) < 1:
            raise RuntimeError(f"{episode.name}: no human correction frames")
        indices = np.arange(0, len(frames), args.stride, dtype=np.int64)
        if indices[-1] != len(frames) - 1:
            indices = np.append(indices, len(frames) - 1)
        positions = np.stack([joint_vector(frame["observation"]["state"]) for frame in frames])
        timestamps = np.asarray([frame["timestamp"] for frame in frames], dtype=np.float64)
        velocity = np.zeros_like(positions)
        if len(frames) > 1:
            velocity[1:] = np.diff(positions, axis=0) / np.maximum(np.diff(timestamps), 1e-3)[:, None]
        states, references = [], []
        for index in indices:
            frame = frames[int(index)]
            observation = {
                "observation.images.top": np.asarray(
                    Image.open(episode / frame["images"]["top"]).convert("RGB"), dtype=np.uint8
                ).copy(),
                "observation.images.wrist": np.asarray(
                    Image.open(episode / frame["images"]["wrist"]).convert("RGB"), dtype=np.uint8
                ).copy(),
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
        reference_array = np.stack(references).astype(np.float32)
        all_human_actions = np.stack(
            [joint_vector(frame["executed_action"]) for frame in human_frames]
        )
        movement = np.max(np.abs(np.diff(all_human_actions, axis=0)), axis=1)
        moving = np.flatnonzero(movement > 0.5)
        correction_start = max(0, int(moving[0] + 1) - 2) if len(moving) else 0
        human_actions = all_human_actions[correction_start : correction_start + 16]
        if len(human_actions) < 16:
            human_actions = np.concatenate(
                (human_actions, np.repeat(human_actions[-1:], 16 - len(human_actions), axis=0))
            )
        correction_chunk = SO101ActionCodec.normalize(human_actions).astype(np.float32)
        actions = reference_array.copy()
        correction = np.zeros(len(indices), dtype=np.float32)
        anchor_count = min(max(args.anchor_decisions, 0), len(indices))
        if anchor_count:
            # These observations immediately precede intervention.  Label them
            # with what the human actually did next instead of reinforcing the
            # failed teacher action with a zero residual target.
            actions[-anchor_count:] = correction_chunk
            correction[-anchor_count:] = 1.0
        atomic_npz(
            target, state=np.stack(states).astype(np.float32),
            reference=reference_array, action=actions,
            intervention=correction, frame_indices=indices,
        )
        metadata = {
            "schema_version": 1, "episode": episode.name,
            "mode": "pre_takeover_mixed_supervision",
            "source_frames": len(frames), "transitions": len(indices),
            "correction_anchors": anchor_count,
            "human_correction_start": correction_start,
            "file": target.name,
        }
        temporary = marker.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, marker)
        print(
            f"rank={args.rank} {ordinal}/{len(episodes)} {episode.name} "
            f"auto={len(frames)} transitions={len(indices)} elapsed={time.time()-started:.1f}s",
            flush=True,
        )
    print(f"rank={args.rank} COMPLETE episodes={len(episodes)}", flush=True)


if __name__ == "__main__":
    main()
