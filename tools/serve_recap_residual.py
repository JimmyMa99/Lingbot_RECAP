#!/usr/bin/env python3
"""Serve frozen LingBot plus phase-specific, bounded RECAP residual actors."""

import argparse
import base64
import io
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lingbot_recap.lingbot_features import LingBotVisualTokenCapture
from lingbot_recap.online_rl import (
    OnlineRLConfig,
    ResidualChunkActor,
    SO101ActionCodec,
    make_state_feature,
)
from lingbot_recap.visual_token import VisualTokenBottleneck, VisualTokenConfig


def load_actor(path: Path, device: torch.device) -> tuple[ResidualChunkActor, OnlineRLConfig]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("kind") == "supervised_intervention_residual":
        config = OnlineRLConfig.from_dict(value["config"])
        actor = ResidualChunkActor(config).to(device).eval()
        actor.load_state_dict(value["actor"], strict=True)
        return actor, config
    if not value.get("bootstrap_completed") or value.get("online_enabled"):
        raise RuntimeError(f"unexpected learner checkpoint state: {path}")
    config = OnlineRLConfig.from_dict(value["agent"]["config"])
    actor = ResidualChunkActor(config).to(device).eval()
    actor.load_state_dict(value["agent"]["actor"], strict=True)
    return actor, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lingbot-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--bottleneck", type=Path, required=True)
    parser.add_argument("--residual-root", type=Path, required=True)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()
    if not 0.0 <= args.residual_scale <= 1.0:
        raise SystemExit("residual-scale must be in [0, 1]")

    os.environ["LINGBOT_TRAINING_CONFIG"] = str(args.training_config.resolve())
    sys.path.insert(0, str(args.lingbot_root.resolve()))
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn

    device = torch.device("cuda")
    server = LingbotVLAv2Server(
        str(args.model_path), robot_norm_path=str(args.norm_stats), use_length=16,
        chunk_ret=True, use_bf16=True, use_fp32=False, use_compile=False,
    )
    os.chdir(args.lingbot_root)
    server.reset("so_arm101")
    capture = LingBotVisualTokenCapture(server)

    bottleneck_value = torch.load(args.bottleneck, map_location="cpu", weights_only=False)
    bottleneck = VisualTokenBottleneck(VisualTokenConfig(**bottleneck_value["config"])).to(device).eval()
    bottleneck.load_state_dict(bottleneck_value["model"], strict=True)
    actors = {}
    config = None
    for phase in ("grasp", "place"):
        direct = args.residual_root / phase / "best.pt"
        actor, phase_config = load_actor(
            direct if direct.exists() else args.residual_root / phase / "checkpoints/latest.pt", device
        )
        if config is not None and phase_config != config:
            raise RuntimeError("grasp/place actor configs differ")
        config = phase_config
        actors[phase] = actor
    assert config is not None

    state_lock = threading.Lock()
    runtime = {
        "phase": "grasp", "previous_state": None, "previous_time": None,
        "closed": 0, "saw_open_gripper": False,
    }

    class InferRequest(BaseModel):
        image: dict[str, str]
        state: list[float]
        task: str
        robo_name: str | None = None
        use_length: int | None = None

    class ResetRequest(BaseModel):
        robo_name: str = "so_arm101"

    app = FastAPI(title="LingBot RECAP bounded residual policy")

    @app.get("/healthz")
    def health():
        return {
            "status": "ok", "model_loaded": True, "phase": runtime["phase"],
            "residual_scale": args.residual_scale, "residual_limit": config.residual_limit,
            "exploration": False, "checkpoint": str(args.residual_root),
        }

    @app.post("/reset")
    def reset(_: ResetRequest):
        with state_lock:
            runtime.update(
                phase="grasp", previous_state=None, previous_time=None,
                closed=0, saw_open_gripper=False,
            )
        return {"status": "ok", "phase": "grasp"}

    @app.post("/infer")
    def infer(request: InferRequest):
        if request.use_length not in (None, 16):
            raise ValueError("RECAP checkpoint requires a 16-step chunk")
        now = time.monotonic()
        position = np.asarray(request.state, dtype=np.float32)
        if position.shape != (6,) or not np.isfinite(position).all():
            raise ValueError("state must be finite 6D SO-101 positions")
        observation = {
            **{
                f"observation.images.{name}": np.asarray(
                    Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB"), dtype=np.uint8
                ).copy()
                for name, value in request.image.items()
            },
            "observation.state": position,
            "task": request.task,
        }
        with state_lock:
            previous = runtime["previous_state"]
            previous_time = runtime["previous_time"]
            velocity = np.zeros(6, dtype=np.float32) if previous is None else (
                (position - previous) / max(now - previous_time, 1e-3)
            )
            # A rollout can start from a closed/parked gripper.  That is not a
            # completed grasp.  Require seeing it open before a later close is
            # allowed to advance the state machine to place.
            if position[-1] >= 5.0:
                runtime["saw_open_gripper"] = True
            if runtime["saw_open_gripper"] and position[-1] <= 2.0:
                runtime["closed"] += 1
            else:
                runtime["closed"] = 0
            if runtime["closed"] >= 1:
                runtime["phase"] = "place"
            phase = runtime["phase"]
            runtime["previous_state"] = position.copy()
            runtime["previous_time"] = now

        started = time.perf_counter()
        features = capture.infer(observation)
        raw = torch.from_numpy(features.visual_tokens).to(device)
        valid = torch.ones(raw.shape[:2], dtype=torch.bool, device=device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            visual = bottleneck.encode(raw, valid)[0].float().cpu().numpy()
        state = make_state_feature(visual, position, velocity, feature_dim=config.feature_dim)
        reference = features.normalized_action[: config.chunk_size]
        with torch.inference_mode():
            predicted = actors[phase].mean(
                torch.from_numpy(state[None]).to(device),
                torch.from_numpy(reference[None]).to(device),
            )[0].cpu().numpy()
        residual = predicted - reference
        normalized = np.clip(reference + args.residual_scale * residual, -1.0, 1.0)
        physical = SO101ActionCodec.denormalize(normalized)
        if not np.isfinite(physical).all() or np.abs(residual).max() > config.residual_limit + 1e-5:
            raise FloatingPointError("residual policy safety contract failed")
        return {
            "action": {"action": physical.tolist()},
            "server_timing_ms": round((time.perf_counter() - started) * 1000, 1),
            "recap": {
                "phase": phase, "residual_scale": args.residual_scale,
                "residual_abs_mean": float(np.abs(residual).mean()),
                "residual_abs_max": float(np.abs(residual).max()),
                "scaled_residual_abs_max": float(np.abs(args.residual_scale * residual).max()),
                "all_finite": True,
            },
        }

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
