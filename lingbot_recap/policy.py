from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import requests

from .hardware import MOTOR_NAMES


@dataclass(frozen=True)
class PolicyResult:
    chunk: np.ndarray
    timing_ms: float


class LingBotHTTPPolicy:
    def __init__(self, server: str, timeout_s: float = 120.0, use_length: int = 16):
        self.server = server.rstrip("/")
        self.timeout_s = timeout_s
        self.use_length = use_length

    def health(self) -> dict:
        response = requests.get(f"{self.server}/healthz", timeout=5)
        response.raise_for_status()
        return response.json()

    def infer(
        self,
        task: str,
        state: Mapping[str, float],
        image_jpegs: Mapping[str, bytes],
    ) -> PolicyResult:
        response = requests.post(
            f"{self.server}/infer",
            json={
                "image": {
                    name: base64.b64encode(data).decode("ascii")
                    for name, data in image_jpegs.items()
                },
                "state": [float(state[name]) for name in MOTOR_NAMES],
                "task": task,
                "robo_name": "so_arm101",
                "use_length": self.use_length,
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        action = payload["action"]
        if "action" in action:
            chunk = np.asarray(action["action"], dtype=np.float32)
        else:
            arm = np.asarray(action["action.arm.position"], dtype=np.float32)
            gripper = np.asarray(action["action.effector.position"], dtype=np.float32)
            chunk = np.concatenate([arm, gripper], axis=-1)
        if chunk.ndim != 2 or chunk.shape[1] != len(MOTOR_NAMES):
            raise RuntimeError(f"unexpected action shape: {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise RuntimeError("policy returned NaN/Inf")
        lower = np.asarray([-100, -100, -100, -100, -100, 0], dtype=np.float32)
        upper = np.asarray([100, 100, 100, 100, 100, 100], dtype=np.float32)
        return PolicyResult(
            chunk=np.clip(chunk, lower, upper),
            timing_ms=float(payload.get("server_timing_ms", -1)),
        )


def action_dict(action: np.ndarray) -> dict[str, float]:
    return {name: float(action[index]) for index, name in enumerate(MOTOR_NAMES)}
