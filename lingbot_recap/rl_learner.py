"""可恢复、幂等提交的 LingBot RECAP 在线 learner。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .online_rl import OnlineRLAgent, OnlineRLConfig, sparse_terminal_rewards


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _batch_digest(batch: dict[str, np.ndarray], keys: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for key in keys:
        value = np.ascontiguousarray(batch[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class LearnerConfig:
    warmup_episodes: int = 20
    warmup_successes: int = 8
    warmup_transitions: int = 400
    updates_per_transition: int = 5


class PhaseLearner:
    """每个子阶段使用独立 replay、actor、critic 和 optimizer。"""

    def __init__(
        self,
        phase: str,
        root: str | Path,
        agent_config: OnlineRLConfig,
        learner_config: LearnerConfig = LearnerConfig(),
        *,
        device: str = "cuda",
        seed: int = 1000,
    ):
        self.phase = phase
        self.root = Path(root) / phase
        self.replay_root = self.root / "replay"
        self.checkpoint_root = self.root / "checkpoints"
        self.metrics_path = self.root / "training.jsonl"
        self.status_path = self.root / "status.json"
        self.replay_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.agent = OnlineRLAgent(agent_config, device=device, seed=seed)
        self.config = learner_config
        self.online_enabled = False
        self.bootstrap_completed = False
        self.episodes: list[dict[str, Any]] = []
        self._load()
        self._write_status()

    def _load(self) -> None:
        manifests = sorted(self.replay_root.glob("episode_*.json"))
        arrays = sorted(self.replay_root.glob("episode_*.npz"))
        if len(manifests) != len(arrays):
            raise RuntimeError(f"{self.phase}: replay manifest/npz 数量不一致")
        for manifest_path, array_path in zip(manifests, arrays, strict=True):
            if manifest_path.stem != array_path.stem:
                raise RuntimeError(f"{self.phase}: replay 文件顺序不一致")
            manifest = json.loads(manifest_path.read_text())
            with np.load(array_path, allow_pickle=False) as data:
                batch = {key: data[key] for key in self.agent.replay.REQUIRED}
            if _batch_digest(batch, self.agent.replay.REQUIRED) != manifest["batch_sha256"]:
                raise RuntimeError(f"{array_path}: replay digest 校验失败")
            self.agent.replay.add_batch(batch)
            self.episodes.append(manifest)
        latest = self.checkpoint_root / "latest.pt"
        if latest.exists():
            checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
            if checkpoint.get("phase") != self.phase:
                raise RuntimeError("checkpoint phase 不一致")
            self.agent.load_checkpoint(checkpoint["agent"])
            self.online_enabled = bool(checkpoint["online_enabled"])
            self.bootstrap_completed = bool(checkpoint["bootstrap_completed"])

    def summary(self) -> dict[str, Any]:
        warmup = [item for item in self.episodes if item["collection_mode"] == "warmup"]
        result = {
            "phase": self.phase,
            "online_enabled": self.online_enabled,
            "episodes": len(self.episodes),
            "warmup_episodes": len(warmup),
            "warmup_successes": sum(bool(item["success"]) for item in warmup),
            "transitions": len(self.agent.replay),
            "update_step": self.agent.update_step,
            "bootstrap_completed": self.bootstrap_completed,
        }
        result["warmup_ready"] = (
            result["warmup_episodes"] >= self.config.warmup_episodes
            and result["warmup_successes"] >= self.config.warmup_successes
            and result["transitions"] >= self.config.warmup_transitions
        )
        return result

    def _write_status(self) -> None:
        _atomic_json(self.status_path, {"schema_version": 1, **self.summary()})

    def save(self) -> None:
        _atomic_torch_save(
            self.checkpoint_root / "latest.pt",
            {
                "schema_version": 1,
                "phase": self.phase,
                "online_enabled": self.online_enabled,
                "bootstrap_completed": self.bootstrap_completed,
                "agent": self.agent.checkpoint(),
            },
        )
        self._write_status()

    def commit(
        self,
        *,
        commit_id: str,
        collection_mode: str,
        success: bool,
        batch: dict[str, np.ndarray],
        operator_note: str = "",
    ) -> dict[str, Any]:
        if collection_mode not in {"warmup", "online"}:
            raise ValueError("collection_mode 必须是 warmup 或 online")
        commit_id = commit_id.strip()
        if not commit_id:
            raise ValueError("必须提供稳定的 commit_id")
        values, count = self.agent.replay._validate(batch)
        expected_reward = sparse_terminal_rewards(values["done"], success)
        if not np.array_equal(values["reward"], expected_reward):
            raise ValueError("reward 与 success/done 合同不一致")
        digest = _batch_digest(values, self.agent.replay.REQUIRED)
        for previous in self.episodes:
            if previous["commit_id"] == commit_id:
                if previous["batch_sha256"] != digest:
                    raise RuntimeError("相同 commit_id 被用于不同数据")
                return {"duplicate": True, "episode": previous, "status": self.summary()}

        stem = f"episode_{len(self.episodes):06d}"
        array_path = self.replay_root / f"{stem}.npz"
        temporary = self.replay_root / f".{stem}.{os.getpid()}.tmp"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **values)
        os.replace(temporary, array_path)
        manifest = {
            "schema_version": 1,
            "phase": self.phase,
            "collection_mode": collection_mode,
            "success": bool(success),
            "transitions": count,
            "intervention_transitions": int(values["intervention"].sum()),
            "commit_id": commit_id,
            "batch_sha256": digest,
            "operator_note": operator_note,
        }
        _atomic_json(self.replay_root / f"{stem}.json", manifest)
        self.agent.replay.add_batch(values)
        self.episodes.append(manifest)

        metrics = []
        if self.online_enabled and collection_mode == "online":
            for _ in range(count * self.config.updates_per_transition):
                item = self.agent.train_step()
                if item is not None:
                    metrics.append(item)
            with self.metrics_path.open("a", encoding="utf-8") as stream:
                for item in metrics:
                    stream.write(json.dumps(item, allow_nan=False) + "\n")
        self.save()
        return {
            "duplicate": False,
            "episode": manifest,
            "updates_completed": len(metrics),
            "last_metrics": metrics[-1] if metrics else None,
            "status": self.summary(),
        }

    def bootstrap(self, updates: int | None = None) -> dict[str, Any]:
        status = self.summary()
        if self.online_enabled:
            raise RuntimeError("online 激活后禁止重新 bootstrap")
        if not status["warmup_ready"]:
            raise RuntimeError(f"warmup 尚未达标: {status}")
        if self.bootstrap_completed:
            return {"already_completed": True, "status": status}
        requested = updates or len(self.agent.replay) * self.config.updates_per_transition
        completed = 0
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            for _ in range(requested):
                item = self.agent.train_step()
                if item is None:
                    break
                stream.write(json.dumps({"stage": "bootstrap", **item}, allow_nan=False) + "\n")
                completed += 1
        if completed != requested:
            raise RuntimeError(f"bootstrap 只完成 {completed}/{requested}")
        self.bootstrap_completed = True
        self.save()
        return {"completed": completed, "status": self.summary()}

    def activate(self) -> dict[str, Any]:
        status = self.summary()
        if not status["warmup_ready"] or not self.bootstrap_completed:
            raise RuntimeError(f"不满足 online 激活门槛: {status}")
        self.online_enabled = True
        self.save()
        return {"activated": True, "status": self.summary()}
