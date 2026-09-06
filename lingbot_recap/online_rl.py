"""LingBot RECAP 的轻量在线 residual actor/critic。

这个模块只依赖 PyTorch，不依赖 OpenPI 或 JAX。LingBot 本体保持冻结；actor
学习在 LingBot action chunk 周围做有界修正，twin critic 用已封存的人工介入
轨迹学习成功回报。
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class OnlineRLConfig:
    feature_dim: int = 512
    proprio_dim: int = 12
    action_dim: int = 6
    chunk_size: int = 16
    hidden_dim: int = 256
    actor_layers: int = 2
    critic_layers: int = 2
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    bc_weight: float = 100.0
    exploration_std: float = 0.03
    reference_dropout: float = 0.5
    residual_limit: float = 0.05
    batch_size: int = 256
    policy_delay: int = 2
    replay_capacity: int = 100_000

    def __post_init__(self) -> None:
        positive = {
            "feature_dim": self.feature_dim,
            "proprio_dim": self.proprio_dim,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "hidden_dim": self.hidden_dim,
            "actor_layers": self.actor_layers,
            "critic_layers": self.critic_layers,
            "batch_size": self.batch_size,
            "policy_delay": self.policy_delay,
            "replay_capacity": self.replay_capacity,
        }
        bad = [name for name, value in positive.items() if value <= 0]
        if bad:
            raise ValueError(f"配置必须为正数: {bad}")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma 必须在 (0, 1] 内")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau 必须在 (0, 1] 内")
        if not 0.0 <= self.reference_dropout <= 1.0:
            raise ValueError("reference_dropout 必须在 [0, 1] 内")
        if not 0.0 <= self.residual_limit <= 1.0:
            raise ValueError("residual_limit 必须在 [0, 1] 内")

    @property
    def state_dim(self) -> int:
        return self.feature_dim + self.proprio_dim

    @property
    def flat_action_dim(self) -> int:
        return self.chunk_size * self.action_dim

    @property
    def chunk_discount(self) -> float:
        return self.gamma**self.chunk_size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OnlineRLConfig":
        fields = cls.__dataclass_fields__
        return cls(**{key: value[key] for key in fields if key in value})


class SO101ActionCodec:
    """校准后 SO-101 关节坐标与 RL 的 [-1, 1] 空间之间互转。"""

    lower = np.asarray([-100, -100, -100, -100, -100, 0], dtype=np.float32)
    upper = np.asarray([100, 100, 100, 100, 100, 100], dtype=np.float32)

    @classmethod
    def normalize(cls, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape[-1] != 6:
            raise ValueError(f"动作最后一维必须为 6，实际为 {array.shape}")
        result = 2.0 * (array - cls.lower) / (cls.upper - cls.lower) - 1.0
        return np.clip(result, -1.0, 1.0)

    @classmethod
    def denormalize(cls, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape[-1] != 6:
            raise ValueError(f"动作最后一维必须为 6，实际为 {array.shape}")
        clipped = np.clip(array, -1.0, 1.0)
        return cls.lower + (clipped + 1.0) * 0.5 * (cls.upper - cls.lower)


def make_state_feature(
    visual_feature: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    *,
    feature_dim: int,
    velocity_scale: float = 300.0,
) -> np.ndarray:
    visual = np.asarray(visual_feature, dtype=np.float32).reshape(-1)
    if visual.shape != (feature_dim,) or not np.isfinite(visual).all():
        raise ValueError(f"视觉表征应为有限的 ({feature_dim},)，实际为 {visual.shape}")
    q = SO101ActionCodec.normalize(position).reshape(-1)
    dq = np.asarray(velocity, dtype=np.float32).reshape(-1)
    if dq.shape != (6,) or velocity_scale <= 0:
        raise ValueError("速度应为 6 维，且 velocity_scale 必须为正数")
    dq = np.clip(dq / velocity_scale, -1.0, 1.0)
    result = np.concatenate((visual, q, dq)).astype(np.float32, copy=False)
    if result.shape != (feature_dim + 12,) or not np.isfinite(result).all():
        raise ValueError("state feature 合同不匹配或包含 NaN/Inf")
    return result


def _mlp(input_dim: int, hidden_dim: int, depth: int, output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for _ in range(depth):
        layers.extend((nn.Linear(width, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()))
        width = hidden_dim
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


class ResidualChunkActor(nn.Module):
    """以冻结 LingBot chunk 为中心、零残差初始化的 actor。"""

    def __init__(self, config: OnlineRLConfig):
        super().__init__()
        self.config = config
        self.feature_norm = nn.LayerNorm(config.feature_dim)
        self.net = _mlp(
            config.state_dim + config.flat_action_dim,
            config.hidden_dim,
            config.actor_layers,
            config.flat_action_dim,
        )
        output = self.net[-1]
        if not isinstance(output, nn.Linear):
            raise RuntimeError("actor 输出层合同被破坏")
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def _state(self, state: torch.Tensor) -> torch.Tensor:
        feature = self.feature_norm(state[..., : self.config.feature_dim])
        proprio = state[..., self.config.feature_dim :]
        return torch.cat((feature, proprio), dim=-1)

    def mean(
        self,
        state: torch.Tensor,
        reference: torch.Tensor,
        *,
        apply_reference_dropout: bool = False,
    ) -> torch.Tensor:
        flat_reference = reference.reshape(reference.shape[0], -1)
        network_reference = flat_reference
        if apply_reference_dropout and self.training and self.config.reference_dropout > 0:
            keep = (
                torch.rand((reference.shape[0], 1), device=reference.device)
                >= self.config.reference_dropout
            )
            network_reference = network_reference * keep
        raw = self.net(torch.cat((self._state(state), network_reference), dim=-1))
        residual = torch.tanh(raw) * self.config.residual_limit
        return torch.clamp(flat_reference + residual, -1.0, 1.0).reshape_as(reference)

    def sample(
        self,
        state: torch.Tensor,
        reference: torch.Tensor,
        *,
        explore: bool,
        exploration_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0.0 <= exploration_scale <= 1.0:
            raise ValueError("exploration_scale 必须在 [0, 1] 内")
        mean = self.mean(state, reference)
        action = mean
        if explore and self.config.exploration_std > 0:
            action = mean + torch.randn_like(mean) * self.config.exploration_std * exploration_scale
        return torch.clamp(action, -1.0, 1.0), mean


class ChunkCritic(nn.Module):
    def __init__(self, config: OnlineRLConfig):
        super().__init__()
        self.config = config
        self.feature_norm = nn.LayerNorm(config.feature_dim)
        self.net = _mlp(
            config.state_dim + config.flat_action_dim,
            config.hidden_dim,
            config.critic_layers,
            1,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        feature = self.feature_norm(state[..., : self.config.feature_dim])
        proprio = state[..., self.config.feature_dim :]
        return self.net(
            torch.cat((feature, proprio, action.reshape(action.shape[0], -1)), dim=-1)
        ).squeeze(-1)


class ReplayBuffer:
    REQUIRED = (
        "state",
        "action",
        "reference",
        "reward",
        "next_state",
        "next_reference",
        "done",
        "intervention",
    )

    def __init__(self, config: OnlineRLConfig):
        self.config = config
        capacity = config.replay_capacity
        self.state = np.empty((capacity, config.state_dim), dtype=np.float32)
        self.action = np.empty((capacity, config.chunk_size, config.action_dim), dtype=np.float32)
        self.reference = np.empty_like(self.action)
        self.reward = np.empty((capacity,), dtype=np.float32)
        self.next_state = np.empty_like(self.state)
        self.next_reference = np.empty_like(self.action)
        self.done = np.empty((capacity,), dtype=np.float32)
        self.intervention = np.empty((capacity,), dtype=np.float32)
        self.size = 0
        self.position = 0

    def __len__(self) -> int:
        return self.size

    def _validate(self, batch: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int]:
        missing = set(self.REQUIRED) - set(batch)
        if missing:
            raise ValueError(f"replay 缺少字段: {sorted(missing)}")
        values = {key: np.asarray(batch[key], dtype=np.float32) for key in self.REQUIRED}
        count = int(values["state"].shape[0])
        shapes = {
            "state": (count, self.config.state_dim),
            "action": (count, self.config.chunk_size, self.config.action_dim),
            "reference": (count, self.config.chunk_size, self.config.action_dim),
            "reward": (count,),
            "next_state": (count, self.config.state_dim),
            "next_reference": (count, self.config.chunk_size, self.config.action_dim),
            "done": (count,),
            "intervention": (count,),
        }
        for key, shape in shapes.items():
            if values[key].shape != shape:
                raise ValueError(f"replay {key} 应为 {shape}，实际为 {values[key].shape}")
            if not np.isfinite(values[key]).all():
                raise ValueError(f"replay {key} 包含 NaN/Inf")
        for key in ("done", "intervention"):
            if not np.all(np.isin(values[key], (0.0, 1.0))):
                raise ValueError(f"replay {key} 必须是 0/1")
        return values, count

    def add_batch(self, batch: dict[str, np.ndarray]) -> int:
        values, count = self._validate(batch)
        if count == 0:
            return 0
        if count > self.config.replay_capacity:
            values = {key: value[-self.config.replay_capacity :] for key, value in values.items()}
            count = self.config.replay_capacity
        indices = (np.arange(count) + self.position) % self.config.replay_capacity
        for key in self.REQUIRED:
            getattr(self, key)[indices] = values[key]
        self.position = int((self.position + count) % self.config.replay_capacity)
        self.size = min(self.config.replay_capacity, self.size + count)
        return count

    def sample(self, count: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        if self.size < count:
            raise RuntimeError(f"replay 只有 {self.size} 条，至少需要 {count} 条")
        indices = rng.integers(0, self.size, size=count)
        return {key: getattr(self, key)[indices] for key in self.REQUIRED}

    def export(self) -> dict[str, np.ndarray]:
        return {key: getattr(self, key)[: self.size].copy() for key in self.REQUIRED}


class OnlineRLAgent:
    def __init__(self, config: OnlineRLConfig, device: str = "cuda", seed: int = 1000):
        self.config = config
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.actor = ResidualChunkActor(config).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).eval()
        self.q1 = ChunkCritic(config).to(self.device)
        self.q2 = ChunkCritic(config).to(self.device)
        self.q1_target = copy.deepcopy(self.q1).eval()
        self.q2_target = copy.deepcopy(self.q2).eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=config.critic_lr
        )
        self.replay = ReplayBuffer(config)
        self.update_step = 0

    @torch.inference_mode()
    def act(
        self,
        state: np.ndarray,
        reference: np.ndarray,
        *,
        explore: bool,
        exploration_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.actor.eval()
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        reference_tensor = torch.as_tensor(reference, dtype=torch.float32, device=self.device).reshape(
            1, self.config.chunk_size, self.config.action_dim
        )
        action, mean = self.actor.sample(
            state_tensor, reference_tensor, explore=explore, exploration_scale=exploration_scale
        )
        return action[0].cpu().numpy(), mean[0].cpu().numpy()

    @staticmethod
    def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for source_parameter, target_parameter in zip(
                source.parameters(), target.parameters(), strict=True
            ):
                target_parameter.lerp_(source_parameter, tau)

    @staticmethod
    def _finite(name: str, value: torch.Tensor) -> float:
        result = float(value.detach())
        if not np.isfinite(result):
            raise FloatingPointError(f"{name} 出现 NaN/Inf")
        return result

    def train_step(self) -> dict[str, float] | None:
        if len(self.replay) < self.config.batch_size:
            return None
        batch = self.replay.sample(self.config.batch_size, self.rng)
        tensors = {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in batch.items()
        }
        self.actor.train()
        self.q1.train()
        self.q2.train()
        with torch.no_grad():
            next_action, _ = self.actor_target.sample(
                tensors["next_state"], tensors["next_reference"], explore=True
            )
            target_q = torch.minimum(
                self.q1_target(tensors["next_state"], next_action),
                self.q2_target(tensors["next_state"], next_action),
            )
            target = tensors["reward"] + (
                1.0 - tensors["done"]
            ) * self.config.chunk_discount * target_q

        q1 = self.q1(tensors["state"], tensors["action"])
        q2 = self.q2(tensors["state"], tensors["action"])
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 10.0)
        self.critic_optimizer.step()

        self.update_step += 1
        metrics = {
            "update_step": float(self.update_step),
            "critic_loss": self._finite("critic_loss", critic_loss),
            "q1_mean": self._finite("q1_mean", q1.mean()),
            "q2_mean": self._finite("q2_mean", q2.mean()),
            "target_q_mean": self._finite("target_q_mean", target.mean()),
        }
        if self.update_step % self.config.policy_delay == 0:
            action_mean = self.actor.mean(
                tensors["state"], tensors["reference"], apply_reference_dropout=True
            )
            actor_q = self.q1(tensors["state"], action_mean)
            bc_loss = F.mse_loss(action_mean, tensors["reference"])
            actor_loss = -actor_q.mean() + self.config.bc_weight * bc_loss
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_optimizer.step()
            self._soft_update(self.actor, self.actor_target, self.config.tau)
            metrics.update(
                actor_loss=self._finite("actor_loss", actor_loss),
                actor_q_mean=self._finite("actor_q_mean", actor_q.mean()),
                bc_loss=self._finite("bc_loss", bc_loss),
            )
        self._soft_update(self.q1, self.q1_target, self.config.tau)
        self._soft_update(self.q2, self.q2_target, self.config.tau)
        return metrics

    def checkpoint(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "config": self.config.to_dict(),
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "update_step": self.update_step,
        }

    def load_checkpoint(self, value: dict[str, Any]) -> None:
        if value.get("schema_version") != 1:
            raise RuntimeError("不支持的在线 RL checkpoint schema")
        if OnlineRLConfig.from_dict(value["config"]) != self.config:
            raise RuntimeError("在线 RL checkpoint 配置不一致，拒绝自动迁移")
        for key, module in (
            ("actor", self.actor),
            ("actor_target", self.actor_target),
            ("q1", self.q1),
            ("q2", self.q2),
            ("q1_target", self.q1_target),
            ("q2_target", self.q2_target),
        ):
            module.load_state_dict(value[key], strict=True)
        self.actor_optimizer.load_state_dict(value["actor_optimizer"])
        self.critic_optimizer.load_state_dict(value["critic_optimizer"])
        self.update_step = int(value["update_step"])


def sparse_terminal_rewards(done: np.ndarray, success: bool) -> np.ndarray:
    """成功只奖励 terminal suffix；失败轨迹奖励保持为 0。"""

    terminal = np.asarray(done, dtype=np.float32).reshape(-1)
    if terminal.size == 0 or not np.all(np.isin(terminal, (0.0, 1.0))):
        raise ValueError("done 必须是非空 0/1 数组")
    indices = np.flatnonzero(terminal > 0.5)
    if indices.size == 0 or not np.all(terminal[indices[0] :] == 1.0):
        raise ValueError("terminal 标记必须是最终连续后缀")
    return terminal.copy() if success else np.zeros_like(terminal)
