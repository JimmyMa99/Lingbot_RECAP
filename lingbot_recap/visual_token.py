"""将 LingBot 最终层视觉 token 压缩为在线 RL 使用的固定维度表征。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class VisualTokenConfig:
    input_dim: int
    token_dim: int = 512
    max_tokens: int = 1024
    heads: int = 8
    encoder_layers: int = 2
    decoder_layers: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if min(
            self.input_dim,
            self.token_dim,
            self.max_tokens,
            self.heads,
            self.encoder_layers,
            self.decoder_layers,
        ) <= 0:
            raise ValueError("visual token 配置必须为正数")
        if self.token_dim % self.heads:
            raise ValueError("token_dim 必须能被 heads 整除")

    def to_dict(self) -> dict:
        return asdict(self)


class VisualTokenBottleneck(nn.Module):
    """带重构门槛的视觉 bottleneck；在线只调用 `encode`。"""

    def __init__(self, config: VisualTokenConfig):
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.token_dim)
        self.rl_token = nn.Parameter(torch.zeros(1, 1, config.token_dim))
        self.encoder_positions = nn.Parameter(
            torch.zeros(1, config.max_tokens + 1, config.token_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.token_dim,
            nhead=config.heads,
            dim_feedforward=config.token_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.token_dim,
            nhead=config.heads,
            dim_feedforward=config.token_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.decoder_layers)
        self.decoder_positions = nn.Parameter(torch.zeros(1, config.max_tokens, config.token_dim))
        self.output_projection = nn.Linear(config.token_dim, config.input_dim)

    def _validate(self, embeddings: torch.Tensor, valid: torch.Tensor) -> None:
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.config.input_dim:
            raise ValueError(f"embedding 合同不匹配: {tuple(embeddings.shape)}")
        if valid.shape != embeddings.shape[:2] or valid.dtype != torch.bool:
            raise ValueError(f"valid mask 合同不匹配: {tuple(valid.shape)} {valid.dtype}")
        if embeddings.shape[1] > self.config.max_tokens:
            raise ValueError("视觉 token 数超过 max_tokens")
        if torch.any(valid.sum(dim=1) <= 0):
            raise ValueError("每个样本至少需要一个有效视觉 token")

    def encode(self, embeddings: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        self._validate(embeddings, valid)
        batch, length, _ = embeddings.shape
        inputs = self.input_projection(embeddings)
        token = self.rl_token.expand(batch, -1, -1)
        sequence = torch.cat((inputs, token), dim=1)
        sequence = sequence + self.encoder_positions[:, : length + 1]
        padding = torch.cat(
            (torch.logical_not(valid), torch.zeros((batch, 1), dtype=torch.bool, device=valid.device)),
            dim=1,
        )
        encoded = self.encoder(sequence, src_key_padding_mask=padding)
        return encoded[:, -1]

    def decode_from_token(self, token: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        if token.shape != (embeddings.shape[0], self.config.token_dim):
            raise ValueError(f"RL token 合同不匹配: {tuple(token.shape)}")
        length = embeddings.shape[1]
        # Learned queries may use only the compressed token. Feeding embeddings
        # here would create a bypass that reconstructs without the 512D bottleneck.
        target = self.decoder_positions[:, :length].expand(embeddings.shape[0], -1, -1)
        memory = token[:, None]
        causal = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=embeddings.device), diagonal=1
        )
        decoded = self.decoder(target, memory, tgt_mask=causal)
        return self.output_projection(decoded)

    def reconstruct(self, embeddings: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token = self.encode(embeddings, valid)
        return self.decode_from_token(token, embeddings), token

    def forward(self, embeddings: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.reconstruction_loss(embeddings, valid)

    def reconstruction_loss(self, embeddings: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        prediction, token = self.reconstruct(embeddings, valid)
        per_token = F.mse_loss(prediction.float(), embeddings.float(), reduction="none").mean(-1)
        weights = valid.float()
        loss = (per_token * weights).sum() / weights.sum().clamp_min(1.0)
        return {"loss": loss, "token_rms": token.float().square().mean().sqrt()}

    @torch.inference_mode()
    def zero_token_loss_ratio(self, embeddings: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        prediction, token = self.reconstruct(embeddings, valid)
        zero_prediction = self.decode_from_token(torch.zeros_like(token), embeddings)
        weights = valid.float()
        normal = (
            F.mse_loss(prediction.float(), embeddings.float(), reduction="none").mean(-1) * weights
        ).sum() / weights.sum().clamp_min(1.0)
        zero = (
            F.mse_loss(zero_prediction.float(), embeddings.float(), reduction="none").mean(-1)
            * weights
        ).sum() / weights.sum().clamp_min(1.0)
        return zero / normal.clamp_min(1e-12)
