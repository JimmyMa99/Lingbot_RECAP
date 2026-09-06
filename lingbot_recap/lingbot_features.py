"""从冻结 LingBot 前向中旁路捕获视觉 token。

不修改 LingBot 模型权重，也不重复做一次 VLA 前向。hook 只接受 prefix 分支
（output[0][0]），明确拒绝 action-expert suffix，避免把未来动作信息泄漏给 RL 状态。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class LingBotReferenceFeatures:
    action: dict[str, Any]
    normalized_action: np.ndarray
    visual_tokens: np.ndarray


class LingBotVisualTokenCapture:
    """为一个 `LingbotVLAv2Server` 安装临时、安全的 forward hook。"""

    def __init__(self, server: Any):
        self.server = server
        self._lock = threading.Lock()
        self._captured: torch.Tensor | None = None

    def _hook(self, _module, _args, kwargs, output) -> None:
        if not isinstance(output, tuple) or not output:
            return
        branches = output[0]
        if not isinstance(branches, (list, tuple)) or not branches:
            return
        prefix = branches[0]
        visual_mask = kwargs.get("visual_pos_masks")
        if prefix is None or visual_mask is None:
            return
        mask = visual_mask.to(device=prefix.device, dtype=torch.bool)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        if prefix.ndim != 3 or mask.shape != prefix.shape[:2]:
            raise RuntimeError(
                f"LingBot visual mask 合同不匹配: prefix={tuple(prefix.shape)} mask={tuple(mask.shape)}"
            )
        counts = mask.sum(dim=1)
        if counts.numel() == 0 or torch.any(counts <= 0) or not torch.all(counts == counts[0]):
            raise RuntimeError(f"每个样本必须具有相同且非空的视觉 token 数: {counts.tolist()}")
        selected = prefix[mask].reshape(prefix.shape[0], int(counts[0]), prefix.shape[-1])
        if not torch.isfinite(selected).all():
            raise FloatingPointError("LingBot 视觉 token 包含 NaN/Inf")
        self._captured = selected.detach().float().cpu()

    def infer(self, observation: dict[str, Any]) -> LingBotReferenceFeatures:
        """执行一次原始推理，同时返回 normalized reference 与视觉 token。"""

        with self._lock:
            self._captured = None
            target = self.server.vla.model.qwenvl_with_expert
            handle = target.register_forward_hook(self._hook, with_kwargs=True)
            try:
                action = self.server.infer(observation, return_normalized=True)
            finally:
                handle.remove()
            if self._captured is None:
                raise RuntimeError("未捕获到 LingBot prefix 视觉 token")
            normalized = action.pop("_normalized_actions", None)
            if normalized is None:
                raise RuntimeError("LingBot 服务未返回 normalized action chunk")
            normalized_np = np.asarray(normalized, dtype=np.float32)
            if normalized_np.ndim != 2 or not np.isfinite(normalized_np).all():
                raise RuntimeError(f"normalized action 合同不匹配: {normalized_np.shape}")
            return LingBotReferenceFeatures(
                action=action,
                normalized_action=normalized_np,
                visual_tokens=self._captured.numpy(),
            )
