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

from .online_rl import SO101ActionCodec


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

    def _capture(self, kwargs, output) -> None:
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
            if getattr(self.server, "use_compile", False):
                raise RuntimeError("视觉 token 捕获要求 LingBot use_compile=False")
            original_forward = target.forward

            def wrapped_forward(*args, **kwargs):
                output = original_forward(*args, **kwargs)
                self._capture(kwargs, output)
                return output

            target.forward = wrapped_forward
            try:
                action = self.server.infer(observation, return_normalized=True)
            finally:
                target.forward = original_forward
            if self._captured is None:
                raise RuntimeError("未捕获到 LingBot prefix 视觉 token")
            # LingBot 的内部 normalized action 是 max_action_dim=55 的填充空间。
            # residual learner 必须使用实际 SO-101 六维动作，不能直接拿那 55 维张量。
            action.pop("_normalized_actions", None)
            if "action" in action:
                physical = np.asarray(action["action"], dtype=np.float32)
            elif "action.arm.position" in action and "action.effector.position" in action:
                physical = np.concatenate(
                    (
                        np.asarray(action["action.arm.position"], dtype=np.float32),
                        np.asarray(action["action.effector.position"], dtype=np.float32),
                    ),
                    axis=-1,
                )
            else:
                raise RuntimeError(f"无法识别 LingBot SO-101 action keys: {sorted(action)}")
            if physical.ndim != 2 or physical.shape[1] != 6 or not np.isfinite(physical).all():
                raise RuntimeError(f"SO-101 action 合同不匹配: {physical.shape}")
            normalized_np = SO101ActionCodec.normalize(physical)
            return LingBotReferenceFeatures(
                action=action,
                normalized_action=normalized_np,
                visual_tokens=self._captured.numpy(),
            )


@torch.inference_mode()
def extract_visual_tokens_from_transformed(
    model: Any,
    observation: dict[str, torch.Tensor],
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> np.ndarray:
    """只执行 LingBot prefix 前向，不运行 flow-matching action denoise。"""

    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import (
        make_att_2d_masks,
    )

    images = observation["images"]
    img_masks = observation["img_masks"]
    lang_tokens = observation["lang_tokens"]
    lang_masks = observation["lang_masks"]
    image_grid_thw = observation.get("image_grid_thw")
    if images.ndim == 4:
        images = images.unsqueeze(0)
        img_masks = img_masks.unsqueeze(0)
    if lang_tokens.ndim == 1:
        lang_tokens = lang_tokens.unsqueeze(0)
        lang_masks = lang_masks.unsqueeze(0)
    (
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        prefix_position_ids,
        visual_pos_masks,
        deepstack_visual_embeds,
    ) = model.embed_prefix(
        images.to(device=device, dtype=dtype),
        img_masks.to(device=device),
        lang_tokens.to(device=device),
        lang_masks.to(device=device),
        image_grid_thw=(
            None if image_grid_thw is None else image_grid_thw.to(device=device, dtype=torch.long)
        ),
    )
    outputs, _, _ = model.qwenvl_with_expert.forward(
        attention_mask=make_att_2d_masks(prefix_pad_masks, prefix_att_masks),
        position_ids=prefix_position_ids,
        vlm_position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=False,
        fill_kv_cache=False,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
    )
    prefix = outputs[0]
    mask = visual_pos_masks.to(device=prefix.device, dtype=torch.bool)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    if prefix.ndim != 3 or mask.shape != prefix.shape[:2]:
        raise RuntimeError(
            f"LingBot prefix/mask 合同不匹配: {tuple(prefix.shape)} / {tuple(mask.shape)}"
        )
    counts = mask.sum(dim=1)
    if torch.any(counts <= 0) or not torch.all(counts == counts[0]):
        raise RuntimeError(f"视觉 token 数不一致: {counts.tolist()}")
    selected = prefix[mask].reshape(prefix.shape[0], int(counts[0]), prefix.shape[-1])
    if not torch.isfinite(selected).all():
        raise FloatingPointError("视觉 token 包含 NaN/Inf")
    return selected.detach().float().cpu().numpy()
