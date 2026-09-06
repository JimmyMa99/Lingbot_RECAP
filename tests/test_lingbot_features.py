from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lingbot_recap.lingbot_features import LingBotVisualTokenCapture  # noqa: E402


class FakeJointModel:
    def forward(self, *, visual_pos_masks=None):
        prefix = torch.arange(30, dtype=torch.float32).reshape(1, 5, 6)
        return ([prefix, None], None, [])


class FakeServer:
    use_compile = False

    def __init__(self):
        self.vla = SimpleNamespace(model=SimpleNamespace(qwenvl_with_expert=FakeJointModel()))

    def infer(self, _observation, return_normalized=False):
        assert return_normalized
        target = self.vla.model.qwenvl_with_expert
        target.forward(visual_pos_masks=torch.tensor([[True, False, True, False, False]]))
        return {
            "action.arm.position": np.zeros((4, 5), dtype=np.float32),
            "action.effector.position": np.full((4, 1), 50, dtype=np.float32),
            "_normalized_actions": np.zeros((4, 55), dtype=np.float32),
        }


def test_capture_uses_visual_prefix_and_six_dimensional_action():
    result = LingBotVisualTokenCapture(FakeServer()).infer({})
    assert result.visual_tokens.shape == (1, 2, 6)
    assert result.normalized_action.shape == (4, 6)
    np.testing.assert_array_equal(result.normalized_action, 0)
