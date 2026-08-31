import json

import numpy as np
import pytest

from lingbot_recap.multi_policy import MultiPolicyRouter, TeacherSpec
from lingbot_recap.policy import PolicyResult


class FakePolicy:
    def __init__(self, spec):
        self.spec = spec

    def health(self):
        return {"model_loaded": True, "checkpoint": self.spec.checkpoint}

    def infer(self, task, state, image_jpegs):
        return PolicyResult(np.ones((16, 6), dtype=np.float32), 1.0)


def test_exact_task_routes_to_one_teacher(tmp_path):
    registry = tmp_path / "teachers.json"
    registry.write_text(
        json.dumps(
            {
                "format": "lingbot_multi_policy_registry_v1",
                "teachers": [
                    {
                        "key": "pick",
                        "checkpoint": "pick-30",
                        "server": "http://pick",
                        "tasks": ["夹起黄色鸭子"],
                    },
                    {
                        "key": "box",
                        "checkpoint": "box-30",
                        "server": "http://box",
                        "tasks": ["把鸭子收拾到箱子里"],
                    },
                ],
            },
            ensure_ascii=False,
        )
    )
    router = MultiPolicyRouter.from_file(registry, policy_factory=FakePolicy)
    assert router.teacher_for("  夹起黄色鸭子  ").key == "pick"
    routed = router.infer("夹起黄色鸭子", {}, {})
    assert routed.teacher.checkpoint == "pick-30"
    assert routed.result.chunk.shape == (16, 6)
    assert set(router.validate_health()) == {"pick", "box"}


def test_unknown_task_is_not_fuzzily_routed():
    router = MultiPolicyRouter(
        [TeacherSpec("pick", "ckpt", "http://pick", ("夹起黄色鸭子",))],
        policy_factory=FakePolicy,
    )
    with pytest.raises(KeyError, match="no teacher"):
        router.teacher_for("夹起鸭子")


def test_duplicate_task_is_rejected():
    with pytest.raises(ValueError, match="assigned to both"):
        MultiPolicyRouter(
            [
                TeacherSpec("a", "a", "http://a", ("task",)),
                TeacherSpec("b", "b", "http://b", ("task",)),
            ],
            policy_factory=FakePolicy,
        )
