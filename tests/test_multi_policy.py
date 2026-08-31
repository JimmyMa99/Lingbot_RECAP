import json

import numpy as np
import pytest

from lingbot_recap.multi_policy import (
    MultiPolicyRouter,
    TeacherSpec,
    sha256_file,
)
from lingbot_recap.policy import PolicyResult


class FakePolicy:
    def __init__(self, spec):
        self.spec = spec

    def health(self):
        return {"model_loaded": True, "checkpoint": self.spec.checkpoint}

    def infer(self, task, state, image_jpegs):
        return PolicyResult(np.ones((16, 6), dtype=np.float32), 1.0)


def registry_payload(tmp_path):
    norm = tmp_path / "norm.json"
    robot = tmp_path / "robot.yaml"
    norm.write_text("{}")
    robot.write_text("robot: so101\n")
    return {
        "format": "lingbot_multi_policy_registry_v2",
        "training_contract": {
            "norm_stats": {"path": str(norm), "sha256": sha256_file(norm)},
            "robot_config": {"path": str(robot), "sha256": sha256_file(robot)},
            "camera_mapping": {"top": "camera_top", "wrist": "camera_wrist_left"},
            "action_space": "so101_calibrated_absolute_6d_v1",
            "joints": [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            ],
        },
        "teachers": [
            {
                "key": "pick",
                "checkpoint": "/models/pick-30",
                "server": "http://pick",
                "tasks": ["夹起黄色鸭子"],
            },
            {
                "key": "box",
                "checkpoint": "/models/box-30",
                "server": "http://box",
                "tasks": ["把鸭子收拾到箱子里"],
            },
        ],
    }


def test_exact_task_routes_to_one_identity_verified_teacher(tmp_path):
    registry = tmp_path / "teachers.json"
    registry.write_text(
        json.dumps(registry_payload(tmp_path), ensure_ascii=False)
    )
    router = MultiPolicyRouter.from_file(registry, policy_factory=FakePolicy)
    assert router.teacher_for("  夹起黄色鸭子  ").key == "pick"
    routed = router.infer("夹起黄色鸭子", {}, {})
    assert routed.teacher.checkpoint == "/models/pick-30"
    assert routed.result.chunk.shape == (16, 6)
    health = router.validate_health()
    assert set(health) == {"pick", "box"}
    assert health["pick"]["verified_checkpoint"] == "/models/pick-30"
    assert len(router.require_training_contract().contract_id) == 64


def test_checkpoint_identity_mismatch_is_rejected(tmp_path):
    payload = registry_payload(tmp_path)
    registry = tmp_path / "teachers.json"
    registry.write_text(json.dumps(payload))

    class WrongPolicy(FakePolicy):
        def health(self):
            return {"model_loaded": True, "checkpoint": "/models/wrong"}

    router = MultiPolicyRouter.from_file(registry, policy_factory=WrongPolicy)
    with pytest.raises(RuntimeError, match="checkpoint mismatch"):
        router.validate_health()


def test_health_without_checkpoint_identity_is_rejected(tmp_path):
    registry = tmp_path / "teachers.json"
    registry.write_text(json.dumps(registry_payload(tmp_path)))

    class AnonymousPolicy(FakePolicy):
        def health(self):
            return {"model_loaded": True}

    router = MultiPolicyRouter.from_file(registry, policy_factory=AnonymousPolicy)
    with pytest.raises(RuntimeError, match="does not expose checkpoint identity"):
        router.validate_health()


def test_training_contract_hash_mismatch_is_rejected(tmp_path):
    payload = registry_payload(tmp_path)
    payload["training_contract"]["norm_stats"]["sha256"] = "0" * 64
    registry = tmp_path / "teachers.json"
    registry.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        MultiPolicyRouter.from_file(registry, policy_factory=FakePolicy)


def test_unknown_task_is_not_fuzzily_routed():
    router = MultiPolicyRouter(
        [TeacherSpec("pick", "/models/ckpt", "http://pick", ("夹起黄色鸭子",))],
        policy_factory=FakePolicy,
    )
    with pytest.raises(KeyError, match="no teacher"):
        router.teacher_for("夹起鸭子")


def test_duplicate_task_is_rejected():
    with pytest.raises(ValueError, match="assigned to both"):
        MultiPolicyRouter(
            [
                TeacherSpec("a", "/models/a", "http://a", ("task",)),
                TeacherSpec("b", "/models/b", "http://b", ("task",)),
            ],
            policy_factory=FakePolicy,
        )
