import json

import numpy as np

from lingbot_recap.distillation import (
    ExperienceRelabeler,
    RelabelConfig,
    contiguous_segments,
    load_labeled_frames,
)
from lingbot_recap.hardware import MOTOR_NAMES
from lingbot_recap.journal import ExperienceJournal
from lingbot_recap.multi_policy import (
    AssetFingerprint,
    MultiPolicyRouter,
    TeacherSpec,
    TrainingContract,
    sha256_file,
)
from lingbot_recap.policy import PolicyResult


class FakeTeacherPolicy:
    def __init__(self, spec):
        self.spec = spec

    def health(self):
        return {"model_loaded": True, "checkpoint": self.spec.checkpoint}

    def infer(self, task, state, image_jpegs):
        base = np.asarray([state[name] for name in MOTOR_NAMES], dtype=np.float32)
        return PolicyResult(np.repeat((base + 1)[None, :], 16, axis=0), 2.5)


def contract(tmp_path):
    norm = tmp_path / "norm.json"
    robot = tmp_path / "robot.yaml"
    norm.write_text("{}")
    robot.write_text("robot: so101\\n")
    return TrainingContract(
        norm_stats=AssetFingerprint(str(norm), sha256_file(norm)),
        robot_config=AssetFingerprint(str(robot), sha256_file(robot)),
        camera_mapping=(("top", "camera_top"), ("wrist", "camera_wrist_left")),
        action_space="so101_calibrated_absolute_6d_v1",
        joints=tuple(MOTOR_NAMES),
    )


def test_student_frames_are_teacher_labeled_and_human_frames_are_excluded(tmp_path):
    journal = ExperienceJournal(tmp_path, "夹起黄色鸭子", "student")
    state = {name: float(index) for index, name in enumerate(MOTOR_NAMES)}
    for source in ("lingbot_policy", "human_intervention", "lingbot_policy"):
        journal.frame(
            observation={"state": state},
            proposed_action={name: 0.0 for name in MOTOR_NAMES},
            executed_action={name: 0.0 for name in MOTOR_NAMES},
            action_source=source,
            control_mode="auto" if source == "lingbot_policy" else "human",
            image_jpegs={"top": b"jpg-top", "wrist": b"jpg-wrist"},
        )
    episode = journal.close("success")
    router = MultiPolicyRouter(
        [TeacherSpec("pick", "/models/teacher-30", "http://teacher", ("夹起黄色鸭子",))],
        training_contract=contract(tmp_path),
        policy_factory=FakeTeacherPolicy,
    )
    summary = ExperienceRelabeler(router, RelabelConfig()).label_episode(episode)
    assert summary.labeled_frames == 2
    assert summary.skipped_frames == 1
    meta, pairs = load_labeled_frames(episode)
    assert meta["task"] == "夹起黄色鸭子"
    assert [frame["frame_index"] for frame, _ in pairs] == [0, 2]
    assert pairs[0][1]["teacher_key"] == "pick"
    assert pairs[0][1]["teacher_action"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    label_meta = json.loads((episode / "teacher_labels.meta.json").read_text())
    assert label_meta["training_use"] == "MULTI_POLICY_ON_POLICY_DISTILLATION_ONLY"
    assert label_meta["teacher_checkpoint_identity_verified"] is True
    assert label_meta["training_contract_id"] == router.training_contract.contract_id


def test_max_frames_creates_preview_without_blocking_full_labeling(tmp_path):
    journal = ExperienceJournal(tmp_path, "夹起黄色鸭子", "student")
    state = {name: float(index) for index, name in enumerate(MOTOR_NAMES)}
    for _ in range(3):
        journal.frame(
            observation={"state": state},
            proposed_action={name: 0.0 for name in MOTOR_NAMES},
            executed_action={name: 0.0 for name in MOTOR_NAMES},
            action_source="lingbot_policy",
            control_mode="auto",
            image_jpegs={"top": b"jpg-top", "wrist": b"jpg-wrist"},
        )
    episode = journal.close("success")
    router = MultiPolicyRouter(
        [TeacherSpec("pick", "/models/teacher-30", "http://teacher", ("夹起黄色鸭子",))],
        training_contract=contract(tmp_path),
        policy_factory=FakeTeacherPolicy,
    )
    preview = ExperienceRelabeler(
        router, RelabelConfig(max_frames=1)
    ).label_episode(episode)
    assert preview.preview is True
    assert preview.output.name == "teacher_labels.preview.jsonl"
    assert not (episode / "teacher_labels.jsonl").exists()

    full = ExperienceRelabeler(router, RelabelConfig()).label_episode(episode)
    assert full.preview is False
    assert full.labeled_frames == 3
    assert (episode / "teacher_labels.preview.jsonl").exists()
    assert (episode / "teacher_labels.jsonl").exists()


def test_contiguous_segments_do_not_bridge_handoff_gap():
    assert contiguous_segments([0, 1, 2, 8, 9, 15]) == [[0, 1, 2], [8, 9], [15]]
