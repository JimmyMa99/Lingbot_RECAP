import sys

import pytest

from lingbot_recap.orchestrator import (
    IterationConfig,
    IterationRunner,
    build_training_manifest,
    iteration_lock,
)


def test_manifest_mixes_distillation_and_explicit_replay(tmp_path):
    replay = tmp_path / "replay.txt"
    replay.write_text(
        "# clean replay\nso_arm101 /data/pick\n\nso_arm101 /data/box\n"
    )
    output = build_training_manifest(
        tmp_path / "train.txt",
        tmp_path / "distilled",
        replay,
        replay_repeat=2,
    )
    assert output.read_text().splitlines() == [
        f"so_arm101 {tmp_path / 'distilled'}",
        "so_arm101 /data/pick",
        "so_arm101 /data/box",
        "so_arm101 /data/pick",
        "so_arm101 /data/box",
    ]


def test_iteration_lock_rejects_a_second_owner(tmp_path):
    lock = tmp_path / ".lock"
    with iteration_lock(lock):
        with pytest.raises(RuntimeError, match="already running"):
            with iteration_lock(lock):
                pass
    assert not lock.exists()


def test_completed_external_stage_is_not_run_twice(tmp_path):
    config = IterationConfig(
        iteration=1,
        teacher_registry=tmp_path / "teachers.json",
        experience_root=tmp_path / "experience",
        run_root=tmp_path / "runs",
        repo_id="mzm/test",
    )
    runner = IterationRunner(config)
    runner.path.mkdir(parents=True)
    state = runner._new_state()
    command = (sys.executable, "-c", "print('first run')")
    runner._run_command(state, "train", command, {})
    assert state["stages"]["train"]["status"] == "complete"
    runner._run_command(
        state,
        "train",
        (sys.executable, "-c", "raise SystemExit(99)"),
        {},
    )
    assert state["stages"]["train"]["status"] == "complete"
