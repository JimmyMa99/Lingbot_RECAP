import json

from lingbot_recap.journal import ExperienceJournal


def test_journal_is_separate_from_sft_and_atomic(tmp_path):
    journal = ExperienceJournal(tmp_path, "夹起黄色鸭子", "checkpoint-8")
    journal.frame(
        observation={"state": [1, 2, 3]},
        proposed_action={"action": [2, 3, 4]},
        executed_action={"action": [2, 3, 4]},
        action_source="policy",
        control_mode="auto",
        image_jpegs={"top": b"jpeg"},
    )
    completed = journal.close("success")
    assert completed.suffix == ".complete"
    assert (completed / "DO_NOT_ADD_TO_SFT").exists()
    metadata = json.loads((completed / "metadata.json").read_text())
    assert metadata["training_use"] == "RECAP_RL_EXPERIENCE_ONLY_NOT_SFT"
    assert len((completed / "frames.jsonl").read_text().splitlines()) == 1


def test_incomplete_episode_is_recoverable(tmp_path):
    journal = ExperienceJournal(tmp_path, "task", "checkpoint")
    journal.abort("simulated crash")
    recovered = ExperienceJournal.recover_incomplete(tmp_path)
    assert recovered == [journal.path]
    assert (journal.path / "RECOVERED_AFTER_UNCLEAN_SHUTDOWN").exists()
