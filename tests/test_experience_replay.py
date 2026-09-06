import numpy as np

from lingbot_recap.experience_replay import (
    PhaseSlice,
    chunk_actions,
    decision_indices,
    find_grasp_boundary,
    replay_batch,
)


def test_split_and_phase_chunk_contract():
    gripper = np.r_[np.full(12, 20.0), np.zeros(8), np.full(10, 5.0)]
    assert find_grasp_boundary(gripper) == 12
    phase = PhaseSlice("grasp", 0, 13)
    indices = decision_indices(phase, 4)
    assert indices.tolist() == [0, 4, 8, 12]
    actions = np.zeros((30, 6), dtype=np.float32)
    chunks = chunk_actions(actions, indices, phase, 16)
    assert chunks.shape == (4, 16, 6)


def test_replay_terminal_success_and_failure():
    states = np.zeros((3, 524), dtype=np.float32)
    actions = references = np.zeros((3, 16, 6), dtype=np.float32)
    success = replay_batch(states=states, actions=actions, references=references, success=True)
    failure = replay_batch(states=states, actions=actions, references=references, success=False)
    assert success["done"].tolist() == [0, 0, 1]
    assert success["reward"].tolist() == [0, 0, 1]
    assert failure["reward"].tolist() == [0, 0, 0]
