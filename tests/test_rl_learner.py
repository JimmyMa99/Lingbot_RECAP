import numpy as np
import pytest

pytest.importorskip("torch")

from lingbot_recap.online_rl import OnlineRLConfig, sparse_terminal_rewards  # noqa: E402
from lingbot_recap.rl_learner import LearnerConfig, PhaseLearner  # noqa: E402


def batch(config: OnlineRLConfig, count: int = 4):
    rng = np.random.default_rng(4)
    done = np.zeros(count, dtype=np.float32)
    done[-1] = 1
    return {
        "state": rng.normal(size=(count, config.state_dim)).astype(np.float32),
        "action": rng.uniform(-1, 1, (count, config.chunk_size, 6)).astype(np.float32),
        "reference": rng.uniform(-1, 1, (count, config.chunk_size, 6)).astype(np.float32),
        "reward": sparse_terminal_rewards(done, True),
        "next_state": rng.normal(size=(count, config.state_dim)).astype(np.float32),
        "next_reference": rng.uniform(-1, 1, (count, config.chunk_size, 6)).astype(np.float32),
        "done": done,
        "intervention": np.asarray([0, 1, 1, 1], dtype=np.float32),
    }


def test_commit_is_idempotent_and_reloadable(tmp_path):
    agent_config = OnlineRLConfig(feature_dim=8, chunk_size=4, hidden_dim=16, batch_size=2)
    learner_config = LearnerConfig(warmup_episodes=1, warmup_successes=1, warmup_transitions=4)
    learner = PhaseLearner("grasp", tmp_path, agent_config, learner_config, device="cpu")
    value = batch(agent_config)
    first = learner.commit(
        commit_id="episode-a", collection_mode="warmup", success=True, batch=value
    )
    second = learner.commit(
        commit_id="episode-a", collection_mode="warmup", success=True, batch=value
    )
    assert first["duplicate"] is False
    assert second["duplicate"] is True
    assert learner.summary()["warmup_ready"] is True

    restored = PhaseLearner("grasp", tmp_path, agent_config, learner_config, device="cpu")
    assert restored.summary()["episodes"] == 1
    assert restored.summary()["transitions"] == 4


def test_rejects_reward_contract_drift(tmp_path):
    config = OnlineRLConfig(feature_dim=8, chunk_size=4, hidden_dim=16, batch_size=2)
    learner = PhaseLearner("place", tmp_path, config, device="cpu")
    value = batch(config)
    value["reward"][:] = 0
    with pytest.raises(ValueError, match="reward"):
        learner.commit(
            commit_id="episode-b", collection_mode="warmup", success=True, batch=value
        )
