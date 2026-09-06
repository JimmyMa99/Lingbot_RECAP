import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lingbot_recap.online_rl import (  # noqa: E402
    OnlineRLAgent,
    OnlineRLConfig,
    SO101ActionCodec,
    sparse_terminal_rewards,
)


def replay_batch(config: OnlineRLConfig, count: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    done = np.zeros(count, dtype=np.float32)
    done[-1] = 1.0
    return {
        "state": rng.normal(size=(count, config.state_dim)).astype(np.float32),
        "action": rng.uniform(-1, 1, (count, config.chunk_size, 6)).astype(np.float32),
        "reference": rng.uniform(-1, 1, (count, config.chunk_size, 6)).astype(np.float32),
        "reward": sparse_terminal_rewards(done, True),
        "next_state": rng.normal(size=(count, config.state_dim)).astype(np.float32),
        "next_reference": rng.uniform(-1, 1, (count, config.chunk_size, 6)).astype(np.float32),
        "done": done,
        "intervention": np.zeros(count, dtype=np.float32),
    }


def test_action_codec_round_trip():
    action = np.asarray([[-80, -20, 0, 40, 90, 75]], dtype=np.float32)
    np.testing.assert_allclose(SO101ActionCodec.denormalize(SO101ActionCodec.normalize(action)), action)


def test_actor_starts_exactly_at_frozen_reference():
    config = OnlineRLConfig(feature_dim=8, chunk_size=4, batch_size=2)
    agent = OnlineRLAgent(config, device="cpu")
    state = np.zeros(config.state_dim, dtype=np.float32)
    reference = np.linspace(-0.8, 0.8, config.flat_action_dim, dtype=np.float32).reshape(4, 6)
    action, mean = agent.act(state, reference, explore=False)
    np.testing.assert_array_equal(action, reference)
    np.testing.assert_array_equal(mean, reference)


def test_training_step_is_finite():
    config = OnlineRLConfig(feature_dim=8, chunk_size=4, hidden_dim=32, batch_size=4)
    agent = OnlineRLAgent(config, device="cpu")
    agent.replay.add_batch(replay_batch(config, 8))
    first = agent.train_step()
    second = agent.train_step()
    assert first is not None and second is not None
    assert all(np.isfinite(value) for value in first.values())
    assert all(np.isfinite(value) for value in second.values())
    assert "actor_loss" not in first
    assert "actor_loss" in second


def test_terminal_reward_contract():
    done = np.asarray([0, 0, 1, 1], dtype=np.float32)
    np.testing.assert_array_equal(sparse_terminal_rewards(done, True), done)
    np.testing.assert_array_equal(sparse_terminal_rewards(done, False), np.zeros(4))
    with pytest.raises(ValueError):
        sparse_terminal_rewards(np.asarray([0, 1, 0], dtype=np.float32), True)
