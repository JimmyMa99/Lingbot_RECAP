"""Convert sealed RECAP intervention episodes into phase replay transitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .online_rl import SO101ActionCodec, sparse_terminal_rewards


JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclass(frozen=True)
class PhaseSlice:
    name: str
    start: int
    stop: int


def joint_vector(value: dict) -> np.ndarray:
    result = np.asarray([value[key] for key in JOINTS], dtype=np.float32)
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ValueError("SO-101 joint vector is invalid")
    return result


def find_grasp_boundary(
    gripper: np.ndarray,
    *,
    threshold: float = 1.0,
    sustained_frames: int = 3,
    minimum_prefix: int = 8,
) -> int:
    """Return the first sustained close; fall back to the strongest close."""

    values = np.asarray(gripper, dtype=np.float32).reshape(-1)
    if values.size < minimum_prefix + sustained_frames + 2 or not np.isfinite(values).all():
        raise ValueError("not enough finite gripper samples to split phases")
    for index in range(minimum_prefix, values.size - sustained_frames + 1):
        if np.all(values[index : index + sustained_frames] <= threshold):
            return index
    return max(minimum_prefix, int(np.argmin(values)))


def phase_slices(frame_count: int, boundary: int) -> tuple[PhaseSlice, PhaseSlice]:
    if not 1 <= boundary < frame_count - 1:
        raise ValueError(f"invalid phase boundary {boundary}/{frame_count}")
    return PhaseSlice("grasp", 0, boundary + 1), PhaseSlice("place", boundary, frame_count)


def decision_indices(phase: PhaseSlice, stride: int) -> np.ndarray:
    if stride <= 0:
        raise ValueError("stride must be positive")
    indices = np.arange(phase.start, phase.stop, stride, dtype=np.int64)
    if indices.size == 0 or indices[-1] != phase.stop - 1:
        indices = np.append(indices, phase.stop - 1)
    return np.unique(indices)


def chunk_actions(actions: np.ndarray, indices: np.ndarray, phase: PhaseSlice, chunk_size: int) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float32)
    result = np.empty((len(indices), chunk_size, 6), dtype=np.float32)
    for row, index in enumerate(indices):
        offsets = np.minimum(np.arange(chunk_size) + index, phase.stop - 1)
        result[row] = SO101ActionCodec.normalize(values[offsets])
    return result


def replay_batch(
    *,
    states: np.ndarray,
    actions: np.ndarray,
    references: np.ndarray,
    success: bool,
) -> dict[str, np.ndarray]:
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    references = np.asarray(references, dtype=np.float32)
    count = states.shape[0]
    if count == 0 or actions.shape[0] != count or references.shape[0] != count:
        raise ValueError("replay arrays have inconsistent sample counts")
    done = np.zeros(count, dtype=np.float32)
    done[-1] = 1.0
    return {
        "state": states,
        "action": actions,
        "reference": references,
        "reward": sparse_terminal_rewards(done, success),
        "next_state": np.concatenate((states[1:], states[-1:])),
        "next_reference": np.concatenate((references[1:], references[-1:])),
        "done": done,
        "intervention": np.ones(count, dtype=np.float32),
    }
