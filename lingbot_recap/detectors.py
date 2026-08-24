from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np

from .types import ChunkSample, Detection


def _rms(vector: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(vector))))


@dataclass(frozen=True)
class OscillationConfig:
    window: int = 8
    min_samples: int = 6
    return_tolerance: float = 2.0
    min_swing: float = 4.0
    min_reversal_ratio: float = 0.60


class OscillationDetector:
    """Detect a repeated A-B-A-B pattern in normalized joint space."""

    def __init__(self, config: OscillationConfig = OscillationConfig()):
        self.config = config
        self.samples: Deque[ChunkSample] = deque(maxlen=config.window)

    def update(self, sample: ChunkSample) -> Detection | None:
        self.samples.append(sample)
        if len(self.samples) < self.config.min_samples:
            return None
        q = np.asarray([item.joints for item in self.samples], dtype=np.float64)
        adjacent = np.asarray([_rms(q[i] - q[i - 1]) for i in range(1, len(q))])
        period_two = np.asarray([_rms(q[i] - q[i - 2]) for i in range(2, len(q))])
        velocities = np.diff(q, axis=0)
        dots = np.sum(velocities[1:] * velocities[:-1], axis=1)
        reversal_ratio = float(np.mean(dots < 0)) if len(dots) else 0.0
        return_error = float(np.median(period_two))
        swing = float(np.median(adjacent))
        if (
            return_error <= self.config.return_tolerance
            and swing >= self.config.min_swing
            and reversal_ratio >= self.config.min_reversal_ratio
        ):
            score = min(1.0, (swing / self.config.min_swing) * reversal_ratio)
            return Detection(
                reason="oscillation",
                score=score,
                details={
                    "return_error": return_error,
                    "swing": swing,
                    "reversal_ratio": reversal_ratio,
                },
            )
        return None

    def reset(self) -> None:
        self.samples.clear()


@dataclass(frozen=True)
class NoProgressConfig:
    window: int = 10
    min_duration_s: float = 4.0
    max_state_displacement: float = 1.0
    min_commanded_displacement: float = 3.0
    max_progress_gain: float = 0.01


class NoProgressDetector:
    """Detect commanded motion without state/task progress.

    This is deliberately conservative. It pauses and asks for confirmation; it
    never grants human control or moves the leader by itself.
    """

    def __init__(self, config: NoProgressConfig = NoProgressConfig()):
        self.config = config
        self.samples: Deque[ChunkSample] = deque(maxlen=config.window)

    def update(self, sample: ChunkSample) -> Detection | None:
        self.samples.append(sample)
        if len(self.samples) < self.config.window:
            return None
        elapsed = self.samples[-1].timestamp - self.samples[0].timestamp
        if elapsed < self.config.min_duration_s:
            return None
        q0 = np.asarray(self.samples[0].joints, dtype=np.float64)
        q1 = np.asarray(self.samples[-1].joints, dtype=np.float64)
        state_displacement = _rms(q1 - q0)
        commanded = [
            _rms(np.asarray(item.proposed_action) - np.asarray(item.joints))
            for item in self.samples
            if item.proposed_action is not None
        ]
        command_displacement = float(np.median(commanded)) if commanded else 0.0
        progress = [item.task_progress for item in self.samples if item.task_progress is not None]
        progress_gain = float(progress[-1] - progress[0]) if len(progress) >= 2 else 0.0
        has_progress_signal = len(progress) >= 2
        no_task_progress = not has_progress_signal or progress_gain <= self.config.max_progress_gain
        if (
            state_displacement <= self.config.max_state_displacement
            and command_displacement >= self.config.min_commanded_displacement
            and no_task_progress
        ):
            return Detection(
                reason="no_progress",
                score=min(1.0, command_displacement / (2 * self.config.min_commanded_displacement)),
                details={
                    "elapsed_s": elapsed,
                    "state_displacement": state_displacement,
                    "command_displacement": command_displacement,
                    "progress_gain": progress_gain,
                },
            )
        return None

    def reset(self) -> None:
        self.samples.clear()


class DetectorSuite:
    def __init__(self):
        self.detectors = [OscillationDetector(), NoProgressDetector()]

    def update(self, sample: ChunkSample) -> Detection | None:
        for detector in self.detectors:
            result = detector.update(sample)
            if result is not None:
                return result
        return None

    def reset(self) -> None:
        for detector in self.detectors:
            detector.reset()
