from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


class ControlMode(str, Enum):
    AUTO = "auto"
    TAKEOVER_PENDING = "takeover_pending"
    ALIGNING_LEADER = "aligning_leader"
    LEADER_ALIGNED = "leader_aligned"
    HUMAN = "human"
    STOPPED = "stopped"
    FAULT = "fault"


class InputEvent(str, Enum):
    ALIGN_LEADER = "align_leader"
    RELEASE_LEADER = "release_leader"
    TAKEOVER_OR_HAND_BACK = "takeover_or_hand_back"
    SUCCESS = "success"
    FAILURE = "failure"
    RESUME_AUTO = "resume_auto"
    QUIT = "quit"


@dataclass(frozen=True)
class Detection:
    reason: str
    score: float
    details: Mapping[str, float]


@dataclass(frozen=True)
class ChunkSample:
    timestamp: float
    joints: Sequence[float]
    proposed_action: Sequence[float] | None = None
    task_progress: float | None = None
