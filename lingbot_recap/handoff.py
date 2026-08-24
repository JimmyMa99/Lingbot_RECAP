from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .hardware import AlignmentConfig, Arm, align_leader_to_follower
from .types import ControlMode


@dataclass(frozen=True)
class HandoffConfig:
    alignment: AlignmentConfig = AlignmentConfig()


class HandoffCoordinator:
    """Single-writer control arbitration for policy and human commands."""

    def __init__(
        self,
        follower: Arm,
        leader: Arm,
        event_callback: Callable[[str, Mapping], None] | None = None,
        config: HandoffConfig = HandoffConfig(),
    ):
        self.follower = follower
        self.leader = leader
        self.config = config
        self.mode = ControlMode.AUTO
        self._emit = event_callback or (lambda _name, _details: None)
        self._hold_position: dict[str, float] | None = None

    def request_takeover(self, reason: str, details: Mapping | None = None) -> None:
        if self.mode is not ControlMode.AUTO:
            return
        self._hold_position = self.follower.read_positions()
        self.follower.command_positions(self._hold_position)
        self.mode = ControlMode.TAKEOVER_PENDING
        self._emit("takeover_requested", {"reason": reason, **dict(details or {})})

    def confirm_takeover(self) -> None:
        if self.mode is not ControlMode.TAKEOVER_PENDING or self._hold_position is None:
            raise RuntimeError(f"cannot confirm takeover from {self.mode}")
        self.mode = ControlMode.ALIGNING_LEADER
        self._emit("leader_alignment_started", {})
        try:
            align_leader_to_follower(self.leader, self._hold_position, self.config.alignment)
            torque_state = self.leader.disable_torque_verified()
        except Exception as exc:
            self.mode = ControlMode.FAULT
            self.follower.command_positions(self._hold_position)
            self._emit("takeover_failed", {"error": repr(exc)})
            raise
        self.mode = ControlMode.HUMAN
        self._emit("human_control_granted", {"leader_torque_enable": torque_state})

    def policy_command(self, positions: Mapping[str, float]) -> None:
        if self.mode is not ControlMode.AUTO:
            raise RuntimeError(f"policy is not controller in mode={self.mode}")
        self.follower.command_positions(positions)

    def human_step(self) -> dict[str, float]:
        if self.mode is not ControlMode.HUMAN:
            raise RuntimeError(f"human is not controller in mode={self.mode}")
        action = self.leader.read_positions()
        self.follower.command_positions(action)
        return action

    def resume_auto(self) -> None:
        if self.mode not in (ControlMode.HUMAN, ControlMode.TAKEOVER_PENDING):
            raise RuntimeError(f"cannot resume auto from {self.mode}")
        self._hold_position = self.follower.read_positions()
        self.follower.command_positions(self._hold_position)
        self.mode = ControlMode.AUTO
        self._emit("auto_control_resumed", {})

    def stop(self, reason: str) -> None:
        self._hold_position = self.follower.read_positions()
        self.follower.command_positions(self._hold_position)
        self.mode = ControlMode.STOPPED
        self._emit("control_stopped", {"reason": reason})
