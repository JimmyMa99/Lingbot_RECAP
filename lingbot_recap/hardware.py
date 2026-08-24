from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np


MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class Arm(Protocol):
    def read_positions(self) -> dict[str, float]: ...
    def command_positions(self, positions: Mapping[str, float]) -> None: ...
    def enable_torque(self) -> None: ...
    def disable_torque_verified(self) -> dict[str, int]: ...


@dataclass(frozen=True)
class AlignmentConfig:
    duration_s: float = 4.0
    frequency_hz: float = 30.0
    tolerance: float = 2.0
    settle_reads: int = 5


class TorqueVerificationError(RuntimeError):
    pass


class LeaderAlignmentError(RuntimeError):
    pass


def align_leader_to_follower(
    leader: Arm,
    follower_positions: Mapping[str, float],
    config: AlignmentConfig = AlignmentConfig(),
) -> None:
    """Actively align leader, then verify it is settled. Does not unload it."""
    leader.enable_torque()
    start = leader.read_positions()
    steps = max(1, round(config.duration_s * config.frequency_hz))
    for step in range(1, steps + 1):
        alpha = step / steps
        target = {
            name: float(start[name] + (follower_positions[name] - start[name]) * alpha)
            for name in MOTOR_NAMES
        }
        leader.command_positions(target)
        time.sleep(1.0 / config.frequency_hz)
    settled = 0
    for _ in range(config.settle_reads * 3):
        actual = leader.read_positions()
        error = max(abs(actual[name] - follower_positions[name]) for name in MOTOR_NAMES)
        if error <= config.tolerance:
            settled += 1
            if settled >= config.settle_reads:
                return
        else:
            settled = 0
        time.sleep(1.0 / config.frequency_hz)
    raise LeaderAlignmentError("leader failed to settle within alignment tolerance")


class SO101BusArm:
    """LeRobot 0.4.x Feetech adapter used by both leader and follower."""

    def __init__(self, port: str, calibration_path: str | Path):
        from lerobot.motors import Motor, MotorCalibration, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus

        calibration = {
            name: MotorCalibration(**value)
            for name, value in json.loads(Path(calibration_path).read_text()).items()
        }
        motors = {
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
            "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
            "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
            "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        }
        self.bus = FeetechMotorsBus(port=port, motors=motors, calibration=calibration)

    def connect(self) -> None:
        self.bus.connect()

    def disconnect(self, disable_torque: bool = True) -> None:
        self.bus.disconnect(disable_torque=disable_torque)

    def read_positions(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.bus.sync_read("Present_Position").items()}

    def command_positions(self, positions: Mapping[str, float]) -> None:
        values = {name: float(positions[name]) for name in MOTOR_NAMES}
        if not np.isfinite(list(values.values())).all():
            raise ValueError("refusing non-finite joint command")
        self.bus.sync_write("Goal_Position", values)

    def enable_torque(self) -> None:
        self.bus.enable_torque(num_retry=3)

    def hold_current_position(self) -> dict[str, float]:
        current = self.read_positions()
        self.command_positions(current)
        return current

    def disable_torque_verified(self) -> dict[str, int]:
        failures = []
        for name in MOTOR_NAMES:
            try:
                self.bus.disable_torque(motors=[name], num_retry=5)
            except Exception as exc:
                failures.append(f"{name}: write failed: {exc}")
        states = {}
        for name in MOTOR_NAMES:
            try:
                states[name] = int(self.bus.read("Torque_Enable", name, normalize=False))
            except Exception as exc:
                failures.append(f"{name}: readback failed: {exc}")
        enabled = [name for name, value in states.items() if value != 0]
        if failures or enabled or len(states) != len(MOTOR_NAMES):
            raise TorqueVerificationError(
                f"leader torque disable not verified; enabled={enabled}; failures={failures}"
            )
        return states
