from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .cameras import OpenCVCameraRig
from .detectors import DetectorSuite
from .handoff import HandoffCoordinator
from .hardware import MOTOR_NAMES, SO101BusArm
from .inputs import EventSource
from .journal import ExperienceJournal
from .notifier import ConsoleNotifier
from .policy import LingBotHTTPPolicy, action_dict
from .types import ChunkSample, ControlMode, InputEvent


@dataclass(frozen=True)
class CollectorConfig:
    task: str
    policy_checkpoint: str
    experience_root: Path
    fps: float = 30.0
    execute_length: int = 16
    detectors_enabled: bool = True


class ExperienceCollector:
    def __init__(
        self,
        config: CollectorConfig,
        follower: SO101BusArm,
        leader: SO101BusArm,
        cameras: OpenCVCameraRig,
        policy: LingBotHTTPPolicy,
        events: EventSource,
        notifier: ConsoleNotifier | None = None,
    ):
        self.config = config
        self.follower = follower
        self.leader = leader
        self.cameras = cameras
        self.policy = policy
        self.events = events
        self.notifier = notifier or ConsoleNotifier()
        self.detectors = DetectorSuite()
        self.journal = ExperienceJournal(
            config.experience_root,
            config.task,
            config.policy_checkpoint,
            extra_metadata={
                "fps": config.fps,
                "execute_length": config.execute_length,
            },
        )
        self.handoff = HandoffCoordinator(
            follower,
            leader,
            event_callback=lambda name, details: self.journal.event(name, details),
        )
        self.running = True
        self.outcome = "aborted"

    def _handle_common_event(self, event: InputEvent | None) -> bool:
        if event is InputEvent.SUCCESS:
            self.outcome = "success"
            self.running = False
            return True
        if event is InputEvent.FAILURE:
            self.outcome = "failure"
            self.running = False
            return True
        if event is InputEvent.QUIT:
            self.outcome = "aborted"
            self.running = False
            return True
        return False

    def _align_leader(self) -> None:
        self.notifier.announce("主臂即将自动对齐，请松手并远离关节")
        time.sleep(1.0)
        self.handoff.align_leader()
        self.notifier.announce("主臂已对齐。按按键 2 卸力并开始人工接管")

    def _release_leader(self) -> None:
        self.handoff.release_leader_for_human()
        self.detectors.reset()
        self.notifier.announce("主臂已卸力并确认，可以人工接管")

    def _pending_step(self) -> None:
        event = self.events.poll()
        if self._handle_common_event(event):
            return
        if event in (InputEvent.ALIGN_LEADER, InputEvent.TAKEOVER_OR_HAND_BACK):
            self._align_leader()
        elif event is InputEvent.RESUME_AUTO:
            self.handoff.resume_auto()
            self.detectors.reset()
            self.notifier.announce("已恢复自动控制")
        time.sleep(0.02)

    def _aligned_step(self) -> None:
        event = self.events.poll()
        if self._handle_common_event(event):
            return
        if event in (InputEvent.RELEASE_LEADER, InputEvent.TAKEOVER_OR_HAND_BACK):
            self._release_leader()
        elif event is InputEvent.RESUME_AUTO:
            self.handoff.resume_auto()
            self.detectors.reset()
            self.notifier.announce("已取消接管并恢复自动控制")
        time.sleep(0.02)

    def _human_step(self) -> None:
        start = time.perf_counter()
        event = self.events.poll()
        if self._handle_common_event(event):
            return
        if event is InputEvent.TAKEOVER_OR_HAND_BACK:
            self.handoff.resume_auto()
            self.detectors.reset()
            self.notifier.announce("人工接管结束，恢复自动控制")
            return
        image_jpegs = self.cameras.capture_jpegs()
        before = self.follower.read_positions()
        executed = self.handoff.human_step()
        self.journal.frame(
            observation={"state": before},
            proposed_action=None,
            executed_action=executed,
            action_source="human_intervention",
            control_mode=self.handoff.mode.value,
            image_jpegs=image_jpegs,
        )
        time.sleep(max(0.0, 1.0 / self.config.fps - (time.perf_counter() - start)))

    def _auto_chunk(self) -> None:
        state = self.follower.read_positions()
        images = self.cameras.capture_jpegs()
        result = self.policy.infer(self.config.task, state, images)
        proposed_first = action_dict(result.chunk[0])
        detection = None
        if self.config.detectors_enabled:
            detection = self.detectors.update(
                ChunkSample(
                    timestamp=time.monotonic(),
                    joints=[state[name] for name in MOTOR_NAMES],
                    proposed_action=[proposed_first[name] for name in MOTOR_NAMES],
                )
            )
        self.journal.event(
            "policy_chunk",
            {"timing_ms": result.timing_ms, "chunk_length": len(result.chunk)},
        )
        if detection is not None:
            self.handoff.request_takeover(detection.reason, detection.details)
            self.notifier.announce(
                "检测到策略卡住，已暂停。按按键 1/空格对齐主臂，按 R 恢复自动"
            )
            return
        for action in result.chunk[: self.config.execute_length]:
            if not self.running or self.handoff.mode is not ControlMode.AUTO:
                break
            event = self.events.poll()
            if self._handle_common_event(event):
                break
            if event is InputEvent.ALIGN_LEADER:
                self.handoff.request_takeover("manual_button_1")
                self._align_leader()
                break
            if event is InputEvent.TAKEOVER_OR_HAND_BACK:
                self.handoff.request_takeover("manual_space_key")
                self._align_leader()
                break
            started = time.perf_counter()
            image_jpegs = self.cameras.capture_jpegs()
            before = self.follower.read_positions()
            proposed = action_dict(action)
            self.handoff.policy_command(proposed)
            self.journal.frame(
                observation={"state": before},
                proposed_action=proposed,
                executed_action=proposed,
                action_source="lingbot_policy",
                control_mode=self.handoff.mode.value,
                image_jpegs=image_jpegs,
            )
            time.sleep(max(0.0, 1.0 / self.config.fps - (time.perf_counter() - started)))

    def run(self) -> Path:
        self.journal.event("controls", {
            "space": "request/confirm takeover or hand back",
            "button_1": "pause policy and align leader to follower",
            "button_2": "release aligned leader and grant human control",
            "s": "success and save",
            "f": "failure and save",
            "r": "resume from detector pause",
            "q_or_esc": "abort but preserve data",
        })
        follower_connected = False
        leader_connected = False
        cameras_connected = False
        try:
            self.follower.connect()
            follower_connected = True
            self.leader.connect()
            leader_connected = True
            self.cameras.connect()
            cameras_connected = True
            leader_torque_state = self.leader.disable_torque_verified()
            self.journal.event(
                "startup_leader_torque_verified",
                {"leader_torque_enable": leader_torque_state},
            )
            self.follower.enable_torque()
            while self.running:
                if self.handoff.mode is ControlMode.AUTO:
                    self._auto_chunk()
                elif self.handoff.mode is ControlMode.TAKEOVER_PENDING:
                    self._pending_step()
                elif self.handoff.mode is ControlMode.LEADER_ALIGNED:
                    self._aligned_step()
                elif self.handoff.mode is ControlMode.HUMAN:
                    self._human_step()
                elif self.handoff.mode is ControlMode.FAULT:
                    raise RuntimeError("handoff entered FAULT; follower is holding position")
                else:
                    break
            self.handoff.stop(self.outcome)
            return self.journal.close(self.outcome)
        except BaseException as exc:
            self.journal.event("runtime_exception", {"error": repr(exc)})
            self.journal.abort(repr(exc))
            raise
        finally:
            if cameras_connected:
                self.cameras.disconnect()
            try:
                if leader_connected:
                    self.leader.disconnect(disable_torque=True)
            finally:
                if follower_connected:
                    self.follower.disconnect(disable_torque=True)
