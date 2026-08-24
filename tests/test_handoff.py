from lingbot_recap.handoff import HandoffCoordinator
from lingbot_recap.hardware import AlignmentConfig
from lingbot_recap.handoff import HandoffConfig
from lingbot_recap.types import ControlMode


class FakeArm:
    def __init__(self, positions=None, torque_state=0):
        self.positions = positions or {name: 0.0 for name in (
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper"
        )}
        self.torque_state = torque_state

    def read_positions(self):
        return dict(self.positions)

    def command_positions(self, positions):
        self.positions = dict(positions)

    def enable_torque(self):
        self.torque_state = 1

    def disable_torque_verified(self):
        self.torque_state = 0
        return {name: 0 for name in self.positions}


class FailingTorqueArm(FakeArm):
    def disable_torque_verified(self):
        raise RuntimeError("readback failed")


def test_takeover_aligns_and_verifies_before_human_control():
    follower = FakeArm({name: 5.0 for name in FakeArm().positions})
    leader = FakeArm()
    events = []
    coordinator = HandoffCoordinator(
        follower,
        leader,
        event_callback=lambda name, details: events.append(name),
        config=HandoffConfig(AlignmentConfig(duration_s=0, settle_reads=1)),
    )
    coordinator.request_takeover("manual")
    assert coordinator.mode is ControlMode.TAKEOVER_PENDING
    coordinator.confirm_takeover()
    assert coordinator.mode is ControlMode.HUMAN
    assert leader.torque_state == 0
    assert leader.positions == follower.positions
    assert events[-1] == "human_control_granted"


def test_two_button_handoff_does_not_unload_until_second_step():
    follower = FakeArm({name: 5.0 for name in FakeArm().positions})
    leader = FakeArm()
    coordinator = HandoffCoordinator(
        follower,
        leader,
        config=HandoffConfig(AlignmentConfig(duration_s=0, settle_reads=1)),
    )
    coordinator.request_takeover("button_1")
    coordinator.align_leader()
    assert coordinator.mode is ControlMode.LEADER_ALIGNED
    assert leader.torque_state == 1
    assert leader.positions == follower.positions
    coordinator.release_leader_for_human()
    assert coordinator.mode is ControlMode.HUMAN
    assert leader.torque_state == 0


def test_failed_torque_readback_never_grants_human_control():
    follower = FakeArm({name: 5.0 for name in FakeArm().positions})
    coordinator = HandoffCoordinator(
        follower,
        FailingTorqueArm(),
        config=HandoffConfig(AlignmentConfig(duration_s=0, settle_reads=1)),
    )
    coordinator.request_takeover("manual")
    try:
        coordinator.confirm_takeover()
    except RuntimeError as exc:
        assert "readback failed" in str(exc)
    else:
        raise AssertionError("takeover unexpectedly succeeded")
    assert coordinator.mode is ControlMode.FAULT
