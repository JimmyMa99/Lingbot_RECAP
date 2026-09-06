import unittest
from unittest.mock import patch

from lingbot_recap.hardware import (
    MOTOR_NAMES, AlignmentConfig, LeaderAlignmentError,
    LeaderAlignmentCancelled, align_leader_to_follower,
)
from lingbot_recap.handoff import HandoffCoordinator, HandoffConfig
from lingbot_recap.types import ControlMode


class Clock:
    now = 0.0

    def sleep(self, seconds):
        self.now += seconds

    def monotonic(self):
        return self.now


class SimArm:
    def __init__(self, clock, *, stuck=None, delay=0, bias=None):
        self.clock = clock
        self.stuck = stuck
        self.delay = delay
        self.bias = bias or {}
        self.positions = dict.fromkeys(MOTOR_NAMES, 0.0)
        self.goal = dict(self.positions)
        self.history = []
        self.torque = False

    def read_positions(self):
        if self.clock.now >= self.delay and self.torque:
            self.positions.update({
                k: v + self.bias.get(k, 0.0)
                for k, v in self.goal.items() if k != self.stuck
            })
        return dict(self.positions)

    def command_positions(self, target):
        self.goal = dict(target)
        self.history.append(("command", dict(target)))

    def enable_torque(self):
        self.history.append(("enable", None))
        self.torque = True

    def disable_torque_verified(self):
        self.torque = False
        return dict.fromkeys(MOTOR_NAMES, 0)


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.patcher = patch("lingbot_recap.hardware.time", self.clock)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.target = dict.fromkeys(MOTOR_NAMES, 5.0)
        self.config = AlignmentConfig(duration_s=0, tolerance=2, settle_reads=2)

    def test_preloads_position_before_torque_and_keeps_torque_after_success(self):
        arm = SimArm(self.clock)
        align_leader_to_follower(arm, self.target, self.config)
        self.assertEqual(arm.history[:2], [("command", dict.fromkeys(MOTOR_NAMES, 0.0)), ("enable", None)])
        self.assertTrue(arm.torque)

    def test_slow_settling_beyond_old_half_second_budget_succeeds(self):
        arm = SimArm(self.clock, delay=1.0)
        align_leader_to_follower(arm, self.target, self.config)
        self.assertGreaterEqual(self.clock.now, 1.0)

    def test_stuck_joint_times_out_and_never_grants_human_control(self):
        leader = SimArm(self.clock, stuck="shoulder_lift")
        follower = SimArm(self.clock)
        follower.positions = dict(self.target)
        reports = []
        handoff = HandoffCoordinator(follower, leader,
            config=HandoffConfig(self.config),
            event_callback=lambda name, details: reports.append((name, details)))
        handoff.request_takeover("test")
        with self.assertRaisesRegex(LeaderAlignmentError, "shoulder_lift"):
            handoff.align_leader()
        self.assertEqual(handoff.mode, ControlMode.FAULT)
        self.assertLess(self.clock.now, 3.0)
        self.assertEqual(follower.goal, self.target)
        progress = [d for n, d in reports if n == "leader_alignment_progress"]
        self.assertEqual(progress[-1]["abs_error"]["shoulder_lift"], 5.0)
        self.assertNotIn("human_control_granted", [n for n, d in reports])

    def test_invalid_target_never_enables_or_commands(self):
        for invalid in (float("nan"), float("inf"), 101):
            arm = SimArm(self.clock)
            target = {**self.target, "shoulder_pan": invalid}
            with self.assertRaises(ValueError):
                align_leader_to_follower(arm, target, self.config)
            self.assertEqual(arm.history, [])

    def test_cancel_before_motion_never_enables(self):
        arm = SimArm(self.clock)
        with self.assertRaises(LeaderAlignmentCancelled):
            align_leader_to_follower(arm, self.target, self.config, cancelled=lambda: True)
        self.assertEqual(arm.history, [])

    def test_cancel_during_motion_stops_commands(self):
        arm = SimArm(self.clock)
        with self.assertRaises(LeaderAlignmentCancelled):
            align_leader_to_follower(arm, self.target, self.config,
                cancelled=lambda: self.clock.now >= 0.1)
        self.assertLess(arm.goal["shoulder_pan"], 5)

    def test_large_move_slows_down(self):
        arm = SimArm(self.clock)
        align_leader_to_follower(arm, dict.fromkeys(MOTOR_NAMES, 100.0), self.config)
        self.assertGreaterEqual(self.clock.now, 5.0)

    def test_success_requires_consecutive_settled_reads(self):
        arm = SimArm(self.clock)
        reports = []
        align_leader_to_follower(arm, self.target, self.config, progress=reports.append)
        self.assertEqual(len([r for r in reports if r["phase"] == "settling"]), 2)

    def test_runtime_polls_quit_during_alignment(self):
        from unittest.mock import Mock
        from lingbot_recap.runtime import ExperienceCollector
        from lingbot_recap.types import InputEvent

        collector = ExperienceCollector.__new__(ExperienceCollector)
        collector.notifier = Mock()
        collector.events = Mock()
        collector.running = True
        collector.outcome = "aborted"
        collector.events.poll.side_effect = lambda: InputEvent.QUIT if self.clock.now > 1.05 else None
        follower = SimArm(self.clock)
        follower.positions = dict(self.target)
        follower.goal = dict(self.target)
        follower.torque = True
        leader = SimArm(self.clock)
        collector.handoff = HandoffCoordinator(follower, leader, config=HandoffConfig(self.config))
        collector.handoff.request_takeover("test")
        with patch("lingbot_recap.runtime.time", self.clock):
            collector._align_leader()
        self.assertFalse(collector.running)
        self.assertEqual(collector.handoff.mode, ControlMode.FAULT)
        self.assertLess(leader.goal["shoulder_pan"], 5)

    def test_runtime_alignment_failure_keeps_both_arms_holding_and_allows_retry(self):
        from unittest.mock import Mock
        from lingbot_recap.runtime import ExperienceCollector

        collector = ExperienceCollector.__new__(ExperienceCollector)
        collector.notifier = Mock()
        collector.events = Mock()
        collector.events.poll.return_value = None
        collector.running = True
        collector.outcome = "aborted"
        follower = SimArm(self.clock)
        follower.positions = dict(self.target)
        follower.goal = dict(self.target)
        follower.torque = True
        leader = SimArm(self.clock, stuck="shoulder_lift")
        collector.handoff = HandoffCoordinator(
            follower, leader, config=HandoffConfig(self.config)
        )
        collector.handoff.request_takeover("test")
        with patch("lingbot_recap.runtime.time", self.clock):
            collector._align_leader()
        self.assertTrue(collector.running)
        self.assertEqual(collector.handoff.mode, ControlMode.TAKEOVER_PENDING)
        self.assertEqual(follower.goal, self.target)
        self.assertTrue(follower.torque)
        self.assertTrue(leader.torque)
        self.assertEqual(leader.goal, leader.positions)

    def test_default_tolerance_accepts_observed_elbow_static_error(self):
        self.assertGreater(AlignmentConfig().tolerance, 4.194)

    def test_bounded_settle_compensation_cancels_gravity_bias(self):
        arm = SimArm(self.clock, bias={"shoulder_lift": 5.1})
        reports = []
        align_leader_to_follower(
            arm, self.target,
            AlignmentConfig(duration_s=0, tolerance=2, settle_reads=2),
            progress=reports.append,
        )
        self.assertLessEqual(
            reports[-1]["abs_error"]["shoulder_lift"], 2
        )
        self.assertGreaterEqual(arm.goal["shoulder_lift"], -3)
        self.assertLess(arm.goal["shoulder_lift"], self.target["shoulder_lift"])

    def test_settle_compensation_is_bounded_for_stuck_joint(self):
        arm = SimArm(self.clock, stuck="shoulder_lift")
        with self.assertRaises(LeaderAlignmentError):
            align_leader_to_follower(
                arm, self.target,
                AlignmentConfig(duration_s=0, tolerance=2, settle_reads=2),
            )
        self.assertGreaterEqual(arm.goal["shoulder_lift"], -3)


if __name__ == "__main__":
    unittest.main()
