import time

from lingbot_recap.detectors import NoProgressDetector, OscillationDetector
from lingbot_recap.types import ChunkSample


def test_oscillation_abab_is_detected():
    detector = OscillationDetector()
    result = None
    for index in range(8):
        value = 0.0 if index % 2 == 0 else 10.0
        result = detector.update(ChunkSample(index, [value] * 6))
    assert result is not None
    assert result.reason == "oscillation"


def test_monotonic_motion_is_not_oscillation():
    detector = OscillationDetector()
    for index in range(10):
        assert detector.update(ChunkSample(index, [float(index)] * 6)) is None


def test_command_without_motion_is_no_progress():
    detector = NoProgressDetector()
    result = None
    for index in range(10):
        result = detector.update(
            ChunkSample(index * 0.5, [0.0] * 6, proposed_action=[10.0] * 6)
        )
    assert result is not None
    assert result.reason == "no_progress"
