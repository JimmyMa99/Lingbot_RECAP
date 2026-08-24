import json

import pytest

from lingbot_recap.inputs import LinuxTwoButtonEventSource
from lingbot_recap.types import InputEvent


def write_config(path, align=2, release=3):
    path.write_text(
        json.dumps(
            {
                "device": "/dev/input/by-id/test-event-kbd",
                "buttons": {
                    "align": {"code": align, "name": "KEY_1"},
                    "release": {"code": release, "name": "KEY_2"},
                },
            }
        )
    )


def test_two_button_config_maps_distinct_actions(tmp_path):
    config = tmp_path / "buttons.json"
    write_config(config)
    source = LinuxTwoButtonEventSource(config)
    assert source.keymap == {
        2: InputEvent.ALIGN_LEADER,
        3: InputEvent.RELEASE_LEADER,
    }


def test_two_button_config_rejects_duplicate_codes(tmp_path):
    config = tmp_path / "buttons.json"
    write_config(config, align=2, release=2)
    with pytest.raises(ValueError, match="different key codes"):
        LinuxTwoButtonEventSource(config)
