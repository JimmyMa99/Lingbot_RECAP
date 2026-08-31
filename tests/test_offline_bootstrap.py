from pathlib import Path

import pytest

from lingbot_recap.offline_bootstrap import (
    parse_episode_selector,
    parse_lingbot_manifest,
)


def test_parse_episode_selector_ranges_and_deduplication():
    assert parse_episode_selector("0-3,5,7-8,5") == (0, 1, 2, 3, 5, 7, 8)


def test_parse_episode_selector_rejects_reverse_range():
    with pytest.raises(ValueError, match="invalid episode range"):
        parse_episode_selector("4-2")


def test_parse_lingbot_manifest(tmp_path: Path):
    manifest = tmp_path / "train.txt"
    manifest.write_text(
        "# clean subset\n"
        "so_arm101 /data/pick::episodes=0-2,4\n"
        "so_arm101 /data/place\n",
        encoding="utf-8",
    )
    parsed = parse_lingbot_manifest(manifest)
    assert parsed[0].root == Path("/data/pick")
    assert parsed[0].episodes == (0, 1, 2, 4)
    assert parsed[1].root == Path("/data/place")
    assert parsed[1].episodes is None
