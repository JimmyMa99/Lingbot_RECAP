#!/usr/bin/env python3
"""Relabel a completed RECAP episode while preserving an append-only audit event."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("outcome", choices=("success", "failure", "aborted"))
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    episode = args.episode.resolve()
    if not episode.is_dir() or not episode.name.endswith(".complete"):
        raise SystemExit("只能重新标注 .complete episode")
    result_path = episode / "result.json"
    events_path = episode / "events.jsonl"
    result = json.loads(result_path.read_text())
    previous = result.get("outcome")
    if previous == args.outcome:
        print(f"unchanged {episode}: {previous}")
        return
    event = {
        "timestamp": time.time(),
        "event": "operator_outcome_relabelled",
        "details": {
            "previous_outcome": previous,
            "new_outcome": args.outcome,
            "reason": args.reason,
        },
    }
    with events_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    result.update(
        outcome=args.outcome,
        relabelled_at_unix=event["timestamp"],
        previous_outcome=previous,
        relabel_reason=args.reason,
    )
    atomic_json(result_path, result)
    print(f"relabelled {episode}: {previous} -> {args.outcome}")


if __name__ == "__main__":
    main()
