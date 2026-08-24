from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Mapping


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


class ExperienceJournal:
    """Append-only RECAP experience storage, intentionally separate from SFT data."""

    FORMAT = "lingbot_recap_experience_v1"

    def __init__(
        self,
        root: str | Path,
        task: str,
        policy_checkpoint: str,
        extra_metadata: Mapping[str, Any] | None = None,
        fsync_every: int = 10,
    ):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.root = Path(root)
        self.path = self.root / f"episode_{stamp}_{uuid.uuid4().hex[:8]}.partial"
        self.path.mkdir(parents=True, exist_ok=False)
        self.images_dir = self.path / "images"
        self.images_dir.mkdir()
        self.frames_path = self.path / "frames.jsonl"
        self.events_path = self.path / "events.jsonl"
        self._frames = self.frames_path.open("a", encoding="utf-8")
        self._events = self.events_path.open("a", encoding="utf-8")
        self.frame_index = 0
        self.fsync_every = max(1, fsync_every)
        metadata = {
            "format": self.FORMAT,
            "task": task,
            "policy_checkpoint": policy_checkpoint,
            "created_at_unix": time.time(),
            "training_use": "RECAP_RL_EXPERIENCE_ONLY_NOT_SFT",
            **dict(extra_metadata or {}),
        }
        _atomic_json(self.path / "metadata.json", metadata)
        (self.path / "DO_NOT_ADD_TO_SFT").touch()
        self.event("episode_started", metadata)

    @staticmethod
    def recover_incomplete(root: str | Path) -> list[Path]:
        recovered = []
        for path in Path(root).glob("episode_*.partial"):
            marker = path / "RECOVERED_AFTER_UNCLEAN_SHUTDOWN"
            if not marker.exists():
                marker.write_text(f"recovered_at_unix={time.time()}\n")
            recovered.append(path)
        return recovered

    def _append(self, handle, payload: Mapping[str, Any], force_sync: bool = False) -> None:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        if force_sync:
            os.fsync(handle.fileno())

    def event(self, name: str, details: Mapping[str, Any] | None = None) -> None:
        self._append(
            self._events,
            {"timestamp": time.time(), "event": name, "details": dict(details or {})},
            force_sync=True,
        )

    def frame(
        self,
        *,
        observation: Mapping[str, Any],
        proposed_action: Mapping[str, Any] | None,
        executed_action: Mapping[str, Any] | None,
        action_source: str,
        control_mode: str,
        image_jpegs: Mapping[str, bytes] | None = None,
    ) -> None:
        image_paths = {}
        for camera, data in (image_jpegs or {}).items():
            camera_dir = self.images_dir / camera
            camera_dir.mkdir(exist_ok=True)
            relative = Path("images") / camera / f"{self.frame_index:08d}.jpg"
            target = self.path / relative
            temporary = target.with_suffix(".jpg.tmp")
            temporary.write_bytes(data)
            os.replace(temporary, target)
            image_paths[camera] = str(relative)
        payload = {
            "frame_index": self.frame_index,
            "timestamp": time.time(),
            "observation": dict(observation),
            "proposed_action": dict(proposed_action or {}),
            "executed_action": dict(executed_action or {}),
            "action_source": action_source,
            "control_mode": control_mode,
            "images": image_paths,
        }
        self._append(self._frames, payload, force_sync=self.frame_index % self.fsync_every == 0)
        self.frame_index += 1

    def close(self, outcome: str, details: Mapping[str, Any] | None = None) -> Path:
        if self._frames.closed:
            return self.path
        self.event("episode_finished", {"outcome": outcome, **dict(details or {})})
        self._frames.flush()
        os.fsync(self._frames.fileno())
        self._frames.close()
        self._events.close()
        _atomic_json(
            self.path / "result.json",
            {"outcome": outcome, "finished_at_unix": time.time(), **dict(details or {})},
        )
        completed = self.path.with_suffix(".complete")
        os.replace(self.path, completed)
        self.path = completed
        return completed

    def abort(self, reason: str) -> None:
        if not self._frames.closed:
            self.event("episode_aborted", {"reason": reason})
            self._frames.close()
            self._events.close()
