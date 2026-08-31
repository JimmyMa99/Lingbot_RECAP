from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .hardware import MOTOR_NAMES
from .multi_policy import MultiPolicyRouter


LABEL_FORMAT = "lingbot_multi_policy_teacher_labels_v1"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _state_mapping(frame: Mapping[str, object]) -> dict[str, float]:
    state = frame.get("observation", {}).get("state")
    if isinstance(state, Mapping):
        missing = [name for name in MOTOR_NAMES if name not in state]
        if missing:
            raise ValueError(f"state is missing joints: {missing}")
        return {name: float(state[name]) for name in MOTOR_NAMES}
    if isinstance(state, list) and len(state) == len(MOTOR_NAMES):
        return {name: float(state[index]) for index, name in enumerate(MOTOR_NAMES)}
    raise ValueError(f"invalid observation.state: {state!r}")


def contiguous_segments(indices: Iterable[int]) -> list[list[int]]:
    segments: list[list[int]] = []
    for index in sorted(indices):
        if not segments or index != segments[-1][-1] + 1:
            segments.append([index])
        else:
            segments[-1].append(index)
    return segments


@dataclass(frozen=True)
class RelabelConfig:
    stride: int = 1
    max_frames: int | None = None
    allowed_action_sources: tuple[str, ...] = ("lingbot_policy",)

    def __post_init__(self):
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be >= 1")


@dataclass(frozen=True)
class RelabelSummary:
    episode: Path
    task: str
    teacher_key: str
    labeled_frames: int
    skipped_frames: int
    output: Path


class ExperienceRelabeler:
    """Label student-visited states with an exact task-routed teacher policy."""

    def __init__(self, router: MultiPolicyRouter, config: RelabelConfig | None = None):
        self.router = router
        self.config = config or RelabelConfig()

    def label_episode(self, episode: str | Path, overwrite: bool = False) -> RelabelSummary:
        episode = Path(episode)
        metadata_path = episode / "metadata.json"
        frames_path = episode / "frames.jsonl"
        if not metadata_path.exists() or not frames_path.exists():
            raise ValueError(f"not a RECAP experience directory: {episode}")
        metadata = _read_json(metadata_path)
        task = str(metadata["task"])
        teacher = self.router.teacher_for(task)
        output = episode / "teacher_labels.jsonl"
        partial = episode / "teacher_labels.partial.jsonl"
        meta_output = episode / "teacher_labels.meta.json"

        if output.exists() and not overwrite:
            count = sum(1 for _ in _iter_jsonl(output))
            return RelabelSummary(episode, task, teacher.key, count, 0, output)
        if overwrite:
            output.unlink(missing_ok=True)
            partial.unlink(missing_ok=True)
            meta_output.unlink(missing_ok=True)

        already_labeled: set[int] = set()
        if partial.exists():
            already_labeled = {
                int(row["frame_index"]) for row in _iter_jsonl(partial)
            }

        labeled = len(already_labeled)
        skipped = 0
        with partial.open("a", encoding="utf-8") as handle:
            for frame in _iter_jsonl(frames_path):
                frame_index = int(frame["frame_index"])
                if frame_index in already_labeled:
                    continue
                if frame_index % self.config.stride:
                    skipped += 1
                    continue
                if frame.get("action_source") not in self.config.allowed_action_sources:
                    skipped += 1
                    continue
                if self.config.max_frames is not None and labeled >= self.config.max_frames:
                    break

                image_jpegs = {}
                for name, relative in frame.get("images", {}).items():
                    image_path = episode / str(relative)
                    if not image_path.exists():
                        raise FileNotFoundError(image_path)
                    image_jpegs[str(name)] = image_path.read_bytes()
                if set(image_jpegs) != {"top", "wrist"}:
                    raise ValueError(
                        f"frame {frame_index} must contain top and wrist images; "
                        f"got {sorted(image_jpegs)}"
                    )

                state = _state_mapping(frame)
                routed = self.router.infer(task, state, image_jpegs)
                chunk = routed.result.chunk
                row = {
                    "format": LABEL_FORMAT,
                    "frame_index": frame_index,
                    "source_timestamp": frame.get("timestamp"),
                    "task": task,
                    "teacher_key": routed.teacher.key,
                    "teacher_checkpoint": routed.teacher.checkpoint,
                    "teacher_server": routed.teacher.server,
                    "teacher_timing_ms": routed.result.timing_ms,
                    "teacher_action": [float(value) for value in chunk[0]],
                    "teacher_chunk": chunk.tolist(),
                    "student_action": frame.get("proposed_action", {}),
                }
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                labeled += 1

        if not partial.exists() or partial.stat().st_size == 0:
            raise RuntimeError(f"no eligible student-policy frames in {episode}")
        os.replace(partial, output)
        _atomic_json(
            meta_output,
            {
                "format": LABEL_FORMAT,
                "episode": episode.name,
                "task": task,
                "teacher_key": teacher.key,
                "teacher_checkpoint": teacher.checkpoint,
                "stride": self.config.stride,
                "allowed_action_sources": list(self.config.allowed_action_sources),
                "labeled_frames": labeled,
                "skipped_frames": skipped,
                "created_at_unix": time.time(),
                "training_use": "MULTI_POLICY_ON_POLICY_DISTILLATION_ONLY",
            },
        )
        return RelabelSummary(episode, task, teacher.key, labeled, skipped, output)


def find_experiences(root: str | Path, include_partial: bool = False) -> list[Path]:
    root = Path(root)
    paths = sorted(root.glob("episode_*.complete"))
    if include_partial:
        paths.extend(sorted(root.glob("episode_*.partial")))
    return paths


def load_labeled_frames(episode: str | Path) -> tuple[dict, list[tuple[dict, dict]]]:
    episode = Path(episode)
    metadata = _read_json(episode / "metadata.json")
    frames = {int(row["frame_index"]): row for row in _iter_jsonl(episode / "frames.jsonl")}
    labels = list(_iter_jsonl(episode / "teacher_labels.jsonl"))
    paired = []
    for label in labels:
        frame_index = int(label["frame_index"])
        if frame_index not in frames:
            raise ValueError(f"teacher label references missing frame {frame_index} in {episode}")
        if label.get("task") != metadata.get("task"):
            raise ValueError(f"task mismatch in teacher label frame {frame_index}")
        paired.append((frames[frame_index], label))
    return metadata, paired
