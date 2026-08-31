from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .hardware import MOTOR_NAMES


@dataclass(frozen=True)
class ManifestDataset:
    root: Path
    episodes: tuple[int, ...] | None


@dataclass(frozen=True)
class ImportSummary:
    output_root: Path
    datasets: int
    episodes: int
    frames: int
    skipped_existing: int


def parse_episode_selector(value: str) -> tuple[int, ...]:
    episodes: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"invalid episode range: {part}")
            episodes.update(range(start, end + 1))
        else:
            episodes.add(int(part))
    if not episodes:
        raise ValueError("empty episode selector")
    return tuple(sorted(episodes))


def parse_lingbot_manifest(path: str | Path) -> list[ManifestDataset]:
    result: list[ManifestDataset] = []
    path = Path(path)
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"invalid manifest line {path}:{line_number}")
        _, dataset_spec = fields
        root_text, marker, selector = dataset_spec.partition("::episodes=")
        episodes = parse_episode_selector(selector) if marker else None
        result.append(ManifestDataset(Path(root_text).resolve(), episodes))
    if not result:
        raise ValueError(f"manifest contains no datasets: {path}")
    return result


def _joint_mapping(values) -> dict[str, float]:
    array = values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)
    array = np.asarray(array, dtype=np.float32).reshape(-1)
    if array.shape != (len(MOTOR_NAMES),) or not np.isfinite(array).all():
        raise ValueError(f"invalid joint vector: shape={array.shape}")
    return {name: float(array[index]) for index, name in enumerate(MOTOR_NAMES)}


def _write_rgb(path: Path, value) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for offline bootstrap import") from exc
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    else:
        array = array.astype(np.uint8)
    if array.shape[-1] != 3:
        raise ValueError(f"expected RGB image, got {array.shape}")
    if not cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write image: {path}")


def import_lingbot_manifest(
    manifest: str | Path,
    output_root: str | Path,
    *,
    sample_stride: int = 15,
    max_episodes_per_dataset: int | None = None,
    overwrite: bool = False,
) -> ImportSummary:
    """Sample clean demos as an explicit offline bootstrap state set."""

    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")
    if max_episodes_per_dataset is not None and max_episodes_per_dataset < 1:
        raise ValueError("max_episodes_per_dataset must be >= 1")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError("LeRobot 0.4.2 is required; install .[robot]") from exc

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    specs = parse_lingbot_manifest(manifest)
    written_episodes = written_frames = skipped_existing = 0

    for spec in specs:
        if not (spec.root / "meta" / "info.json").exists():
            raise FileNotFoundError(f"not a LeRobot dataset: {spec.root}")
        dataset = LeRobotDataset(repo_id=f"offline/{spec.root.name}", root=spec.root)
        episode_column = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
        available = set(int(value) for value in np.unique(episode_column))
        selected = sorted(available if spec.episodes is None else set(spec.episodes))
        missing = sorted(set(selected) - available)
        if missing:
            raise ValueError(f"episodes missing from {spec.root}: {missing}")
        if max_episodes_per_dataset is not None:
            selected = selected[:max_episodes_per_dataset]

        digest = hashlib.sha256(str(spec.root).encode()).hexdigest()[:10]
        for episode_index in selected:
            name = f"episode_offline_{digest}_{episode_index:06d}.complete"
            complete = output_root / name
            partial = output_root / name.replace(".complete", ".partial")
            if complete.exists() and not overwrite:
                skipped_existing += 1
                continue
            if overwrite:
                shutil.rmtree(complete, ignore_errors=True)
                shutil.rmtree(partial, ignore_errors=True)
            partial.mkdir(parents=True)
            for camera in ("top", "wrist"):
                (partial / "images" / camera).mkdir(parents=True)

            row_indices = np.flatnonzero(episode_column == episode_index).tolist()
            sampled = row_indices[::sample_stride]
            if row_indices and sampled[-1] != row_indices[-1]:
                sampled.append(row_indices[-1])
            if len(sampled) < 2:
                shutil.rmtree(partial)
                continue

            first = dataset[sampled[0]]
            metadata = {
                "format": "lingbot_recap_experience_v1",
                "task": str(first["task"]),
                "policy_checkpoint": "offline-demonstration-bootstrap",
                "created_at_unix": time.time(),
                "training_use": "OFFLINE_BOOTSTRAP_FOR_DISTILLATION_NOT_ON_POLICY",
                "fps": float(dataset.fps) / sample_stride,
                "execute_length": 16,
                "source": "clean_lerobot_demonstration",
                "source_dataset": str(spec.root),
                "source_episode": episode_index,
                "source_fps": float(dataset.fps),
                "sample_stride": sample_stride,
            }
            (partial / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            (partial / "OFFLINE_BOOTSTRAP_NOT_ON_POLICY").touch()
            with (partial / "frames.jsonl").open("w", encoding="utf-8") as handle:
                for output_index, row_index in enumerate(sampled):
                    item = first if row_index == sampled[0] else dataset[row_index]
                    top_relative = f"images/top/{output_index:08d}.jpg"
                    wrist_relative = f"images/wrist/{output_index:08d}.jpg"
                    _write_rgb(partial / top_relative, item["observation.images.top"])
                    _write_rgb(partial / wrist_relative, item["observation.images.wrist"])
                    action = _joint_mapping(item["action"])
                    row = {
                        "frame_index": output_index,
                        "source_frame_index": int(item["frame_index"]),
                        "timestamp": float(item["timestamp"]),
                        "observation": {"state": _joint_mapping(item["observation.state"])},
                        "proposed_action": action,
                        "executed_action": action,
                        "action_source": "offline_demonstration",
                        "control_mode": "offline_bootstrap",
                        "images": {"top": top_relative, "wrist": wrist_relative},
                    }
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    written_frames += 1
            os.replace(partial, complete)
            written_episodes += 1

    return ImportSummary(output_root, len(specs), written_episodes, written_frames, skipped_existing)
