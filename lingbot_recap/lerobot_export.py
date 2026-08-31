from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .distillation import load_labeled_frames
from .hardware import MOTOR_NAMES


@dataclass(frozen=True)
class ExportSummary:
    output_root: Path
    source_experiences: int
    output_episodes: int
    output_frames: int


def _read_rgb(path: Path) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for LeRobot export; install .[robot]") from exc
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _segments(pairs: list[tuple[dict, dict]], stride: int) -> list[list[tuple[dict, dict]]]:
    result: list[list[tuple[dict, dict]]] = []
    for pair in sorted(pairs, key=lambda value: int(value[0]["frame_index"])):
        index = int(pair[0]["frame_index"])
        if not result or index != int(result[-1][-1][0]["frame_index"]) + stride:
            result.append([pair])
        else:
            result[-1].append(pair)
    return result


def export_lerobot_distillation_dataset(
    experiences: list[str | Path],
    output_root: str | Path,
    repo_id: str,
    *,
    use_videos: bool = True,
    min_segment_frames: int = 2,
) -> ExportSummary:
    """Export teacher actions at student states as a LeRobot v3 dataset.

    The resulting `action` is the routed teacher's first action at each
    student-visited observation. Existing LingBot L1 flow-matching training can
    consume this dataset without changing the action head. Replay data should be
    mixed in by the training manifest to limit forgetting.
    """

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError("LeRobot 0.4.2 is required; install .[robot]") from exc

    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {output_root}")
    experience_paths = [Path(value) for value in experiences]
    if not experience_paths:
        raise ValueError("no labeled experiences supplied")

    loaded = []
    source_fps = None
    source_stride = None
    first_image_shape = None
    training_contract_id = None
    training_contract = None
    training_uses: set[str] = set()
    eligible_segment_count = 0
    for episode in experience_paths:
        metadata, pairs = load_labeled_frames(episode)
        label_meta = json.loads((episode / "teacher_labels.meta.json").read_text(encoding="utf-8"))
        if label_meta.get("preview"):
            raise ValueError(f"preview labels cannot be exported for training: {episode}")
        if label_meta.get("teacher_checkpoint_identity_verified") is not True:
            raise ValueError(f"teacher checkpoint identity was not verified: {episode}")
        training_uses.add(str(label_meta.get("training_use", "")))
        contract_id = str(label_meta.get("training_contract_id", ""))
        contract = label_meta.get("training_contract")
        if not contract_id or not isinstance(contract, dict):
            raise ValueError(f"training contract metadata is missing: {episode}")
        if training_contract_id is None:
            training_contract_id = contract_id
            training_contract = contract
        elif contract_id != training_contract_id or contract != training_contract:
            raise ValueError("all experiences must use the same training contract")
        fps = float(metadata.get("fps", 30.0))
        stride = int(label_meta.get("stride", 1))
        if source_fps is None:
            source_fps, source_stride = fps, stride
        elif (fps, stride) != (source_fps, source_stride):
            raise ValueError("all experiences must use the same fps and relabel stride")
        segments = [
            segment
            for segment in _segments(pairs, stride)
            if len(segment) >= min_segment_frames
        ]
        eligible_segment_count += len(segments)
        if segments and first_image_shape is None:
            first_frame = segments[0][0][0]
            first_image_shape = _read_rgb(episode / first_frame["images"]["top"]).shape
        loaded.append((episode, metadata, segments))

    assert source_fps is not None and source_stride is not None
    effective_fps = source_fps / source_stride
    if abs(effective_fps - round(effective_fps)) > 1e-6:
        raise ValueError(f"effective fps must be an integer, got {effective_fps}")
    if first_image_shape is None:
        raise RuntimeError(
            f"no labeled segment contains at least {min_segment_frames} frames"
        )
    if eligible_segment_count == 0:
        raise RuntimeError("no eligible labeled segments found")
    height, width, channels = first_image_shape
    image_dtype = "video" if use_videos else "image"
    joint_names = list(MOTOR_NAMES)
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(MOTOR_NAMES),),
            "names": joint_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(MOTOR_NAMES),),
            "names": joint_names,
        },
        "observation.images.top": {
            "dtype": image_dtype,
            "shape": (height, width, channels),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": image_dtype,
            "shape": (height, width, channels),
            "names": ["height", "width", "channels"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(round(effective_fps)),
        features=features,
        root=output_root,
        robot_type="so101_follower",
        use_videos=use_videos,
        image_writer_threads=4,
    )

    output_episodes = 0
    output_frames = 0
    provenance_episodes = []
    for episode, metadata, segments in loaded:
        for segment in segments:
            segment_teacher = segment[0][1]["teacher_key"]
            segment_checkpoint = segment[0][1]["teacher_checkpoint"]
            for frame, label in segment:
                if (
                    label.get("teacher_key") != segment_teacher
                    or label.get("teacher_checkpoint") != segment_checkpoint
                ):
                    raise ValueError(f"teacher identity changes within segment in {episode}")
                state = frame["observation"]["state"]
                if isinstance(state, dict):
                    state_values = [float(state[name]) for name in MOTOR_NAMES]
                else:
                    state_values = [float(value) for value in state]
                action = [float(value) for value in label["teacher_action"]]
                if len(state_values) != 6 or len(action) != 6:
                    raise ValueError(f"invalid state/action dimensions in {episode}")
                if not np.isfinite(state_values).all() or not np.isfinite(action).all():
                    raise ValueError(f"state/action contains NaN or Inf in {episode}")
                dataset.add_frame(
                    {
                        "observation.state": np.asarray(state_values, dtype=np.float32),
                        "action": np.asarray(action, dtype=np.float32),
                        "observation.images.top": _read_rgb(episode / frame["images"]["top"]),
                        "observation.images.wrist": _read_rgb(episode / frame["images"]["wrist"]),
                        "task": str(metadata["task"]),
                    }
                )
                output_frames += 1
            dataset.save_episode(parallel_encoding=False)
            output_episodes += 1
            provenance_episodes.append(
                {
                    "source": str(episode),
                    "source_first_frame": int(segment[0][0]["frame_index"]),
                    "source_last_frame": int(segment[-1][0]["frame_index"]),
                    "frames": len(segment),
                    "task": metadata["task"],
                    "teacher_key": segment_teacher,
                    "teacher_checkpoint": segment_checkpoint,
                }
            )

    provenance = {
        "format": "lingbot_multi_policy_distillation_dataset_v1",
        "created_at_unix": time.time(),
        "training_use": (
            "MULTI_POLICY_OFFLINE_BOOTSTRAP_DISTILLATION"
            if training_uses == {"MULTI_POLICY_OFFLINE_BOOTSTRAP_DISTILLATION"}
            else "MULTI_POLICY_ON_POLICY_DISTILLATION_ONLY"
        ),
        "repo_id": repo_id,
        "source_fps": source_fps,
        "source_stride": source_stride,
        "effective_fps": effective_fps,
        "training_contract_id": training_contract_id,
        "training_contract": training_contract,
        "source_experiences": len(experience_paths),
        "output_episodes": output_episodes,
        "output_frames": output_frames,
        "episodes": provenance_episodes,
    }
    (output_root / "DISTILLATION_DATASET_ONLY").touch()
    (output_root / "distillation_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return ExportSummary(output_root, len(experience_paths), output_episodes, output_frames)
