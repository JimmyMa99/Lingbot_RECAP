from __future__ import annotations

import json
import os
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .distillation import ExperienceRelabeler, RelabelConfig, find_experiences
from .lerobot_export import export_lerobot_distillation_dataset
from .multi_policy import MultiPolicyRouter


STATE_FORMAT = "lingbot_mopd_iteration_v1"


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _clean_manifest_lines(lines: Iterable[str]) -> list[str]:
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]


def build_training_manifest(
    output: str | Path,
    distilled_dataset: str | Path,
    replay_manifest: str | Path | None = None,
    replay_repeat: int = 1,
) -> Path:
    if replay_repeat < 0:
        raise ValueError("replay_repeat must be >= 0")
    lines = [f"so_arm101 {Path(distilled_dataset)}"]
    if replay_manifest is not None:
        replay_path = Path(replay_manifest)
        if not replay_path.is_file():
            raise FileNotFoundError(replay_path)
        replay_lines = _clean_manifest_lines(
            replay_path.read_text(encoding="utf-8").splitlines()
        )
        if not replay_lines:
            raise ValueError(f"replay manifest has no datasets: {replay_path}")
        for _ in range(replay_repeat):
            lines.extend(replay_lines)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return output


@contextmanager
def iteration_lock(path: Path):
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise RuntimeError(
            f"iteration is already running or left a stale lock: {path}"
        ) from exc
    try:
        os.write(descriptor, f"pid={os.getpid()} started={time.time()}\n".encode())
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)


@dataclass(frozen=True)
class IterationConfig:
    iteration: int
    teacher_registry: Path
    experience_root: Path
    run_root: Path
    repo_id: str
    replay_manifest: Path | None = None
    replay_repeat: int = 1
    use_videos: bool = True
    min_segment_frames: int = 2
    train_command: tuple[str, ...] = ()
    post_train_command: tuple[str, ...] = ()

    def __post_init__(self):
        if self.iteration < 1:
            raise ValueError("iteration must be >= 1")
        if self.replay_repeat < 0:
            raise ValueError("replay_repeat must be >= 0")
        if self.min_segment_frames < 1:
            raise ValueError("min_segment_frames must be >= 1")


class IterationRunner:
    """Durable offline MOPD iteration: verify, label, export, train and post-process."""

    def __init__(self, config: IterationConfig):
        self.config = config
        self.path = config.run_root / f"iteration_{config.iteration:03d}"
        self.state_path = self.path / "iteration_state.json"
        self.lock_path = self.path / ".iteration.lock"

    def _new_state(self) -> dict:
        return {
            "format": STATE_FORMAT,
            "iteration": self.config.iteration,
            "status": "running",
            "created_at_unix": time.time(),
            "updated_at_unix": time.time(),
            "stages": {},
        }

    def _load_or_create_state(self, resume: bool) -> dict:
        if self.state_path.exists():
            if not resume:
                raise FileExistsError(
                    f"iteration state already exists; use --resume: {self.state_path}"
                )
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("format") != STATE_FORMAT:
                raise ValueError(f"unsupported iteration state: {self.state_path}")
            return state
        if self.path.exists() and any(self.path.iterdir()):
            raise RuntimeError(
                f"iteration directory is non-empty but has no valid state: {self.path}"
            )
        self.path.mkdir(parents=True, exist_ok=True)
        state = self._new_state()
        _atomic_json(self.state_path, state)
        return state

    def _stage(self, state: dict, name: str, status: str, **details) -> None:
        state["stages"][name] = {
            "status": status,
            "updated_at_unix": time.time(),
            **details,
        }
        state["updated_at_unix"] = time.time()
        _atomic_json(self.state_path, state)

    def _run_command(
        self,
        state: dict,
        stage: str,
        command: Sequence[str],
        environment: dict[str, str],
    ) -> None:
        if state.get("stages", {}).get(stage, {}).get("status") == "complete":
            return
        if not command:
            self._stage(state, stage, "skipped")
            return
        log_path = self.path / f"{stage}.log"
        self._stage(state, stage, "running", command=list(command), log=str(log_path))
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                list(command),
                cwd=self.path,
                env={**os.environ, **environment},
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode != 0:
            self._stage(state, stage, "failed", returncode=result.returncode)
            raise RuntimeError(
                f"{stage} command failed with exit code {result.returncode}; see {log_path}"
            )
        self._stage(state, stage, "complete", returncode=0)

    def run(self, resume: bool = False) -> Path:
        state = self._load_or_create_state(resume)
        with iteration_lock(self.lock_path):
            try:
                router = MultiPolicyRouter.from_file(self.config.teacher_registry)
                health = router.validate_health()
                self._stage(
                    state,
                    "validate_teachers",
                    "complete",
                    teachers=health,
                    training_contract_id=router.require_training_contract().contract_id,
                )

                episodes = find_experiences(self.config.experience_root)
                if not episodes:
                    raise RuntimeError(
                        f"no completed student experiences in {self.config.experience_root}"
                    )
                relabeler = ExperienceRelabeler(router, RelabelConfig())
                summaries = [
                    relabeler.label_episode(episode, overwrite=False)
                    for episode in episodes
                ]
                self._stage(
                    state,
                    "relabel",
                    "complete",
                    source_episodes=len(summaries),
                    labeled_frames=sum(item.labeled_frames for item in summaries),
                )

                dataset_root = self.path / "distilled_lerobot"
                if not dataset_root.exists():
                    export = export_lerobot_distillation_dataset(
                        episodes,
                        dataset_root,
                        f"{self.config.repo_id}-iter{self.config.iteration:03d}",
                        use_videos=self.config.use_videos,
                        min_segment_frames=self.config.min_segment_frames,
                    )
                    export_details = {
                        "episodes": export.output_episodes,
                        "frames": export.output_frames,
                    }
                else:
                    provenance_path = dataset_root / "distillation_provenance.json"
                    if not provenance_path.is_file():
                        raise RuntimeError(
                            f"existing dataset lacks provenance: {dataset_root}"
                        )
                    provenance = json.loads(
                        provenance_path.read_text(encoding="utf-8")
                    )
                    export_details = {
                        "episodes": provenance["output_episodes"],
                        "frames": provenance["output_frames"],
                        "resumed": True,
                    }
                self._stage(
                    state,
                    "export",
                    "complete",
                    dataset=str(dataset_root),
                    **export_details,
                )

                manifest = build_training_manifest(
                    self.path / "train_manifest.txt",
                    dataset_root,
                    self.config.replay_manifest,
                    self.config.replay_repeat,
                )
                self._stage(
                    state,
                    "manifest",
                    "complete",
                    path=str(manifest),
                    replay_repeat=self.config.replay_repeat,
                )

                environment = {
                    "MOPD_ITERATION": str(self.config.iteration),
                    "MOPD_ITERATION_DIR": str(self.path),
                    "MOPD_TRAIN_MANIFEST": str(manifest),
                    "MOPD_DATASET_ROOT": str(dataset_root),
                    "MOPD_TEACHER_REGISTRY": str(self.config.teacher_registry),
                }
                self._run_command(
                    state, "train", self.config.train_command, environment
                )
                self._run_command(
                    state,
                    "post_train",
                    self.config.post_train_command,
                    environment,
                )
                state["status"] = "complete"
                state["completed_at_unix"] = time.time()
                state["updated_at_unix"] = time.time()
                _atomic_json(self.state_path, state)
                return self.path
            except BaseException as exc:
                state["status"] = "failed"
                state["error"] = repr(exc)
                state["updated_at_unix"] = time.time()
                _atomic_json(self.state_path, state)
                raise
