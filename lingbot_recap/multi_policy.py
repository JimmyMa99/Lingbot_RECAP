from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from .hardware import MOTOR_NAMES
from .policy import LingBotHTTPPolicy, PolicyResult


def normalize_task(task: str) -> str:
    """Normalize harmless whitespace without changing task semantics."""

    return " ".join(task.strip().split())


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class AssetFingerprint:
    path: str
    sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], name: str) -> "AssetFingerprint":
        path = str(value.get("path", "")).strip()
        digest = str(value.get("sha256", "")).strip().lower()
        if not path or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"{name} requires an absolute path and a 64-character SHA-256")
        if not Path(path).is_absolute():
            raise ValueError(f"{name} path must be absolute: {path}")
        return cls(path=path, sha256=digest)

    def verify(self, name: str) -> None:
        path = Path(self.path)
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist or is not a file: {path}")
        actual = sha256_file(path)
        if actual != self.sha256:
            raise RuntimeError(
                f"{name} SHA-256 mismatch for {path}: expected {self.sha256}, got {actual}"
            )

    def to_mapping(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class TrainingContract:
    norm_stats: AssetFingerprint
    robot_config: AssetFingerprint
    camera_mapping: tuple[tuple[str, str], ...]
    action_space: str
    joints: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TrainingContract":
        raw_mapping = value.get("camera_mapping")
        if not isinstance(raw_mapping, Mapping) or set(raw_mapping) != {"top", "wrist"}:
            raise ValueError("training_contract.camera_mapping must define exactly top and wrist")
        camera_mapping = tuple(
            (name, str(raw_mapping[name]).strip()) for name in ("top", "wrist")
        )
        if any(not target for _, target in camera_mapping):
            raise ValueError("camera mapping targets must not be empty")
        action_space = str(value.get("action_space", "")).strip()
        joints = tuple(str(name).strip() for name in value.get("joints", []))
        if not action_space:
            raise ValueError("training_contract.action_space is required")
        if joints != tuple(MOTOR_NAMES):
            raise ValueError(
                f"training_contract.joints must exactly match {list(MOTOR_NAMES)!r}"
            )
        return cls(
            norm_stats=AssetFingerprint.from_mapping(
                value.get("norm_stats", {}), "normalization stats"
            ),
            robot_config=AssetFingerprint.from_mapping(
                value.get("robot_config", {}), "robot config"
            ),
            camera_mapping=camera_mapping,
            action_space=action_space,
            joints=joints,
        )

    def verify_assets(self) -> None:
        self.norm_stats.verify("normalization stats")
        self.robot_config.verify("robot config")

    def semantic_mapping(self) -> dict[str, object]:
        return {
            "norm_stats_sha256": self.norm_stats.sha256,
            "robot_config_sha256": self.robot_config.sha256,
            "camera_mapping": dict(self.camera_mapping),
            "action_space": self.action_space,
            "joints": list(self.joints),
        }

    @property
    def contract_id(self) -> str:
        encoded = json.dumps(
            self.semantic_mapping(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_mapping(self) -> dict[str, object]:
        return {
            "contract_id": self.contract_id,
            "norm_stats": self.norm_stats.to_mapping(),
            "robot_config": self.robot_config.to_mapping(),
            "camera_mapping": dict(self.camera_mapping),
            "action_space": self.action_space,
            "joints": list(self.joints),
        }


@dataclass(frozen=True)
class TeacherSpec:
    key: str
    checkpoint: str
    server: str
    tasks: tuple[str, ...]
    use_length: int = 16

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TeacherSpec":
        tasks = tuple(normalize_task(str(task)) for task in value.get("tasks", []))
        if not tasks:
            raise ValueError("every teacher must declare at least one exact task")
        key = str(value.get("key", "")).strip()
        checkpoint = str(value.get("checkpoint", "")).strip()
        server = str(value.get("server", "")).strip().rstrip("/")
        use_length = int(value.get("use_length", 16))
        if not key or not checkpoint or not server:
            raise ValueError("teacher key, checkpoint and server are required")
        if not Path(checkpoint).is_absolute():
            raise ValueError(f"teacher checkpoint must be an absolute path: {checkpoint}")
        if use_length < 1:
            raise ValueError("teacher use_length must be >= 1")
        return cls(
            key=key,
            checkpoint=checkpoint,
            server=server,
            tasks=tasks,
            use_length=use_length,
        )


@dataclass(frozen=True)
class RoutedPolicyResult:
    teacher: TeacherSpec
    result: PolicyResult


PolicyFactory = Callable[[TeacherSpec], LingBotHTTPPolicy]


class MultiPolicyRouter:
    """Exact, identity-verified task-to-teacher routing."""

    FORMAT = "lingbot_multi_policy_registry_v2"
    CHECKPOINT_HEALTH_FIELDS = (
        "checkpoint",
        "checkpoint_path",
        "model_path",
        "adapter_path",
        "policy_checkpoint",
    )

    def __init__(
        self,
        teachers: list[TeacherSpec],
        training_contract: TrainingContract | None = None,
        policy_factory: PolicyFactory | None = None,
    ):
        if not teachers:
            raise ValueError("at least one teacher is required")
        self.teachers = tuple(teachers)
        self.training_contract = training_contract
        self._by_task: dict[str, TeacherSpec] = {}
        self._policies: dict[str, LingBotHTTPPolicy] = {}
        self._verified_health: dict[str, dict] = {}
        self._policy_factory = policy_factory or (
            lambda spec: LingBotHTTPPolicy(spec.server, use_length=spec.use_length)
        )
        keys: set[str] = set()
        for teacher in self.teachers:
            if teacher.key in keys:
                raise ValueError(f"duplicate teacher key: {teacher.key}")
            keys.add(teacher.key)
            for task in teacher.tasks:
                normalized = normalize_task(task)
                existing = self._by_task.get(normalized)
                if existing is not None:
                    raise ValueError(
                        f"task {normalized!r} is assigned to both "
                        f"{existing.key!r} and {teacher.key!r}"
                    )
                self._by_task[normalized] = teacher

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        policy_factory: PolicyFactory | None = None,
    ) -> "MultiPolicyRouter":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != cls.FORMAT:
            raise ValueError(
                f"unsupported teacher registry format: {payload.get('format')!r}; "
                f"expected {cls.FORMAT!r}"
            )
        contract = TrainingContract.from_mapping(payload.get("training_contract", {}))
        contract.verify_assets()
        teachers = [TeacherSpec.from_mapping(value) for value in payload["teachers"]]
        return cls(teachers, training_contract=contract, policy_factory=policy_factory)

    def require_training_contract(self) -> TrainingContract:
        if self.training_contract is None:
            raise RuntimeError("a verified training contract is required for distillation")
        return self.training_contract

    def teacher_for(self, task: str) -> TeacherSpec:
        normalized = normalize_task(task)
        try:
            return self._by_task[normalized]
        except KeyError as exc:
            known = ", ".join(repr(value) for value in sorted(self._by_task))
            raise KeyError(f"no teacher for task {normalized!r}; known tasks: {known}") from exc

    @staticmethod
    def _canonical_identity(value: str) -> str:
        return os.path.realpath(os.path.expanduser(value)).rstrip("/")

    def _validate_teacher_health(
        self, teacher: TeacherSpec, policy: LingBotHTTPPolicy
    ) -> dict:
        if teacher.key in self._verified_health:
            return self._verified_health[teacher.key]
        health = policy.health()
        if not health.get("model_loaded"):
            raise RuntimeError(f"teacher {teacher.key!r} is not ready: {health}")
        reported = next(
            (
                str(health[field]).strip()
                for field in self.CHECKPOINT_HEALTH_FIELDS
                if health.get(field)
            ),
            "",
        )
        if not reported:
            raise RuntimeError(
                f"teacher {teacher.key!r} health response does not expose checkpoint identity; "
                f"expected one of {self.CHECKPOINT_HEALTH_FIELDS}"
            )
        expected_identity = self._canonical_identity(teacher.checkpoint)
        reported_identity = self._canonical_identity(reported)
        if reported_identity != expected_identity:
            raise RuntimeError(
                f"teacher {teacher.key!r} checkpoint mismatch: registry={expected_identity!r}, "
                f"server={reported_identity!r}"
            )
        verified = dict(health)
        verified["verified_checkpoint"] = expected_identity
        self._verified_health[teacher.key] = verified
        return verified

    def policy_for(self, task: str) -> tuple[TeacherSpec, LingBotHTTPPolicy]:
        teacher = self.teacher_for(task)
        if teacher.key not in self._policies:
            self._policies[teacher.key] = self._policy_factory(teacher)
        policy = self._policies[teacher.key]
        self._validate_teacher_health(teacher, policy)
        return teacher, policy

    def infer(self, task, state, image_jpegs) -> RoutedPolicyResult:
        teacher, policy = self.policy_for(task)
        return RoutedPolicyResult(teacher, policy.infer(task, state, image_jpegs))

    def validate_health(self) -> dict[str, dict]:
        self.require_training_contract().verify_assets()
        results = {}
        for teacher in self.teachers:
            if teacher.key not in self._policies:
                self._policies[teacher.key] = self._policy_factory(teacher)
            results[teacher.key] = self._validate_teacher_health(
                teacher, self._policies[teacher.key]
            )
        return results
