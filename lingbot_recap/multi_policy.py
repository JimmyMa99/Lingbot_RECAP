from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from .policy import LingBotHTTPPolicy, PolicyResult


def normalize_task(task: str) -> str:
    """Normalize harmless whitespace without changing task semantics."""

    return " ".join(task.strip().split())


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
        if not key or not checkpoint or not server:
            raise ValueError("teacher key, checkpoint and server are required")
        return cls(
            key=key,
            checkpoint=checkpoint,
            server=server,
            tasks=tasks,
            use_length=int(value.get("use_length", 16)),
        )


@dataclass(frozen=True)
class RoutedPolicyResult:
    teacher: TeacherSpec
    result: PolicyResult


PolicyFactory = Callable[[TeacherSpec], LingBotHTTPPolicy]


class MultiPolicyRouter:
    """Exact task-to-teacher routing for multi-policy distillation.

    Exact routing is deliberate: silently sending a physical observation to the
    wrong specialist creates plausible-looking but invalid supervision.
    """

    FORMAT = "lingbot_multi_policy_registry_v1"

    def __init__(
        self,
        teachers: list[TeacherSpec],
        policy_factory: PolicyFactory | None = None,
    ):
        if not teachers:
            raise ValueError("at least one teacher is required")
        self.teachers = tuple(teachers)
        self._by_task: dict[str, TeacherSpec] = {}
        self._policies: dict[str, LingBotHTTPPolicy] = {}
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
                f"unsupported teacher registry format: {payload.get('format')!r}"
            )
        teachers = [TeacherSpec.from_mapping(value) for value in payload["teachers"]]
        return cls(teachers, policy_factory=policy_factory)

    def teacher_for(self, task: str) -> TeacherSpec:
        normalized = normalize_task(task)
        try:
            return self._by_task[normalized]
        except KeyError as exc:
            known = ", ".join(repr(value) for value in sorted(self._by_task))
            raise KeyError(f"no teacher for task {normalized!r}; known tasks: {known}") from exc

    def policy_for(self, task: str) -> tuple[TeacherSpec, LingBotHTTPPolicy]:
        teacher = self.teacher_for(task)
        if teacher.key not in self._policies:
            self._policies[teacher.key] = self._policy_factory(teacher)
        return teacher, self._policies[teacher.key]

    def infer(self, task, state, image_jpegs) -> RoutedPolicyResult:
        teacher, policy = self.policy_for(task)
        return RoutedPolicyResult(teacher, policy.infer(task, state, image_jpegs))

    def validate_health(self) -> dict[str, dict]:
        results = {}
        for teacher in self.teachers:
            _, policy = self.policy_for(teacher.tasks[0])
            health = policy.health()
            if not health.get("model_loaded"):
                raise RuntimeError(f"teacher {teacher.key!r} is not ready: {health}")
            results[teacher.key] = health
        return results
