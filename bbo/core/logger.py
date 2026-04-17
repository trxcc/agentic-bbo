"""Logging and replay primitives for benchmark-oriented BBO."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .description import TaskDescriptionBundle
from .task import ObjectiveDirection, ObjectiveSpec, TaskSpec
from .trial import TrialObservation, TrialRecord, TrialStatus


@dataclass(frozen=True)
class ResumeState:
    """Recovered state needed to continue a prior experiment."""

    n_completed: int = 0
    next_trial_id: int = 0
    total_eval_time: float = 0.0
    best_primary_objective: float | None = None
    best_trial_id: int | None = None
    best_objectives: dict[str, float] = field(default_factory=dict)


class MetricLogger(ABC):
    """Abstract append-only metric logger."""

    def bind_run(
        self,
        *,
        task_spec: TaskSpec,
        algorithm_name: str,
        seed: int,
        description_bundle: TaskDescriptionBundle | None = None,
    ) -> None:
        """Bind run-level context before the experiment loop starts."""

    @abstractmethod
    def log(self, observation: TrialObservation) -> None:
        """Persist one observation."""

    @abstractmethod
    def load_history(self) -> list[TrialObservation]:
        """Load prior observations in replay order."""

    @abstractmethod
    def resume_state(self) -> ResumeState:
        """Return summary state needed to resume a run."""

    @abstractmethod
    def summary(self) -> dict[str, Any]:
        """Return aggregate run statistics."""


class JsonlMetricLogger(MetricLogger):
    """Append-only JSONL logger with replay and resume helpers."""

    def __init__(
        self,
        path: Path,
        *,
        task_name: str | None = None,
        algorithm_name: str | None = None,
        seed: int | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.task_name = task_name
        self.algorithm_name = algorithm_name
        self.seed = seed
        self.objectives: tuple[ObjectiveSpec, ...] = ()
        self.description_fingerprint: str | None = None
        self.description_paths: list[str] = []

    def bind_run(
        self,
        *,
        task_spec: TaskSpec,
        algorithm_name: str,
        seed: int,
        description_bundle: TaskDescriptionBundle | None = None,
    ) -> None:
        self.task_name = task_spec.name
        self.algorithm_name = algorithm_name
        self.seed = seed
        self.objectives = task_spec.objectives
        if description_bundle is not None:
            self.description_fingerprint = description_bundle.fingerprint or None
            self.description_paths = [str(doc.path) for doc in description_bundle.all_docs]

    def log(self, observation: TrialObservation) -> None:
        record = TrialRecord.from_observation(
            observation,
            task_name=self.task_name,
            algorithm=self.algorithm_name,
            seed=self.seed,
            description_fingerprint=self.description_fingerprint,
            description_paths=self.description_paths,
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), default=self._json_default, sort_keys=True) + "\n")

    def load_records(self) -> list[TrialRecord]:
        if not self.path.exists():
            return []
        records: list[TrialRecord] = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(TrialRecord.from_dict(json.loads(line)))
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
                raise ValueError(f"Malformed JSONL at {self.path}:{line_number}") from exc
        return records

    def load_history(self) -> list[TrialObservation]:
        return [record.to_observation() for record in self.load_records()]

    def resume_state(self) -> ResumeState:
        records = self.load_records()
        if not records:
            return ResumeState()

        next_trial_id = max((record.trial_id for record in records if record.trial_id >= 0), default=-1) + 1
        total_eval_time = sum(record.elapsed_seconds or 0.0 for record in records)
        best_objectives = self._best_objectives(records)
        best_primary_objective, best_trial_id = self._best_primary(records)
        return ResumeState(
            n_completed=len(records),
            next_trial_id=next_trial_id,
            total_eval_time=total_eval_time,
            best_primary_objective=best_primary_objective,
            best_trial_id=best_trial_id,
            best_objectives=best_objectives,
        )

    def summary(self) -> dict[str, Any]:
        records = self.load_records()
        state = self.resume_state()
        counts = {
            TrialStatus.SUCCESS.value: 0,
            TrialStatus.FAILED.value: 0,
            TrialStatus.INVALID.value: 0,
        }
        for record in records:
            counts[record.status] = counts.get(record.status, 0) + 1
        return {
            "path": str(self.path),
            "task_name": self.task_name,
            "algorithm": self.algorithm_name,
            "seed": self.seed,
            "total_trials": len(records),
            "successful_trials": counts.get(TrialStatus.SUCCESS.value, 0),
            "failed_trials": counts.get(TrialStatus.FAILED.value, 0),
            "invalid_trials": counts.get(TrialStatus.INVALID.value, 0),
            "total_eval_time": state.total_eval_time,
            "best_primary_objective": state.best_primary_objective,
            "best_trial_id": state.best_trial_id,
            "best_objectives": state.best_objectives,
            "description_fingerprint": self.description_fingerprint,
        }

    @staticmethod
    def _json_default(obj: object) -> object:
        import numpy as np

        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def _best_primary(self, records: list[TrialRecord]) -> tuple[float | None, int | None]:
        if not self.objectives:
            return None, None
        primary = self.objectives[0]
        scored: list[tuple[float, int]] = []
        for record in records:
            value = record.objectives.get(primary.name)
            if value is None:
                continue
            scored.append((float(value), record.trial_id))
        if not scored:
            return None, None
        if primary.direction == ObjectiveDirection.MAXIMIZE:
            value, trial_id = max(scored, key=lambda item: item[0])
        else:
            value, trial_id = min(scored, key=lambda item: item[0])
        return value, trial_id

    def _best_objectives(self, records: list[TrialRecord]) -> dict[str, float]:
        best: dict[str, float] = {}
        direction_map = {objective.name: objective.direction for objective in self.objectives}
        for record in records:
            for name, value in record.objectives.items():
                numeric_value = float(value)
                if name not in best:
                    best[name] = numeric_value
                    continue
                direction = direction_map.get(name, ObjectiveDirection.MINIMIZE)
                if direction == ObjectiveDirection.MAXIMIZE and numeric_value > best[name]:
                    best[name] = numeric_value
                elif direction != ObjectiveDirection.MAXIMIZE and numeric_value < best[name]:
                    best[name] = numeric_value
        return best


__all__ = ["JsonlMetricLogger", "MetricLogger", "ResumeState"]
