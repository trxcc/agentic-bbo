"""Trial dataclasses shared across tasks, algorithms, and loggers."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TrialStatus(str, Enum):
    """Outcome status for one trial."""

    SUCCESS = "success"
    FAILED = "failed"
    INVALID = "invalid"


@dataclass
class TrialSuggestion:
    """One algorithm proposal to evaluate."""

    config: dict[str, Any]
    trial_id: int | None = None
    budget: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.config, dict):
            raise TypeError("TrialSuggestion.config must be a dict.")
        if self.trial_id is not None and self.trial_id < 0:
            raise ValueError("trial_id must be non-negative when provided.")
        if self.budget is not None and self.budget <= 0:
            raise ValueError("budget must be positive when provided.")


@dataclass
class EvaluationResult:
    """Result returned by a task's evaluate() method."""

    status: TrialStatus = TrialStatus.SUCCESS
    objectives: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.elapsed_seconds is not None:
            if self.elapsed_seconds < 0:
                raise ValueError("elapsed_seconds must be non-negative.")
            if not math.isfinite(self.elapsed_seconds):
                raise ValueError("elapsed_seconds must be finite.")
        for name, value in self.objectives.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"Objective `{name}` must be finite, got {value!r}.")

    @property
    def success(self) -> bool:
        return self.status == TrialStatus.SUCCESS


@dataclass
class TrialObservation:
    """Normalized trial observation consumed by tell(), replay(), and logging."""

    suggestion: TrialSuggestion
    status: TrialStatus
    objectives: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == TrialStatus.SUCCESS

    @classmethod
    def from_evaluation(
        cls,
        suggestion: TrialSuggestion,
        result: EvaluationResult,
        *,
        timestamp: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "TrialObservation":
        merged_metadata = dict(result.metadata)
        if metadata:
            merged_metadata.update(metadata)
        return cls(
            suggestion=suggestion,
            status=result.status,
            objectives=dict(result.objectives),
            metrics=dict(result.metrics),
            elapsed_seconds=result.elapsed_seconds,
            error_type=result.error_type,
            error_message=result.error_message,
            timestamp=time.time() if timestamp is None else timestamp,
            metadata=merged_metadata,
        )


@dataclass
class TrialRecord:
    """Serialized observation with run-level context."""

    trial_id: int
    task_name: str | None
    algorithm: str | None
    seed: int | None
    config: dict[str, Any]
    budget: float | None
    status: str
    suggestion_metadata: dict[str, Any] = field(default_factory=dict)
    objectives: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    timestamp: float = field(default_factory=time.time)
    description_fingerprint: str | None = None
    description_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_observation(
        cls,
        observation: TrialObservation,
        *,
        task_name: str | None = None,
        algorithm: str | None = None,
        seed: int | None = None,
        description_fingerprint: str | None = None,
        description_paths: list[str] | None = None,
    ) -> "TrialRecord":
        if observation.suggestion.trial_id is None:
            raise ValueError("TrialObservation must have a trial_id before it can be logged.")
        return cls(
            trial_id=observation.suggestion.trial_id,
            task_name=task_name,
            algorithm=algorithm,
            seed=seed,
            config=dict(observation.suggestion.config),
            budget=observation.suggestion.budget,
            status=observation.status.value,
            suggestion_metadata=dict(observation.suggestion.metadata),
            objectives=dict(observation.objectives),
            metrics=dict(observation.metrics),
            elapsed_seconds=observation.elapsed_seconds,
            error_type=observation.error_type,
            error_message=observation.error_message,
            timestamp=observation.timestamp,
            description_fingerprint=description_fingerprint,
            description_paths=list(description_paths or []),
            metadata=dict(observation.metadata),
        )

    def to_observation(self) -> TrialObservation:
        return TrialObservation(
            suggestion=TrialSuggestion(
                config=dict(self.config),
                trial_id=self.trial_id,
                budget=self.budget,
                metadata=dict(self.suggestion_metadata),
            ),
            status=TrialStatus(self.status),
            objectives=dict(self.objectives),
            metrics=dict(self.metrics),
            elapsed_seconds=self.elapsed_seconds,
            error_type=self.error_type,
            error_message=self.error_message,
            timestamp=self.timestamp,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "task_name": self.task_name,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "config": self.config,
            "budget": self.budget,
            "status": self.status,
            "suggestion_metadata": self.suggestion_metadata,
            "objectives": self.objectives,
            "metrics": self.metrics,
            "elapsed_seconds": self.elapsed_seconds,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
            "description_fingerprint": self.description_fingerprint,
            "description_paths": self.description_paths,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrialRecord":
        return cls(
            trial_id=data["trial_id"],
            task_name=data.get("task_name"),
            algorithm=data.get("algorithm"),
            seed=data.get("seed"),
            config=data.get("config", {}),
            budget=data.get("budget"),
            status=data["status"],
            suggestion_metadata=data.get("suggestion_metadata", {}),
            objectives=data.get("objectives", {}),
            metrics=data.get("metrics", {}),
            elapsed_seconds=data.get("elapsed_seconds"),
            error_type=data.get("error_type"),
            error_message=data.get("error_message"),
            timestamp=data.get("timestamp", time.time()),
            description_fingerprint=data.get("description_fingerprint"),
            description_paths=data.get("description_paths", []),
            metadata=data.get("metadata", {}),
        )


__all__ = [
    "EvaluationResult",
    "TrialObservation",
    "TrialRecord",
    "TrialStatus",
    "TrialSuggestion",
]
