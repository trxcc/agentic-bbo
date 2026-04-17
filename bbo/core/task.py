"""Task definitions for benchmark-oriented BBO."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .description import (
    MarkdownDescriptionLoader,
    TaskDescriptionBundle,
    TaskDescriptionRef,
)
from .space import SearchSpace
from .trial import EvaluationResult, TrialSuggestion


class ObjectiveDirection(str, Enum):
    """Supported objective directions."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass(frozen=True)
class ObjectiveSpec:
    """Definition of one optimization objective."""

    name: str
    direction: ObjectiveDirection = ObjectiveDirection.MINIMIZE

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Objective name must be non-empty.")


@dataclass(frozen=True)
class TaskSpec:
    """Static task metadata shared with algorithms and experiment runners."""

    name: str
    search_space: SearchSpace
    objectives: tuple[ObjectiveSpec, ...]
    max_evaluations: int
    default_budget: float | None = None
    budget_range: tuple[float, float] | None = None
    description_ref: TaskDescriptionRef | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Task name must be non-empty.")
        if self.max_evaluations <= 0:
            raise ValueError("Task max_evaluations must be greater than zero.")
        if not self.objectives:
            raise ValueError("Task must define at least one optimization objective.")
        if self.default_budget is not None and self.default_budget <= 0:
            raise ValueError("default_budget must be positive when provided.")
        if self.budget_range is not None:
            low, high = self.budget_range
            if low <= 0 or high <= 0 or low > high:
                raise ValueError(f"budget_range must satisfy 0 < low <= high, got {self.budget_range!r}.")
            if self.default_budget is not None and not (low <= self.default_budget <= high):
                raise ValueError(
                    f"default_budget {self.default_budget} is outside budget_range {self.budget_range!r}."
                )

    @property
    def primary_objective(self) -> ObjectiveSpec:
        return self.objectives[0]


@dataclass(frozen=True)
class SanityCheckIssue:
    """One sanity-check issue discovered for a task."""

    severity: str
    code: str
    message: str


@dataclass
class SanityCheckReport:
    """Sanity-check report for a task."""

    errors: list[SanityCheckIssue] = field(default_factory=list)
    warnings: list[SanityCheckIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, code: str, message: str) -> None:
        self.errors.append(SanityCheckIssue(severity="error", code=code, message=message))

    def add_warning(self, code: str, message: str) -> None:
        self.warnings.append(SanityCheckIssue(severity="warning", code=code, message=message))


class Task(ABC):
    """Abstract black-box task definition."""

    description_loader_cls = MarkdownDescriptionLoader

    @property
    @abstractmethod
    def spec(self) -> TaskSpec:
        """Return the task specification."""

    @abstractmethod
    def evaluate(self, suggestion: TrialSuggestion) -> EvaluationResult:
        """Evaluate one suggestion and return the observed result."""

    def get_description(self) -> TaskDescriptionBundle:
        if self.spec.description_ref is None:
            return TaskDescriptionBundle.empty(task_id=self.spec.name)
        return self.description_loader_cls(schema=self.spec.description_ref.schema).load(self.spec.description_ref)

    def cleanup(self) -> None:
        """Release any task-owned runtime resources."""

    def sanity_check(self) -> SanityCheckReport:
        spec = self.spec
        report = SanityCheckReport(metadata={"task_name": spec.name})

        try:
            defaults = spec.search_space.defaults()
            spec.search_space.validate_config(defaults)
            report.metadata["default_config"] = defaults
        except Exception as exc:  # pragma: no cover - defensive guard
            report.add_error("invalid_default_config", f"Task default config is invalid: {exc}")

        names = [objective.name for objective in spec.objectives]
        if len(names) != len(set(names)):
            report.add_error("duplicate_objectives", f"Task has duplicate objective names: {names!r}")

        if spec.description_ref is None:
            report.add_error(
                "missing_description",
                "Task must provide a markdown description so agentic methods can use structured task context.",
            )
        else:
            if not spec.description_ref.primary_path.exists():
                report.add_error(
                    "missing_primary_description",
                    f"Primary task description does not exist: {spec.description_ref.primary_path}",
                )
            missing_sections = spec.description_ref.missing_sections()
            if missing_sections:
                report.add_error(
                    "missing_required_sections",
                    "Task description is missing required sections: " + ", ".join(missing_sections),
                )
            missing_extras = [str(path) for path in spec.description_ref.extra_paths if not path.exists()]
            if missing_extras:
                report.add_warning(
                    "missing_extra_descriptions",
                    f"Some extra task descriptions are missing and will be ignored: {missing_extras!r}",
                )
            try:
                description = self.get_description()
                report.metadata["description_fingerprint"] = description.fingerprint
                report.metadata["description_sections"] = sorted(description.section_map)
            except Exception as exc:
                report.add_error("unreadable_description", f"Task description could not be loaded: {exc}")
            self._check_environment_provisioning(report)

        return report

    def _check_environment_provisioning(self, report: SanityCheckReport) -> None:
        ref = self.spec.description_ref
        if ref is None or ref.directory is None:
            return

        directory = ref.directory
        has_environment_doc = (directory / "environment.md").exists()
        docker_candidates = (
            directory / "Dockerfile",
            directory / "docker-compose.yml",
            directory / "docker-compose.yaml",
            directory / "compose.yml",
            directory / "compose.yaml",
        )
        has_docker_assets = any(path.exists() for path in docker_candidates) or (directory / "docker").exists()

        if not has_environment_doc and not has_docker_assets:
            report.add_error(
                "missing_environment_setup",
                "Each task must provide either a task-local Docker setup or an `environment.md` with setup instructions.",
            )
            return

        if has_environment_doc and has_docker_assets:
            report.metadata["environment_provisioning"] = "docker_and_manual"
        elif has_docker_assets:
            report.metadata["environment_provisioning"] = "docker"
        else:
            report.metadata["environment_provisioning"] = "manual"


__all__ = [
    "ObjectiveDirection",
    "ObjectiveSpec",
    "SanityCheckIssue",
    "SanityCheckReport",
    "Task",
    "TaskSpec",
]
