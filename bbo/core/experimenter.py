"""Serial experiment orchestration for benchmark-oriented BBO."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from .algo import Algorithm, Incumbent
from .description import TaskDescriptionBundle
from .logger import MetricLogger
from .task import ObjectiveDirection, Task
from .trial import EvaluationResult, TrialObservation, TrialStatus, TrialSuggestion

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for one benchmark experiment."""

    seed: int = 0
    resume: bool = True
    stop_on_task_budget: bool = True
    time_budget: float | None = None
    fail_fast_on_sanity: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunSummary:
    """Aggregate result of one experiment run."""

    task_name: str
    algorithm_name: str
    seed: int
    n_completed: int
    total_eval_time: float
    best_primary_objective: float | None
    incumbents: list[Incumbent]
    stop_reason: str
    description_fingerprint: str | None
    logger_summary: dict[str, Any] = field(default_factory=dict)


class Experimenter:
    """Serial ask/evaluate/tell loop with sanity checks and replay."""

    def __init__(self, task: Task, algorithm: Algorithm, logger_backend: MetricLogger, config: ExperimentConfig):
        self.task = task
        self.algorithm = algorithm
        self.logger = logger_backend
        self.config = config

    def run(self) -> RunSummary:
        if not self.config.stop_on_task_budget and self.config.time_budget is None:
            raise ValueError("At least one stopping criterion must be enabled.")

        report = self.task.sanity_check()
        if not report.ok:
            message = "; ".join(issue.message for issue in report.errors)
            if self.config.fail_fast_on_sanity:
                raise ValueError(f"Task sanity check failed: {message}")
            logger.warning("Task sanity check found errors but execution will continue: %s", message)
        for issue in report.warnings:
            logger.warning("Task sanity warning [%s]: %s", issue.code, issue.message)

        task_spec = self.task.spec
        try:
            description = self.task.get_description()
        except Exception as exc:
            if self.config.fail_fast_on_sanity:
                raise
            logger.warning("Falling back to an empty task description after load failure: %s", exc)
            description = TaskDescriptionBundle.empty(task_id=task_spec.name)

        self.algorithm.setup(task_spec, seed=self.config.seed, task_description=description)
        self.logger.bind_run(
            task_spec=task_spec,
            algorithm_name=self.algorithm.name,
            seed=self.config.seed,
            description_bundle=description,
        )

        next_trial_id = 0
        total_eval_time = 0.0
        n_completed = 0
        if self.config.resume:
            history = self.logger.load_history()
            if history:
                self.algorithm.replay(history)
            state = self.logger.resume_state()
            next_trial_id = state.next_trial_id
            total_eval_time = state.total_eval_time
            n_completed = state.n_completed

        max_evaluations = task_spec.max_evaluations if self.config.stop_on_task_budget else float("inf")
        stop_reason = "task_budget"

        while n_completed < max_evaluations:
            if self.config.time_budget is not None and total_eval_time >= self.config.time_budget:
                stop_reason = "time_budget"
                break

            try:
                suggestion = self.algorithm.ask()
            except Exception as exc:  # pragma: no cover - exceptional failure path.
                raise RuntimeError(f"Algorithm `{self.algorithm.name}` failed during ask().") from exc
            try:
                suggestion = self._normalize_suggestion(suggestion, next_trial_id)
            except Exception as exc:
                invalid_suggestion = TrialSuggestion(
                    trial_id=next_trial_id,
                    config=dict(suggestion.config),
                    budget=suggestion.budget,
                    metadata=dict(suggestion.metadata),
                )
                observation = TrialObservation.from_evaluation(
                    invalid_suggestion,
                    EvaluationResult(
                        status=TrialStatus.INVALID,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    ),
                    metadata=self._runtime_metadata(description),
                )
                self.algorithm.tell(observation)
                self.logger.log(observation)
                next_trial_id += 1
                n_completed += 1
                continue

            observation = self._evaluate_safely(suggestion, description)
            self.algorithm.tell(observation)
            self.logger.log(observation)

            next_trial_id = max(next_trial_id, (observation.suggestion.trial_id or 0) + 1)
            total_eval_time += observation.elapsed_seconds or 0.0
            n_completed += 1

        logger_summary = self.logger.summary()
        return RunSummary(
            task_name=task_spec.name,
            algorithm_name=self.algorithm.name,
            seed=self.config.seed,
            n_completed=n_completed,
            total_eval_time=total_eval_time,
            best_primary_objective=logger_summary.get("best_primary_objective"),
            incumbents=self.algorithm.incumbents(),
            stop_reason=stop_reason,
            description_fingerprint=description.fingerprint or None,
            logger_summary=logger_summary,
        )

    def _normalize_suggestion(self, suggestion: TrialSuggestion, next_trial_id: int) -> TrialSuggestion:
        trial_id = next_trial_id if suggestion.trial_id is None else suggestion.trial_id
        budget = suggestion.budget if suggestion.budget is not None else self.task.spec.default_budget
        if budget is not None and self.task.spec.budget_range is not None:
            low, high = self.task.spec.budget_range
            if budget < low or budget > high:
                raise ValueError(f"Suggested budget {budget} is outside task budget_range {self.task.spec.budget_range!r}.")

        normalized_config = self.task.spec.search_space.coerce_config(suggestion.config, use_defaults=True)
        return TrialSuggestion(
            trial_id=trial_id,
            config=normalized_config,
            budget=budget,
            metadata=dict(suggestion.metadata),
        )

    def _evaluate_safely(
        self,
        suggestion: TrialSuggestion,
        description: TaskDescriptionBundle,
    ) -> TrialObservation:
        try:
            result = self.task.evaluate(suggestion)
            result = self._validate_result(result)
        except Exception as exc:
            result = EvaluationResult(
                status=TrialStatus.FAILED,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        return TrialObservation.from_evaluation(
            suggestion,
            result,
            metadata=self._runtime_metadata(description),
        )

    def _validate_result(self, result: EvaluationResult) -> EvaluationResult:
        primary = self.task.spec.primary_objective
        allowed_objectives = {objective.name for objective in self.task.spec.objectives}
        for name, value in result.objectives.items():
            if name not in allowed_objectives:
                raise ValueError(f"Task returned unknown objective `{name}`.")
            if not math.isfinite(float(value)):
                raise ValueError(f"Objective `{name}` must be finite, got {value!r}.")
        if result.status == TrialStatus.SUCCESS and primary.name not in result.objectives:
            raise ValueError(f"Successful evaluation must include the primary objective `{primary.name}`.")
        if result.elapsed_seconds is not None and not math.isfinite(result.elapsed_seconds):
            raise ValueError("elapsed_seconds must be finite when provided.")
        return result

    def _runtime_metadata(self, description: TaskDescriptionBundle) -> dict[str, Any]:
        return {
            "task_name": self.task.spec.name,
            "algorithm": self.algorithm.name,
            "description_fingerprint": description.fingerprint,
            **self.config.metadata,
        }


__all__ = ["ExperimentConfig", "Experimenter", "RunSummary"]
