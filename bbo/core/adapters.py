"""Adapter base classes and interoperability helpers."""

from __future__ import annotations

import math
from abc import ABC
from typing import Any

from .algo import Algorithm, Incumbent
from .space import SearchSpace, from_configspace, to_configspace
from .task import ObjectiveDirection, TaskSpec
from .trial import TrialObservation, TrialSuggestion


class ExternalOptimizerAdapter(Algorithm, ABC):
    """Base class for wrapping external optimizers behind the core protocol.

    Subclasses are expected to implement `setup()`, `ask()`, and `tell()` while
    reusing the common task binding, replay, and incumbent bookkeeping helpers.
    """

    def __init__(self) -> None:
        self._task_spec: TaskSpec | None = None
        self._search_space: SearchSpace | None = None
        self._primary_name: str | None = None
        self._primary_direction = ObjectiveDirection.MINIMIZE
        self._best: Incumbent | None = None

    def bind_task_spec(self, task_spec: TaskSpec) -> None:
        """Bind the task metadata shared by most external-optimizer adapters."""

        if not task_spec.objectives:
            raise ValueError("Adapter algorithms require at least one objective.")
        self._task_spec = task_spec
        self._search_space = task_spec.search_space
        self._primary_name = task_spec.primary_objective.name
        self._primary_direction = task_spec.primary_objective.direction
        self._best = None

    def replay(self, history: list[TrialObservation]) -> None:
        """Rebuild adapter state by deterministically re-asking and re-telling."""

        for observation in history:
            expected = self.ask()
            self.assert_matching_config(expected.config, observation.suggestion.config)
            self.tell(self.make_replay_observation(expected, observation))

    def incumbents(self) -> list[Incumbent]:
        return [self._best] if self._best is not None else []

    def make_replay_observation(
        self,
        expected: TrialSuggestion,
        observation: TrialObservation,
    ) -> TrialObservation:
        """Merge deterministic ask-side metadata into a replayed observation."""

        return TrialObservation(
            suggestion=TrialSuggestion(
                config=dict(observation.suggestion.config),
                trial_id=observation.suggestion.trial_id,
                budget=observation.suggestion.budget,
                metadata=dict(expected.metadata),
            ),
            status=observation.status,
            objectives=dict(observation.objectives),
            metrics=dict(observation.metrics),
            elapsed_seconds=observation.elapsed_seconds,
            error_type=observation.error_type,
            error_message=observation.error_message,
            timestamp=observation.timestamp,
            metadata=dict(observation.metadata),
        )

    def objective_to_minimization_score(self, observation: TrialObservation, *, failure_penalty: float) -> float:
        """Convert the primary objective into a minimization score for external optimizers."""

        assert self._primary_name is not None
        if not observation.success or self._primary_name not in observation.objectives:
            return float(failure_penalty)
        value = float(observation.objectives[self._primary_name])
        if self._primary_direction == ObjectiveDirection.MAXIMIZE:
            return -value
        return value

    def update_best_incumbent(self, observation: TrialObservation) -> None:
        """Track the best valid observation seen so far."""

        assert self._primary_name is not None
        if not observation.success or self._primary_name not in observation.objectives:
            return
        score = float(observation.objectives[self._primary_name])
        incumbent = Incumbent(
            config=dict(observation.suggestion.config),
            score=score,
            objectives=dict(observation.objectives),
            trial_id=observation.suggestion.trial_id,
            metadata={"algorithm": self.name},
        )
        if self._best is None:
            self._best = incumbent
            return
        if self._primary_direction == ObjectiveDirection.MINIMIZE and score < float(self._best.score):
            self._best = incumbent
        if self._primary_direction == ObjectiveDirection.MAXIMIZE and score > float(self._best.score):
            self._best = incumbent

    def require_task_spec(self) -> TaskSpec:
        if self._task_spec is None:
            raise RuntimeError(f"{type(self).__name__}.setup() must be called before ask/tell.")
        return self._task_spec

    def require_search_space(self) -> SearchSpace:
        if self._search_space is None:
            raise RuntimeError(f"{type(self).__name__}.setup() must be called before ask/tell.")
        return self._search_space

    @staticmethod
    def assert_matching_config(expected: dict[str, Any], actual: dict[str, Any]) -> None:
        """Validate that a replayed suggestion matches the deterministic ask() output."""

        if expected.keys() != actual.keys():
            raise ValueError("Replay history does not match the optimizer search-space keys.")
        for key in expected:
            expected_value = expected[key]
            actual_value = actual[key]
            if isinstance(expected_value, float) or isinstance(actual_value, float):
                if not math.isclose(float(expected_value), float(actual_value), rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(
                        f"Replay history diverged for `{key}`: expected {expected_value!r}, got {actual_value!r}."
                    )
            elif expected_value != actual_value:
                raise ValueError(
                    f"Replay history diverged for `{key}`: expected {expected_value!r}, got {actual_value!r}."
                )


__all__ = ["ExternalOptimizerAdapter", "from_configspace", "to_configspace"]
