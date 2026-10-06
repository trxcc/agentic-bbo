"""Uniform random search baseline."""

from __future__ import annotations

import math
import random
from typing import Any

from ...core import Incumbent, ObjectiveDirection, TaskSpec, TrialObservation, TrialSuggestion
from ...core.algo import Algorithm
from ..benchmark_protocol import FixedInitializationProtocol, resolve_fixed_initialization


class RandomSearchAlgorithm(Algorithm):
    """Uniform random search with deterministic replay semantics."""

    def __init__(self):
        self._rng = random.Random()
        self._task_spec: TaskSpec | None = None
        self._best: Incumbent | None = None
        self._primary_name: str | None = None
        self._direction = ObjectiveDirection.MINIMIZE
        self._fixed_initialization: FixedInitializationProtocol | None = None
        self._observation_count = 0

    @property
    def name(self) -> str:
        return "random_search"

    def setup(self, task_spec: TaskSpec, seed: int = 0, **kwargs: Any) -> None:
        self._task_spec = task_spec
        self._rng = random.Random(seed)
        self._best = None
        self._primary_name = task_spec.primary_objective.name
        self._direction = task_spec.primary_objective.direction
        self._fixed_initialization = resolve_fixed_initialization(task_spec, seed=int(seed))
        self._observation_count = 0

    def ask(self) -> TrialSuggestion:
        if self._task_spec is None:
            raise RuntimeError("RandomSearchAlgorithm.setup() must be called before ask().")
        if self._fixed_initialization is not None and self._observation_count < len(
            self._fixed_initialization.configurations
        ):
            suggestion = self._fixed_initialization.suggestion(self._observation_count, algorithm=self.name)
            suggestion.metadata["random_search_phase"] = "benchmark_initialization"
            return suggestion
        return TrialSuggestion(config=self._task_spec.search_space.sample(self._rng))

    def tell(self, observation: TrialObservation) -> None:
        self._observation_count += 1
        if self._primary_name is None or not observation.success:
            return
        if self._primary_name not in observation.objectives:
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
        if self._direction == ObjectiveDirection.MINIMIZE and score < float(self._best.score):
            self._best = incumbent
        if self._direction == ObjectiveDirection.MAXIMIZE and score > float(self._best.score):
            self._best = incumbent

    def replay(self, history: list[TrialObservation]) -> None:
        for observation in history:
            expected = self.ask()
            self._assert_same_config(expected.config, observation.suggestion.config)
            self.tell(observation)

    def incumbents(self) -> list[Incumbent]:
        return [self._best] if self._best is not None else []

    @staticmethod
    def _assert_same_config(expected: dict[str, Any], actual: dict[str, Any]) -> None:
        if expected.keys() != actual.keys():
            raise ValueError("Replay history does not match the random-search search space.")
        for key in expected:
            expected_value = expected[key]
            actual_value = actual[key]
            if isinstance(expected_value, float) or isinstance(actual_value, float):
                if not math.isclose(float(expected_value), float(actual_value), rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(
                        f"Replay history diverged for `{key}`: expected {expected_value!r}, got {actual_value!r}."
                    )
            elif expected_value != actual_value:
                raise ValueError(
                    f"Replay history diverged for `{key}`: expected {expected_value!r}, got {actual_value!r}."
                )
