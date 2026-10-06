"""Trust-region GP-EI (TuRBO-style) built on the curated support API.

The reference GP-EI searches the whole domain with a single stationary GP.
For 10-D anonymous tasks a trust-region wrapper concentrates the acquisition
search around the incumbent while still allowing global exploration early and
after restarts.  Candidate generation is restricted through the supported
``set_candidate_search_space`` hook, so the underlying GP fit, acquisition
optimization and replay semantics are unchanged.
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import Any

from solver_support import (
    CategoricalParam,
    FloatParam,
    IntParam,
    ObjectiveDirection,
    SearchSpace,
    GpEiAlgorithm,
)


def _stable_int(*parts: object) -> int:
    text = ":".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**31 - 1)


class TurboGpEi:
    """TuRBO-flavoured wrapper around :class:`GpEiAlgorithm`."""

    name = "turbo_gp_ei"

    def __init__(
        self,
        *,
        length_init: float = 0.8,
        length_min: float = 0.5**7,
        length_max: float = 1.6,
        success_tolerance: int = 3,
        failure_tolerance: int | None = None,
        inner_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.length_init = float(length_init)
        self.length_min = float(length_min)
        self.length_max = float(length_max)
        self.success_tolerance = int(success_tolerance)
        self.failure_tolerance = failure_tolerance
        base = {
            "startup_trials": 2,
            "acquisition": "logei",
            "pool_size": 4096,
            "acqf_num_restarts": 24,
            "max_acqf_attempts": 24,
            "xi": 0.0,
            "alpha": 1e-6,
            "n_restarts_optimizer": 0,
        }
        if inner_kwargs:
            base.update(inner_kwargs)
        self._inner = GpEiAlgorithm(**base)
        self._inner_kwargs = base
        self._space: SearchSpace | None = None
        self._seed = 0
        self._primary_name: str | None = None
        self._direction = ObjectiveDirection.MINIMIZE
        self._length = self.length_init
        self._center: dict[str, Any] | None = None
        self._best_score: float | None = None
        self._n_success = 0
        self._n_failure = 0

    # -- protocol -----------------------------------------------------------
    def setup(self, task_spec, seed: int = 0, **kwargs: Any) -> None:
        self._inner.setup(task_spec, seed, **kwargs)
        self._space = task_spec.search_space
        self._seed = int(seed)
        self._primary_name = task_spec.primary_objective.name
        self._direction = task_spec.primary_objective.direction
        self._length = self.length_init
        self._center = None
        self._best_score = None
        self._n_success = 0
        self._n_failure = 0

    def ask(self):
        self._inner.set_candidate_search_space(self._region_space())
        suggestion = self._inner.ask()
        reason = str(suggestion.metadata.get("gp_ei_random_reason", ""))
        if reason.startswith("gp_ei_fallback") and self._inner.acquisition == "logei":
            # LogEI unavailable/broken in this backend: fall back to plain EI.
            self._inner.acquisition = "ei"
            self._inner_kwargs["acquisition"] = "ei"
            suggestion = self._inner.ask()
        return suggestion

    def tell(self, observation) -> None:
        self._inner.tell(observation)
        self._absorb(observation)

    def replay(self, history) -> None:
        for observation in history:
            self.tell(observation)

    def incumbents(self):
        return self._inner.incumbents() if self._inner.incumbents() else []

    def seed(self, observation) -> None:
        self.tell(observation)

    # -- trust region bookkeeping ------------------------------------------
    def _absorb(self, observation) -> None:
        if not observation.success or self._primary_name is None:
            return
        if self._primary_name not in observation.objectives:
            return
        value = float(observation.objectives[self._primary_name])
        minimize = self._direction == ObjectiveDirection.MINIMIZE
        improved = (
            self._best_score is None
            or (value < self._best_score if minimize else value > self._best_score)
        )
        if improved:
            self._best_score = value
            self._center = dict(observation.suggestion.config)
            self._n_success += 1
            self._n_failure = 0
            if self._n_success >= self.success_tolerance:
                self._length = min(self._length * 2.0, self.length_max)
                self._n_success = 0
        else:
            self._n_failure += 1
            self._n_success = 0
            if self._n_failure >= self._failure_tolerance():
                self._length = max(self._length / 2.0, self.length_min)
                self._n_failure = 0
                if self._length <= self.length_min * 1.0001:
                    self._restart()

    def _failure_tolerance(self) -> int:
        if self.failure_tolerance is not None:
            return max(1, int(self.failure_tolerance))
        dimension = max(1, len(self._space) if self._space is not None else 1)
        return max(4, int(math.ceil(1.5 * dimension)))

    def _restart(self) -> None:
        assert self._space is not None
        rng = random.Random(_stable_int(self.name, self._seed, len(self._inner._history), "restart"))
        self._center = self._space.coerce_config(self._space.sample(rng), use_defaults=False)
        self._length = self.length_init
        self._n_success = 0
        self._n_failure = 0

    def _region_space(self) -> SearchSpace:
        assert self._space is not None
        if self._center is None:
            return self._space
        half = 0.5 * self._length
        parameters = []
        for param in self._space:
            center_value = self._center.get(param.name)
            if center_value is None:
                parameters.append(param)
                continue
            if isinstance(param, FloatParam):
                low, high = float(param.low), float(param.high)
                span = high - low
                width = half * span
                new_low = max(low, float(center_value) - width)
                new_high = min(high, float(center_value) + width)
                if new_high - new_low < 1e-12 * max(1.0, span):
                    mid = min(max(float(center_value), low), high)
                    new_low = max(low, mid - 1e-9 * max(1.0, span))
                    new_high = min(high, mid + 1e-9 * max(1.0, span))
                parameters.append(FloatParam(name=param.name, default=None, low=new_low, high=new_high, log=param.log))
            elif isinstance(param, IntParam):
                low, high = int(param.low), int(param.high)
                width = half * (high - low)
                new_low = max(low, int(math.floor(float(center_value) - width)))
                new_high = min(high, int(math.ceil(float(center_value) + width)))
                if new_high < new_low:
                    new_low = new_high = min(max(int(round(float(center_value))), low), high)
                parameters.append(IntParam(name=param.name, default=None, low=new_low, high=new_high, log=param.log))
            else:
                parameters.append(param)
        return SearchSpace(parameters)


def create_optimizer():
    return TurboGpEi()
