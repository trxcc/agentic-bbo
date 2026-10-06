"""Adaptive GP-EI optimizer for reusable black-box optimization.

The supplied, benchmark-agnostic ``GpEiAlgorithm`` is kept as the surrogate
engine (feature encoding, posterior fit, acquisition optimisation, duplicate
handling, deterministic replay) and two improvements are layered on top.

1. Adaptive objective compression.  When the initial observations span a wide
   dynamic range (``max/min >= compression_ratio``) objective values are passed
   through ``arcsinh(value / scale)`` with ``scale = median(|y|) / 16`` recomputed
   from the current history.  ``arcsinh`` is strictly monotone and sign-safe, so
   the argmin/argmax is unchanged, but it behaves like a log transform for large
   magnitudes and linearly near zero.  This stops a few heavy-tailed observations
   from dominating the GP fit and lets the surrogate resolve the (relatively) low
   region that actually matters.  The decision is frozen from the initial
   observations, so a smooth task that later happens to span a wider range is not
   switched mid-run.

2. A trust-region candidate policy.  The first proposal is made on the full
   domain (this matches the reference proposal at the first ask) and, after a
   sustained run of non-improving trials, the acquisition is optimised inside a
   box around the incumbent.  A run of successes widens the box again.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from bbo.core import (
    Algorithm,
    FloatParam,
    Incumbent,
    IntParam,
    ObjectiveDirection,
    SearchSpace,
    TrialObservation,
    TrialSuggestion,
)
from bbo.algorithms.model_based.gp_ei import GpEiAlgorithm

LOG_SCALE_DIVISOR = 16.0


class _TrustRegionGpEi(GpEiAlgorithm):
    """GP-EI whose candidate proposals are restricted to a shrinking box."""

    def __init__(
        self,
        *,
        tr_init: float = 0.8,
        tr_min: float = 0.03,
        tr_max: float = 1.0,
        lookahead_successes: int = 3,
        lookahead_failures: int | None = None,
        expand_factor: float = 1.5,
        shrink_factor: float = 0.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not 0.0 < tr_min <= tr_init <= tr_max:
            raise ValueError("Require 0 < tr_min <= tr_init <= tr_max.")
        if expand_factor <= 1.0 or not 0.0 < shrink_factor < 1.0:
            raise ValueError("expand_factor must be > 1 and shrink_factor in (0, 1).")
        self._tr_init = float(tr_init)
        self._tr_min = float(tr_min)
        self._tr_max = float(tr_max)
        self._success_tolerance = max(1, int(lookahead_successes))
        self._configured_failure_tolerance = lookahead_failures
        self._expand_factor = float(expand_factor)
        self._shrink_factor = float(shrink_factor)
        self._length = self._tr_init
        self._failure_tolerance = 5
        self._success_streak = 0
        self._failure_streak = 0

    def setup(self, task_spec: Any, seed: int = 0, **kwargs: Any) -> None:
        super().setup(task_spec, seed=seed, **kwargs)
        dimension = len(task_spec.search_space)
        self._failure_tolerance = (
            max(1, int(self._configured_failure_tolerance))
            if self._configured_failure_tolerance is not None
            else max(5, int(math.ceil(dimension / 2.0)))
        )
        self._reset_trust_region()

    def _reset_trust_region(self) -> None:
        self._length = self._tr_init
        self._success_streak = 0
        self._failure_streak = 0

    def replay(self, history: list[TrialObservation]) -> None:
        self._reset_trust_region()
        super().replay(history)

    def tell(self, observation: TrialObservation) -> None:
        super().tell(observation)
        accepted = self._best is not None and self._best.trial_id == observation.suggestion.trial_id
        if accepted:
            self._success_streak += 1
            self._failure_streak = 0
        else:
            self._failure_streak += 1
            self._success_streak = 0
        if self._success_streak >= self._success_tolerance:
            self._length = min(self._tr_max, self._length * self._expand_factor)
            self._success_streak = 0
        elif self._failure_streak >= self._failure_tolerance:
            self._length = max(self._tr_min, self._length * self._shrink_factor)
            self._failure_streak = 0

    def ask(self) -> TrialSuggestion:
        self._apply_trust_region()
        return super().ask()

    def _apply_trust_region(self) -> None:
        full_space = self.require_search_space()
        if self._best is None:
            self.set_candidate_search_space(full_space)
            return
        best_config = self._best.config
        params: list[Any] = []
        for param in full_space:
            if isinstance(param, FloatParam):
                low = float(param.low)
                high = float(param.high)
                center = min(max(float(best_config[param.name]), low), high)
                half = 0.5 * self._length * (high - low)
                box_low = max(low, center - half)
                box_high = min(high, center + half)
                if box_high < box_low:
                    box_low = box_high = center
                params.append(
                    FloatParam(
                        name=param.name,
                        low=box_low,
                        high=box_high,
                        log=param.log,
                        default=min(max(center, box_low), box_high),
                    )
                )
            elif isinstance(param, IntParam):
                low = int(param.low)
                high = int(param.high)
                center = min(max(int(round(float(best_config[param.name]))), low), high)
                half = 0.5 * self._length * (high - low)
                box_low = min(max(low, int(math.floor(center - half))), center)
                box_high = max(min(high, int(math.ceil(center + half))), center)
                if box_high < box_low:
                    box_low = box_high = center
                params.append(
                    IntParam(
                        name=param.name,
                        low=box_low,
                        high=box_high,
                        log=param.log,
                        default=min(max(center, box_low), box_high),
                    )
                )
            else:
                params.append(param)
        self.set_candidate_search_space(SearchSpace(params))


class AdaptiveGpEiAlgorithm(Algorithm):
    """Wrapper adding adaptive objective compression around trust-region GP-EI."""

    def __init__(
        self,
        *,
        compression_ratio: float = 20.0,
        scale_divisor: float = LOG_SCALE_DIVISOR,
        min_history_for_transform: int = 6,
        **gp_params: Any,
    ) -> None:
        self._inner = _TrustRegionGpEi(**gp_params)
        self._compression_ratio = float(compression_ratio)
        self._scale_divisor = float(scale_divisor)
        if self._scale_divisor <= 0.0:
            raise ValueError("scale_divisor must be positive.")
        self._min_history_for_transform = max(2, int(min_history_for_transform))
        self._history: list[TrialObservation] = []
        self._primary_name: str | None = None
        self._direction = ObjectiveDirection.MINIMIZE
        self._scale: float | None = None
        self._compression_enabled: bool | None = None

    @property
    def name(self) -> str:
        return "adaptive_gp_ei"

    def setup(self, task_spec: Any, seed: int = 0, **kwargs: Any) -> None:
        if len(task_spec.objectives) != 1:
            raise ValueError("AdaptiveGpEiAlgorithm supports exactly one objective.")
        self._inner.setup(task_spec, seed=seed, **kwargs)
        self._primary_name = task_spec.primary_objective.name
        self._direction = task_spec.primary_objective.direction
        self._history = []
        self._scale = None
        self._compression_enabled = None

    def replay(self, history: list[TrialObservation]) -> None:
        self._history = list(history)
        self._update_scale()

    def tell(self, observation: TrialObservation) -> None:
        self._history.append(observation)

    def ask(self) -> TrialSuggestion:
        self._update_scale()
        transformed = [self._compress(obs) for obs in self._history]
        self._inner.replay(transformed)
        return self._inner.ask()

    def incumbents(self) -> list[Incumbent]:
        best_value: float | None = None
        best_obs: TrialObservation | None = None
        for obs in self._history:
            if not self._observed(obs):
                continue
            value = self._value(obs)
            if best_obs is None or self._is_better(value, best_value):
                best_value = value
                best_obs = obs
        if best_obs is None or best_value is None:
            return []
        return [
            Incumbent(
                config=dict(best_obs.suggestion.config),
                score=best_value,
                objectives=dict(best_obs.objectives),
                trial_id=best_obs.suggestion.trial_id,
                metadata={"algorithm": self.name},
            )
        ]

    def _observed(self, obs: TrialObservation) -> bool:
        return (
            obs.success
            and self._primary_name is not None
            and self._primary_name in obs.objectives
        )

    def _value(self, obs: TrialObservation) -> float:
        assert self._primary_name is not None
        return float(obs.objectives[self._primary_name])

    def _is_better(self, value: float, reference: float | None) -> bool:
        if reference is None:
            return True
        if self._direction == ObjectiveDirection.MAXIMIZE:
            return value > reference
        return value < reference

    def _update_scale(self) -> None:
        values = [
            self._value(obs)
            for obs in self._history
            if self._observed(obs) and math.isfinite(self._value(obs))
        ]
        if len(values) < self._min_history_for_transform:
            self._scale = None
            return
        values_np = np.asarray(values, dtype=float)
        if self._compression_enabled is None:
            low = float(np.min(values_np))
            high = float(np.max(values_np))
            self._compression_enabled = bool(
                low > 0.0 and self._compression_ratio > 0.0 and high / low >= self._compression_ratio
            )
        if not self._compression_enabled:
            self._scale = None
            return
        magnitude = float(np.median(np.abs(values_np)))
        if not math.isfinite(magnitude) or magnitude <= 0.0:
            magnitude = max(abs(float(np.min(values_np))), abs(float(np.max(values_np))), 1e-12)
        self._scale = magnitude / self._scale_divisor

    def _compress(self, obs: TrialObservation) -> TrialObservation:
        if self._scale is None or self._scale <= 0.0 or not self._observed(obs):
            return obs
        value = self._value(obs)
        compressed = float(np.arcsinh(value / self._scale))
        if not math.isfinite(compressed) or compressed == value:
            return obs
        clone = copy.copy(obs)
        objectives = dict(obs.objectives)
        assert self._primary_name is not None
        objectives[self._primary_name] = compressed
        clone.objectives = objectives
        return clone


def create_optimizer() -> Algorithm:
    """Return the optimizer submitted to the host."""

    return AdaptiveGpEiAlgorithm(
        acquisition="logei",
        kernel="matern52",
        xi=0.0,
        alpha=1e-6,
        acqf_num_restarts=16,
        max_acqf_attempts=8,
        tr_init=0.8,
        tr_min=0.03,
        tr_max=1.0,
        lookahead_successes=3,
        lookahead_failures=None,
        expand_factor=1.5,
        shrink_factor=0.7,
        compression_ratio=20.0,
        scale_divisor=LOG_SCALE_DIVISOR,
        min_history_for_transform=6,
    )
