"""Incumbent-centred local perturbation baseline."""

from __future__ import annotations

import hashlib
import random
from typing import Any

from ...core import (
    CategoricalParam,
    ExternalOptimizerAdapter,
    FloatParam,
    IntParam,
    StringParam,
    TrialObservation,
    TrialSuggestion,
)
from ..benchmark_protocol import FixedInitializationProtocol, resolve_fixed_initialization


class LocalPerturbationAlgorithm(ExternalOptimizerAdapter):
    """Deterministic Gaussian perturbations around the current incumbent."""

    def __init__(self, *, jitter_fraction: float = 0.1, max_duplicate_attempts: int = 512) -> None:
        super().__init__()
        if not 0.0 < float(jitter_fraction) <= 1.0:
            raise ValueError("jitter_fraction must be in (0, 1].")
        if int(max_duplicate_attempts) <= 0:
            raise ValueError("max_duplicate_attempts must be positive.")
        self.jitter_fraction = float(jitter_fraction)
        self.max_duplicate_attempts = int(max_duplicate_attempts)
        self._seed = 0
        self._history: list[TrialObservation] = []
        self._seen: set[str] = set()
        self._fixed_initialization: FixedInitializationProtocol | None = None

    @property
    def name(self) -> str:
        return "local_perturbation"

    def setup(self, task_spec, seed: int = 0, **kwargs: Any) -> None:
        if len(task_spec.objectives) != 1:
            raise ValueError("LocalPerturbationAlgorithm supports exactly one objective.")
        self.bind_task_spec(task_spec)
        self._seed = int(seed)
        self._fixed_initialization = resolve_fixed_initialization(task_spec, seed=self._seed)
        self._history = []
        self._seen = set()

    def ask(self) -> TrialSuggestion:
        if self._fixed_initialization is not None and len(self._history) < len(
            self._fixed_initialization.configurations
        ):
            suggestion = self._fixed_initialization.suggestion(len(self._history), algorithm=self.name)
            suggestion.metadata["local_perturbation_phase"] = "benchmark_initialization"
            self._seen.add(_identity(suggestion.config))
            return suggestion

        space = self.require_search_space()
        incumbents = self.incumbents()
        anchor = dict(incumbents[0].config) if incumbents else space.defaults()
        rng = random.Random(self._stable_seed("ask", len(self._history)))
        last: dict[str, Any] | None = None
        for attempt in range(self.max_duplicate_attempts):
            candidate: dict[str, Any] = {}
            for param in space:
                current = anchor.get(param.name, param.effective_default())
                if isinstance(param, FloatParam):
                    width = float(param.high) - float(param.low)
                    value = float(current) + rng.gauss(0.0, self.jitter_fraction * width)
                    candidate[param.name] = min(max(value, float(param.low)), float(param.high))
                elif isinstance(param, IntParam):
                    width = int(param.high) - int(param.low)
                    value = int(round(int(current) + rng.gauss(0.0, max(1.0, self.jitter_fraction * width))))
                    candidate[param.name] = min(max(value, int(param.low)), int(param.high))
                elif isinstance(param, CategoricalParam):
                    candidate[param.name] = rng.choice(param.choices) if rng.random() < 0.2 else current
                elif isinstance(param, StringParam):
                    raise TypeError("LocalPerturbationAlgorithm does not support open string parameters.")
            normalized = space.coerce_config(candidate, use_defaults=False)
            last = normalized
            if _identity(normalized) not in self._seen:
                self._seen.add(_identity(normalized))
                return TrialSuggestion(
                    config=normalized,
                    metadata={
                        "local_perturbation_phase": "perturbation",
                        "local_perturbation_jitter_fraction": self.jitter_fraction,
                        "local_perturbation_attempt": attempt,
                    },
                )
        assert last is not None
        return TrialSuggestion(
            config=last,
            metadata={
                "local_perturbation_phase": "duplicate_fallback",
                "local_perturbation_jitter_fraction": self.jitter_fraction,
            },
        )

    def tell(self, observation: TrialObservation) -> None:
        self._history.append(observation)
        self._seen.add(_identity(observation.suggestion.config))
        self.update_best_incumbent(observation)

    def replay(self, history: list[TrialObservation]) -> None:
        self._history = []
        self._seen = set()
        self._best = None
        for observation in history:
            self.tell(observation)

    def _stable_seed(self, *parts: object) -> int:
        value = ":".join(str(part) for part in (self.name, self._seed, *parts))
        return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16) % (2**31 - 1)


def _identity(config: dict[str, Any]) -> str:
    return repr(sorted(config.items()))


__all__ = ["LocalPerturbationAlgorithm"]
