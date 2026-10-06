"""CMA-ES baseline backed by the `pycma` package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cma
import numpy as np

from ...core import (
    ContinuousSearchSpaceConverter,
    ExternalOptimizerAdapter,
    TrialObservation,
    TrialSuggestion,
    build_continuous_converter,
)
from ..benchmark_protocol import FixedInitializationProtocol, resolve_fixed_initialization


@dataclass
class _PendingCandidate:
    batch_id: int
    candidate_index: int
    vector: np.ndarray
    suggestion: TrialSuggestion


class PyCmaAlgorithm(ExternalOptimizerAdapter):
    """Traditional CMA-ES optimizer implemented on top of the core adapter base class."""

    def __init__(
        self,
        *,
        sigma_fraction: float = 0.25,
        popsize: int | None = 2,
        inopts: dict[str, Any] | None = None,
        failure_penalty: float = 1e12,
    ) -> None:
        super().__init__()
        if sigma_fraction <= 0:
            raise ValueError("sigma_fraction must be positive.")
        self.sigma_fraction = sigma_fraction
        self.popsize = popsize
        self.inopts = dict(inopts or {})
        self.failure_penalty = float(failure_penalty)
        self._strategy: cma.CMAEvolutionStrategy | None = None
        self._pending: list[_PendingCandidate] = []
        self._evaluated_batch: dict[int, tuple[np.ndarray, float]] = {}
        self._current_batch_size: int = 0
        self._batch_id: int = 0
        self._continuous_converter: ContinuousSearchSpaceConverter | None = None
        self._seed = 0
        self._fixed_initialization: FixedInitializationProtocol | None = None
        self._initialization_count = 0
        self._seeded_observation_count = 0

    @property
    def name(self) -> str:
        return "pycma"

    def setup(self, task_spec, seed: int = 0, **kwargs: Any) -> None:
        if len(task_spec.objectives) != 1:
            raise ValueError("PyCmaAlgorithm currently supports exactly one objective.")
        self.bind_task_spec(task_spec)
        self._seed = int(seed)
        search_space = self.require_search_space()
        try:
            bounds = search_space.numeric_bounds()
            self._continuous_converter = None
        except TypeError:
            self._continuous_converter = build_continuous_converter(search_space, strategy="onehot")
            bounds = self._continuous_converter.continuous_bounds()
        self._fixed_initialization = resolve_fixed_initialization(
            task_spec, seed=self._seed
        )
        self._strategy = None
        self._pending = []
        self._evaluated_batch = {}
        self._current_batch_size = 0
        self._batch_id = 0
        self._initialization_count = 0
        self._seeded_observation_count = 0

    def _initialize_strategy(self) -> None:
        search_space = self.require_search_space()
        if self._continuous_converter is None:
            bounds = search_space.numeric_bounds()
        else:
            bounds = self._continuous_converter.continuous_bounds()
        incumbents = self.incumbents()
        initial_config = (
            dict(incumbents[0].config)
            if incumbents
            else self._task_spec.metadata.get(
                "cma_initial_config", search_space.defaults()
            )
        )
        if self._continuous_converter is None:
            x0 = search_space.to_numeric_vector(initial_config)
        else:
            x0 = self._continuous_converter.encode_vector(initial_config)
        widths = bounds[:, 1] - bounds[:, 0]
        sigma0 = float(np.mean(widths) * self.sigma_fraction)
        sigma0 = max(sigma0, 1e-6)

        options = {
            "bounds": [bounds[:, 0].tolist(), bounds[:, 1].tolist()],
            "seed": self._seed,
            "verbose": -9,
            "verb_disp": 0,
        }
        if self.popsize is not None:
            options["popsize"] = int(self.popsize)
        options.update(self.inopts)

        self._strategy = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, options)
        self._pending = []
        self._evaluated_batch = {}
        self._current_batch_size = 0
        self._batch_id = 0

    def ask(self) -> TrialSuggestion:
        if (
            self._fixed_initialization is not None
            and self._initialization_count
            < len(self._fixed_initialization.configurations)
        ):
            suggestion = self._fixed_initialization.suggestion(
                self._initialization_count, algorithm=self.name
            )
            suggestion.metadata["pycma_phase"] = "benchmark_initialization"
            return suggestion
        if self._strategy is None:
            self._initialize_strategy()
        strategy = self._require_strategy()
        search_space = self.require_search_space()
        converter = self._continuous_converter
        if not self._pending:
            raw_vectors = strategy.ask()
            self._pending = []
            self._evaluated_batch = {}
            self._current_batch_size = len(raw_vectors)
            for candidate_index, raw_vector in enumerate(raw_vectors):
                latent_vector = np.asarray(raw_vector, dtype=float)
                if converter is None:
                    config = search_space.from_numeric_vector(latent_vector, clip=True)
                else:
                    config = converter.decode_vector(latent_vector, clip=True)
                suggestion = TrialSuggestion(
                    config=config,
                    metadata={
                        "pycma_batch_id": self._batch_id,
                        "pycma_candidate_index": candidate_index,
                        "pycma_seeded_observations": self._seeded_observation_count,
                        # Fitness belongs to the exact latent point returned by
                        # CMA ask(). Integer rounding and categorical decoding
                        # only affect the evaluated config. Telling CMA the
                        # re-encoded config turns every rounded point into an
                        # injected solution and corrupts ask/tell bookkeeping.
                        "pycma_vector": latent_vector.tolist(),
                    },
                )
                self._pending.append(
                    _PendingCandidate(
                        batch_id=self._batch_id,
                        candidate_index=candidate_index,
                        vector=latent_vector,
                        suggestion=suggestion,
                    )
                )
            self._batch_id += 1
        pending = self._pending.pop(0)
        return TrialSuggestion(
            config=dict(pending.suggestion.config),
            budget=pending.suggestion.budget,
            metadata=dict(pending.suggestion.metadata),
        )

    def tell(self, observation: TrialObservation) -> None:
        if (
            self._fixed_initialization is not None
            and self._initialization_count
            < len(self._fixed_initialization.configurations)
        ):
            self._initialization_count += 1
            self.update_best_incumbent(observation)
            return
        strategy = self._require_strategy()
        candidate_index = observation.suggestion.metadata.get("pycma_candidate_index")
        if observation.suggestion.metadata.get("pycma_batch_id") is None or candidate_index is None:
            raise ValueError("PyCMA suggestions must preserve `pycma_batch_id` and `pycma_candidate_index` metadata.")
        vector = observation.suggestion.metadata.get("pycma_vector")
        if vector is None:
            if self._continuous_converter is None:
                vector_array = self.require_search_space().to_numeric_vector(observation.suggestion.config)
            else:
                vector_array = self._continuous_converter.encode_vector(observation.suggestion.config)
        else:
            vector_array = np.asarray(vector, dtype=float)

        fitness = self.objective_to_minimization_score(observation, failure_penalty=self.failure_penalty)
        self._evaluated_batch[int(candidate_index)] = (vector_array, fitness)
        self.update_best_incumbent(observation)

        if len(self._evaluated_batch) == self._current_batch_size and self._current_batch_size > 0:
            ordered = [self._evaluated_batch[index] for index in range(self._current_batch_size)]
            vectors = [vector.tolist() for vector, _ in ordered]
            fitnesses = [fitness_value for _, fitness_value in ordered]
            strategy.tell(vectors, fitnesses)
            self._evaluated_batch = {}
            self._current_batch_size = 0

    def seed(self, observation: TrialObservation) -> None:
        """Warm-start CMA from shared/external history without inventing fitness."""

        if (
            self._fixed_initialization is not None
            and self._initialization_count
            < len(self._fixed_initialization.configurations)
        ):
            expected = self.ask()
            self.assert_matching_config(
                expected.config, observation.suggestion.config
            )
            self.tell(self.make_replay_observation(expected, observation))
            return
        self._seeded_observation_count += 1
        self.update_best_incumbent(observation)

    def _require_strategy(self) -> cma.CMAEvolutionStrategy:
        if self._strategy is None:
            raise RuntimeError(f"{type(self).__name__}.setup() must be called before ask/tell.")
        return self._strategy


__all__ = ["PyCmaAlgorithm"]
