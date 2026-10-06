"""Standalone Graph GA algorithm for SMILES tasks.

Adapted from PMO mol_opt/main/graph_ga/run.py.

PMO's original loop evaluates whole offspring batches by calling the oracle
inside the optimizer. This wrapper preserves the molecular operators and
selection rule, but moves true task evaluation into the benchmark ask/tell loop:
`ask()` proposes one SMILES and `tell()` supplies the real score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ...core import ExternalOptimizerAdapter, ObjectiveDirection, StringParam, TrialObservation, TrialSuggestion
from .graph_ga_ops import generate_offspring_smiles, load_smiles_file, sanitize_smiles

GRAPH_GA_DEFAULT_POPULATION_SIZE = 120
GRAPH_GA_DEFAULT_OFFSPRING_SIZE = 70
GRAPH_GA_DEFAULT_MUTATION_RATE = 0.067


class GraphGAAlgorithm(ExternalOptimizerAdapter):
    """Graph genetic algorithm over a direct SMILES `StringParam` search space."""

    def __init__(
        self,
        *,
        initial_smiles: Sequence[str] | None = None,
        initial_smiles_path: str | Path | None = None,
        population_size: int = GRAPH_GA_DEFAULT_POPULATION_SIZE,
        offspring_size: int = GRAPH_GA_DEFAULT_OFFSPRING_SIZE,
        mutation_rate: float = GRAPH_GA_DEFAULT_MUTATION_RATE,
        max_generation_attempts: int = 8,
        crossover_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if initial_smiles is not None and initial_smiles_path is not None:
            raise ValueError("Pass either initial_smiles or initial_smiles_path, not both.")
        if population_size <= 0:
            raise ValueError("population_size must be positive.")
        if offspring_size <= 0:
            raise ValueError("offspring_size must be positive.")
        if mutation_rate < 0:
            raise ValueError("mutation_rate must be non-negative.")
        if max_generation_attempts <= 0:
            raise ValueError("max_generation_attempts must be positive.")
        self.initial_smiles = tuple(str(item) for item in initial_smiles) if initial_smiles is not None else None
        self.initial_smiles_path = Path(initial_smiles_path) if initial_smiles_path is not None else None
        self.population_size = int(population_size)
        self.offspring_size = int(offspring_size)
        self.mutation_rate = float(mutation_rate)
        self.max_generation_attempts = int(max_generation_attempts)
        self.crossover_kwargs = dict(crossover_kwargs or {})
        self._seed = 0
        self._smiles_param_name: str | None = None
        self._initial_queue: list[str] = []
        self._generation_queue: list[str] = []
        self._population: dict[str, float] = {}
        self._seen_smiles: set[str] = set()
        self._generation_index = 0
        self._active_generation_expected = 0
        self._active_generation_scores: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "graph_ga"

    def setup(self, task_spec, seed: int = 0, **kwargs: Any) -> None:
        if len(task_spec.objectives) != 1:
            raise ValueError("GraphGAAlgorithm currently supports exactly one objective.")
        self.bind_task_spec(task_spec)
        string_params = [param for param in task_spec.search_space if isinstance(param, StringParam)]
        if len(string_params) != 1:
            raise ValueError("GraphGAAlgorithm requires exactly one StringParam SMILES search parameter.")
        self._smiles_param_name = string_params[0].name
        self._seed = int(seed)
        self._population = {}
        self._seen_smiles = set()
        self._generation_queue = []
        self._generation_index = 0
        self._active_generation_expected = 0
        self._active_generation_scores = {}

        if self.initial_smiles_path is not None:
            source_smiles = load_smiles_file(self.initial_smiles_path)
        elif self.initial_smiles is not None:
            source_smiles = list(self.initial_smiles)
        else:
            raise ValueError(
                "GraphGAAlgorithm requires explicit initial_smiles or initial_smiles_path. "
                "No default molecular population is invented by the framework."
            )
        initial = sanitize_smiles(source_smiles)[: self.population_size]
        if len(initial) < 2:
            raise ValueError("GraphGAAlgorithm requires at least two valid initial SMILES.")
        self._initial_queue = list(initial)

    def ask(self) -> TrialSuggestion:
        param_name = self._require_smiles_param_name()
        if self._initial_queue:
            smiles = self._initial_queue.pop(0)
            self._seen_smiles.add(smiles)
            return TrialSuggestion(
                config={param_name: smiles},
                metadata={
                    "graph_ga_phase": "initial_population",
                    "graph_ga_generation": -1,
                    "graph_ga_population_size": self.population_size,
                    "graph_ga_offspring_size": self.offspring_size,
                    "graph_ga_mutation_rate": self.mutation_rate,
                },
            )

        if not self._generation_queue:
            self._start_generation()
        smiles = self._generation_queue.pop(0)
        self._seen_smiles.add(smiles)
        return TrialSuggestion(
            config={param_name: smiles},
            metadata={
                "graph_ga_phase": "offspring",
                "graph_ga_generation": self._generation_index,
                "graph_ga_population_size": self.population_size,
                "graph_ga_offspring_size": self.offspring_size,
                "graph_ga_mutation_rate": self.mutation_rate,
            },
        )

    def tell(self, observation: TrialObservation) -> None:
        self.update_best_incumbent(observation)
        if not observation.success:
            return
        score = self._score_for_maximization(observation)
        if score is None:
            return
        param_name = self._require_smiles_param_name()
        smiles = sanitize_smiles([str(observation.suggestion.config.get(param_name, ""))])
        if not smiles:
            return
        canonical = smiles[0]
        phase = observation.suggestion.metadata.get("graph_ga_phase")
        if phase == "offspring":
            self._active_generation_scores[canonical] = score
            if len(self._active_generation_scores) >= self._active_generation_expected:
                self._population.update(self._active_generation_scores)
                self._trim_population()
                self._active_generation_scores = {}
                self._active_generation_expected = 0
        else:
            self._population[canonical] = score
            self._trim_population()

    def _start_generation(self) -> None:
        if not self._population:
            raise RuntimeError("GraphGAAlgorithm has no scored population. Evaluate initial SMILES first.")
        population_items = sorted(self._population.items(), key=lambda item: item[1], reverse=True)
        population_smiles = [item[0] for item in population_items[: self.population_size]]
        population_scores = [item[1] for item in population_items[: self.population_size]]

        candidates: list[str] = []
        for attempt in range(self.max_generation_attempts):
            offspring = generate_offspring_smiles(
                population_smiles=population_smiles,
                population_scores=population_scores,
                offspring_size=self.offspring_size,
                mutation_rate=self.mutation_rate,
                mating_pool_size=self.population_size,
                min_func_val=0.0,
                crossover_kwargs=self.crossover_kwargs,
                seed=self._seed + self._generation_index * 1009 + attempt,
            )
            for smiles in offspring:
                if smiles not in self._seen_smiles and smiles not in candidates:
                    candidates.append(smiles)
            if candidates:
                break
        if not candidates:
            raise RuntimeError("GraphGAAlgorithm could not generate any unseen valid offspring SMILES.")
        self._generation_queue = candidates
        self._active_generation_expected = len(candidates)
        self._active_generation_scores = {}
        self._generation_index += 1

    def _score_for_maximization(self, observation: TrialObservation) -> float | None:
        assert self._primary_name is not None
        if self._primary_name not in observation.objectives:
            return None
        value = float(observation.objectives[self._primary_name])
        if self._primary_direction == ObjectiveDirection.MINIMIZE:
            return -value
        return value

    def _trim_population(self) -> None:
        ordered = sorted(self._population.items(), key=lambda item: item[1], reverse=True)[: self.population_size]
        self._population = dict(ordered)

    def _require_smiles_param_name(self) -> str:
        if self._smiles_param_name is None:
            raise RuntimeError("GraphGAAlgorithm.setup() must be called before ask/tell.")
        return self._smiles_param_name


__all__ = [
    "GRAPH_GA_DEFAULT_MUTATION_RATE",
    "GRAPH_GA_DEFAULT_OFFSPRING_SIZE",
    "GRAPH_GA_DEFAULT_POPULATION_SIZE",
    "GraphGAAlgorithm",
]
