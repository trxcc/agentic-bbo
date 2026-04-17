"""Shared task wrappers for synthetic black-box objectives."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ...core import (
    EvaluationResult,
    ObjectiveDirection,
    ObjectiveSpec,
    SearchSpace,
    Task,
    TaskDescriptionRef,
    TaskSpec,
    TrialStatus,
    TrialSuggestion,
)


ArrayObjective = Callable[[np.ndarray], float]
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
TASK_DESCRIPTION_ROOT = PACKAGE_ROOT / "task_descriptions"


@dataclass(frozen=True)
class SyntheticFunctionDefinition:
    """Static definition for one synthetic benchmark function."""

    key: str
    display_name: str
    description: str
    search_space: SearchSpace
    objective: ArrayObjective
    known_optimum: float
    known_optima: tuple[tuple[float, ...], ...]
    description_dir: Path
    default_max_evaluations: int = 40
    objective_name: str = "loss"
    plot_resolution: int = 180

    @property
    def dimension(self) -> int:
        return len(self.search_space)


@dataclass
class SyntheticFunctionTaskConfig:
    """Configuration for one synthetic benchmark task instance."""

    problem: str = "branin_demo"
    max_evaluations: int | None = None
    seed: int = 0
    noise_std: float = 0.0
    description_dir: Path | None = None
    metadata: dict[str, str] = field(default_factory=dict)


class SyntheticFunctionTask(Task):
    """Task wrapper around a deterministic synthetic objective."""

    def __init__(
        self,
        config: SyntheticFunctionTaskConfig,
        definition: SyntheticFunctionDefinition | None = None,
    ):
        self.config = config
        if definition is None:
            from ..registry import get_synthetic_problem

            definition = get_synthetic_problem(config.problem)
        self.definition = definition
        self._rng = np.random.default_rng(config.seed)
        description_dir = config.description_dir or self.definition.description_dir
        search_space = self.definition.search_space
        self._spec = TaskSpec(
            name=self.definition.key,
            search_space=search_space,
            objectives=(ObjectiveSpec(self.definition.objective_name, ObjectiveDirection.MINIMIZE),),
            max_evaluations=config.max_evaluations or self.definition.default_max_evaluations,
            description_ref=TaskDescriptionRef.from_directory(self.definition.key, description_dir),
            metadata={
                "problem_key": self.definition.key,
                "display_name": self.definition.display_name,
                "dimension": self.definition.dimension,
                "known_optimum": self.definition.known_optimum,
                "known_optima": [list(point) for point in self.definition.known_optima],
                "plot_resolution": self.definition.plot_resolution,
                "cma_initial_config": search_space.defaults(),
                **config.metadata,
            },
        )

    @property
    def spec(self) -> TaskSpec:
        return self._spec

    def evaluate(self, suggestion: TrialSuggestion) -> EvaluationResult:
        start = time.perf_counter()
        config = self.spec.search_space.coerce_config(suggestion.config, use_defaults=False)
        vector = self.spec.search_space.to_numeric_vector(config)
        value = float(self.definition.objective(vector))
        if self.config.noise_std > 0:
            value += float(self._rng.normal(0.0, self.config.noise_std))

        elapsed = time.perf_counter() - start
        best_distance = self._distance_to_known_optimum(vector)
        metrics = {
            "regret": value - float(self.definition.known_optimum),
            "distance_to_known_optimum": best_distance,
            "dimension": float(self.definition.dimension),
        }
        for name, scalar in zip(self.spec.search_space.names(), vector, strict=True):
            metrics[f"coord::{name}"] = float(scalar)

        return EvaluationResult(
            status=TrialStatus.SUCCESS,
            objectives={self.definition.objective_name: value},
            metrics=metrics,
            elapsed_seconds=elapsed,
            metadata={
                "problem_key": self.definition.key,
                "display_name": self.definition.display_name,
            },
        )

    def surface_grid(self, *, resolution: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.definition.dimension != 2:
            raise TypeError("surface_grid() is only available for 2D synthetic tasks.")
        bounds = self.spec.search_space.numeric_bounds()
        resolution = resolution or int(self.spec.metadata.get("plot_resolution", self.definition.plot_resolution))
        xs = np.linspace(bounds[0, 0], bounds[0, 1], resolution)
        ys = np.linspace(bounds[1, 0], bounds[1, 1], resolution)
        xx, yy = np.meshgrid(xs, ys)
        zz = np.zeros_like(xx)
        for index in range(xx.shape[0]):
            stacked = np.stack([xx[index], yy[index]], axis=1)
            zz[index, :] = [self.definition.objective(row) for row in stacked]
        return xx, yy, zz

    def sanity_check(self):
        report = super().sanity_check()
        if self.definition.dimension <= 0:
            report.add_error("invalid_dimension", "Synthetic function dimension must be positive.")
        try:
            bounds = self.spec.search_space.numeric_bounds()
            if bounds.shape[0] != self.definition.dimension:
                report.add_error(
                    "dimension_mismatch",
                    f"Search-space dimension {bounds.shape[0]} does not match definition {self.definition.dimension}.",
                )
            for optimum in self.definition.known_optima:
                if len(optimum) != self.definition.dimension:
                    report.add_error("known_optimum_shape", "Known optimum has the wrong dimensionality.")
                    continue
                inside = all(low <= value <= high for value, (low, high) in zip(optimum, bounds, strict=True))
                if not inside:
                    report.add_warning("optimum_out_of_bounds", f"Known optimum {optimum!r} lies outside the search space.")
        except Exception as exc:  # pragma: no cover - defensive guard
            report.add_error("invalid_numeric_bounds", f"Search-space numeric bounds could not be derived: {exc}")
        return report

    def _distance_to_known_optimum(self, vector: np.ndarray) -> float:
        distances = [float(np.linalg.norm(vector - np.asarray(point, dtype=float))) for point in self.definition.known_optima]
        return min(distances) if distances else math.nan


__all__ = [
    "ArrayObjective",
    "SyntheticFunctionDefinition",
    "SyntheticFunctionTask",
    "SyntheticFunctionTaskConfig",
    "TASK_DESCRIPTION_ROOT",
]
