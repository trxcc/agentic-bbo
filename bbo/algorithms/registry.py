"""Algorithm registry grouped by algorithm family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..core.algo import Algorithm
from .agentic import LlamboAlgorithm, OproAlgorithm
from .model_based import OptunaTpeAlgorithm
from .traditional import PyCmaAlgorithm, RandomSearchAlgorithm


@dataclass(frozen=True)
class AlgorithmSpec:
    """Metadata for one algorithm entrypoint."""

    factory: Callable[..., Algorithm]
    description: str
    family: str
    numeric_only: bool = False


ALGORITHM_REGISTRY: dict[str, AlgorithmSpec] = {
    "random_search": AlgorithmSpec(
        factory=RandomSearchAlgorithm,
        description="Uniform random search over the declared search space.",
        family="traditional",
    ),
    "random": AlgorithmSpec(
        factory=RandomSearchAlgorithm,
        description="Alias for random_search.",
        family="traditional",
    ),
    "pycma": AlgorithmSpec(
        factory=PyCmaAlgorithm,
        description="CMA-ES via the external `pycma` package.",
        family="traditional",
        numeric_only=True,
    ),
    "cma_es": AlgorithmSpec(
        factory=PyCmaAlgorithm,
        description="Alias for pycma.",
        family="traditional",
        numeric_only=True,
    ),
    "optuna_tpe": AlgorithmSpec(
        factory=OptunaTpeAlgorithm,
        description="Optuna TPE via the optional `optuna` package.",
        family="model_based",
    ),
    "llambo": AlgorithmSpec(
        factory=LlamboAlgorithm,
        description="LLAMBO-style prompt optimizer with pluggable chat backends and an offline heuristic mode.",
        family="agentic",
    ),
    "opro": AlgorithmSpec(
        factory=OproAlgorithm,
        description="OPRO-style prompt optimizer over prior configuration/objective pairs.",
        family="agentic",
    ),
}


def create_algorithm(name: str, **kwargs: Any) -> Algorithm:
    if name not in ALGORITHM_REGISTRY:
        available = ", ".join(sorted(ALGORITHM_REGISTRY))
        raise ValueError(f"Unknown algorithm `{name}`. Available: {available}")
    return ALGORITHM_REGISTRY[name].factory(**kwargs)


def algorithms_by_family() -> dict[str, dict[str, AlgorithmSpec]]:
    grouped: dict[str, dict[str, AlgorithmSpec]] = {}
    for name, spec in ALGORITHM_REGISTRY.items():
        grouped.setdefault(spec.family, {})[name] = spec
    return grouped


__all__ = ["ALGORITHM_REGISTRY", "AlgorithmSpec", "algorithms_by_family", "create_algorithm"]
