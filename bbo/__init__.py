"""Standalone benchmark core for agentic black-box optimization."""

from . import core
from .algorithms import (
    ALGORITHM_REGISTRY,
    AlgorithmSpec,
    LlamboAlgorithm,
    OproAlgorithm,
    PyCmaAlgorithm,
    RandomSearchAlgorithm,
    algorithms_by_family,
    create_algorithm,
)
from .tasks import (
    BRANIN_DEFINITION,
    SPHERE_DEFINITION,
    SYNTHETIC_PROBLEM_REGISTRY,
    TASK_FAMILIES,
    SyntheticFunctionDefinition,
    SyntheticFunctionTask,
    SyntheticFunctionTaskConfig,
    create_demo_task,
    get_synthetic_problem,
)

__all__ = [
    "ALGORITHM_REGISTRY",
    "AlgorithmSpec",
    "BRANIN_DEFINITION",
    "LlamboAlgorithm",
    "OproAlgorithm",
    "PyCmaAlgorithm",
    "RandomSearchAlgorithm",
    "SPHERE_DEFINITION",
    "SYNTHETIC_PROBLEM_REGISTRY",
    "TASK_FAMILIES",
    "SyntheticFunctionDefinition",
    "SyntheticFunctionTask",
    "SyntheticFunctionTaskConfig",
    "algorithms_by_family",
    "core",
    "create_algorithm",
    "create_demo_task",
    "get_synthetic_problem",
    "run_demo_suite",
    "run_single_experiment",
]


def __getattr__(name: str):
    if name in {"run_demo_suite", "run_single_experiment"}:
        from .run import run_demo_suite, run_single_experiment

        return {"run_demo_suite": run_demo_suite, "run_single_experiment": run_single_experiment}[name]
    raise AttributeError(f"module 'bbo' has no attribute {name!r}")
