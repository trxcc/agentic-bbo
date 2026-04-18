"""Task registries and convenience constructors."""

from __future__ import annotations

from ..core import Task
from .bboplace import BBOPLACE_TASK_KEY, create_bboplace_task
from .synthetic import BRANIN_DEFINITION, SPHERE_DEFINITION, SyntheticFunctionDefinition, SyntheticFunctionTask, SyntheticFunctionTaskConfig


SYNTHETIC_PROBLEM_REGISTRY: dict[str, SyntheticFunctionDefinition] = {
    BRANIN_DEFINITION.key: BRANIN_DEFINITION,
    SPHERE_DEFINITION.key: SPHERE_DEFINITION,
}

TASK_FAMILIES: dict[str, tuple[str, ...]] = {
    "synthetic": tuple(sorted(SYNTHETIC_PROBLEM_REGISTRY)),
    "bboplace": (BBOPLACE_TASK_KEY,),
}

ALL_DEMO_TASK_NAMES: tuple[str, ...] = tuple(
    sorted([*SYNTHETIC_PROBLEM_REGISTRY.keys(), BBOPLACE_TASK_KEY]),
)


def get_synthetic_problem(name: str) -> SyntheticFunctionDefinition:
    if name not in SYNTHETIC_PROBLEM_REGISTRY:
        available = ", ".join(sorted(SYNTHETIC_PROBLEM_REGISTRY))
        raise ValueError(f"Unknown synthetic problem `{name}`. Available: {available}")
    return SYNTHETIC_PROBLEM_REGISTRY[name]


def create_demo_task(
    problem: str = "branin_demo",
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    noise_std: float = 0.0,
) -> Task:
    if problem == BBOPLACE_TASK_KEY:
        return create_bboplace_task(max_evaluations=max_evaluations, seed=seed)
    config = SyntheticFunctionTaskConfig(
        problem=problem,
        max_evaluations=max_evaluations,
        seed=seed,
        noise_std=noise_std,
    )
    return SyntheticFunctionTask(config=config, definition=get_synthetic_problem(problem))


__all__ = [
    "ALL_DEMO_TASK_NAMES",
    "BBOPLACE_TASK_KEY",
    "SYNTHETIC_PROBLEM_REGISTRY",
    "TASK_FAMILIES",
    "create_demo_task",
    "get_synthetic_problem",
]
