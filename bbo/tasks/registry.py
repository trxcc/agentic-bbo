"""Task registries and convenience constructors."""

from __future__ import annotations

from .synthetic import BRANIN_DEFINITION, SPHERE_DEFINITION, SyntheticFunctionDefinition, SyntheticFunctionTask, SyntheticFunctionTaskConfig


SYNTHETIC_PROBLEM_REGISTRY: dict[str, SyntheticFunctionDefinition] = {
    BRANIN_DEFINITION.key: BRANIN_DEFINITION,
    SPHERE_DEFINITION.key: SPHERE_DEFINITION,
}

TASK_FAMILIES: dict[str, tuple[str, ...]] = {
    "synthetic": tuple(sorted(SYNTHETIC_PROBLEM_REGISTRY)),
}


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
) -> SyntheticFunctionTask:
    config = SyntheticFunctionTaskConfig(
        problem=problem,
        max_evaluations=max_evaluations,
        seed=seed,
        noise_std=noise_std,
    )
    return SyntheticFunctionTask(config=config, definition=get_synthetic_problem(problem))


__all__ = [
    "SYNTHETIC_PROBLEM_REGISTRY",
    "TASK_FAMILIES",
    "create_demo_task",
    "get_synthetic_problem",
]
