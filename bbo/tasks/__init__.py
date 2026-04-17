"""Task packages and registries."""

from .registry import SYNTHETIC_PROBLEM_REGISTRY, TASK_FAMILIES, create_demo_task, get_synthetic_problem
from .synthetic import BRANIN_DEFINITION, SPHERE_DEFINITION, SyntheticFunctionDefinition, SyntheticFunctionTask, SyntheticFunctionTaskConfig

__all__ = [
    "BRANIN_DEFINITION",
    "SPHERE_DEFINITION",
    "SYNTHETIC_PROBLEM_REGISTRY",
    "TASK_FAMILIES",
    "SyntheticFunctionDefinition",
    "SyntheticFunctionTask",
    "SyntheticFunctionTaskConfig",
    "create_demo_task",
    "get_synthetic_problem",
]
