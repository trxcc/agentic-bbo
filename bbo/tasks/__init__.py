"""Task packages and registries."""

from .bboplace import (
    BBOPLACE_DEFAULT_DEFINITION,
    BBOPLACE_TASK_KEY,
    BBOPlaceDefinition,
    BBOPlaceTask,
    BBOPlaceTaskConfig,
    create_bboplace_task,
    default_bboplace_definition,
)
from .registry import ALL_DEMO_TASK_NAMES, SYNTHETIC_PROBLEM_REGISTRY, TASK_FAMILIES, create_demo_task, get_synthetic_problem
from .synthetic import BRANIN_DEFINITION, SPHERE_DEFINITION, SyntheticFunctionDefinition, SyntheticFunctionTask, SyntheticFunctionTaskConfig

__all__ = [
    "ALL_DEMO_TASK_NAMES",
    "BBOPLACE_DEFAULT_DEFINITION",
    "BBOPLACE_TASK_KEY",
    "BRANIN_DEFINITION",
    "SPHERE_DEFINITION",
    "BBOPlaceDefinition",
    "BBOPlaceTask",
    "BBOPlaceTaskConfig",
    "SYNTHETIC_PROBLEM_REGISTRY",
    "TASK_FAMILIES",
    "SyntheticFunctionDefinition",
    "SyntheticFunctionTask",
    "SyntheticFunctionTaskConfig",
    "create_bboplace_task",
    "create_demo_task",
    "default_bboplace_definition",
    "get_synthetic_problem",
]
