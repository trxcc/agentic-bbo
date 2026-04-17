"""Synthetic benchmark task families."""

from .base import SyntheticFunctionDefinition, SyntheticFunctionTask, SyntheticFunctionTaskConfig
from .branin import BRANIN_DEFINITION
from .sphere import SPHERE_DEFINITION

__all__ = [
    "BRANIN_DEFINITION",
    "SPHERE_DEFINITION",
    "SyntheticFunctionDefinition",
    "SyntheticFunctionTask",
    "SyntheticFunctionTaskConfig",
]
