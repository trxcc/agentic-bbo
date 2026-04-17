"""Sphere synthetic benchmark definition."""

from __future__ import annotations

import numpy as np

from ...core import FloatParam, SearchSpace
from .base import SyntheticFunctionDefinition, TASK_DESCRIPTION_ROOT


def sphere_objective(vector: np.ndarray) -> float:
    return float(np.dot(vector, vector))


SPHERE_DEFINITION = SyntheticFunctionDefinition(
    key="sphere_demo",
    display_name="Shifted Sphere (4D)",
    description="A convex synthetic benchmark used for smoke tests and resume validation.",
    search_space=SearchSpace(
        [
            FloatParam("x1", low=-5.0, high=5.0, default=2.0),
            FloatParam("x2", low=-5.0, high=5.0, default=-1.5),
            FloatParam("x3", low=-5.0, high=5.0, default=1.0),
            FloatParam("x4", low=-5.0, high=5.0, default=-0.5),
        ]
    ),
    objective=sphere_objective,
    known_optimum=0.0,
    known_optima=((0.0, 0.0, 0.0, 0.0),),
    description_dir=TASK_DESCRIPTION_ROOT / "sphere_demo",
    default_max_evaluations=35,
    plot_resolution=120,
)


__all__ = ["SPHERE_DEFINITION", "sphere_objective"]
