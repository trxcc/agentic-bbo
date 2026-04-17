"""Branin synthetic benchmark definition."""

from __future__ import annotations

import math

import numpy as np

from ...core import FloatParam, SearchSpace
from .base import SyntheticFunctionDefinition, TASK_DESCRIPTION_ROOT


def branin_objective(vector: np.ndarray) -> float:
    x1, x2 = vector
    a = 1.0
    b = 5.1 / (4.0 * math.pi**2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)
    return float(a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1.0 - t) * math.cos(x1) + s)


BRANIN_DEFINITION = SyntheticFunctionDefinition(
    key="branin_demo",
    display_name="Branin-Hoo (2D)",
    description="A two-dimensional multimodal benchmark suitable for visualization and optimizer comparisons.",
    search_space=SearchSpace(
        [
            FloatParam("x1", low=-5.0, high=10.0, default=2.5),
            FloatParam("x2", low=0.0, high=15.0, default=7.5),
        ]
    ),
    objective=branin_objective,
    known_optimum=0.39788735772973816,
    known_optima=(
        (-math.pi, 12.275),
        (math.pi, 2.275),
        (9.42478, 2.475),
    ),
    description_dir=TASK_DESCRIPTION_ROOT / "branin_demo",
    default_max_evaluations=45,
    plot_resolution=220,
)


__all__ = ["BRANIN_DEFINITION", "branin_objective"]
