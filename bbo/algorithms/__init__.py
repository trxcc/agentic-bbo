"""Optimization algorithms and shared numerical defaults."""
from .registry import ALGORITHM_REGISTRY, AlgorithmSpec, create_algorithm, algorithms_by_family
from .baseline_factory import (COMPARABLE_BASELINE_BACKENDS, COMPARABLE_BASELINE_DEFAULTS,
    create_comparable_baseline, comparable_baseline_kwargs, normalize_comparable_backend)
from .traditional import RandomSearchAlgorithm, SobolSearchAlgorithm, PyCmaAlgorithm
from .model_based import GpEiAlgorithm, GitBoAlgorithm, BotorchTurboAlgorithm, OptunaTpeAlgorithm
from .agentic import GeneralAgentBBOAlgorithm, CodexBBOAlgorithm
