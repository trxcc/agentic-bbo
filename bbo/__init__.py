"""Standalone benchmark core for agentic black-box optimization."""

from __future__ import annotations

from . import core

# NOTE:
# `bbo` 顶层 import 不应强制依赖所有可选算法依赖（例如 `cma`）。
# 因此这里采用惰性导入，避免在只装部分 extra（比如 surrogate）时 import 失败。

__all__ = ["core", "run_demo_suite", "run_single_experiment"]


def __getattr__(name: str):
    if name in {"run_demo_suite", "run_single_experiment"}:
        from .run import run_demo_suite, run_single_experiment

        return {"run_demo_suite": run_demo_suite, "run_single_experiment": run_single_experiment}[name]
    if name in {
        "ALGORITHM_REGISTRY",
        "AlgorithmSpec",
        "PyCmaAlgorithm",
        "RandomSearchAlgorithm",
        "algorithms_by_family",
        "create_algorithm",
    }:
        from . import algorithms as _algorithms

        return getattr(_algorithms, name)
    if name in {
        "BRANIN_DEFINITION",
        "SPHERE_DEFINITION",
        "SYNTHETIC_PROBLEM_REGISTRY",
        "TASK_FAMILIES",
        "SyntheticFunctionDefinition",
        "SyntheticFunctionTask",
        "SyntheticFunctionTaskConfig",
        "create_demo_task",
        "get_synthetic_problem",
    }:
        from . import tasks as _tasks

        return getattr(_tasks, name)
    raise AttributeError(f"module 'bbo' has no attribute {name!r}")
