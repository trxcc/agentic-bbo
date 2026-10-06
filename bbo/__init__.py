"""Agentic black-box optimization benchmark."""
from . import core

__all__ = ['core', 'create_task', 'create_algorithm', 'ALL_TASK_NAMES', 'ALGORITHM_REGISTRY']


def __getattr__(name):
    if name in {'create_task', 'ALL_TASK_NAMES'}:
        from . import tasks
        return getattr(tasks, name)
    if name in {'create_algorithm', 'ALGORITHM_REGISTRY'}:
        from . import algorithms
        return getattr(algorithms, name)
    raise AttributeError(name)
