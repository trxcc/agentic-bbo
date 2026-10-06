"""Task definitions and the paper task registry."""
from .registry import (ALL_TASK_NAMES, ALL_DEMO_TASK_NAMES, TASK_REGISTRY, TASK_FAMILIES,
    BBOPLACE_TASK_IDS, DBTUNE_SURROGATE_SERVICE_TASK_IDS, HTTP_SURROGATE_TASK_IDS,
    HPO_TASK_IDS, SCIENTIFIC_TASK_REGISTRY, SURROGATE_TASK_IDS, SYNTHETIC_PROBLEM_REGISTRY,
    create_task, create_demo_task, get_synthetic_problem)
from .synthetic import *
from .bboplace import *
from .hpo import *
from .scientific import *
