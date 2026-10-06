"""The paper's 77 tasks: 24 BBOB, 25 HPO, 6 DBTune, 12 placement, 10 molecular."""
from .synthetic import BBOB_PROBLEM_REGISTRY, BBOB_TASK_IDS, create_bbob_task
from .hpo import HPO_TASK_IDS, create_hpo_task
from .scientific import SCIENTIFIC_TASK_REGISTRY, create_scientific_task
from .dbtune.http_surrogate_specs import DBTUNE_SURROGATE_SERVICE_TASK_IDS, HTTP_SURROGATE_TASK_IDS
from .dbtune.http_surrogate_task import create_http_surrogate_knob_task
from .dbtune.catalog import SURROGATE_BENCHMARKS
from .bboplace import create_bboplace_task, default_bboplace_definition

BBOPLACE_BENCHMARKS = ("adaptec1", "adaptec3", "adaptec4", "bigblue1", "bigblue2", "bigblue4",
                      "superblue1", "superblue4", "superblue5", "superblue7", "superblue16", "superblue18")
BBOPLACE_TASK_IDS = tuple(f"bboplace_{name}_n32" for name in BBOPLACE_BENCHMARKS)
TASK_FAMILIES = dict(BBOB=BBOB_TASK_IDS, HPO=HPO_TASK_IDS,
    DBTune=DBTUNE_SURROGATE_SERVICE_TASK_IDS, BBOPlace=BBOPLACE_TASK_IDS,
    GuacaMol=tuple(SCIENTIFIC_TASK_REGISTRY))
TASK_REGISTRY = {name: family for family, names in TASK_FAMILIES.items() for name in names}
ALL_TASK_NAMES = tuple(sorted(TASK_REGISTRY))
ALL_DEMO_TASK_NAMES = ALL_TASK_NAMES
SYNTHETIC_PROBLEM_REGISTRY = BBOB_PROBLEM_REGISTRY
SURROGATE_TASK_IDS = tuple(SURROGATE_BENCHMARKS)


def create_task(name, *, max_evaluations=None, seed=2, noise_std=0.0, **kwargs):
    if noise_std:
        raise ValueError("Paper tasks do not add observation noise")
    family = TASK_REGISTRY.get(name)
    if max_evaluations is None:
        max_evaluations = {'BBOB':120, 'HPO':30, 'DBTune':250, 'BBOPlace':250, 'GuacaMol':250}.get(family)
    common = dict(max_evaluations=max_evaluations, seed=seed, **kwargs)
    if family == "BBOB":
        return create_bbob_task(name, **common)
    if family == "HPO":
        return create_hpo_task(name, **common)
    if family == "GuacaMol":
        return create_scientific_task(name, **common)
    if family == "DBTune":
        return create_http_surrogate_knob_task(name, **common)
    if family == "BBOPlace":
        base_url = common.pop("base_url", None)
        definition = default_bboplace_definition(key=name, benchmark=name[9:-4],
            n_macro=32, base_url=base_url, default_max_evaluations=max_evaluations or 250)
        return create_bboplace_task(definition=definition, **common)
    raise ValueError(f"Unknown task {name!r}; available: {', '.join(ALL_TASK_NAMES)}")


def create_demo_task(problem="bbob_f01_d10", **kwargs):
    return create_task(problem, **kwargs)


def get_synthetic_problem(name):
    return BBOB_PROBLEM_REGISTRY[name]


def get_scientific_task(name):
    return SCIENTIFIC_TASK_REGISTRY[name]
