"""Paper algorithms, with aliases only for established public names."""
from dataclasses import dataclass
from typing import Callable
from .traditional import RandomSearchAlgorithm, SobolSearchAlgorithm, PyCmaAlgorithm, LocalPerturbationAlgorithm
from .model_based import GpEiAlgorithm, BotorchTurboAlgorithm, OptunaTpeAlgorithm, GitBoAlgorithm
from .molecular import GraphGAAlgorithm, GraphGPBOAlgorithm
from .agentic.raw_agentic_bbo import create_raw_agentic_bbo


@dataclass(frozen=True)
class AlgorithmSpec:
    factory: Callable
    description: str
    family: str
    numeric_only: bool = False
    categorical_to_continuous: str | None = None


ALGORITHM_REGISTRY = {
    "random": AlgorithmSpec(RandomSearchAlgorithm, "Uniform random search", "traditional"),
    "sobol": AlgorithmSpec(SobolSearchAlgorithm, "Scrambled Sobol search", "traditional", True),
    "local_perturbation": AlgorithmSpec(LocalPerturbationAlgorithm, "Incumbent-centered local search", "traditional"),
    "cma_es": AlgorithmSpec(PyCmaAlgorithm, "CMA-ES", "traditional", True, "onehot"),
    "gp_ei": AlgorithmSpec(GpEiAlgorithm, "Fixed GP with EI", "model_based", False, "onehot"),
    "optuna_tpe": AlgorithmSpec(OptunaTpeAlgorithm, "TPE", "model_based"),
    "turbo": AlgorithmSpec(BotorchTurboAlgorithm, "TuRBO-1", "model_based", True),
    "git_bo": AlgorithmSpec(GitBoAlgorithm, "Gradient-informed TabPFN BO", "model_based", True),
    "graph_ga": AlgorithmSpec(GraphGAAlgorithm, "Graph GA", "molecular"),
    "gpbo": AlgorithmSpec(GraphGPBOAlgorithm, "Tanimoto GPBO", "molecular"),
    "raw_agentic_bbo": AlgorithmSpec(create_raw_agentic_bbo, "Persistent Docker workspace agent", "agentic"),
}
for alias, name in {"random_search":"random", "sobol_search":"sobol", "pycma":"cma_es",
    "gp_bo":"gp_ei", "gpei":"gp_ei", "botorch_turbo":"turbo", "tpe":"optuna_tpe",
    "graph_gpbo":"gpbo", "agentic":"raw_agentic_bbo"}.items():
    ALGORITHM_REGISTRY[alias] = ALGORITHM_REGISTRY[name]


def create_algorithm(name, **kwargs):
    return ALGORITHM_REGISTRY[name].factory(**kwargs)


def algorithms_by_family():
    return {family: {k:v for k,v in ALGORITHM_REGISTRY.items() if v.family == family}
            for family in sorted({v.family for v in ALGORITHM_REGISTRY.values()})}
