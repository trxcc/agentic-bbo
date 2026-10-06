"""Published LLAMBO/Bayesmark HPO task definitions.

The paper is the authority for the optimizer-facing search spaces.  The
released LLAMBO code accidentally comments out two MLP parameters, while the
paper and the pinned Bayesmark source both define the eight-dimensional MLP
space.  We intentionally keep the paper definition here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ...core import FloatParam, IntParam, SearchSpace


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = Path(__file__).resolve().parent / "assets"
DESCRIPTION_ROOT = PACKAGE_ROOT / "task_descriptions"

BAYESMARK_SOURCE_COMMIT = "8c420e935718f0d6867153b781e58943ecaf2338"
LLAMBO_SOURCE_COMMIT = "196fe237f60a3d3a2fa53cbf8f474ec20a01dd57"

TransformName = Literal["linear", "log", "logit"]


@dataclass(frozen=True)
class BayesmarkDatasetDefinition:
    key: str
    display_name: str
    problem_type: Literal["classification", "regression"]
    total_samples: int
    train_samples: int
    test_samples: int
    feature_count: int
    class_counts_train: tuple[int, ...] = ()
    class_counts_test: tuple[int, ...] = ()

    @property
    def objective_name(self) -> str:
        return "accuracy" if self.problem_type == "classification" else "mse"


@dataclass(frozen=True)
class BayesmarkModelDefinition:
    key: str
    display_name: str
    search_space: SearchSpace
    transforms: dict[str, TransformName]
    fixed_parameters: dict[str, object]

    @property
    def dimension(self) -> int:
        return len(self.search_space)


def _log_midpoint(low: float, high: float) -> float:
    return math.sqrt(low * high)


def _logit(value: float) -> float:
    return math.log(value / (1.0 - value))


def _inverse_logit(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _logit_midpoint(low: float, high: float) -> float:
    return _inverse_logit((_logit(low) + _logit(high)) / 2.0)


TREE_PARAMETERS = (
    IntParam("max_depth", low=1, high=15, default=8),
    FloatParam("min_samples_split", low=0.01, high=0.99, default=_logit_midpoint(0.01, 0.99)),
    FloatParam("min_samples_leaf", low=0.01, high=0.49, default=_logit_midpoint(0.01, 0.49)),
    FloatParam(
        "min_weight_fraction_leaf",
        low=0.01,
        high=0.49,
        default=_logit_midpoint(0.01, 0.49),
    ),
    FloatParam("max_features", low=0.01, high=0.99, default=_logit_midpoint(0.01, 0.99)),
    FloatParam("min_impurity_decrease", low=0.0, high=0.5, default=0.25),
)

TREE_TRANSFORMS: dict[str, TransformName] = {
    "max_depth": "linear",
    "min_samples_split": "logit",
    "min_samples_leaf": "logit",
    "min_weight_fraction_leaf": "logit",
    "max_features": "logit",
    "min_impurity_decrease": "linear",
}


DATASETS: dict[str, BayesmarkDatasetDefinition] = {
    "breast": BayesmarkDatasetDefinition(
        key="breast",
        display_name="Breast Cancer Wisconsin",
        problem_type="classification",
        total_samples=569,
        train_samples=455,
        test_samples=114,
        feature_count=30,
        class_counts_train=(165, 290),
        class_counts_test=(47, 67),
    ),
    "wine": BayesmarkDatasetDefinition(
        key="wine",
        display_name="Wine Recognition",
        problem_type="classification",
        total_samples=178,
        train_samples=142,
        test_samples=36,
        feature_count=13,
        class_counts_train=(45, 55, 42),
        class_counts_test=(14, 16, 6),
    ),
    "iris": BayesmarkDatasetDefinition(
        key="iris",
        display_name="Iris",
        problem_type="classification",
        total_samples=150,
        train_samples=120,
        test_samples=30,
        feature_count=4,
        class_counts_train=(39, 37, 44),
        class_counts_test=(11, 13, 6),
    ),
    "digits": BayesmarkDatasetDefinition(
        key="digits",
        display_name="Optical Digits",
        problem_type="classification",
        total_samples=1797,
        train_samples=1437,
        test_samples=360,
        feature_count=64,
        class_counts_train=(151, 147, 141, 154, 151, 142, 137, 140, 135, 139),
        class_counts_test=(27, 35, 36, 29, 30, 40, 44, 39, 39, 41),
    ),
    "diabetes": BayesmarkDatasetDefinition(
        key="diabetes",
        display_name="Diabetes Progression",
        problem_type="regression",
        total_samples=442,
        train_samples=353,
        test_samples=89,
        feature_count=10,
    ),
}


MODELS: dict[str, BayesmarkModelDefinition] = {
    "random_forest": BayesmarkModelDefinition(
        key="random_forest",
        display_name="RandomForest",
        search_space=SearchSpace(TREE_PARAMETERS),
        transforms=dict(TREE_TRANSFORMS),
        fixed_parameters={"n_estimators": 10, "max_leaf_nodes": None, "random_state": 0},
    ),
    "svm": BayesmarkModelDefinition(
        key="svm",
        display_name="SVM",
        search_space=SearchSpace(
            (
                FloatParam("C", low=1.0, high=1e3, log=True, default=_log_midpoint(1.0, 1e3)),
                FloatParam("gamma", low=1e-4, high=1e-3, log=True, default=_log_midpoint(1e-4, 1e-3)),
                FloatParam("tol", low=1e-5, high=1e-1, log=True, default=_log_midpoint(1e-5, 1e-1)),
            )
        ),
        transforms={"C": "log", "gamma": "log", "tol": "log"},
        fixed_parameters={"kernel": "rbf"},
    ),
    "decision_tree": BayesmarkModelDefinition(
        key="decision_tree",
        display_name="DecisionTree",
        search_space=SearchSpace(TREE_PARAMETERS),
        transforms=dict(TREE_TRANSFORMS),
        fixed_parameters={"max_leaf_nodes": None, "random_state": 0},
    ),
    "mlp_sgd": BayesmarkModelDefinition(
        key="mlp_sgd",
        display_name="MLP-SGD",
        search_space=SearchSpace(
            (
                IntParam("hidden_layer_sizes", low=50, high=200, default=125),
                FloatParam("alpha", low=1e-5, high=1e1, log=True, default=_log_midpoint(1e-5, 1e1)),
                IntParam("batch_size", low=10, high=250, default=130),
                FloatParam(
                    "learning_rate_init",
                    low=1e-5,
                    high=1e-1,
                    log=True,
                    default=_log_midpoint(1e-5, 1e-1),
                ),
                FloatParam("power_t", low=0.1, high=0.9, default=_logit_midpoint(0.1, 0.9)),
                FloatParam("tol", low=1e-5, high=1e-1, log=True, default=_log_midpoint(1e-5, 1e-1)),
                FloatParam("momentum", low=0.001, high=0.999, default=_logit_midpoint(0.001, 0.999)),
                FloatParam(
                    "validation_fraction",
                    low=0.1,
                    high=0.9,
                    default=_logit_midpoint(0.1, 0.9),
                ),
            )
        ),
        transforms={
            "hidden_layer_sizes": "linear",
            "alpha": "log",
            "batch_size": "linear",
            "learning_rate_init": "log",
            "power_t": "logit",
            "tol": "log",
            "momentum": "logit",
            "validation_fraction": "logit",
        },
        fixed_parameters={
            "solver": "sgd",
            "early_stopping": True,
            "max_iter": 40,
            "learning_rate": "invscaling",
            "nesterovs_momentum": True,
            "random_state": 0,
        },
    ),
    "adaboost": BayesmarkModelDefinition(
        key="adaboost",
        display_name="AdaBoost",
        search_space=SearchSpace(
            (
                IntParam("n_estimators", low=10, high=100, default=55),
                FloatParam(
                    "learning_rate",
                    low=1e-4,
                    high=1e1,
                    log=True,
                    default=_log_midpoint(1e-4, 1e1),
                ),
            )
        ),
        transforms={"n_estimators": "linear", "learning_rate": "log"},
        fixed_parameters={"random_state": 0},
    ),
}


HPO_TASK_IDS: tuple[str, ...] = tuple(
    f"hpo_bayesmark_{dataset_key}_{model_key}"
    for dataset_key in DATASETS
    for model_key in MODELS
)


def task_id(dataset_key: str, model_key: str) -> str:
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown Bayesmark dataset `{dataset_key}`.")
    if model_key not in MODELS:
        raise ValueError(f"Unknown Bayesmark model `{model_key}`.")
    return f"hpo_bayesmark_{dataset_key}_{model_key}"


def parse_task_id(name: str) -> tuple[BayesmarkDatasetDefinition, BayesmarkModelDefinition]:
    prefix = "hpo_bayesmark_"
    if not name.startswith(prefix):
        raise ValueError(f"Not a Bayesmark HPO task id: {name!r}")
    suffix = name[len(prefix) :]
    for dataset_key, dataset in DATASETS.items():
        marker = f"{dataset_key}_"
        if suffix.startswith(marker):
            model_key = suffix[len(marker) :]
            if model_key in MODELS:
                return dataset, MODELS[model_key]
    available = ", ".join(HPO_TASK_IDS)
    raise ValueError(f"Unknown Bayesmark HPO task `{name}`. Available: {available}")


__all__ = [
    "ASSET_ROOT",
    "BAYESMARK_SOURCE_COMMIT",
    "BayesmarkDatasetDefinition",
    "BayesmarkModelDefinition",
    "DATASETS",
    "DESCRIPTION_ROOT",
    "HPO_TASK_IDS",
    "LLAMBO_SOURCE_COMMIT",
    "MODELS",
    "TransformName",
    "parse_task_id",
    "task_id",
]
