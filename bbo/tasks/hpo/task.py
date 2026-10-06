"""Offline scikit-learn tasks for the 25 public LLAMBO Bayesmark pairs."""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from ...core import (
    EvaluationResult,
    ObjectiveDirection,
    ObjectiveSpec,
    SanityCheckReport,
    Task,
    TaskDescriptionRef,
    TaskSpec,
    TrialStatus,
    TrialSuggestion,
)
from .catalog import (
    ASSET_ROOT,
    BAYESMARK_SOURCE_COMMIT,
    DATASETS,
    DESCRIPTION_ROOT,
    HPO_TASK_IDS,
    LLAMBO_SOURCE_COMMIT,
    MODELS,
    BayesmarkDatasetDefinition,
    BayesmarkModelDefinition,
    parse_task_id,
    task_id,
)
from .initialization import INITIALIZATION_COUNT, initialization_protocol_metadata


DEFAULT_MAX_EVALUATIONS = 30
CV_SPLITS = 5
_SKLEARN_RUNTIME_LOCK = Lock()
_SKLEARN_RUNTIME_READY = False


@dataclass
class BayesmarkHpoTaskConfig:
    task_name: str
    max_evaluations: int | None = None
    seed: int = 0
    asset_root: Path = ASSET_ROOT
    description_root: Path = DESCRIPTION_ROOT


class BayesmarkHpoTask(Task):
    """One deterministic (dataset, sklearn model) Bayesmark HPO task."""

    def __init__(self, config: BayesmarkHpoTaskConfig):
        self.config = config
        self.dataset_definition, self.model_definition = parse_task_id(config.task_name)
        self._asset_path = Path(config.asset_root) / f"{self.dataset_definition.key}.npz"
        self._asset_sha256 = _sha256_file(self._asset_path)
        self._asset_manifest = _load_asset_manifest(Path(config.asset_root) / "manifest.json")
        self._data = _load_dataset_asset(self._asset_path)
        objective_direction = (
            ObjectiveDirection.MAXIMIZE
            if self.dataset_definition.problem_type == "classification"
            else ObjectiveDirection.MINIMIZE
        )
        description_dir = Path(config.description_root) / config.task_name
        benchmark_protocol = initialization_protocol_metadata(
            self.model_definition,
            seed=int(config.seed),
        )
        self._spec = TaskSpec(
            name=config.task_name,
            search_space=self.model_definition.search_space,
            objectives=(ObjectiveSpec(self.dataset_definition.objective_name, objective_direction),),
            max_evaluations=config.max_evaluations or DEFAULT_MAX_EVALUATIONS,
            description_ref=TaskDescriptionRef.from_directory(config.task_name, description_dir),
            metadata={
                "task_family": "hpo",
                "benchmark": "bayesmark",
                "paper": "LLAMBO",
                "dataset": self.dataset_definition.key,
                "dataset_display_name": self.dataset_definition.display_name,
                "model": self.model_definition.key,
                "model_display_name": self.model_definition.display_name,
                "problem_type": self.dataset_definition.problem_type,
                "dimension": self.model_definition.dimension,
                "parameter_transforms": dict(self.model_definition.transforms),
                "fixed_model_parameters": dict(self.model_definition.fixed_parameters),
                "cv_splits": CV_SPLITS,
                "initial_evaluations": INITIALIZATION_COUNT,
                "optimizer_evaluations": 25,
                "benchmark_protocol": benchmark_protocol,
                "data_split": "published_llambo_80_20_random_state_0",
                "asset_path": str(self._asset_path),
                "asset_sha256": self._asset_sha256,
                "source_pickle_sha256": self._asset_manifest["datasets"][self.dataset_definition.key][
                    "source_pickle_sha256"
                ],
                "bayesmark_source_commit": BAYESMARK_SOURCE_COMMIT,
                "llambo_source_commit": LLAMBO_SOURCE_COMMIT,
                "task_seed": int(config.seed),
            },
        )

    @property
    def spec(self) -> TaskSpec:
        return self._spec

    def evaluate(self, suggestion: TrialSuggestion) -> EvaluationResult:
        start = time.perf_counter()
        candidate = self.spec.search_space.coerce_config(suggestion.config, use_defaults=False)
        preload_sklearn_runtime()
        model = _build_estimator(self.dataset_definition, self.model_definition, candidate)
        try:
            from sklearn.model_selection import cross_val_score

            scoring = "accuracy" if self.dataset_definition.problem_type == "classification" else "neg_mean_squared_error"
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                fold_scores = cross_val_score(
                    model,
                    self._data["train_x"],
                    self._data["train_y"],
                    scoring=scoring,
                    cv=CV_SPLITS,
                    error_score="raise",
                )
            if self.dataset_definition.problem_type == "classification":
                objective_value = float(np.mean(fold_scores))
                metrics = {
                    "cv_accuracy_mean": objective_value,
                    "cv_accuracy_std": float(np.std(fold_scores)),
                }
            else:
                objective_value = float(-np.mean(fold_scores))
                metrics = {
                    "cv_mse_mean": objective_value,
                    "cv_mse_std": float(np.std(-fold_scores)),
                }
        except Exception as exc:  # noqa: BLE001 - task failures are serialized for the harness.
            return EvaluationResult(
                status=TrialStatus.FAILED,
                elapsed_seconds=time.perf_counter() - start,
                error_type=type(exc).__name__,
                error_message=str(exc),
                metadata={
                    "dataset": self.dataset_definition.key,
                    "model": self.model_definition.key,
                    "asset_sha256": self._asset_sha256,
                },
            )

        metrics.update(
            {
                "cv_splits": CV_SPLITS,
                "train_samples": self.dataset_definition.train_samples,
            }
        )
        return EvaluationResult(
            status=TrialStatus.SUCCESS,
            objectives={self.dataset_definition.objective_name: objective_value},
            metrics=metrics,
            elapsed_seconds=time.perf_counter() - start,
            metadata={
                "dataset": self.dataset_definition.key,
                "model": self.model_definition.key,
                "asset_sha256": self._asset_sha256,
                "bayesmark_source_commit": BAYESMARK_SOURCE_COMMIT,
                "llambo_source_commit": LLAMBO_SOURCE_COMMIT,
            },
        )

    def evaluate_final(self, suggestion: TrialSuggestion) -> EvaluationResult:
        """Evaluate the frozen CV incumbent once on the hidden test split."""

        start = time.perf_counter()
        candidate = self.spec.search_space.coerce_config(suggestion.config, use_defaults=False)
        preload_sklearn_runtime()
        model = _build_estimator(self.dataset_definition, self.model_definition, candidate)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                model.fit(self._data["train_x"], self._data["train_y"])
            if self.dataset_definition.problem_type == "classification":
                metrics = {
                    "holdout_accuracy": float(
                        model.score(self._data["test_x"], self._data["test_y"])
                    )
                }
            else:
                from sklearn.metrics import mean_squared_error

                predictions = model.predict(self._data["test_x"])
                metrics = {
                    "holdout_mse": float(
                        mean_squared_error(
                            self._data["test_y"],
                            predictions,
                        )
                    )
                }
        except Exception as exc:  # noqa: BLE001 - serialized final-evaluation failure.
            return EvaluationResult(
                status=TrialStatus.FAILED,
                elapsed_seconds=time.perf_counter() - start,
                error_type=type(exc).__name__,
                error_message=str(exc),
                metadata={
                    "dataset": self.dataset_definition.key,
                    "model": self.model_definition.key,
                    "evaluation_split": "test",
                    "optimizer_feedback": False,
                    "asset_sha256": self._asset_sha256,
                },
            )
        return EvaluationResult(
            status=TrialStatus.SUCCESS,
            metrics=metrics,
            elapsed_seconds=time.perf_counter() - start,
            metadata={
                "dataset": self.dataset_definition.key,
                "model": self.model_definition.key,
                "evaluation_split": "test",
                "optimizer_feedback": False,
                "train_samples": self.dataset_definition.train_samples,
                "test_samples": self.dataset_definition.test_samples,
                "asset_sha256": self._asset_sha256,
                "bayesmark_source_commit": BAYESMARK_SOURCE_COMMIT,
                "llambo_source_commit": LLAMBO_SOURCE_COMMIT,
            },
        )

    def sanity_check(self) -> SanityCheckReport:
        report = super().sanity_check()
        expected = self.dataset_definition
        observed_shapes = {
            "train_x": (expected.train_samples, expected.feature_count),
            "test_x": (expected.test_samples, expected.feature_count),
            "train_y": (expected.train_samples,),
            "test_y": (expected.test_samples,),
        }
        for key, expected_shape in observed_shapes.items():
            if tuple(self._data[key].shape) != expected_shape:
                report.add_error(
                    "dataset_shape_mismatch",
                    f"{key} expected shape {expected_shape}, got {self._data[key].shape}.",
                )
            if not np.all(np.isfinite(self._data[key])):
                report.add_error("nonfinite_dataset", f"{key} contains non-finite values.")
        if set(self.model_definition.transforms) != set(self.spec.search_space.names()):
            report.add_error("transform_mismatch", "Every HPO parameter must define exactly one transform.")
        protocol = self.spec.metadata.get("benchmark_protocol", {})
        initialization = protocol.get("initialization", {}) if isinstance(protocol, dict) else {}
        configurations = initialization.get("configurations", []) if isinstance(initialization, dict) else []
        if len(configurations) != INITIALIZATION_COUNT:
            report.add_error(
                "initialization_count_mismatch",
                f"Expected {INITIALIZATION_COUNT} fixed LLAMBO initial configurations, got {len(configurations)}.",
            )
        for index, candidate in enumerate(configurations):
            try:
                self.spec.search_space.validate_config(candidate)
            except Exception as exc:
                report.add_error(
                    "invalid_initialization_config",
                    f"Fixed initialization config {index} is invalid: {exc}",
                )
        manifest_entry = self._asset_manifest.get("datasets", {}).get(self.dataset_definition.key, {})
        if manifest_entry.get("asset_sha256") != self._asset_sha256:
            report.add_error(
                "asset_checksum_mismatch",
                f"Asset checksum does not match manifest for {self.dataset_definition.key}.",
            )
        if expected.problem_type == "classification":
            train_counts = tuple(int(value) for value in np.unique(self._data["train_y"], return_counts=True)[1])
            test_counts = tuple(int(value) for value in np.unique(self._data["test_y"], return_counts=True)[1])
            if train_counts != expected.class_counts_train:
                report.add_error(
                    "train_class_distribution_mismatch",
                    f"Expected train class counts {expected.class_counts_train}, got {train_counts}.",
                )
            if test_counts != expected.class_counts_test:
                report.add_error(
                    "test_class_distribution_mismatch",
                    f"Expected test class counts {expected.class_counts_test}, got {test_counts}.",
                )
        report.metadata.update(
            {
                "asset_sha256": self._asset_sha256,
                "dataset_shapes": {key: list(value.shape) for key, value in self._data.items()},
                "parameter_transforms": dict(self.model_definition.transforms),
                "benchmark_protocol": protocol,
            }
        )
        return report


def _build_estimator(
    dataset: BayesmarkDatasetDefinition,
    model: BayesmarkModelDefinition,
    candidate: dict[str, Any],
) -> Any:
    classification = dataset.problem_type == "classification"
    kwargs = {**model.fixed_parameters, **candidate}
    if model.key == "random_forest":
        if classification:
            from sklearn.ensemble import RandomForestClassifier as Estimator
        else:
            from sklearn.ensemble import RandomForestRegressor as Estimator
    elif model.key == "decision_tree":
        if classification:
            from sklearn.tree import DecisionTreeClassifier as Estimator
        else:
            from sklearn.tree import DecisionTreeRegressor as Estimator
    elif model.key == "svm":
        if classification:
            from sklearn.svm import SVC as Estimator

            kwargs.update({"probability": True, "random_state": 0})
        else:
            from sklearn.svm import SVR as Estimator
    elif model.key == "mlp_sgd":
        if classification:
            from sklearn.neural_network import MLPClassifier as Estimator
        else:
            from sklearn.neural_network import MLPRegressor as Estimator

            kwargs["activation"] = "tanh"
    elif model.key == "adaboost":
        if classification:
            from sklearn.ensemble import AdaBoostClassifier as Estimator
        else:
            from sklearn.ensemble import AdaBoostRegressor as Estimator
    else:  # pragma: no cover - guarded by the catalog.
        raise ValueError(f"Unknown model: {model.key}")
    return Estimator(**kwargs)


def preload_sklearn_runtime() -> None:
    """Load every HPO sklearn dependency once before threaded evaluation.

    Python's import lock protects individual modules, but sklearn's package-wide
    lazy imports can still expose partially initialized sibling modules when
    different estimators are first imported concurrently.  A single explicit
    warm-up makes the subsequent per-task imports cache hits.
    """

    global _SKLEARN_RUNTIME_READY
    if _SKLEARN_RUNTIME_READY:
        return
    with _SKLEARN_RUNTIME_LOCK:
        if _SKLEARN_RUNTIME_READY:
            return
        from sklearn.ensemble import (  # noqa: F401
            AdaBoostClassifier,
            AdaBoostRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
        )
        from sklearn.metrics import mean_squared_error  # noqa: F401
        from sklearn.model_selection import cross_val_score  # noqa: F401
        from sklearn.neural_network import MLPClassifier, MLPRegressor  # noqa: F401
        from sklearn.svm import SVC, SVR  # noqa: F401
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # noqa: F401

        _SKLEARN_RUNTIME_READY = True


def _load_dataset_asset(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Bayesmark dataset asset is missing: {path}. "
            "Regenerate it with tools/generate_bayesmark_hpo_assets.py."
        )
    with np.load(path, allow_pickle=False) as loaded:
        required = ("train_x", "train_y", "test_x", "test_y")
        missing = [key for key in required if key not in loaded]
        if missing:
            raise ValueError(f"Bayesmark dataset asset {path} is missing arrays: {missing}")
        return {key: np.asarray(loaded[key]) for key in required}


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_asset_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Bayesmark asset manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), dict):
        raise ValueError(f"Invalid Bayesmark asset manifest: {path}")
    return payload


def create_hpo_task(
    name: str,
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    asset_root: Path = ASSET_ROOT,
    description_root: Path = DESCRIPTION_ROOT,
) -> BayesmarkHpoTask:
    return BayesmarkHpoTask(
        BayesmarkHpoTaskConfig(
            task_name=name,
            max_evaluations=max_evaluations,
            seed=seed,
            asset_root=asset_root,
            description_root=description_root,
        )
    )


__all__ = [
    "BayesmarkHpoTask",
    "BayesmarkHpoTaskConfig",
    "CV_SPLITS",
    "DEFAULT_MAX_EVALUATIONS",
    "HPO_TASK_IDS",
    "create_hpo_task",
    "preload_sklearn_runtime",
    "task_id",
]
