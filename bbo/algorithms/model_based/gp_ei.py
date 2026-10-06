"""BoTorch Gaussian-process expected-improvement baseline."""

from __future__ import annotations

import hashlib
import json
import random
import warnings
from typing import Any

import numpy as np

from ...core import (
    CategoricalParam,
    ExternalOptimizerAdapter,
    FloatParam,
    IntParam,
    ObjectiveDirection,
    SearchSpace,
    TrialObservation,
    TrialSuggestion,
)
from ...core.conversion import (
    ContinuousSearchSpaceConverter,
    UnitCubeSearchSpaceConverter,
    build_continuous_converter,
)
from ..benchmark_protocol import (
    FixedInitializationProtocol,
    resolve_fixed_initialization,
    resolve_model_candidate_budget,
)


def require_botorch() -> dict[str, Any]:
    try:
        import torch
        from botorch.acquisition import (
            ExpectedImprovement,
            LogExpectedImprovement,
            ProbabilityOfImprovement,
            UpperConfidenceBound,
            qLogNoisyExpectedImprovement,
        )
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.optim import optimize_acqf
        from botorch.models.transforms.outcome import Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise ImportError(
            "`gp_ei` requires BoTorch, GPyTorch, and torch. Install them with "
            "`uv sync --extra molecular` in this repo."
        ) from exc
    return {
        "torch": torch,
        "SingleTaskGP": SingleTaskGP,
        "Standardize": Standardize,
        "ExactMarginalLogLikelihood": ExactMarginalLogLikelihood,
        "ExpectedImprovement": ExpectedImprovement,
        "LogExpectedImprovement": LogExpectedImprovement,
        "ProbabilityOfImprovement": ProbabilityOfImprovement,
        "UpperConfidenceBound": UpperConfidenceBound,
        "qLogNoisyExpectedImprovement": qLogNoisyExpectedImprovement,
        "fit_gpytorch_mll": fit_gpytorch_mll,
        "optimize_acqf": optimize_acqf,
    }


class GpEiAlgorithm(ExternalOptimizerAdapter):
    """BoTorch SingleTaskGP optimizer over one-hot standardized features."""

    def __init__(
        self,
        *,
        pool_size: int | None = None,
        startup_trials: int = 2,
        xi: float = 0.0,
        alpha: float = 1e-6,
        n_restarts_optimizer: int = 0,
        acqf_num_restarts: int = 10,
        max_acqf_attempts: int = 8,
        candidate_attempt_multiplier: int = 20,
        acquisition: str = "ei",
        acquisition_beta: float = 2.0,
        device: str = "cpu",
        kernel: str = "default",
        input_scaling: str = "train_mean_std",
        parameter_transforms: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        if pool_size is not None and pool_size <= 0:
            raise ValueError("pool_size/raw_samples must be positive.")
        if startup_trials < 0:
            raise ValueError("startup_trials must be non-negative.")
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if n_restarts_optimizer < 0:
            raise ValueError("n_restarts_optimizer must be non-negative.")
        if acqf_num_restarts <= 0:
            raise ValueError("acqf_num_restarts must be positive.")
        if max_acqf_attempts <= 0:
            raise ValueError("max_acqf_attempts must be positive.")
        if candidate_attempt_multiplier <= 0:
            raise ValueError("candidate_attempt_multiplier must be positive.")
        if not str(device).strip():
            raise ValueError("device must be a non-empty torch device string.")
        normalized_acquisition = str(acquisition).strip().lower().replace("-", "_")
        normalized_acquisition = {
            "expectedimprovement": "ei",
            "logexpectedimprovement": "logei",
            "noisylogei": "noisy_logei",
            "noisy_logexpectedimprovement": "noisy_logei",
            "noisylogexpectedimprovement": "noisy_logei",
            "probabilityofimprovement": "pi",
            "upperconfidencebound": "ucb",
        }.get(normalized_acquisition, normalized_acquisition)
        if normalized_acquisition not in {
            "ei",
            "logei",
            "noisy_logei",
            "pi",
            "ucb",
            "sobol",
        }:
            raise ValueError(
                "acquisition must be one of: ei, logei, noisy_logei, pi, ucb, sobol."
            )
        if float(acquisition_beta) <= 0:
            raise ValueError("acquisition_beta must be positive.")
        self.pool_size = None if pool_size is None else int(pool_size)
        self.startup_trials = int(startup_trials)
        self.xi = float(xi)
        self.alpha = float(alpha)
        self.n_restarts_optimizer = int(n_restarts_optimizer)
        self.acqf_num_restarts = int(acqf_num_restarts)
        self.max_acqf_attempts = int(max_acqf_attempts)
        self.candidate_attempt_multiplier = int(candidate_attempt_multiplier)
        self.acquisition = normalized_acquisition
        self.acquisition_beta = float(acquisition_beta)
        self.device = str(device).strip()
        if kernel not in {"default", "matern52"}:
            raise ValueError("kernel must be default or matern52.")
        if input_scaling not in {"train_mean_std", "unit_cube"}:
            raise ValueError("input_scaling must be train_mean_std or unit_cube.")
        self.kernel = kernel
        self.input_scaling = input_scaling
        # Opt-in geometry; preserve the historical numeric-identity path by default.
        self.parameter_transforms = None if parameter_transforms is None else dict(parameter_transforms)
        self._seed = 0
        self._converter: ContinuousSearchSpaceConverter | None = None
        self._history: list[TrialObservation] = []
        self._seen_config_ids: set[str] = set()
        self._fixed_initialization: FixedInitializationProtocol | None = None
        self._candidate_budget = 0
        self._candidate_budget_policy = "uninitialized"
        self._candidate_search_space: SearchSpace | None = None

    @property
    def name(self) -> str:
        return "gp_ei"

    @property
    def candidate_budget(self) -> int:
        """Effective acquisition raw-sample budget after setup."""

        return self._require_candidate_budget()

    @property
    def candidate_budget_policy(self) -> str:
        return self._candidate_budget_policy

    def setup(self, task_spec, seed: int = 0, **kwargs: Any) -> None:
        if len(task_spec.objectives) != 1:
            raise ValueError("GpEiAlgorithm currently supports exactly one objective.")
        self.bind_task_spec(task_spec)
        self._converter = (
            build_continuous_converter(task_spec.search_space, strategy="onehot")
            if self.parameter_transforms is None
            else UnitCubeSearchSpaceConverter(task_spec.search_space, transforms=self.parameter_transforms)
        )
        self._seed = int(seed)
        self._fixed_initialization = resolve_fixed_initialization(task_spec, seed=self._seed)
        self._candidate_budget, self._candidate_budget_policy = resolve_model_candidate_budget(
            task_spec,
            target="gp_ei.raw_samples",
            configured=self.pool_size,
        )
        self._history = []
        self._seen_config_ids = set()
        self._candidate_search_space = task_spec.search_space

    def set_candidate_search_space(self, search_space: SearchSpace) -> None:
        """Restrict candidate generation while retaining the full training history."""

        original = self.require_search_space()
        _validate_candidate_search_space(original, search_space)
        self._candidate_search_space = search_space

    def ask(self) -> TrialSuggestion:
        successful = self._successful_history()
        if self._fixed_initialization is not None and len(self._history) < len(
            self._fixed_initialization.configurations
        ):
            suggestion = self._fixed_initialization.suggestion(len(self._history), algorithm=self.name)
            suggestion.metadata.update(
                {
                    "gp_ei_phase": "benchmark_initialization",
                    "gp_ei_backend": "botorch",
                    "gp_ei_training_points": len(successful),
                }
            )
            self._seen_config_ids.add(_config_identity(suggestion.config))
            return suggestion
        if self._fixed_initialization is not None and len(successful) < 2:
            return self._random_suggestion(reason="benchmark_initialization_recovery")
        if self._fixed_initialization is None and len(successful) < self.startup_trials:
            return self._random_suggestion(reason="startup")

        try:
            suggestion = (
                self._sobol_suggestion(successful)
                if self.acquisition == "sobol"
                else self._botorch_ei_suggestion(successful)
            )
        except Exception as exc:  # noqa: BLE001
            suggestion = self._random_suggestion(reason=f"gp_ei_fallback:{type(exc).__name__}")
        self._seen_config_ids.add(_config_identity(suggestion.config))
        return suggestion

    def tell(self, observation: TrialObservation) -> None:
        self._seen_config_ids.add(_config_identity(observation.suggestion.config))
        self._history.append(observation)
        self.update_best_incumbent(observation)

    def replay(self, history: list[TrialObservation]) -> None:
        self._history = []
        self._seen_config_ids = set()
        self._best = None
        for observation in history:
            self.tell(observation)

    def evaluate_virtual_configs(
        self,
        configs: list[dict[str, Any]],
        *,
        include_acquisition: bool = True,
    ) -> list[dict[str, Any]]:
        """Predict and optionally score configs with this exact baseline model."""

        successful = self._successful_history()
        if len(successful) < 2:
            return [
                {
                    "config": self.require_search_space().coerce_config(
                        config, use_defaults=False
                    ),
                    "mean": None,
                    "std": None,
                    "acquisition_score": None if include_acquisition else None,
                    "data_sufficient": False,
                }
                for config in configs
            ]

        deps = require_botorch()
        torch = deps["torch"]
        device = torch.device(self.device)
        converter = self._require_converter()
        primary_name = self._require_primary_name()
        x_train = np.vstack(
            [converter.encode_vector(item.suggestion.config) for item in successful]
        )
        y_values = np.asarray(
            [float(item.objectives[primary_name]) for item in successful],
            dtype=float,
        )
        if self._primary_direction == ObjectiveDirection.MINIMIZE:
            y_values = -y_values
        x_train_scaled, mean, scale = self._scale_training_features(x_train)
        train_x = torch.as_tensor(x_train_scaled, dtype=torch.double, device=device)
        train_y = torch.as_tensor(
            y_values.reshape(-1, 1), dtype=torch.double, device=device
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = self._make_model(deps, train_x, train_y)
            mll = deps["ExactMarginalLogLikelihood"](model.likelihood, model)
            deps["fit_gpytorch_mll"](mll)
        acquisition = None
        if self.acquisition != "sobol":
            acquisition, _ = self._build_acquisition(
                deps,
                model,
                y_values,
                train_x=train_x,
            )

        normalized = [
            self.require_search_space().coerce_config(config, use_defaults=False)
            for config in configs
        ]
        x_test = np.vstack([converter.encode_vector(config) for config in normalized])
        x_test_scaled = (x_test - mean) / scale
        test_x = torch.as_tensor(x_test_scaled, dtype=torch.double, device=device)
        with torch.no_grad():
            posterior = model.posterior(test_x)
            utility_means = posterior.mean.detach().cpu().numpy().reshape(-1)
            stds = (
                posterior.variance.clamp_min(0.0).sqrt().detach().cpu().numpy().reshape(-1)
            )
            acquisition_values = (
                acquisition(test_x.unsqueeze(-2)).detach().cpu().numpy().reshape(-1)
                if include_acquisition and acquisition is not None
                else [None] * len(normalized)
            )
        predictions: list[dict[str, Any]] = []
        for config, utility_mean, std, acquisition_value in zip(
            normalized, utility_means, stds, acquisition_values
        ):
            objective_mean = (
                -float(utility_mean)
                if self._primary_direction == ObjectiveDirection.MINIMIZE
                else float(utility_mean)
            )
            predictions.append(
                {
                    "config": config,
                    "mean": objective_mean,
                    "std": float(std),
                    "acquisition_score": (
                        None
                        if acquisition_value is None
                        else float(acquisition_value)
                    ),
                    "data_sufficient": True,
                }
            )
        return predictions

    def diagnose(self, *, max_cv_folds: int = 5) -> dict[str, Any]:
        """Return read-only fit, calibration, and sensitivity diagnostics."""
        successful = self._successful_history()
        converter = self._require_converter()
        count = len(successful)
        result: dict[str, Any] = {
            "model": "SingleTaskGP", "backend": "botorch",
            "training_points": count,
            "encoded_dimensions": len(converter.feature_names),
            "feature_names": list(converter.feature_names),
            "data_sufficient": count >= 3, "cv_r2": None, "cv_folds": 0,
            "feature_sensitivities": None, "hyperparameters": None,
            "fit_status": "insufficient_data" if count < 3 else "pending",
        }
        if count < 3:
            return result
        try:
            x_values, y_values = self._diagnostic_training_arrays(successful)
            deps = require_botorch()
            model, _, _ = self._fit_diagnostic_model(deps, x_values, y_values)
            result["hyperparameters"] = _model_hyperparameters(model)
            result["feature_sensitivities"] = _feature_sensitivities(model, converter.feature_names)
            result["cv_r2"], result["cv_folds"] = self._cross_validated_r2(
                deps, x_values, y_values, max_folds=max_cv_folds
            )
            result["fit_status"] = "ok"
        except Exception as exc:  # noqa: BLE001 - diagnostics must degrade safely.
            result.update(fit_status="fit_failed", error={"type": type(exc).__name__, "message": str(exc)})
        return result

    def cross_validate_landscape(self, *, local_bounds: dict[str, list[float | int]] | None = None, max_folds: int = 5) -> dict[str, Any]:
        """Compare global and local SingleTaskGP models on held-out observations."""
        successful = self._successful_history()
        converter = self._require_converter()
        minimum = max(8, len(converter.feature_names) + 3)
        result: dict[str, Any] = {
            "backend": "botorch", "model": "SingleTaskGP",
            "sample_count": len(successful), "minimum_sample_count": minimum,
            "folds": 0, "global_gp_cv_r2": None,
            "global_gp_ranking_accuracy": None, "local_scope_count": 0,
            "global_gp_on_local_r2": None, "local_gp_r2": None,
            "local_gp_gain_over_global": None,
            "status": "insufficient_data" if len(successful) < minimum else "pending",
            "limitations": [],
        }
        if len(successful) < minimum:
            return result
        try:
            deps = require_botorch()
            x_values, y_values = self._diagnostic_training_arrays(successful)
            fold_count = min(max(int(max_folds), 2), len(successful))
            rng = np.random.default_rng(self._seed)
            all_indices = np.arange(len(successful))
            folds = np.array_split(rng.permutation(all_indices), fold_count)
            predictions = np.empty_like(y_values, dtype=float)
            for held_out in folds:
                train = np.setdiff1d(all_indices, held_out)
                model, mean, scale = self._fit_diagnostic_model(deps, x_values[train], y_values[train])
                predictions[held_out] = self._diagnostic_posterior_means(deps, model, x_values[held_out], mean, scale)
            result.update(status="ok", folds=fold_count,
                          global_gp_cv_r2=_r2_score(y_values, predictions),
                          global_gp_ranking_accuracy=_ranking_accuracy(y_values, predictions))
            bounds = dict(local_bounds or {})
            if bounds:
                local_indices = np.asarray([
                    index for index, observation in enumerate(successful)
                    if _config_within_numeric_bounds(observation.suggestion.config, bounds)
                ], dtype=int)
                result["local_scope_count"] = int(len(local_indices))
                local_minimum = max(6, min(len(converter.feature_names) + 2, 12))
                if len(local_indices) >= local_minimum:
                    local_folds = np.array_split(rng.permutation(local_indices), min(fold_count, len(local_indices)))
                    global_predictions = np.empty(len(local_indices), dtype=float)
                    local_predictions = np.empty(len(local_indices), dtype=float)
                    positions = {int(index): position for position, index in enumerate(local_indices)}
                    for held_out in local_folds:
                        global_train = np.setdiff1d(all_indices, held_out)
                        local_train = np.setdiff1d(local_indices, held_out)
                        target = [positions[int(index)] for index in held_out]
                        global_model, global_mean, global_scale = self._fit_diagnostic_model(deps, x_values[global_train], y_values[global_train])
                        local_model, local_mean, local_scale = self._fit_diagnostic_model(deps, x_values[local_train], y_values[local_train])
                        global_predictions[target] = self._diagnostic_posterior_means(deps, global_model, x_values[held_out], global_mean, global_scale)
                        local_predictions[target] = self._diagnostic_posterior_means(deps, local_model, x_values[held_out], local_mean, local_scale)
                    local_y = y_values[local_indices]
                    global_r2 = _r2_score(local_y, global_predictions)
                    local_r2 = _r2_score(local_y, local_predictions)
                    result.update(global_gp_on_local_r2=global_r2, local_gp_r2=local_r2,
                                  local_gp_gain_over_global=None if global_r2 is None or local_r2 is None else local_r2 - global_r2)
                else:
                    result["limitations"].append(f"Need at least {local_minimum} observations inside the candidate subspace.")
        except Exception as exc:  # noqa: BLE001
            result.update(status="fit_failed", error={"type": type(exc).__name__, "message": str(exc)})
        return result

    def _diagnostic_posterior_means(self, deps: dict[str, Any], model: Any, x_values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
        torch = deps["torch"]
        test_x = torch.as_tensor((x_values - mean) / scale, dtype=torch.double, device=torch.device(self.device))
        with torch.no_grad():
            return model.posterior(test_x).mean.detach().cpu().numpy().reshape(-1)

    def _diagnostic_training_arrays(self, successful: list[TrialObservation]) -> tuple[np.ndarray, np.ndarray]:
        converter = self._require_converter()
        primary_name = self._require_primary_name()
        x_values = np.vstack([converter.encode_vector(item.suggestion.config) for item in successful])
        y_values = np.asarray([float(item.objectives[primary_name]) for item in successful], dtype=float)
        if self._primary_direction == ObjectiveDirection.MINIMIZE:
            y_values = -y_values
        return x_values, y_values

    def _fit_diagnostic_model(self, deps: dict[str, Any], x_values: np.ndarray, y_values: np.ndarray) -> tuple[Any, np.ndarray, np.ndarray]:
        torch = deps["torch"]
        x_scaled, mean, scale = self._scale_training_features(x_values)
        device = torch.device(self.device)
        train_x = torch.as_tensor(x_scaled, dtype=torch.double, device=device)
        train_y = torch.as_tensor(y_values.reshape(-1, 1), dtype=torch.double, device=device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = self._make_model(deps, train_x, train_y)
            mll = deps["ExactMarginalLogLikelihood"](model.likelihood, model)
            deps["fit_gpytorch_mll"](mll)
        return model, mean, scale

    def _scale_training_features(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.input_scaling == "train_mean_std":
            return _standardize_training_features(values)
        # Always use the original domain, including for CV and temporary regions.
        bounds = _candidate_feature_bounds(self._require_converter(), self.require_search_space())
        offset = bounds[:, 0]
        scale = bounds[:, 1] - offset
        scale = np.where(scale == 0, 1.0, scale)
        return (values - offset) / scale, offset, scale

    def _make_model(self, deps: dict[str, Any], train_x: Any, train_y: Any) -> Any:
        options = {}
        if self.kernel == "matern52":
            from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
            options["covar_module"] = get_covar_module_with_dim_scaled_prior(
                ard_num_dims=train_x.shape[-1], use_rbf_kernel=False
            )
        return deps["SingleTaskGP"](
            train_x, train_y, outcome_transform=deps["Standardize"](m=1), **options
        )

    def _cross_validated_r2(self, deps: dict[str, Any], x_values: np.ndarray, y_values: np.ndarray, *, max_folds: int) -> tuple[float | None, int]:
        fold_count = min(max(int(max_folds), 2), len(y_values))
        folds = np.array_split(np.arange(len(y_values)), fold_count)
        predictions = np.empty_like(y_values, dtype=float)
        torch = deps["torch"]
        for held_out in folds:
            train_indices = np.setdiff1d(np.arange(len(y_values)), held_out)
            model, mean, scale = self._fit_diagnostic_model(deps, x_values[train_indices], y_values[train_indices])
            test_x = torch.as_tensor((x_values[held_out] - mean) / scale, dtype=torch.double, device=torch.device(self.device))
            with torch.no_grad():
                predictions[held_out] = model.posterior(test_x).mean.detach().cpu().numpy().reshape(-1)
        denominator = float(np.sum((y_values - np.mean(y_values)) ** 2))
        if denominator <= 1e-12:
            return None, fold_count
        return 1.0 - float(np.sum((y_values - predictions) ** 2)) / denominator, fold_count

    def _build_acquisition(
        self,
        deps: dict[str, Any],
        model: Any,
        y_values: np.ndarray,
        *,
        train_x: Any | None = None,
    ) -> tuple[Any, str]:
        if self.acquisition == "ucb":
            return (
                deps["UpperConfidenceBound"](
                    model=model,
                    beta=self.acquisition_beta,
                    maximize=True,
                ),
                "UpperConfidenceBound",
            )
        if self.acquisition == "noisy_logei":
            if train_x is None:
                raise ValueError("noisy_logei requires the encoded training inputs.")
            from botorch.sampling.normal import SobolQMCNormalSampler
            return (
                deps["qLogNoisyExpectedImprovement"](
                    model=model,
                    X_baseline=train_x,
                    prune_baseline=True,
                    sampler=SobolQMCNormalSampler(
                        sample_shape=deps["torch"].Size([128]),
                        seed=self._stable_int("noisy_logei_mc", len(self._history)),
                    ),
                ),
                "qLogNoisyExpectedImprovement",
            )
        if self.acquisition == "pi":
            return (
                deps["ProbabilityOfImprovement"](
                    model=model,
                    best_f=float(np.max(y_values) + self.xi),
                    maximize=True,
                ),
                "ProbabilityOfImprovement",
            )
        if self.acquisition == "sobol":
            raise ValueError("Sobol sampling does not define a GP acquisition score.")
        acquisition_class = (
            deps["LogExpectedImprovement"]
            if self.acquisition == "logei"
            else deps["ExpectedImprovement"]
        )
        return (
            acquisition_class(
                model=model,
                best_f=float(np.max(y_values) + self.xi),
                maximize=True,
            ),
            (
                "LogExpectedImprovement"
                if self.acquisition == "logei"
                else "ExpectedImprovement"
            ),
        )

    def _botorch_ei_suggestion(self, successful: list[TrialObservation]) -> TrialSuggestion:
        deps = require_botorch()
        torch = deps["torch"]
        SingleTaskGP = deps["SingleTaskGP"]
        Standardize = deps["Standardize"]
        ExactMarginalLogLikelihood = deps["ExactMarginalLogLikelihood"]
        fit_gpytorch_mll = deps["fit_gpytorch_mll"]
        optimize_acqf = deps["optimize_acqf"]
        device = torch.device(self.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(f"GP-EI requested {self.device}, but CUDA is unavailable.")
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"GP-EI requested {self.device}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
                )
        converter = self._require_converter()
        search_space = self.require_search_space()
        primary_name = self._require_primary_name()

        x_train = np.vstack([converter.encode_vector(item.suggestion.config) for item in successful])
        y_values = np.asarray([float(item.objectives[primary_name]) for item in successful], dtype=float)
        if self._primary_direction == ObjectiveDirection.MINIMIZE:
            y_values = -y_values
        x_train_scaled, mean, scale = self._scale_training_features(x_train)
        candidate_space = self._require_candidate_search_space()
        bounds = _standardized_candidate_bounds(
            converter,
            candidate_space,
            mean=mean,
            scale=scale,
        )

        train_x = torch.as_tensor(x_train_scaled, dtype=torch.double, device=device)
        train_y = torch.as_tensor(y_values.reshape(-1, 1), dtype=torch.double, device=device)
        bounds_t = torch.as_tensor(bounds.T, dtype=torch.double, device=device)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = self._make_model(deps, train_x, train_y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

        acquisition, acquisition_label = self._build_acquisition(
            deps,
            model,
            y_values,
            train_x=train_x,
        )
        fixed_features = {
            index: float(interval[0])
            for index, interval in enumerate(bounds)
            if float(interval[0]) == float(interval[1])
        }
        last_candidate = None
        last_acq_value = None
        for attempt_index in range(self.max_acqf_attempts):
            torch.manual_seed(self._stable_int("acqf", len(self._history), attempt_index))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(self._stable_int("acqf", len(self._history), attempt_index))
            if len(fixed_features) == len(bounds):
                candidate_t = torch.as_tensor(
                    bounds[:, 0].reshape(1, -1),
                    dtype=torch.double,
                    device=device,
                )
                acq_value_t = acquisition(candidate_t.unsqueeze(0))
            else:
                candidate_t, acq_value_t = optimize_acqf(
                    acq_function=acquisition,
                    bounds=bounds_t,
                    q=1,
                    num_restarts=self.acqf_num_restarts,
                    raw_samples=self._require_candidate_budget(),
                    options={"batch_limit": 5, "maxiter": 200},
                    fixed_features=fixed_features or None,
                )
            vector = candidate_t.detach().cpu().numpy().reshape(-1)
            config = search_space.coerce_config(
                converter.decode_vector(vector * scale + mean, clip=True),
                use_defaults=False,
            )
            config = _repair_candidate_roundoff(config, candidate_space)
            candidate_space.validate_config(config)
            last_candidate = config
            last_acq_value = float(acq_value_t.detach().cpu().reshape(-1)[0])
            if _config_identity(config) not in self._seen_config_ids:
                return TrialSuggestion(
                    config=config,
                    metadata={
                        "gp_ei_phase": "acquisition",
                        "gp_ei_backend": "botorch",
                        "gp_ei_model": "SingleTaskGP",
                        "gp_ei_acquisition": acquisition_label,
                        "gp_ei_acquisition_optimizer": "optimize_acqf",
                        "gp_ei_training_points": len(successful),
                        "gp_ei_raw_samples": self._require_candidate_budget(),
                        "gp_ei_candidate_budget_policy": self._candidate_budget_policy,
                        "gp_ei_num_restarts": self.acqf_num_restarts,
                        "gp_ei_best_acquisition": last_acq_value,
                        "gp_ei_xi": self.xi,
                        "gp_ei_beta": self.acquisition_beta if self.acquisition == "ucb" else None,
                        "gp_ei_feature_encoder": converter.strategy_name,
                        "gp_ei_feature_standardization": self.input_scaling,
                        "gp_ei_kernel": self.kernel,
                        "gp_ei_device": str(device),
                        "gp_ei_candidate_subspace": _search_space_summary(
                            candidate_space
                        ),
                    },
                )
        if last_candidate is not None:
            return TrialSuggestion(
                config=last_candidate,
                metadata={
                    "gp_ei_phase": "acquisition_duplicate_fallback",
                    "gp_ei_backend": "botorch",
                    "gp_ei_model": "SingleTaskGP",
                    "gp_ei_acquisition": acquisition_label,
                    "gp_ei_acquisition_optimizer": "optimize_acqf",
                    "gp_ei_training_points": len(successful),
                    "gp_ei_raw_samples": self._require_candidate_budget(),
                    "gp_ei_candidate_budget_policy": self._candidate_budget_policy,
                    "gp_ei_num_restarts": self.acqf_num_restarts,
                    "gp_ei_best_acquisition": last_acq_value,
                    "gp_ei_xi": self.xi,
                    "gp_ei_beta": self.acquisition_beta if self.acquisition == "ucb" else None,
                    "gp_ei_feature_encoder": converter.strategy_name,
                    "gp_ei_feature_standardization": "train_mean_std",
                    "gp_ei_device": str(device),
                    "gp_ei_duplicate_attempts": self.max_acqf_attempts,
                    "gp_ei_candidate_subspace": _search_space_summary(
                        candidate_space
                    ),
                },
            )
        raise RuntimeError("BoTorch optimize_acqf did not return a candidate.")

    def _sobol_suggestion(
        self, successful: list[TrialObservation]
    ) -> TrialSuggestion:
        deps = require_botorch()
        torch = deps["torch"]
        converter = self._require_converter()
        candidate_space = self._require_candidate_search_space()
        raw_bounds = _candidate_feature_bounds(converter, candidate_space)
        engine = torch.quasirandom.SobolEngine(
            dimension=len(raw_bounds),
            scramble=True,
            seed=self._stable_int("sobol", len(self._history)),
        )
        last_config: dict[str, Any] | None = None
        attempts = max(self.candidate_attempt_multiplier, 1)
        for attempt_index in range(attempts):
            unit = engine.draw(1).detach().cpu().numpy().reshape(-1)
            vector = raw_bounds[:, 0] + unit * (
                raw_bounds[:, 1] - raw_bounds[:, 0]
            )
            config = self.require_search_space().coerce_config(
                converter.decode_vector(vector, clip=True),
                use_defaults=False,
            )
            candidate_space.validate_config(config)
            last_config = config
            if _config_identity(config) not in self._seen_config_ids:
                return TrialSuggestion(
                    config=config,
                    metadata={
                        "gp_ei_phase": "acquisition",
                        "gp_ei_backend": "botorch",
                        "gp_ei_model": "SingleTaskGP",
                        "gp_ei_acquisition": "Sobol",
                        "gp_ei_acquisition_optimizer": (
                            "torch.quasirandom.SobolEngine"
                        ),
                        "gp_ei_training_points": len(successful),
                        "gp_ei_sobol_attempt": attempt_index,
                        "gp_ei_candidate_subspace": _search_space_summary(
                            candidate_space
                        ),
                    },
                )
        if last_config is None:  # pragma: no cover - attempts is always positive.
            raise RuntimeError("Sobol candidate generation produced no candidate.")
        return TrialSuggestion(
            config=last_config,
            metadata={
                "gp_ei_phase": "acquisition_duplicate_fallback",
                "gp_ei_backend": "botorch",
                "gp_ei_model": "SingleTaskGP",
                "gp_ei_acquisition": "Sobol",
                "gp_ei_acquisition_optimizer": "torch.quasirandom.SobolEngine",
                "gp_ei_training_points": len(successful),
                "gp_ei_duplicate_attempts": attempts,
                "gp_ei_candidate_subspace": _search_space_summary(
                    candidate_space
                ),
            },
        )

    def _random_suggestion(self, *, reason: str) -> TrialSuggestion:
        config = self._sample_random_unseen_config()
        self._seen_config_ids.add(_config_identity(config))
        return TrialSuggestion(
            config=config,
            metadata={
                "gp_ei_phase": "random",
                "gp_ei_random_reason": reason,
                "gp_ei_training_points": len(self._successful_history()),
                "gp_ei_backend": "botorch",
                "gp_ei_feature_encoder": self._require_converter().strategy_name,
                "gp_ei_feature_standardization": "train_mean_std",
                "gp_ei_device": self.device,
            },
        )

    def _sample_random_unseen_config(self) -> dict[str, Any]:
        search_space = self._require_candidate_search_space()
        rng = random.Random(self._stable_int("random", len(self._history)))
        attempts = max(self.candidate_attempt_multiplier, 1)
        for _ in range(attempts):
            config = search_space.sample(rng)
            if _config_identity(config) not in self._seen_config_ids:
                return search_space.coerce_config(config, use_defaults=False)
        return search_space.coerce_config(search_space.sample(rng), use_defaults=False)

    def _successful_history(self) -> list[TrialObservation]:
        primary_name = self._require_primary_name()
        return [
            item
            for item in self._history
            if item.success and primary_name in item.objectives
        ]

    def _require_converter(self) -> ContinuousSearchSpaceConverter:
        if self._converter is None:
            raise RuntimeError("GpEiAlgorithm.setup() must be called before ask/tell.")
        return self._converter

    def _require_candidate_search_space(self) -> SearchSpace:
        if self._candidate_search_space is None:
            raise RuntimeError("GpEiAlgorithm.setup() must be called before ask/tell.")
        return self._candidate_search_space

    def _require_primary_name(self) -> str:
        if self._primary_name is None:
            raise RuntimeError("GpEiAlgorithm.setup() must be called before ask/tell.")
        return self._primary_name

    def _require_candidate_budget(self) -> int:
        if self._candidate_budget <= 0:
            raise RuntimeError("GpEiAlgorithm.setup() must be called before candidate generation.")
        return self._candidate_budget

    def _stable_int(self, *parts: object) -> int:
        text = ":".join(str(part) for part in (self.name, self._seed, *parts))
        return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**31 - 1)


def _repair_candidate_roundoff(config: dict[str, Any], space: SearchSpace) -> dict[str, Any]:
    """Snap machine-precision float overshoots; reject substantive violations."""
    repaired = dict(config)
    for param in space:
        if not isinstance(param, FloatParam):
            continue
        value = float(repaired[param.name])
        tolerance = 16 * np.finfo(float).eps * max(1.0, abs(param.low), abs(param.high))
        if value < param.low and param.low - value <= tolerance:
            repaired[param.name] = float(param.low)
        elif value > param.high and value - param.high <= tolerance:
            repaired[param.name] = float(param.high)
    space.validate_config(repaired)
    return repaired


def _standardize_training_features(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (x_train - mean) / scale, mean, scale


def _standardized_candidate_bounds(
    converter: ContinuousSearchSpaceConverter,
    candidate_space: SearchSpace,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    bounds = _candidate_feature_bounds(converter, candidate_space)
    scaled = (bounds - mean[:, None]) / scale[:, None]
    return np.asarray([(low, high) if low <= high else (high, low) for low, high in scaled], dtype=float)


def _candidate_feature_bounds(
    converter: ContinuousSearchSpaceConverter,
    candidate_space: SearchSpace,
) -> np.ndarray:
    original_space = converter.search_space
    _validate_candidate_search_space(original_space, candidate_space)
    if isinstance(converter, UnitCubeSearchSpaceConverter):
        # Candidate boxes use the SAME original-domain warp as all training data.
        # Fitting a new converter on the box would silently move historical points.
        low = {p.name: p.low for p in candidate_space}
        high = {p.name: p.high for p in candidate_space}
        return np.column_stack((converter.encode_vector(low), converter.encode_vector(high)))
    candidate_by_name = {param.name: param for param in candidate_space}
    bounds: list[tuple[float, float]] = []
    for original in original_space:
        candidate = candidate_by_name[original.name]
        if isinstance(original, FloatParam):
            assert isinstance(candidate, FloatParam)
            bounds.append((float(candidate.low), float(candidate.high)))
        elif isinstance(original, IntParam):
            assert isinstance(candidate, IntParam)
            bounds.append((float(candidate.low), float(candidate.high)))
        elif isinstance(original, CategoricalParam):
            assert isinstance(candidate, CategoricalParam)
            allowed = set(candidate.choices)
            for choice in original.choices:
                if choice not in allowed:
                    bounds.append((0.0, 0.0))
                elif len(allowed) == 1:
                    bounds.append((1.0, 1.0))
                elif len(allowed) < len(original.choices):
                    bounds.append((1e-6, 1.0))
                else:
                    bounds.append((0.0, 1.0))
        else:  # pragma: no cover - the converter rejects open strings earlier.
            raise TypeError(
                f"Unsupported candidate parameter type {type(original).__name__}."
            )
    return np.asarray(bounds, dtype=float)


def _validate_candidate_search_space(
    original_space: SearchSpace,
    candidate_space: SearchSpace,
) -> None:
    if original_space.names() != candidate_space.names():
        raise ValueError("Candidate search space must preserve parameter names and order.")
    for original, candidate in zip(original_space, candidate_space, strict=True):
        if type(original) is not type(candidate):
            raise TypeError(
                f"Candidate parameter {candidate.name!r} changed type from "
                f"{type(original).__name__} to {type(candidate).__name__}."
            )
        if isinstance(original, (FloatParam, IntParam)):
            if candidate.low < original.low or candidate.high > original.high:
                raise ValueError(
                    f"Candidate bounds for {original.name!r} exceed the original domain."
                )
        elif isinstance(original, CategoricalParam):
            if not set(candidate.choices).issubset(set(original.choices)):
                raise ValueError(
                    f"Candidate choices for {original.name!r} exceed the original domain."
                )


def _search_space_summary(space: SearchSpace) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for param in space:
        if isinstance(param, (FloatParam, IntParam)):
            summary[param.name] = [param.low, param.high]
        elif isinstance(param, CategoricalParam):
            summary[param.name] = list(param.choices)
    return summary


def _config_identity(config: dict[str, Any]) -> str:
    return json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return None if denominator <= 1e-12 else 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denominator


def _ranking_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    pairs = [(float(y_true[i] - y_true[j]), float(y_pred[i] - y_pred[j])) for i in range(len(y_true)) for j in range(i + 1, len(y_true)) if abs(float(y_true[i] - y_true[j])) > 1e-12]
    return None if not pairs else sum(truth * pred > 0 for truth, pred in pairs) / len(pairs)


def _config_within_numeric_bounds(config: dict[str, Any], bounds: dict[str, list[float | int]]) -> bool:
    return all(name in config and len(interval) == 2 and float(interval[0]) <= float(config[name]) <= float(interval[1]) for name, interval in bounds.items())


def _model_hyperparameters(model: Any) -> dict[str, Any]:
    def serializable(tensor: Any) -> float | list[float]:
        values = tensor.detach().cpu().numpy().reshape(-1)
        return float(values[0]) if values.size == 1 else [float(item) for item in values]

    kernel = model.covar_module
    while not hasattr(kernel, "lengthscale") and hasattr(kernel, "base_kernel"):
        kernel = kernel.base_kernel
    result: dict[str, Any] = {"noise": serializable(model.likelihood.noise)}
    if hasattr(kernel, "lengthscale"):
        result["lengthscale"] = serializable(kernel.lengthscale)
    if hasattr(model.covar_module, "outputscale"):
        result["outputscale"] = serializable(model.covar_module.outputscale)
    return result


def _feature_sensitivities(model: Any, feature_names: tuple[str, ...]) -> dict[str, float] | None:
    kernel = model.covar_module
    while not hasattr(kernel, "lengthscale") and hasattr(kernel, "base_kernel"):
        kernel = kernel.base_kernel
    lengthscale = getattr(kernel, "lengthscale", None)
    if lengthscale is None:
        return None
    values = lengthscale.detach().cpu().numpy().reshape(-1)
    if values.size == 1:
        values = np.repeat(values, len(feature_names))
    if values.size != len(feature_names):
        return None
    inverse = 1.0 / np.maximum(values.astype(float), 1e-12)
    total = float(np.sum(inverse))
    normalized = inverse / total if total > 0 else np.zeros_like(inverse)
    return {name: float(value) for name, value in zip(feature_names, normalized, strict=True)}


__all__ = ["GpEiAlgorithm"]
