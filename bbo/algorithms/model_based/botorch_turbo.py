"""Ask/tell adapter for the pinned BoTorch official TuRBO-1 tutorial."""

from __future__ import annotations

import hashlib
import json
import warnings
from typing import Any

import numpy as np

from ...core import (
    ExternalOptimizerAdapter,
    ObjectiveDirection,
    TrialObservation,
    TrialSuggestion,
    UnitCubeSearchSpaceConverter,
)
from ..benchmark_protocol import (
    FixedInitializationProtocol,
    resolve_fixed_initialization,
    resolve_model_candidate_budget,
)


def require_botorch_turbo() -> dict[str, Any]:
    try:
        import gpytorch
        import torch
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from gpytorch.constraints import Interval
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.likelihoods import GaussianLikelihood
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from torch.quasirandom import SobolEngine

        from ._vendor import botorch_turbo as vendor
    except ImportError as exc:  # pragma: no cover - optional dependency path.
        raise ImportError(
            "`botorch_turbo` requires BoTorch, GPyTorch, and torch. "
            "Install them with `uv sync --extra hpo`."
        ) from exc
    return {
        "gpytorch": gpytorch,
        "torch": torch,
        "fit_gpytorch_mll": fit_gpytorch_mll,
        "SingleTaskGP": SingleTaskGP,
        "Interval": Interval,
        "MaternKernel": MaternKernel,
        "ScaleKernel": ScaleKernel,
        "GaussianLikelihood": GaussianLikelihood,
        "ExactMarginalLogLikelihood": ExactMarginalLogLikelihood,
        "SobolEngine": SobolEngine,
        "vendor": vendor,
    }


class BotorchTurboAlgorithm(ExternalOptimizerAdapter):
    """TuRBO-1 with the official BoTorch tutorial core and a v3 adapter."""

    def __init__(
        self,
        *,
        startup_trials: int = 5,
        n_candidates: int | None = None,
        max_duplicate_attempts: int = 8,
        max_cholesky_size: float = float("inf"),
        device: str = "cpu",
        success_tolerance: int = 10,
    ) -> None:
        super().__init__()
        if startup_trials < 2:
            raise ValueError("startup_trials must be at least 2 for GP standardization.")
        if n_candidates is not None and n_candidates <= 0:
            raise ValueError("n_candidates must be positive when provided.")
        if max_duplicate_attempts <= 0:
            raise ValueError("max_duplicate_attempts must be positive.")
        if not str(device).strip():
            raise ValueError("device must be a non-empty torch device string.")
        if isinstance(success_tolerance, bool) or int(success_tolerance) != success_tolerance or success_tolerance < 1:
            raise ValueError("success_tolerance must be a positive integer.")
        self.startup_trials = int(startup_trials)
        self.n_candidates = None if n_candidates is None else int(n_candidates)
        self.max_duplicate_attempts = int(max_duplicate_attempts)
        self.max_cholesky_size = float(max_cholesky_size)
        self.device = str(device).strip()
        self.success_tolerance = int(success_tolerance)
        self._seed = 0
        self._converter: UnitCubeSearchSpaceConverter | None = None
        self._startup_engine: Any | None = None
        self._history: list[TrialObservation] = []
        self._seen: set[str] = set()
        self._state: Any | None = None
        self._ask_count = 0
        self._restart_count = 0
        self._fixed_initialization: FixedInitializationProtocol | None = None
        self._candidate_budget = 0
        self._candidate_budget_policy = "uninitialized"

    @property
    def name(self) -> str:
        return "botorch_turbo"

    @property
    def candidate_budget(self) -> int:
        """Effective Thompson-sampling pool size after setup."""

        return self._require_candidate_budget()

    @property
    def candidate_budget_policy(self) -> str:
        return self._candidate_budget_policy

    def setup(self, task_spec, seed: int = 0, **kwargs: Any) -> None:
        if len(task_spec.objectives) != 1:
            raise ValueError("BotorchTurboAlgorithm supports exactly one objective.")
        self.bind_task_spec(task_spec)
        transforms = task_spec.metadata.get("parameter_transforms")
        if transforms is not None and not isinstance(transforms, dict):
            raise TypeError("Task metadata `parameter_transforms` must be a mapping.")
        self._converter = UnitCubeSearchSpaceConverter(
            task_spec.search_space,
            transforms=transforms,
        )
        self._seed = int(seed)
        self._fixed_initialization = resolve_fixed_initialization(task_spec, seed=self._seed)
        self._candidate_budget, self._candidate_budget_policy = resolve_model_candidate_budget(
            task_spec,
            target="botorch_turbo.n_candidates",
            configured=self.n_candidates,
        )
        self._reset_runtime()

    def ask(self) -> TrialSuggestion:
        successful = self._successful_history()
        if self._fixed_initialization is not None and len(self._history) < len(
            self._fixed_initialization.configurations
        ):
            suggestion = self._fixed_initialization.suggestion(len(self._history), algorithm=self.name)
            suggestion.metadata.update(
                {
                    "turbo_phase": "benchmark_initialization",
                    "turbo_backend": "botorch_official_tutorial",
                    "turbo_training_points": len(successful),
                }
            )
        elif self._fixed_initialization is not None and len(successful) < 2:
            suggestion = self._sobol_suggestion(phase="startup_recovery")
        elif self._fixed_initialization is None and len(successful) < self.startup_trials:
            suggestion = self._sobol_suggestion(phase="startup")
        elif self._state is not None and self._state.restart_triggered:
            self._restart_count += 1
            self._state = None
            suggestion = self._sobol_suggestion(phase="restart")
        else:
            try:
                suggestion = self._turbo_suggestion(successful)
            except Exception as exc:  # noqa: BLE001 - deterministic, explicit baseline fallback.
                suggestion = self._sobol_suggestion(
                    phase="sobol_fallback",
                    reason=f"{type(exc).__name__}: {exc}",
                )
        self._ask_count += 1
        self._seen.add(_config_identity(suggestion.config))
        return suggestion

    def tell(self, observation: TrialObservation) -> None:
        self._seen.add(_config_identity(observation.suggestion.config))
        self._history.append(observation)
        self.update_best_incumbent(observation)
        if not observation.success or self._primary_name not in observation.objectives:
            return

        phase = observation.suggestion.metadata.get("turbo_phase")
        if phase == "acquisition":
            deps = require_botorch_turbo()
            if self._state is None:
                previous = self._successful_history()[:-1]
                best_value = max(self._objective_to_maximization(item) for item in previous)
                self._state = deps["vendor"].TurboState(
                    dim=len(self._require_converter().feature_specs),
                    batch_size=1,
                    best_value=best_value,
                    success_tolerance=self.success_tolerance,
                )
            y_next = deps["torch"].tensor(
                [self._objective_to_maximization(observation)],
                dtype=deps["torch"].double,
            )
            self._state = deps["vendor"].update_state(self._state, y_next)
        elif self._state is not None:
            self._state.best_value = max(
                float(self._state.best_value),
                self._objective_to_maximization(observation),
            )

    def replay(self, history: list[TrialObservation]) -> None:
        self._best = None
        self._reset_runtime()
        sobol_draws = 0
        for observation in history:
            phase = observation.suggestion.metadata.get("turbo_phase")
            if phase in {"startup", "restart", "sobol_fallback"}:
                sobol_draws += 1
            self.tell(observation)
        if sobol_draws:
            self._require_startup_engine().fast_forward(sobol_draws)
        self._ask_count = len(history)

    def _turbo_suggestion(self, successful: list[TrialObservation]) -> TrialSuggestion:
        deps = require_botorch_turbo()
        torch = deps["torch"]
        gpytorch = deps["gpytorch"]
        converter = self._require_converter()
        device = torch.device(self.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"TuRBO requested {self.device}, but CUDA is unavailable.")

        x_values = np.vstack([converter.encode_vector(item.suggestion.config) for item in successful])
        y_values = np.asarray(
            [self._objective_to_maximization(item) for item in successful],
            dtype=float,
        )
        x_train = torch.tensor(x_values, dtype=torch.double, device=device)
        y_train = torch.tensor(y_values, dtype=torch.double, device=device).unsqueeze(-1)
        y_std = y_train.std()
        if not torch.isfinite(y_std) or y_std < 1e-12:
            y_standardized = y_train - y_train.mean()
        else:
            y_standardized = (y_train - y_train.mean()) / y_std

        if self._state is None:
            self._state = deps["vendor"].TurboState(
                dim=x_train.shape[-1],
                batch_size=1,
                best_value=float(y_train.max().item()),
                success_tolerance=self.success_tolerance,
            )

        likelihood = deps["GaussianLikelihood"](
            noise_constraint=deps["Interval"](1e-8, 1e-3)
        )
        covar_module = deps["ScaleKernel"](
            deps["MaternKernel"](
                nu=2.5,
                ard_num_dims=x_train.shape[-1],
                lengthscale_constraint=deps["Interval"](0.005, 4.0),
            )
        )
        model = deps["SingleTaskGP"](
            x_train,
            y_standardized,
            covar_module=covar_module,
            likelihood=likelihood,
        )
        mll = deps["ExactMarginalLogLikelihood"](model.likelihood, model)
        fit_seed = self._stable_seed("fit", len(self._history))
        torch.manual_seed(fit_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(fit_seed)
        with warnings.catch_warnings(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            deps["fit_gpytorch_mll"](mll)

            last_config: dict[str, Any] | None = None
            for attempt in range(self.max_duplicate_attempts):
                candidate_seed = self._stable_seed("candidate", len(self._history), attempt)
                torch.manual_seed(candidate_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(candidate_seed)
                candidate = deps["vendor"].generate_batch(
                    state=self._state,
                    model=model,
                    X=x_train,
                    Y=y_standardized,
                    batch_size=1,
                    n_candidates=self._require_candidate_budget(),
                    acqf="ts",
                )
                unit = candidate.detach().cpu().numpy().reshape(-1)
                config = converter.decode_vector(unit, clip=True)
                last_config = config
                if _config_identity(config) not in self._seen:
                    return TrialSuggestion(
                        config=config,
                        metadata={
                            **self._source_metadata(deps["vendor"]),
                            "turbo_phase": "acquisition",
                            "turbo_acquisition": "thompson_sampling",
                            "turbo_training_points": len(successful),
                            "turbo_trust_region_length": float(self._state.length),
                            "turbo_candidate_attempt": attempt,
                            "turbo_n_candidates": self._require_candidate_budget(),
                            "turbo_candidate_budget_policy": self._candidate_budget_policy,
                            "turbo_device": str(device),
                            "parameter_transforms": dict(converter.transforms),
                        },
                    )
        if last_config is not None:
            raise RuntimeError("TuRBO generated only duplicate decoded configurations.")
        raise RuntimeError("TuRBO did not generate a candidate.")

    def _sobol_suggestion(self, *, phase: str, reason: str | None = None) -> TrialSuggestion:
        converter = self._require_converter()
        engine = self._require_startup_engine()
        last_config: dict[str, Any] | None = None
        for attempt in range(max(32, self.max_duplicate_attempts)):
            unit = engine.draw(1).detach().cpu().numpy().reshape(-1)
            config = converter.decode_vector(unit, clip=True)
            last_config = config
            if _config_identity(config) not in self._seen:
                metadata = {
                    "turbo_phase": phase,
                    "turbo_backend": "botorch_official_tutorial",
                    "turbo_startup_backend": "torch.quasirandom.SobolEngine",
                    "turbo_seed": self._seed,
                    "turbo_startup_trials": self.startup_trials,
                    "turbo_sobol_duplicate_attempt": attempt,
                    "turbo_restart_count": self._restart_count,
                    "parameter_transforms": dict(converter.transforms),
                }
                if reason is not None:
                    metadata["turbo_fallback_reason"] = reason
                try:
                    metadata.update(self._source_metadata(require_botorch_turbo()["vendor"]))
                except ImportError:
                    pass
                return TrialSuggestion(config=config, metadata=metadata)
        assert last_config is not None
        return TrialSuggestion(
            config=last_config,
            metadata={
                "turbo_phase": phase,
                "turbo_backend": "botorch_official_tutorial",
                "turbo_duplicate_exhausted": True,
                "turbo_fallback_reason": reason,
            },
        )

    def _reset_runtime(self) -> None:
        deps = require_botorch_turbo()
        self._startup_engine = deps["SobolEngine"](
            dimension=len(self._require_converter().feature_specs),
            scramble=True,
            seed=self._seed,
        )
        self._history = []
        self._seen = set()
        self._state = None
        self._ask_count = 0
        self._restart_count = 0

    def _successful_history(self) -> list[TrialObservation]:
        if self._primary_name is None:
            return []
        return [
            item for item in self._history
            if item.success and self._primary_name in item.objectives
        ]

    def _objective_to_maximization(self, observation: TrialObservation) -> float:
        assert self._primary_name is not None
        value = float(observation.objectives[self._primary_name])
        return -value if self._primary_direction == ObjectiveDirection.MINIMIZE else value

    def _require_converter(self) -> UnitCubeSearchSpaceConverter:
        if self._converter is None:
            raise RuntimeError("BotorchTurboAlgorithm.setup() must be called before ask/tell.")
        return self._converter

    def _require_startup_engine(self) -> Any:
        if self._startup_engine is None:
            raise RuntimeError("BotorchTurboAlgorithm.setup() must be called before ask/tell.")
        return self._startup_engine

    def _require_candidate_budget(self) -> int:
        if self._candidate_budget <= 0:
            raise RuntimeError("BotorchTurboAlgorithm.setup() must be called before candidate generation.")
        return self._candidate_budget

    def _stable_seed(self, *parts: object) -> int:
        value = ":".join(str(part) for part in (self.name, self._seed, *parts))
        return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16) % (2**31 - 1)

    @staticmethod
    def _source_metadata(vendor: Any) -> dict[str, Any]:
        return {
            "turbo_backend": "botorch_official_tutorial",
            "turbo_source_tag": vendor.SOURCE_TAG,
            "turbo_source_commit": vendor.SOURCE_COMMIT,
            "turbo_source_notebook_blob_sha": vendor.SOURCE_NOTEBOOK_BLOB_SHA,
            "turbo_source_url": vendor.SOURCE_URL,
        }


def _config_identity(config: dict[str, Any]) -> str:
    return json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = ["BotorchTurboAlgorithm", "require_botorch_turbo"]
