"""Gradient-Informed TabPFN Bayesian optimization (GIT-BO).

This adapter follows the GIT-BO paper's main configuration while preserving
the benchmark-owned initialization prefix and candidate-budget protocol.
"""

from __future__ import annotations

import hashlib
import json
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


GIT_BO_PAPER_SUBSPACE_DIM = 10
GIT_BO_PAPER_BETA = 2.33
GIT_BO_PAPER_N_ESTIMATORS = 1
GIT_BO_DEFAULT_INFERENCE_BATCH_SIZE = 128
GIT_BO_TABPFN_COMMIT = "f87b137bc7c4fe021dda76e62720098541575d37"


def require_git_bo_dependencies() -> dict[str, Any]:
    """Import the pinned differentiable TabPFN v2 stack lazily."""

    try:
        import tabpfn
        import torch
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        from tabpfn.regressor import translate_probs_across_borders
    except ImportError as exc:  # pragma: no cover - optional dependency path.
        raise ImportError(
            "`git_bo` requires torch and the pinned differentiable TabPFN build. "
            "Install it with `uv sync --extra hpo --extra tabpfn`."
        ) from exc
    if not hasattr(TabPFNRegressor, "fit_with_differentiable_input"):
        raise RuntimeError(
            "The installed TabPFN regressor does not expose "
            "fit_with_differentiable_input(); sync the pinned `tabpfn` extra."
        )
    return {
        "torch": torch,
        "tabpfn": tabpfn,
        "TabPFNRegressor": TabPFNRegressor,
        "ModelVersion": ModelVersion,
        "translate_probs_across_borders": translate_probs_across_borders,
    }


class GitBoAlgorithm(ExternalOptimizerAdapter):
    """TabPFN v2 UCB search in a gradient-informed active subspace."""

    def __init__(
        self,
        *,
        subspace_dim: int = GIT_BO_PAPER_SUBSPACE_DIM,
        beta: float = GIT_BO_PAPER_BETA,
        n_candidates: int | None = None,
        n_estimators: int = GIT_BO_PAPER_N_ESTIMATORS,
        inference_batch_size: int = GIT_BO_DEFAULT_INFERENCE_BATCH_SIZE,
        startup_trials: int = 2,
        max_duplicate_attempts: int = 128,
        device: str = "auto",
    ) -> None:
        super().__init__()
        if subspace_dim <= 0:
            raise ValueError("subspace_dim must be positive.")
        if beta < 0:
            raise ValueError("beta must be non-negative.")
        if n_candidates is not None and n_candidates <= 0:
            raise ValueError("n_candidates must be positive when provided.")
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")
        if inference_batch_size <= 0:
            raise ValueError("inference_batch_size must be positive.")
        if startup_trials < 2:
            raise ValueError("startup_trials must be at least 2.")
        if max_duplicate_attempts <= 0:
            raise ValueError("max_duplicate_attempts must be positive.")
        if not str(device).strip():
            raise ValueError("device must be a non-empty torch device string.")

        self.subspace_dim = int(subspace_dim)
        self.beta = float(beta)
        self.n_candidates = None if n_candidates is None else int(n_candidates)
        self.n_estimators = int(n_estimators)
        self.inference_batch_size = int(inference_batch_size)
        self.startup_trials = int(startup_trials)
        self.max_duplicate_attempts = int(max_duplicate_attempts)
        self.device = str(device).strip()

        self._seed = 0
        self._device = "cpu"
        self._converter: UnitCubeSearchSpaceConverter | None = None
        self._history: list[TrialObservation] = []
        self._seen: set[str] = set()
        self._startup_engine: Any | None = None
        self._regressor: Any | None = None
        self._fixed_initialization: FixedInitializationProtocol | None = None
        self._candidate_budget = 0
        self._candidate_budget_policy = "uninitialized"
        self._fallback_count = 0

    @property
    def name(self) -> str:
        return "git_bo"

    @property
    def candidate_budget(self) -> int:
        """Effective gradient/acquisition pool size after setup."""

        if self._candidate_budget <= 0:
            raise RuntimeError("GitBoAlgorithm.setup() must be called before candidate_budget.")
        return self._candidate_budget

    @property
    def candidate_budget_policy(self) -> str:
        return self._candidate_budget_policy

    def setup(self, task_spec, seed: int = 0, **kwargs: Any) -> None:
        if len(task_spec.objectives) != 1:
            raise ValueError("GitBoAlgorithm supports exactly one objective.")
        self.bind_task_spec(task_spec)
        transforms = task_spec.metadata.get("parameter_transforms")
        if transforms is not None and not isinstance(transforms, dict):
            raise TypeError("Task metadata `parameter_transforms` must be a mapping.")
        self._converter = UnitCubeSearchSpaceConverter(
            task_spec.search_space,
            transforms=transforms,
        )
        self._seed = int(seed)
        self._device = self._resolve_device(self.device)
        self._fixed_initialization = resolve_fixed_initialization(task_spec, seed=self._seed)
        self._candidate_budget, self._candidate_budget_policy = resolve_model_candidate_budget(
            task_spec,
            target="git_bo.n_candidates",
            configured=self.n_candidates,
        )
        self._history = []
        self._seen = set()
        self._regressor = None
        self._fallback_count = 0
        self._startup_engine = self._new_sobol_engine(
            seed=self._stable_seed("startup"),
        )

    def ask(self) -> TrialSuggestion:
        successful = self._successful_history()
        if self._fixed_initialization is not None and len(self._history) < len(
            self._fixed_initialization.configurations
        ):
            suggestion = self._fixed_initialization.suggestion(
                len(self._history),
                algorithm=self.name,
            )
            suggestion.metadata.update(
                {
                    "git_bo_phase": "benchmark_initialization",
                    "git_bo_backend": "tabpfn_v2_gradient_informed",
                }
            )
        elif len(successful) < self.startup_trials:
            suggestion = self._sobol_suggestion(reason="startup")
        elif self._target_is_degenerate(successful):
            suggestion = self._sobol_suggestion(reason="degenerate_target")
        else:
            suggestion = self._git_bo_suggestion(successful)
        self._seen.add(_config_identity(suggestion.config))
        return suggestion

    def tell(self, observation: TrialObservation) -> None:
        self._seen.add(_config_identity(observation.suggestion.config))
        self._history.append(observation)
        self.update_best_incumbent(observation)

    def replay(self, history: list[TrialObservation]) -> None:
        self._best = None
        self._history = []
        self._seen = set()
        self._regressor = None
        self._fallback_count = 0
        self._startup_engine = self._new_sobol_engine(seed=self._stable_seed("startup"))
        for observation in history:
            self.tell(observation)

    def _git_bo_suggestion(
        self,
        successful: list[TrialObservation],
    ) -> TrialSuggestion:
        deps = require_git_bo_dependencies()
        torch = deps["torch"]
        converter = self._require_converter()
        device = torch.device(self._device)

        x_train = torch.as_tensor(
            np.vstack(
                [converter.encode_vector(observation.suggestion.config) for observation in successful]
            ),
            dtype=torch.float32,
            device=device,
        )
        y_train = torch.as_tensor(
            [self._objective_to_maximization(observation) for observation in successful],
            dtype=torch.float32,
            device=device,
        )
        model = self._fit_tabpfn(x_train=x_train, y_train=y_train, deps=deps)

        gradient_points = self._sobol_points(
            count=self.candidate_budget,
            seed=self._stable_seed("gradient_pool", len(self._history)),
            device=device,
        )
        dimension = int(x_train.shape[1])
        fisher = torch.zeros((dimension, dimension), dtype=torch.float32, device=device)
        for raw_chunk in gradient_points.split(self.inference_batch_size):
            chunk = raw_chunk.detach().requires_grad_(True)
            gradient_mean, _ = self._tabpfn_posterior(
                model=model,
                queries=chunk,
                deps=deps,
            )
            gradients = torch.autograd.grad(
                gradient_mean.sum(),
                chunk,
                retain_graph=False,
                create_graph=False,
            )[0]
            if not bool(torch.isfinite(gradients).all().item()):
                return self._sobol_suggestion(reason="non_finite_gradient")
            fisher.add_(gradients.transpose(0, 1) @ gradients)
        fisher.div_(max(self.candidate_budget, 1))
        eigenvalues, eigenvectors = torch.linalg.eigh(fisher)
        if not bool(torch.isfinite(eigenvalues).all().item()):
            return self._sobol_suggestion(reason="non_finite_fisher")
        effective_dim = min(self.subspace_dim, int(x_train.shape[1]))
        top_indices = torch.argsort(eigenvalues, descending=True)[:effective_dim]
        basis = eigenvectors[:, top_indices]

        x_reference = x_train.mean(dim=0)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._stable_seed("subspace_pool", len(self._history)))
        z = torch.rand(
            (self.candidate_budget, effective_dim),
            generator=generator,
            dtype=torch.float32,
        )
        z = (2.0 * z - 1.0).to(device)
        candidates = torch.clamp(x_reference.unsqueeze(0) + z @ basis.transpose(0, 1), 0.0, 1.0)
        mean_chunks: list[Any] = []
        std_chunks: list[Any] = []
        with torch.no_grad():
            for chunk in candidates.split(self.inference_batch_size):
                chunk_mean, chunk_std = self._tabpfn_posterior(
                    model=model,
                    queries=chunk,
                    deps=deps,
                )
                mean_chunks.append(chunk_mean)
                std_chunks.append(chunk_std)
        mean = torch.cat(mean_chunks, dim=0)
        std = torch.cat(std_chunks, dim=0)
        acquisition = mean + self.beta * std

        order = torch.argsort(acquisition, descending=True).detach().cpu().tolist()
        selected: tuple[dict[str, Any], int] | None = None
        for rank, candidate_index in enumerate(order):
            unit = candidates[int(candidate_index)].detach().cpu().numpy()
            config = self.require_search_space().coerce_config(
                converter.decode_vector(unit, clip=True),
                use_defaults=False,
            )
            if _config_identity(config) not in self._seen:
                selected = (config, rank)
                break
        if selected is None:
            return self._sobol_suggestion(reason="decoded_candidate_pool_exhausted")

        config, selected_rank = selected
        selected_index = int(order[selected_rank])
        top_eigenvalues = eigenvalues[top_indices].detach().cpu().tolist()
        return TrialSuggestion(
            config=config,
            metadata={
                "git_bo_phase": "acquisition",
                "git_bo_backend": "official_tabpfn_differentiable_input",
                "git_bo_paper": "GIT-BO (ICLR 2026)",
                "git_bo_tabpfn_commit": GIT_BO_TABPFN_COMMIT,
                "git_bo_tabpfn_package_version": str(getattr(deps["tabpfn"], "__version__", "unknown")),
                "git_bo_tabpfn_model_version": "v2",
                "git_bo_tabpfn_n_estimators": self.n_estimators,
                "git_bo_inference_batch_size": self.inference_batch_size,
                "git_bo_gradient_backend": "torch_autograd_predictive_mean",
                "git_bo_fisher_points": self.candidate_budget,
                "git_bo_candidate_points": self.candidate_budget,
                "git_bo_candidate_budget_policy": self._candidate_budget_policy,
                "git_bo_subspace_dim_requested": self.subspace_dim,
                "git_bo_subspace_dim_effective": effective_dim,
                "git_bo_top_fisher_eigenvalues": [float(value) for value in top_eigenvalues],
                "git_bo_acquisition": "UCB",
                "git_bo_beta": self.beta,
                "git_bo_selected_rank": selected_rank,
                "git_bo_selected_mean": float(mean[selected_index].detach().cpu()),
                "git_bo_selected_std": float(std[selected_index].detach().cpu()),
                "git_bo_selected_ucb": float(acquisition[selected_index].detach().cpu()),
                "git_bo_reference_point": "observed_centroid",
                "git_bo_gradient_sampling": "scrambled_sobol_unit_cube",
                "git_bo_subspace_sampling": "uniform_minus1_plus1",
                "git_bo_training_points": len(successful),
                "git_bo_device": str(device),
                "feature_transform": "unit_cube",
                "parameter_transforms": dict(converter.transforms),
            },
        )

    def _fit_tabpfn(
        self,
        *,
        x_train: Any,
        y_train: Any,
        deps: dict[str, Any],
    ) -> Any:
        if self._regressor is None:
            self._regressor = deps["TabPFNRegressor"].create_default_for_version(
                deps["ModelVersion"].V2,
                device=self._device,
                n_estimators=self.n_estimators,
                ignore_pretraining_limits=True,
                fit_mode="fit_preprocessors",
                differentiable_input=True,
                random_state=self._seed,
                show_progress_bar=False,
            )
        self._regressor.fit_with_differentiable_input(x_train, y_train)
        return self._regressor

    @staticmethod
    def _tabpfn_posterior(
        *,
        model: Any,
        queries: Any,
        deps: dict[str, Any],
    ) -> tuple[Any, Any]:
        """Return differentiable posterior mean/std from the pinned v2 engine."""

        torch = deps["torch"]
        translate = deps["translate_probs_across_borders"]
        executor_name = type(model.executor_).__name__
        if executor_name == "InferenceEngineBatchedNoPreprocessing":
            forward_input: Any = [queries]
            use_inference_mode = False
        else:
            forward_input = queries
            use_inference_mode = True

        accumulated = None
        n_estimators = 0
        for borders, output in model._iter_forward_executor(
            forward_input,
            use_inference_mode=use_inference_mode,
        ):
            transformed = translate(
                output,
                frm=torch.as_tensor(borders, device=output.device),
                to=model.znorm_space_bardist_.borders.to(output.device),
            )
            if model.average_before_softmax:
                transformed = transformed.log()
            accumulated = transformed if accumulated is None else accumulated + transformed
            n_estimators += 1
        if accumulated is None or n_estimators <= 0:
            raise RuntimeError("TabPFN returned no posterior estimators.")
        if model.average_before_softmax:
            probabilities = (accumulated / n_estimators).softmax(dim=-1)
        else:
            probabilities = accumulated / n_estimators
        logits = probabilities.clamp_min(1e-30).log()
        if logits.dtype == torch.float16:
            logits = logits.float()
        criterion = model.raw_space_bardist_.to(logits.device)
        mean = criterion.mean(logits)
        variance = criterion.variance(logits).clamp_min(0.0)
        return mean, variance.sqrt()

    def _sobol_suggestion(self, *, reason: str) -> TrialSuggestion:
        converter = self._require_converter()
        engine = self._require_startup_engine()
        last_config: dict[str, Any] | None = None
        for attempt in range(self.max_duplicate_attempts):
            unit = engine.draw(1).detach().cpu().numpy().reshape(-1)
            config = self.require_search_space().coerce_config(
                converter.decode_vector(unit, clip=True),
                use_defaults=False,
            )
            last_config = config
            if _config_identity(config) not in self._seen:
                self._fallback_count += 1
                return TrialSuggestion(
                    config=config,
                    metadata={
                        "git_bo_phase": "sobol_fallback",
                        "git_bo_fallback_reason": reason,
                        "git_bo_fallback_count": self._fallback_count,
                        "git_bo_backend": "official_tabpfn_differentiable_input",
                        "git_bo_tabpfn_model_version": "v2",
                        "git_bo_subspace_dim_requested": self.subspace_dim,
                        "git_bo_beta": self.beta,
                        "git_bo_device": self._device,
                        "feature_transform": "unit_cube",
                        "parameter_transforms": dict(converter.transforms),
                    },
                )
        assert last_config is not None
        raise RuntimeError("GIT-BO Sobol fallback generated only duplicate configurations.")

    def _successful_history(self) -> list[TrialObservation]:
        primary_name = self._require_primary_name()
        return [
            observation
            for observation in self._history
            if observation.success and primary_name in observation.objectives
        ]

    def _objective_to_maximization(self, observation: TrialObservation) -> float:
        value = float(observation.objectives[self._require_primary_name()])
        return -value if self._primary_direction == ObjectiveDirection.MINIMIZE else value

    def _target_is_degenerate(self, successful: list[TrialObservation]) -> bool:
        values = np.asarray(
            [self._objective_to_maximization(observation) for observation in successful],
            dtype=float,
        )
        return bool(not np.all(np.isfinite(values)) or np.std(values) <= 1e-12)

    def _sobol_points(self, *, count: int, seed: int, device: Any) -> Any:
        engine = self._new_sobol_engine(seed=seed)
        return engine.draw(count).to(device=device, dtype=require_git_bo_dependencies()["torch"].float32)

    def _new_sobol_engine(self, *, seed: int) -> Any:
        torch = require_git_bo_dependencies()["torch"]
        return torch.quasirandom.SobolEngine(
            dimension=len(self._require_converter().feature_specs),
            scramble=True,
            seed=seed,
        )

    @staticmethod
    def _resolve_device(requested: str) -> str:
        torch = require_git_bo_dependencies()["torch"]
        if requested == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        device = torch.device(requested)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(f"GIT-BO requested {requested}, but CUDA is unavailable.")
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"GIT-BO requested {requested}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
                )
        return str(device)

    def _stable_seed(self, *parts: object) -> int:
        payload = json.dumps([self._seed, *parts], ensure_ascii=False, separators=(",", ":"))
        return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big") & 0x7FFFFFFF

    def _require_converter(self) -> UnitCubeSearchSpaceConverter:
        if self._converter is None:
            raise RuntimeError("GitBoAlgorithm.setup() must be called before ask/tell.")
        return self._converter

    def _require_startup_engine(self) -> Any:
        if self._startup_engine is None:
            raise RuntimeError("GitBoAlgorithm.setup() must be called before ask/tell.")
        return self._startup_engine

    def _require_primary_name(self) -> str:
        if self._primary_name is None:
            raise RuntimeError("GitBoAlgorithm.setup() must be called before ask/tell.")
        return self._primary_name


def _config_identity(config: dict[str, Any]) -> str:
    return json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "GIT_BO_PAPER_BETA",
    "GIT_BO_DEFAULT_INFERENCE_BATCH_SIZE",
    "GIT_BO_PAPER_N_ESTIMATORS",
    "GIT_BO_PAPER_SUBSPACE_DIM",
    "GIT_BO_TABPFN_COMMIT",
    "GitBoAlgorithm",
    "require_git_bo_dependencies",
]
