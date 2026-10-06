"""GPBO over SMILES with Graph GA acquisition optimization.

Adapted from local PMO sources:

- PMO mol_opt/main/gpbo/run.py
- PMO mol_opt/main/gpbo/bo/gp_bo.py
- PMO mol_opt/main/gpbo/gp/tanimoto_gp.py
- PMO mol_opt/main/gpbo/gp/gp_utils.py
- PMO mol_opt/main/gpbo/bo/acquisition_funcs.py
- PMO mol_opt/main/gpbo/fingerprints.py

The framework adaptation is the same ask/tell split used for Graph GA: GPBO
only asks the real task for the final selected SMILES. The internal Graph GA
scores candidates with the acquisition function.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ...core import ExternalOptimizerAdapter, ObjectiveDirection, StringParam, TrialObservation, TrialSuggestion
from .graph_ga_ops import (
    CachedBatchScoringFunction,
    GraphGACandidateOptimizer,
    load_smiles_file,
    morgan_fingerprint_array,
    sanitize_smiles,
)

GPBO_DEFAULT_INITIAL_POPULATION_SIZE = 340
GPBO_DEFAULT_N_TRAIN_GP_BEST = 2200
GPBO_DEFAULT_N_TRAIN_GP_RAND = 1350
GPBO_DEFAULT_BO_BATCH_SIZE = 1180
GPBO_DEFAULT_GA_MAX_GENERATIONS = 60
GPBO_DEFAULT_GA_OFFSPRING_SIZE = 150
GPBO_DEFAULT_GA_MUTATION_RATE = 0.01
GPBO_DEFAULT_GA_POPULATION_SIZE = 820
GPBO_DEFAULT_GA_POOL_NUM_BEST = 250
GPBO_DEFAULT_GA_POOL_NUM_CARRYOVER = 250
GPBO_DEFAULT_MAX_GA_START_POPULATION_SIZE = 1000
GPBO_DEFAULT_FP_RADIUS = 2
GPBO_DEFAULT_FP_NBITS = 4096


def _require_gp_deps():
    try:
        import botorch
        import gpytorch
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional GP deps.
        raise ImportError(
            "Graph GPBO requires torch, gpytorch, and botorch, matching the local PMO GPBO implementation."
        ) from exc
    return torch, gpytorch, botorch


def _make_tanimoto_gp_class():
    torch, gpytorch, botorch = _require_gp_deps()

    def batch_tanimoto_sim(x1, x2):
        dot_prod = torch.matmul(x1, torch.transpose(x2, -1, -2))
        x1_sum = torch.sum(x1**2, dim=-1, keepdims=True)
        x2_sum = torch.sum(x2**2, dim=-1, keepdims=True)
        return dot_prod / (x1_sum + torch.transpose(x2_sum, -1, -2) - dot_prod)

    class TanimotoKernel(gpytorch.kernels.Kernel):
        is_stationary = False
        has_lengthscale = False

        def forward(self, x1, x2, diag=False, **params):
            if diag:
                if x1.size() != x2.size() or not torch.equal(x1, x2):
                    raise ValueError("TanimotoKernel diag=True expects identical inputs.")
                return torch.ones(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device)
            return batch_tanimoto_sim(x1, x2)

    class TanimotoGP(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
        _num_outputs = 1

        def __init__(self, train_x, train_y, likelihood=None):
            if likelihood is None:
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
            botorch.models.gpytorch.GPyTorchModel.__init__(self)
            gpytorch.models.ExactGP.__init__(self, train_x, train_y, likelihood)
            self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())
            self.mean_module = gpytorch.means.ConstantMean()

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    return TanimotoGP


def _fit_gp_hyperparameters(gp_model: Any) -> None:
    torch, gpytorch, botorch = _require_gp_deps()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    fit_module = getattr(getattr(botorch, "optim", None), "fit", None)
    fit_scipy = getattr(fit_module, "fit_gpytorch_scipy", None)
    if fit_scipy is not None:
        fit_scipy(mll)
        return
    try:  # pragma: no cover - compatibility with newer botorch.
        from botorch.fit import fit_gpytorch_mll
    except ImportError:
        raise ImportError("Installed botorch does not expose a compatible GP fitting helper.")
    fit_gpytorch_mll(mll)


def _batch_predict_mu_var_numpy(gp_model: Any, x: Any, *, batch_size: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    torch, gpytorch, _ = _require_gp_deps()
    gp_model.eval()
    mu = []
    var = []
    with gpytorch.settings.fast_computations(False, False, False), torch.no_grad():
        for batch_start in range(0, len(x), batch_size):
            batch_end = batch_start + batch_size
            output = gp_model(x[batch_start:batch_end])
            mu.append(output.mean.detach().cpu().numpy())
            var.append(output.variance.detach().cpu().numpy())
    return np.concatenate(mu, axis=0), np.concatenate(var, axis=0)


def _upper_confidence_bound(mu: np.ndarray, var: np.ndarray, *, beta: float) -> np.ndarray:
    return mu + np.sqrt(beta * var)


class GraphGPBOAlgorithm(ExternalOptimizerAdapter):
    """GPBO with Morgan fingerprints, Tanimoto GP, UCB, and Graph GA acquisition search."""

    def __init__(
        self,
        *,
        initial_smiles: Sequence[str] | None = None,
        initial_smiles_path: str | Path | None = None,
        initial_population_size: int = GPBO_DEFAULT_INITIAL_POPULATION_SIZE,
        n_train_gp_best: int = GPBO_DEFAULT_N_TRAIN_GP_BEST,
        n_train_gp_rand: int = GPBO_DEFAULT_N_TRAIN_GP_RAND,
        bo_batch_size: int = GPBO_DEFAULT_BO_BATCH_SIZE,
        ga_max_generations: int = GPBO_DEFAULT_GA_MAX_GENERATIONS,
        ga_offspring_size: int = GPBO_DEFAULT_GA_OFFSPRING_SIZE,
        ga_mutation_rate: float = GPBO_DEFAULT_GA_MUTATION_RATE,
        ga_population_size: int = GPBO_DEFAULT_GA_POPULATION_SIZE,
        ga_pool_num_best: int = GPBO_DEFAULT_GA_POOL_NUM_BEST,
        ga_pool_num_carryover: int = GPBO_DEFAULT_GA_POOL_NUM_CARRYOVER,
        max_ga_start_population_size: int = GPBO_DEFAULT_MAX_GA_START_POPULATION_SIZE,
        fp_radius: int = GPBO_DEFAULT_FP_RADIUS,
        fp_nbits: int = GPBO_DEFAULT_FP_NBITS,
        graph_ga_optimizer_cls: type[GraphGACandidateOptimizer] = GraphGACandidateOptimizer,
    ) -> None:
        super().__init__()
        if initial_smiles is not None and initial_smiles_path is not None:
            raise ValueError("Pass either initial_smiles or initial_smiles_path, not both.")
        self.initial_smiles = tuple(str(item) for item in initial_smiles) if initial_smiles is not None else None
        self.initial_smiles_path = Path(initial_smiles_path) if initial_smiles_path is not None else None
        self.initial_population_size = int(initial_population_size)
        self.n_train_gp_best = int(n_train_gp_best)
        self.n_train_gp_rand = int(n_train_gp_rand)
        self.bo_batch_size = int(bo_batch_size)
        self.ga_max_generations = int(ga_max_generations)
        self.ga_offspring_size = int(ga_offspring_size)
        self.ga_mutation_rate = float(ga_mutation_rate)
        self.ga_population_size = int(ga_population_size)
        self.ga_pool_num_best = int(ga_pool_num_best)
        self.ga_pool_num_carryover = int(ga_pool_num_carryover)
        self.max_ga_start_population_size = int(max_ga_start_population_size)
        self.fp_radius = int(fp_radius)
        self.fp_nbits = int(fp_nbits)
        self.graph_ga_optimizer_cls = graph_ga_optimizer_cls
        self._rng = random.Random()
        self._np_rng = np.random.default_rng()
        self._seed = 0
        self._smiles_param_name: str | None = None
        self._initial_queue: list[str] = []
        self._observed_scores: dict[str, float] = {}
        self._smiles_pool: set[str] = set()
        self._carryover_smiles: set[str] = set()
        self._bo_queue: list[tuple[str, float]] = []
        self._bo_iteration = 0
        self._active_batch_expected = 0
        self._active_batch_scores: dict[str, float] = {}
        self._last_acq_smiles: list[str] = []

    @property
    def name(self) -> str:
        return "gpbo"

    def setup(self, task_spec, seed: int = 0, **kwargs: Any) -> None:
        if len(task_spec.objectives) != 1:
            raise ValueError("GraphGPBOAlgorithm currently supports exactly one objective.")
        self.bind_task_spec(task_spec)
        string_params = [param for param in task_spec.search_space if isinstance(param, StringParam)]
        if len(string_params) != 1:
            raise ValueError("GraphGPBOAlgorithm requires exactly one StringParam SMILES search parameter.")
        self._smiles_param_name = string_params[0].name
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        self._np_rng = np.random.default_rng(self._seed)
        self._observed_scores = {}
        self._carryover_smiles = set()
        self._bo_queue = []
        self._bo_iteration = 0
        self._active_batch_expected = 0
        self._active_batch_scores = {}
        self._last_acq_smiles = []

        if self.initial_smiles_path is not None:
            source_smiles = load_smiles_file(self.initial_smiles_path)
        elif self.initial_smiles is not None:
            source_smiles = list(self.initial_smiles)
        else:
            raise ValueError(
                "GraphGPBOAlgorithm requires explicit initial_smiles or initial_smiles_path. "
                "No default molecular population is invented by the framework."
            )
        initial = sanitize_smiles(source_smiles)[: self.initial_population_size]
        if not initial:
            raise ValueError("GraphGPBOAlgorithm requires at least one valid initial SMILES.")
        self._initial_queue = list(initial)
        self._smiles_pool = set(initial)

    def ask(self) -> TrialSuggestion:
        param_name = self._require_smiles_param_name()
        if self._initial_queue:
            smiles = self._initial_queue.pop(0)
            return TrialSuggestion(
                config={param_name: smiles},
                metadata={
                    "gpbo_phase": "initial_population",
                    "gpbo_bo_iteration": 0,
                    "gpbo_fp_radius": self.fp_radius,
                    "gpbo_fp_nbits": self.fp_nbits,
                },
            )
        if not self._bo_queue:
            self._build_bo_batch()
        smiles, acquisition = self._bo_queue.pop(0)
        return TrialSuggestion(
            config={param_name: smiles},
            metadata={
                "gpbo_phase": "acquisition_batch",
                "gpbo_bo_iteration": self._bo_iteration,
                "gpbo_acquisition": float(acquisition),
                "gpbo_acquisition_name": "ucb",
                "gpbo_fp_radius": self.fp_radius,
                "gpbo_fp_nbits": self.fp_nbits,
            },
        )

    def tell(self, observation: TrialObservation) -> None:
        self.update_best_incumbent(observation)
        if not observation.success:
            return
        score = self._score_for_maximization(observation)
        if score is None:
            return
        param_name = self._require_smiles_param_name()
        smiles = sanitize_smiles([str(observation.suggestion.config.get(param_name, ""))])
        if not smiles:
            return
        canonical = smiles[0]
        self._observed_scores[canonical] = score
        self._smiles_pool.add(canonical)

        if observation.suggestion.metadata.get("gpbo_phase") == "acquisition_batch":
            self._active_batch_scores[canonical] = score
            if len(self._active_batch_scores) >= self._active_batch_expected:
                self._finish_active_batch()

    def _build_bo_batch(self) -> None:
        if not self._observed_scores:
            raise RuntimeError("GraphGPBOAlgorithm has no observed scores. Evaluate initial SMILES first.")
        torch, _, _ = _require_gp_deps()
        torch_dtype = torch.float64
        np_dtype = np.float64
        known_smiles = list(self._observed_scores)
        y_all = np.asarray([self._observed_scores[smiles] for smiles in known_smiles], dtype=np_dtype)
        train_indices = self._get_train_indices(y_all)
        train_smiles = [known_smiles[index] for index in train_indices]
        x_train = np.stack([self._fingerprint(smiles) for smiles in train_smiles]).astype(np_dtype)
        y_train = np.asarray([self._observed_scores[smiles] for smiles in train_smiles], dtype=np_dtype)

        TanimotoGP = _make_tanimoto_gp_class()
        gp_model = TanimotoGP(
            train_x=torch.as_tensor(x_train, dtype=torch_dtype),
            train_y=torch.as_tensor(y_train, dtype=torch_dtype),
        )
        _fit_gp_hyperparameters(gp_model)
        gp_model.eval()

        beta_curr = 10 ** float(self._np_rng.uniform(-0.5, 1.5))

        def acquisition_for_smiles(smiles_list: list[str]) -> list[float]:
            fp_array = np.stack([self._fingerprint(smiles) for smiles in smiles_list]).astype(np_dtype)
            mu_pred, var_pred = _batch_predict_mu_var_numpy(
                gp_model,
                torch.as_tensor(fp_array, dtype=torch_dtype),
                batch_size=2**15,
            )
            values = _upper_confidence_bound(mu_pred, var_pred, beta=beta_curr**2)
            return [float(value) for value in values]

        top_known = [
            smiles
            for _, smiles in sorted(
                [(score, smiles) for smiles, score in self._observed_scores.items()],
                reverse=True,
            )[: self.ga_pool_num_best]
        ]
        ga_start_smiles = set(top_known)
        ga_start_smiles.update(self._carryover_smiles)
        if len(ga_start_smiles) < self.max_ga_start_population_size:
            pool = list(self._smiles_pool)
            sample_size = min(len(pool), self.max_ga_start_population_size)
            for smiles in self._rng.sample(pool, sample_size):
                ga_start_smiles.add(smiles)
                if len(ga_start_smiles) >= self.max_ga_start_population_size:
                    break

        optimizer = self.graph_ga_optimizer_cls(
            max_generations=self.ga_max_generations,
            population_size=self.ga_population_size,
            offspring_size=self.ga_offspring_size,
            mutation_rate=self.ga_mutation_rate,
        )
        result = optimizer.maximize(
            starting_population_smiles=list(ga_start_smiles),
            scoring_function=CachedBatchScoringFunction(acquisition_for_smiles),
            seed=self._seed + self._bo_iteration * 1009,
        )
        sorted_acq = sorted(result.scores_by_smiles.items(), key=lambda item: item[1], reverse=True)
        self._last_acq_smiles = [smiles for smiles, _ in sorted_acq]
        self._smiles_pool.update(self._last_acq_smiles)

        selected: list[tuple[str, float]] = []
        observed = set(self._observed_scores)
        for smiles, acquisition in sorted_acq:
            if smiles not in observed and smiles not in {item[0] for item in selected}:
                selected.append((smiles, float(acquisition)))
            if len(selected) >= self.bo_batch_size:
                break
        if not selected:
            raise RuntimeError("GraphGPBOAlgorithm could not select any unseen acquisition candidate.")

        self._bo_iteration += 1
        self._bo_queue = selected
        self._active_batch_expected = len(selected)
        self._active_batch_scores = {}

    def _finish_active_batch(self) -> None:
        observed = set(self._observed_scores)
        carryover: set[str] = set()
        for smiles in self._last_acq_smiles:
            if len(carryover) >= self.ga_pool_num_carryover:
                break
            if smiles not in observed:
                carryover.add(smiles)
        self._carryover_smiles = carryover
        self._active_batch_expected = 0
        self._active_batch_scores = {}

    def _get_train_indices(self, y: np.ndarray) -> list[int]:
        argsort = np.argsort(-y)
        best = list(argsort[: self.n_train_gp_best])
        remaining = list(argsort[self.n_train_gp_best :])
        if len(remaining) <= self.n_train_gp_rand:
            rand = remaining
        else:
            rand = self._rng.sample(remaining, k=self.n_train_gp_rand)
        return sorted(int(index) for index in best + rand)

    def _fingerprint(self, smiles: str) -> np.ndarray:
        return morgan_fingerprint_array(smiles, radius=self.fp_radius, n_bits=self.fp_nbits)

    def _score_for_maximization(self, observation: TrialObservation) -> float | None:
        assert self._primary_name is not None
        if self._primary_name not in observation.objectives:
            return None
        value = float(observation.objectives[self._primary_name])
        if not math.isfinite(value):
            return None
        if self._primary_direction == ObjectiveDirection.MINIMIZE:
            return -value
        return value

    def _require_smiles_param_name(self) -> str:
        if self._smiles_param_name is None:
            raise RuntimeError("GraphGPBOAlgorithm.setup() must be called before ask/tell.")
        return self._smiles_param_name


__all__ = [
    "GraphGPBOAlgorithm",
]
