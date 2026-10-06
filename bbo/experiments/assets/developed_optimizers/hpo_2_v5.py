"""Trust-region Gaussian-process expected-improvement optimizer.

The reference GP-EI baseline fits a *global* SingleTaskGP over raw, unwarped
parameter axes and maximises expected improvement over the whole box.  On
small budgets with plateau objectives that model is easily misled: expected
improvement is largest at hostile, far-away corners where the posterior
variance is huge, so the search repeatedly evaluates extremely poor
configurations instead of refining promising ones.

This version keeps the strong parts (BoTorch SingleTaskGP + expected
improvement) and changes the search logic:

* unit-cube warped encoding (declared ``log``/``logit``/``linear`` geometry)
  so the GP sees distances that match the parameter spaces,
* several maximum-likelihood restarts for the GP hyper-parameters,
* an adaptive trust region around the best observation so acquisition search
  cannot jump to unexplored, hostile corners; the region expands while the
  search makes progress, shrinks while it stalls, and re-opens with a fresh
  quasi-random probe when it has fully collapsed,
* deterministic, validated fallbacks so an invalid or duplicate configuration
  is never emitted.

All algorithm state is derived from the observation history, so
``setup()`` + ``replay(history)`` reproduces exactly the same next proposal.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from solver_support import (
    Algorithm,
    FloatParam,
    Incumbent,
    IntParam,
    ObjectiveDirection,
    TrialObservation,
    TrialSuggestion,
)

_LOGIT_EPS = 1e-9
_DEPS: dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
# deterministic helpers
# --------------------------------------------------------------------------- #
def _stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


def _warp(value: float, kind: str) -> float:
    if kind == "log":
        return math.log(max(value, 1e-300))
    if kind == "logit":
        clipped = min(max(value, _LOGIT_EPS), 1.0 - _LOGIT_EPS)
        return math.log(clipped / (1.0 - clipped))
    return value


def _unwarp(value: float, kind: str) -> float:
    if kind == "log":
        return math.exp(value)
    if kind == "logit":
        if value >= 0.0:
            inverse = math.exp(-value)
            return 1.0 / (1.0 + inverse)
        inverse = math.exp(value)
        return inverse / (1.0 + inverse)
    return value


def _require_botorch() -> dict[str, Any]:
    """Import BoTorch lazily so module import never depends on the runtime env."""

    global _DEPS
    if _DEPS is not None:
        return _DEPS

    import torch
    from botorch.acquisition import ExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood

    log_ei = ExpectedImprovement
    try:
        from botorch.acquisition import LogExpectedImprovement

        log_ei = LogExpectedImprovement
    except Exception:  # pragma: no cover - newer/older BoTorch variants.
        log_ei = ExpectedImprovement

    _DEPS = {
        "torch": torch,
        "SingleTaskGP": SingleTaskGP,
        "Standardize": Standardize,
        "fit_gpytorch_mll": fit_gpytorch_mll,
        "ExactMarginalLogLikelihood": ExactMarginalLogLikelihood,
        "optimize_acqf": optimize_acqf,
        "EI": ExpectedImprovement,
        "LogEI": log_ei,
    }
    return _DEPS


# --------------------------------------------------------------------------- #
# encoding
# --------------------------------------------------------------------------- #
class _ParamEncoder:
    """Encode configs into a unit cube (numeric warps) plus one-hot categories."""

    def __init__(self, space: Any) -> None:
        self.space = space
        self.features: list[tuple[Any, ...]] = []
        dimension = 0
        for param in space:
            if isinstance(param, (FloatParam, IntParam)):
                low = float(param.low)
                high = float(param.high)
                is_log = bool(getattr(param, "log", False))
                if high <= low:
                    kind = "linear"
                elif is_log and low > 0.0:
                    kind = "log"
                elif (not is_log) and 0.0 < low <= high < 1.0:
                    kind = "logit"
                else:
                    kind = "linear"
                self.features.append(
                    ("num", param, kind, low, high, isinstance(param, IntParam))
                )
                dimension += 1
            else:
                choices = list(getattr(param, "choices", ()) or ())
                if not choices:
                    choices = [param.effective_default()]
                self.features.append(("cat", param, choices))
                dimension += len(choices)
        self.dim = dimension

    def encode(self, config: dict[str, Any]) -> np.ndarray:
        normalized = self.space.coerce_config(config, use_defaults=False)
        values: list[float] = []
        for feature in self.features:
            if feature[0] == "num":
                _, param, kind, low, high, _ = feature
                value = min(max(float(normalized[param.name]), low), high)
                warped_low = _warp(low, kind)
                warped_high = _warp(high, kind)
                if warped_high <= warped_low:
                    values.append(0.0)
                else:
                    unit = (_warp(value, kind) - warped_low) / (warped_high - warped_low)
                    values.append(min(max(unit, 0.0), 1.0))
            else:
                _, param, choices = feature
                current = normalized[param.name]
                values.extend(1.0 if current == choice else 0.0 for choice in choices)
        return np.asarray(values, dtype=float)

    def decode(self, unit_vector: np.ndarray) -> dict[str, Any]:
        vector = np.clip(np.asarray(unit_vector, dtype=float).reshape(-1), 0.0, 1.0)
        if vector.shape[0] != self.dim:
            raise ValueError("Encoded vector has the wrong dimension.")
        config: dict[str, Any] = {}
        cursor = 0
        for feature in self.features:
            if feature[0] == "num":
                _, param, kind, low, high, is_int = feature
                warped_low = _warp(low, kind)
                warped_high = _warp(high, kind)
                if warped_high <= warped_low:
                    value = low
                else:
                    value = _unwarp(
                        warped_low + float(vector[cursor]) * (warped_high - warped_low),
                        kind,
                    )
                value = min(max(value, low), high)
                config[param.name] = int(round(value)) if is_int else float(value)
                cursor += 1
            else:
                _, param, choices = feature
                count = len(choices)
                block = vector[cursor : cursor + count]
                index = int(np.argmax(block)) if count > 1 else 0
                config[param.name] = choices[index]
                cursor += count
        return self.space.coerce_config(config, use_defaults=False)


class _State:
    """Trust-region state derived purely from the observation history."""

    __slots__ = (
        "best_loss", "best_config", "best_value", "length",
        "successes", "failures",
    )

    def __init__(self) -> None:
        self.best_loss: float | None = None
        self.best_config: dict[str, Any] | None = None
        self.best_value: float | None = None
        self.length = 0.0
        self.successes = 0
        self.failures = 0


# --------------------------------------------------------------------------- #
# algorithm
# --------------------------------------------------------------------------- #
class TrustRegionEi(Algorithm):
    """GP expected improvement confined to an adaptive trust region."""

    L_INIT = 0.30
    L_MAX = 0.70
    L_MIN = 0.05
    EXPAND_FACTOR = 1.35
    SHRINK_FACTOR = 0.65
    EXPAND_AFTER = 1
    STALL_RESET = 8
    MIN_DISTANCE = 0.03

    def __init__(
        self,
        *,
        n_gp_restarts: int = 3,
        num_restarts: int = 16,
        raw_samples: int = 512,
    ) -> None:
        self._n_gp_restarts = max(1, int(n_gp_restarts))
        self._num_restarts = max(1, int(num_restarts))
        self._raw_samples = max(32, int(raw_samples))
        self._space: Any = None
        self._encoder: _ParamEncoder | None = None
        self._history: list[TrialObservation] = []
        self._seed = 0
        self._objective_name = ""
        self._direction = ObjectiveDirection.MINIMIZE
        self._state = _State()

    # -- protocol ---------------------------------------------------------- #
    @property
    def name(self) -> str:
        return "trust_region_gp_ei"

    def setup(self, task_spec: Any, seed: int = 0, **kwargs: Any) -> None:
        if len(task_spec.objectives) != 1:
            raise ValueError("TrustRegionEi supports exactly one objective.")
        objective = task_spec.primary_objective
        self._space = task_spec.search_space
        self._encoder = _ParamEncoder(self._space)
        self._objective_name = objective.name
        self._direction = objective.direction
        self._seed = int(seed)
        self._history = []
        self._state = _State()

    def ask(self) -> TrialSuggestion:
        state = self._state
        successful = self._successful()
        if len(successful) < 2:
            return self._random_suggestion(phase="startup")
        try:
            config = self._propose(successful, state)
            phase = "acquisition"
        except Exception:  # noqa: BLE001 - a proposal failure must not lose the round.
            return self._random_suggestion(phase="fallback")
        return TrialSuggestion(
            config=config,
            metadata={
                "algorithm": self.name,
                "phase": phase,
                "trust_region_length": float(state.length),
                "training_points": len(successful),
            },
        )

    def tell(self, observation: TrialObservation) -> None:
        self._history.append(observation)
        self._state = self._derive_state()

    def incumbents(self) -> list[Incumbent]:
        state = self._state
        if state.best_config is None or state.best_value is None:
            return []
        return [
            Incumbent(
                config=dict(state.best_config),
                score=float(state.best_value),
                objectives={self._objective_name: float(state.best_value)},
                metadata={"algorithm": self.name},
            )
        ]

    # -- state ------------------------------------------------------------- #
    def _derive_state(self) -> _State:
        state = _State()
        state.length = self.L_INIT
        tolerance = self._failure_tolerance()
        for observation in self._history:
            if not self._usable(observation):
                continue
            loss = self._loss(observation)
            value = float(observation.objectives[self._objective_name])
            if state.best_loss is None or loss < state.best_loss:
                state.best_loss = loss
                state.best_config = dict(observation.suggestion.config)
                state.best_value = value
                state.successes += 1
                state.failures = 0
                if state.successes >= self.EXPAND_AFTER:
                    state.length = min(state.length * self.EXPAND_FACTOR, self.L_MAX)
                    state.successes = 0
            else:
                state.failures += 1
                state.successes = 0
                if state.failures % tolerance == 0:
                    state.length = max(state.length * self.SHRINK_FACTOR, self.L_MIN)
                if self.STALL_RESET and state.failures % self.STALL_RESET == 0:
                    state.length = self.L_INIT
        return state

    # -- proposal ---------------------------------------------------------- #
    def _propose(self, successful: list[TrialObservation], state: _State) -> dict[str, Any]:
        deps = _require_botorch()
        torch = deps["torch"]
        encoder = self._encoder
        assert encoder is not None

        features = np.vstack(
            [encoder.encode(item.suggestion.config) for item in successful]
        )
        losses = np.asarray([self._loss(item) for item in successful], dtype=float)

        center = features[int(np.argmin(losses))]
        half = max(float(state.length), 1e-3)
        lower = np.clip(center - half, 0.0, 1.0)
        upper = np.clip(center + half, 0.0, 1.0)
        upper = np.maximum(upper, lower)

        train_x = torch.as_tensor(features, dtype=torch.double)
        train_y = torch.as_tensor((-losses).reshape(-1, 1), dtype=torch.double)
        model = self._fit_gp(torch, deps, train_x, train_y)
        acquisition = self._build_acquisition(deps, model, float(train_y.max()))

        pool = self._candidate_pool(torch, deps, acquisition, lower, upper)
        scores = self._score_pool(torch, acquisition, pool)
        if scores is None:
            return encoder.decode(self._random_in_box(lower, upper))
        distances = np.min(
            np.linalg.norm(features[None, :, :] - pool[:, None, :], axis=2), axis=1
        )
        acceptable = distances >= self.MIN_DISTANCE
        if np.any(acceptable):
            scores = np.where(acceptable, scores, -np.inf)
        return encoder.decode(pool[int(np.argmax(scores))])

    def _candidate_pool(
        self, torch: Any, deps: dict[str, Any], acquisition: Any,
        lower: np.ndarray, upper: np.ndarray,
    ) -> np.ndarray:
        """Sample the trust region and add analytic acquisition optima."""

        encoder = self._encoder
        assert encoder is not None
        count = max(512, 192 * encoder.dim)
        rng = np.random.default_rng(_stable_seed(self._seed, "pool", len(self._history)))
        pool = lower + (upper - lower) * rng.random((count, encoder.dim))
        bounds = torch.as_tensor(np.vstack([lower, upper]), dtype=torch.double)
        for attempt in range(3):
            torch.manual_seed(_stable_seed(self._seed, "acqf", len(self._history), attempt))
            try:
                points, _ = deps["optimize_acqf"](
                    acq_function=acquisition,
                    bounds=bounds,
                    q=1,
                    num_restarts=self._num_restarts,
                    raw_samples=self._raw_samples,
                    options={"batch_limit": 5, "maxiter": 200},
                )
            except Exception:  # noqa: BLE001
                continue
            extra = np.clip(
                points.detach().cpu().numpy().reshape(1, -1), lower, upper
            )
            pool = np.vstack([pool, extra])
        return pool

    @staticmethod
    def _score_pool(torch: Any, acquisition: Any, pool: np.ndarray) -> np.ndarray | None:
        try:
            with torch.no_grad():
                tensor = torch.as_tensor(pool, dtype=torch.double).unsqueeze(-2)
                values = acquisition(tensor).detach().cpu().numpy().reshape(-1)
        except Exception:  # noqa: BLE001
            return None
        values = np.asarray(values, dtype=float)
        return np.where(np.isfinite(values), values, -np.inf)

    def _random_in_box(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(_stable_seed(self._seed, "box", len(self._history)))
        return lower + (upper - lower) * rng.random(len(lower))

    def _fit_gp(self, torch: Any, deps: dict[str, Any], train_x: Any, train_y: Any) -> Any:
        best_model = None
        best_value = -math.inf
        for restart in range(self._n_gp_restarts):
            torch.manual_seed(_stable_seed(self._seed, "gp", len(self._history), restart))
            model = deps["SingleTaskGP"](
                train_x, train_y, outcome_transform=deps["Standardize"](m=1)
            )
            if restart > 0:
                self._perturb_hyperparameters(torch, model)
            mll = deps["ExactMarginalLogLikelihood"](model.likelihood, model)
            try:
                deps["fit_gpytorch_mll"](mll)
            except Exception:  # noqa: BLE001
                continue
            value = self._mll_value(torch, mll, model, train_x, train_y)
            if best_model is None or value > best_value:
                best_value = value
                best_model = model
        if best_model is None:
            raise RuntimeError("GP hyper-parameter fitting failed for every restart.")
        return best_model

    @staticmethod
    def _mll_value(torch: Any, mll: Any, model: Any, train_x: Any, train_y: Any) -> float:
        """Total marginal log likelihood, robust to per-datapoint tensor shapes."""

        try:
            with torch.no_grad():
                raw = mll(model(train_x), train_y)
            total = float(torch.as_tensor(raw).detach().reshape(-1).sum())
        except Exception:  # noqa: BLE001
            return -math.inf
        return total if math.isfinite(total) else -math.inf

    @staticmethod
    def _perturb_hyperparameters(torch: Any, model: Any) -> None:
        with torch.no_grad():
            try:
                covar = model.covar_module
                base = getattr(covar, "base_kernel", covar)
                shape = tuple(base.lengthscale.shape)
                base.lengthscale = torch.exp(
                    torch.empty(shape, dtype=torch.double).uniform_(
                        math.log(0.03), math.log(4.0)
                    )
                )
            except Exception:  # noqa: BLE001
                pass
            try:
                covar = model.covar_module
                if hasattr(covar, "outputscale"):
                    shape = tuple(covar.outputscale.shape)
                    covar.outputscale = torch.exp(
                        torch.empty(shape, dtype=torch.double).uniform_(
                            math.log(0.05), math.log(5.0)
                        )
                    )
            except Exception:  # noqa: BLE001
                pass
            try:
                noise = model.likelihood.noise
                shape = tuple(noise.shape)
                model.likelihood.noise = torch.exp(
                    torch.empty(shape, dtype=torch.double).uniform_(
                        math.log(1e-5), math.log(1e-1)
                    )
                )
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _build_acquisition(deps: dict[str, Any], model: Any, best_f: float) -> Any:
        try:
            return deps["LogEI"](model=model, best_f=best_f, maximize=True)
        except Exception:  # noqa: BLE001
            return deps["EI"](model=model, best_f=best_f, maximize=True)

    # -- helpers ----------------------------------------------------------- #
    def _failure_tolerance(self) -> int:
        dimension = self._encoder.dim if self._encoder is not None else 1
        return max(2, int(math.ceil(0.5 * dimension)))

    def _successful(self) -> list[TrialObservation]:
        return [item for item in self._history if self._usable(item)]

    def _usable(self, observation: TrialObservation) -> bool:
        if not observation.success:
            return False
        value = observation.objectives.get(self._objective_name)
        return value is not None and math.isfinite(float(value))

    def _loss(self, observation: TrialObservation) -> float:
        value = float(observation.objectives[self._objective_name])
        return value if self._direction == ObjectiveDirection.MINIMIZE else -value

    @staticmethod
    def _identity(config: dict[str, Any]) -> str:
        return repr(sorted((str(key), repr(value)) for key, value in config.items()))

    def _is_new(self, config: dict[str, Any]) -> bool:
        identity = self._identity(config)
        return all(
            self._identity(item.suggestion.config) != identity for item in self._history
        )

    def _incumbent_config(self) -> dict[str, Any]:
        if self._state.best_config is not None:
            return dict(self._state.best_config)
        return self._space.defaults()

    def _random_suggestion(self, *, phase: str) -> TrialSuggestion:
        encoder = self._encoder
        assert encoder is not None
        for attempt in range(64):
            rng = np.random.default_rng(
                _stable_seed(self._seed, "random", phase, len(self._history), attempt)
            )
            try:
                config = encoder.decode(rng.random(encoder.dim))
            except Exception:  # noqa: BLE001
                continue
            if self._is_new(config):
                return TrialSuggestion(
                    config=config,
                    metadata={"algorithm": self.name, "phase": phase},
                )
        return TrialSuggestion(
            config=self._incumbent_config(),
            metadata={"algorithm": self.name, "phase": f"{phase}_duplicate"},
        )


def create_optimizer() -> Algorithm:
    return TrustRegionEi()


__all__ = ["create_optimizer", "TrustRegionEi"]
