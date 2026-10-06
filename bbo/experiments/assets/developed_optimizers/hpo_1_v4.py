"""Reusable black-box optimizer: warped Gaussian process with Expected Improvement.

The optimizer is a self-contained GP-EI implementation:

* every declared parameter is mapped to a unit cube with a geometry-aware warp
  (``log`` for log-scaled parameters, ``logit`` for bounded fractions, ``linear``
  otherwise) and categorical parameters become one-hot blocks;
* a Matern-5/2 ARD Gaussian process is fitted by maximum marginal likelihood
  with several restarts;
* Expected Improvement is maximised over a quasi-random candidate pool that is
  augmented with local perturbations of the incumbent and of the best points
  seen so far, followed by a short greedy refinement;
* every proposal is validated against the task search space and any internal
  failure degrades gracefully to a deterministic space-filling proposal.

Only ``numpy``, ``scipy`` and ``scikit-learn`` are required at proposal time.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from solver_support import Algorithm, Incumbent, TrialObservation, TrialSuggestion

__all__ = ["create_optimizer", "WarpedGpEi"]

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# search-space helpers
# --------------------------------------------------------------------------- #
def _param_kind(param: Any) -> str:
    """Classify a search-space parameter without relying on private modules."""

    name = type(param).__name__.lower()
    if hasattr(param, "choices"):
        return "categorical"
    if "string" in name or (hasattr(param, "pattern") and not hasattr(param, "low")):
        return "string"
    if "int" in name:
        return "int"
    low = getattr(param, "low", None)
    high = getattr(param, "high", None)
    if isinstance(low, int) and isinstance(high, int) and not isinstance(low, bool):
        return "int"
    return "float"


def _transform_for(param: Any, kind: str) -> str:
    if kind == "int":
        return "log" if bool(getattr(param, "log", False)) else "linear"
    if bool(getattr(param, "log", False)):
        return "log"
    low = float(getattr(param, "low", 0.0))
    high = float(getattr(param, "high", 1.0))
    if 0.0 < low and high < 1.0:
        return "logit"
    return "linear"


def _warp(value: float, transform: str) -> float:
    if transform == "log":
        return math.log(max(value, 1e-300))
    if transform == "logit":
        clipped = min(max(value, 1e-12), 1.0 - 1e-12)
        return math.log(clipped / (1.0 - clipped))
    return value


def _inverse_warp(value: float, transform: str) -> float:
    if transform == "log":
        return math.exp(min(max(value, -700.0), 700.0))
    if transform == "logit":
        if value >= 0.0:
            return 1.0 / (1.0 + math.exp(-min(value, 700.0)))
        exp_value = math.exp(max(value, -700.0))
        return exp_value / (1.0 + exp_value)
    return value


class _Encoder:
    """Encode/decode configurations as points in a warped unit cube."""

    def __init__(self, space: Any) -> None:
        self.space = space
        self.entries: list[dict[str, Any]] = []
        self.names: list[str] = []
        self.dim = 0
        self.has_unsupported = False
        for param in space:
            kind = _param_kind(param)
            self.names.append(param.name)
            if kind == "categorical":
                choices = list(getattr(param, "choices"))
                self.entries.append(
                    {
                        "param": param,
                        "kind": kind,
                        "start": self.dim,
                        "size": len(choices),
                        "choices": choices,
                    }
                )
                self.dim += len(choices)
            elif kind == "string":
                self.has_unsupported = True
                self.entries.append(
                    {"param": param, "kind": kind, "start": self.dim, "size": 0}
                )
            else:
                low = float(getattr(param, "low"))
                high = float(getattr(param, "high"))
                self.entries.append(
                    {
                        "param": param,
                        "kind": kind,
                        "start": self.dim,
                        "size": 1,
                        "low": low,
                        "high": high,
                        "transform": _transform_for(param, kind),
                    }
                )
                self.dim += 1

    # -- numeric <-> unit cube -------------------------------------------------
    @staticmethod
    def _to_unit(value: float, entry: dict[str, Any]) -> float:
        low = entry["low"]
        high = entry["high"]
        if high <= low:
            return 0.0
        transform = entry["transform"]
        warped = _warp(value, transform)
        warped_low = _warp(low, transform)
        warped_high = _warp(high, transform)
        if warped_high <= warped_low:
            return 0.0
        return float(np.clip((warped - warped_low) / (warped_high - warped_low), 0.0, 1.0))

    @staticmethod
    def _from_unit(unit: float, entry: dict[str, Any]) -> float:
        low = entry["low"]
        high = entry["high"]
        if high <= low:
            return low
        transform = entry["transform"]
        warped_low = _warp(low, transform)
        warped_high = _warp(high, transform)
        warped = warped_low + float(np.clip(unit, 0.0, 1.0)) * (warped_high - warped_low)
        return float(min(max(_inverse_warp(warped, transform), low), high))

    # -- config <-> vector -----------------------------------------------------
    def encode(self, config: dict[str, Any]) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=float)
        for entry in self.entries:
            name = entry["param"].name
            if entry["kind"] == "string":
                continue
            value = config[name]
            if entry["kind"] == "categorical":
                choices = entry["choices"]
                try:
                    index = choices.index(value)
                except ValueError:
                    index = 0
                vector[entry["start"] + index] = 1.0
            else:
                vector[entry["start"]] = self._to_unit(float(value), entry)
        return vector

    def decode(self, vector: np.ndarray) -> dict[str, Any]:
        config: dict[str, Any] = {}
        for entry in self.entries:
            param = entry["param"]
            if entry["kind"] == "string":
                config[param.name] = self._string_value(param)
            elif entry["kind"] == "categorical":
                block = vector[entry["start"] : entry["start"] + entry["size"]]
                index = int(np.argmax(block)) if block.size else 0
                index = min(max(index, 0), len(entry["choices"]) - 1)
                config[param.name] = entry["choices"][index]
            else:
                unit = float(vector[entry["start"]])
                physical = self._from_unit(unit, entry)
                if entry["kind"] == "int":
                    rounded = int(round(physical))
                    rounded = min(max(rounded, int(entry["low"])), int(entry["high"]))
                    config[param.name] = param.coerce(rounded)
                else:
                    config[param.name] = param.coerce(physical)
        return config

    @staticmethod
    def _string_value(param: Any) -> Any:
        default = getattr(param, "default", None)
        if default is not None:
            return param.coerce(default)
        return param.coerce("")

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        config: dict[str, Any] = {}
        for entry in self.entries:
            param = entry["param"]
            kind = entry["kind"]
            if kind == "string":
                config[param.name] = self._string_value(param)
            elif kind == "categorical":
                index = int(rng.integers(0, len(entry["choices"])))
                config[param.name] = entry["choices"][index]
            elif kind == "int":
                unit = float(rng.random())
                physical = self._from_unit(unit, entry)
                value = min(max(int(round(physical)), int(entry["low"])), int(entry["high"]))
                config[param.name] = param.coerce(value)
            else:
                unit = float(rng.random())
                config[param.name] = param.coerce(self._from_unit(unit, entry))
        return config


# --------------------------------------------------------------------------- #
# surrogate model
# --------------------------------------------------------------------------- #
def _fit_gp(x_values: np.ndarray, y_values: np.ndarray, seed: int) -> Any:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern

    dimension = x_values.shape[1]
    length_scale = np.full(dimension, 0.25, dtype=float)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=length_scale,
        length_scale_bounds=(5e-3, 1e2),
        nu=2.5,
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,
        normalize_y=True,
        n_restarts_optimizer=6,
        random_state=int(seed) % (2**31 - 1),
    )
    model.fit(x_values, y_values)
    return model


def _expected_improvement(
    model: Any,
    candidates: np.ndarray,
    best: float,
    xi: float = 0.0,
) -> np.ndarray:
    from scipy.stats import norm

    mean, std = model.predict(candidates, return_std=True)
    std = np.maximum(np.asarray(std, dtype=float), 1e-9)
    improvement = best - np.asarray(mean, dtype=float) - xi
    z_value = improvement / std
    return improvement * norm.cdf(z_value) + std * norm.pdf(z_value)


# --------------------------------------------------------------------------- #
# algorithm
# --------------------------------------------------------------------------- #
class WarpedGpEi(Algorithm):
    """GP-EI over a geometry-aware unit-cube encoding of the search space."""

    tr_init_radius = 0.5
    tr_min_dim = 4
    tr_expand = 1.4
    tr_shrink = 0.7
    tr_max_failures = 3
    tr_global_period = 4
    tr_anchor_scales = (0.08, 0.2)
    local_acquisition = "greedy"

    def __init__(
        self,
        *,
        n_sobol: int = 1024,
        n_random: int = 256,
        n_local_points: int = 3,
        local_scales: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2, 0.35),
        local_per_scale: int = 12,
        refine_steps: int = 24,
        refine_population: int = 16,
        top_candidates: int = 200,
        xi: float = 0.0,
    ) -> None:
        self.n_sobol = int(n_sobol)
        self.n_random = int(n_random)
        self.n_local_points = int(n_local_points)
        self.local_scales = tuple(float(scale) for scale in local_scales)
        self.local_per_scale = int(local_per_scale)
        self.refine_steps = int(refine_steps)
        self.refine_population = int(refine_population)
        self.top_candidates = int(top_candidates)
        self.xi = float(xi)
        self._spec: Any = None
        self._space: Any = None
        self._encoder: _Encoder | None = None
        self._seed = 0
        self._minimize = True
        self._primary = ""
        self._history: list[TrialObservation] = []
        self._seen: set[Any] = set()
        self._best: Incumbent | None = None

    # -- Algorithm protocol ----------------------------------------------------
    @property
    def name(self) -> str:
        return "warped_gp_ei"

    def setup(self, task_spec: Any, seed: int = 0, **kwargs: Any) -> None:
        del kwargs
        self._spec = task_spec
        self._space = task_spec.search_space
        self._seed = int(seed)
        objectives = list(getattr(task_spec, "objectives", []) or [])
        objective = getattr(task_spec, "primary_objective", None)
        if objective is None and objectives:
            objective = objectives[0]
        self._primary = str(getattr(objective, "name", ""))
        direction = getattr(objective, "direction", None)
        direction_text = str(getattr(direction, "value", direction)).lower()
        self._minimize = "max" not in direction_text
        self._encoder = _Encoder(self._space)
        self._history = []
        self._seen = set()
        self._best = None

    def replay(self, history: list[TrialObservation]) -> None:
        self._history = []
        self._seen = set()
        self._best = None
        for observation in history or []:
            self.tell(observation)

    def tell(self, observation: TrialObservation) -> None:
        self._history.append(observation)
        self._seen.add(self._identity(observation.suggestion.config))
        self.update_best_incumbent(observation)

    def incumbents(self) -> list[Incumbent]:
        return [self._best] if self._best is not None else []

    # -- helpers ---------------------------------------------------------------
    def update_best_incumbent(self, observation: TrialObservation) -> None:
        if not getattr(observation, "success", False):
            return
        if self._primary not in observation.objectives:
            return
        score = float(observation.objectives[self._primary])
        if not math.isfinite(score):
            return
        incumbent = Incumbent(
            config=dict(observation.suggestion.config),
            score=score,
            objectives=dict(observation.objectives),
            trial_id=observation.suggestion.trial_id,
            metadata={"algorithm": self.name},
        )
        if self._best is None:
            self._best = incumbent
            return
        current = float(self._best.score)
        if self._minimize and score < current:
            self._best = incumbent
        elif not self._minimize and score > current:
            self._best = incumbent

    def loss_of(self, observation: TrialObservation) -> float:
        return float(observation.objectives[self._primary])

    @staticmethod
    def _identity(config: dict[str, Any]) -> Any:
        try:
            return tuple(sorted((str(key), value) for key, value in config.items()))
        except TypeError:
            return tuple(
                sorted((str(key), repr(value)) for key, value in config.items())
            )

    def _rng(self, salt: int = 0) -> np.random.Generator:
        state = (self._seed * 1000003 + len(self._history) * 7919 + salt * 104729) % (2**32 - 1)
        return np.random.default_rng(state)

    def _training_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        assert self._encoder is not None
        rows: list[np.ndarray] = []
        targets: list[float] = []
        for observation in self._history:
            if not getattr(observation, "success", False):
                continue
            if self._primary not in observation.objectives:
                continue
            value = float(observation.objectives[self._primary])
            if not math.isfinite(value):
                continue
            config = observation.suggestion.config
            try:
                if any(self._encoder.names[index] not in config for index in range(len(self._encoder.names))):
                    continue
                row = self._encoder.encode(config)
            except Exception:
                continue
            rows.append(row)
            targets.append(value if self._minimize else -value)
        if not rows:
            return np.zeros((0, 0)), np.zeros(0)
        return np.vstack(rows), np.asarray(targets, dtype=float)

    def _validated(self, config: dict[str, Any]) -> dict[str, Any] | None:
        try:
            normalized = self._space.coerce_config(config, use_defaults=False)
        except Exception:
            return None
        return dict(normalized)

    # -- candidate generation --------------------------------------------------
    def _trust_region_state(self) -> tuple[float, int, int]:
        """Derive the trust-region radius, failure count and step index from history.

        The state is a pure function of the recorded observations, so replaying a
        history always reproduces identical proposals.  Initialization
        observations (when the host marks them) do not count as search steps.
        """

        radius = self.tr_init_radius
        failures = 0
        step = 0
        initialized = 0
        best: float | None = None
        for observation in self._history:
            if not getattr(observation, "success", False):
                continue
            if self._primary not in observation.objectives:
                continue
            value = float(observation.objectives[self._primary])
            if not math.isfinite(value):
                continue
            loss = value if self._minimize else -value
            improved = best is None or loss < best - 1e-12
            if improved:
                radius = min(radius * self.tr_expand, 0.8)
                failures = 0
                best = loss
            else:
                failures += 1
                radius *= self.tr_shrink
                if failures >= self.tr_max_failures:
                    radius = self.tr_init_radius
                    failures = 0
            if self._is_initialization(observation):
                initialized += 1
            else:
                step += 1
        if initialized == 0:
            # The host did not mark its initialization prefix: fall back to the
            # raw number of successful observations as the step counter.
            step = sum(
                1
                for observation in self._history
                if getattr(observation, "success", False)
                and self._primary in getattr(observation, "objectives", {})
            )
        return radius, failures, step

    @staticmethod
    def _is_initialization(observation: TrialObservation) -> bool:
        suggestion = getattr(observation, "suggestion", None)
        candidates = (
            getattr(observation, "metadata", None),
            getattr(suggestion, "metadata", None),
        )
        for source in candidates:
            if not isinstance(source, dict):
                continue
            if source.get("benchmark_initialization"):
                return True
            if str(source.get("phase", "")).lower() == "initialization":
                return True
            if "initialization_index" in source:
                return True
        return False

    def _quasi_random(self, rng: np.random.Generator, count: int, dimension: int) -> np.ndarray:
        if count <= 0 or dimension <= 0:
            return np.zeros((0, dimension))
        try:
            from scipy.stats.qmc import Sobol

            sobol = Sobol(dimension, scramble=True, seed=int(rng.integers(1, 2**31 - 1)))
            return np.asarray(sobol.random(count), dtype=float)
        except Exception:
            return rng.random((count, dimension))

    def _candidate_matrix(
        self,
        rng: np.random.Generator,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        assert self._encoder is not None
        dimension = self._encoder.dim
        if dimension <= 0:
            return np.zeros((1, 0)), "empty"
        order = np.argsort(y_train)
        local_mode = False
        radius = self.tr_init_radius
        if dimension >= self.tr_min_dim and len(order) > 0:
            radius, failures, step = self._trust_region_state()
            if failures < self.tr_max_failures and step % self.tr_global_period != 0:
                local_mode = True

        if local_mode:
            center = x_train[order[0]]
            half = radius / 2.0
            low = np.clip(center - half, 0.0, 1.0)
            high = np.clip(center + half, 0.0, 1.0)
            blocks: list[np.ndarray] = [
                low + self._quasi_random(rng, self.n_sobol, dimension) * (high - low)
            ]
            anchors = order[: min(self.n_local_points, len(order))]
            for anchor in anchors:
                base = x_train[anchor]
                for scale in self.tr_anchor_scales:
                    noise = rng.normal(0.0, max(radius, 1e-3) * scale, size=(self.local_per_scale, dimension))
                    blocks.append(np.clip(base + noise, 0.0, 1.0))
                span = rng.random((self.local_per_scale, dimension)) - 0.5
                blocks.append(np.clip(base + span * radius, 0.0, 1.0))
            # pattern-search probes: push single coordinates to a bound or a
            # fraction of the current best, which is where HPO optima often sit.
            for anchor in order[:1]:
                base = x_train[anchor]
                for index in range(dimension):
                    for step_value in (-base[index], 1.0 - base[index], radius, -radius):
                        probe = np.tile(base, (1, 1))
                        probe[0, index] = float(np.clip(base[index] + step_value, 0.0, 1.0))
                        blocks.append(probe)
            return np.vstack(blocks), "local"

        blocks = [self._quasi_random(rng, self.n_sobol, dimension)]
        if self.n_random > 0:
            blocks.append(rng.random((self.n_random, dimension)))
        anchors = order[: min(self.n_local_points, len(order))]
        for anchor in anchors:
            base = x_train[anchor]
            for scale in self.local_scales:
                if scale <= 0:
                    continue
                noise = rng.normal(0.0, scale, size=(self.local_per_scale, dimension))
                blocks.append(np.clip(base + noise, 0.0, 1.0))
        if len(order):
            best_point = x_train[order[0]]
            for index in range(dimension):
                for scale in (0.05, 0.2, 0.4):
                    step = np.zeros((2, dimension))
                    step[0, index] = scale
                    step[1, index] = -scale
                    blocks.append(np.clip(best_point + step, 0.0, 1.0))
        return np.vstack(blocks), "global"

    def _model_proposal(
        self,
        rng: np.random.Generator,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ) -> dict[str, Any] | None:
        model = _fit_gp(x_train, y_train, seed=int(rng.integers(1, 2**31 - 1)))
        best = float(np.min(y_train))
        candidates, mode = self._candidate_matrix(rng, x_train, y_train)
        if candidates.size == 0:
            return None
        if mode == "local" and self.local_acquisition == "greedy":
            mean = np.asarray(model.predict(candidates), dtype=float)
            scores = -mean
        elif mode == "local" and self.local_acquisition == "mixed" and candidates.shape[0] > 0:
            mean = np.asarray(model.predict(candidates), dtype=float)
            _, std = model.predict(candidates, return_std=True)
            scores = -mean / np.maximum(std, 1e-9)
        else:
            scores = _expected_improvement(model, candidates, best, self.xi)
        order = np.argsort(-np.asarray(scores, dtype=float))

        chosen_vector: np.ndarray | None = None
        chosen_score = -np.inf
        for index in order[: self.top_candidates]:
            config = self._validated(self._encoder.decode(candidates[index]))
            if config is None or self._identity(config) in self._seen:
                continue
            chosen_vector = np.asarray(candidates[index], dtype=float)
            chosen_score = float(scores[index])
            break
        if chosen_vector is None:
            return None

        # short greedy refinement of the acquisition surface
        if self.refine_steps > 0 and self.refine_population > 0:
            use_greedy = mode == "local" and self.local_acquisition in ("greedy", "mixed")

            def _score(values: np.ndarray) -> np.ndarray:
                if use_greedy:
                    mean = np.asarray(model.predict(values), dtype=float)
                    return -mean
                return _expected_improvement(model, values, best, self.xi)

            scale = 0.1
            vector = chosen_vector
            for _ in range(self.refine_steps):
                noise = rng.normal(0.0, scale, size=(self.refine_population, vector.shape[0]))
                trials = np.clip(vector + noise, 0.0, 1.0)
                values = _score(trials)
                position = int(np.argmax(values))
                if float(values[position]) > chosen_score:
                    vector = trials[position]
                    chosen_score = float(values[position])
                scale *= 0.9
            refined = self._validated(self._encoder.decode(vector))
            if refined is not None and self._identity(refined) not in self._seen:
                return refined
        config = self._validated(self._encoder.decode(chosen_vector))
        if config is not None and self._identity(config) not in self._seen:
            return config
        return None

    def _space_filling_proposal(self, rng: np.random.Generator) -> dict[str, Any]:
        assert self._encoder is not None
        for _ in range(64):
            config = self._validated(self._encoder.sample(rng))
            if config is not None and self._identity(config) not in self._seen:
                return config
        for _ in range(64):
            row = rng.random(self._encoder.dim) if self._encoder.dim else np.zeros(0)
            config = self._validated(self._encoder.decode(row))
            if config is not None:
                return config
        defaults = self._space.defaults()
        return dict(self._space.coerce_config(defaults, use_defaults=False))

    def ask(self) -> TrialSuggestion:
        suggestion: dict[str, Any] | None = None
        phase = "space_filling"
        rng = self._rng()
        try:
            x_train, y_train = self._training_arrays()
            if x_train.shape[0] >= 2 and x_train.shape[1] > 0 and not self._encoder.has_unsupported:
                model_rng = self._rng(salt=1)
                proposal = self._model_proposal(model_rng, x_train, y_train)
                if proposal is not None:
                    suggestion = proposal
                    phase = "acquisition"
        except Exception:
            suggestion = None
        if suggestion is None:
            try:
                suggestion = self._space_filling_proposal(rng)
            except Exception:
                suggestion = None
        if suggestion is None:
            suggestion = dict(self._space.defaults())
        return TrialSuggestion(
            config=dict(suggestion),
            metadata={
                "algorithm": self.name,
                "phase": phase,
                "training_points": int(len(self._history)),
                "search_dimension": int(self._encoder.dim) if self._encoder else 0,
            },
        )


def create_optimizer() -> Algorithm:
    """Return the optimizer instance used by the host."""

    return WarpedGpEi()
