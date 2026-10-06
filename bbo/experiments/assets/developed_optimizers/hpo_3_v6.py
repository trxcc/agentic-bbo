"""Reference GP-EI geometry with a scheduled low-discrepancy coverage pass.

Version 0 (the reference) remains the strongest configuration observed: raw-axis
one-hot encoding, training-statistic input standardisation and plain expected
improvement.  Its proposals cluster in the region where its acquisition surface
peaks, so it can miss small pockets of the search space entirely (for example it
never probed small ``C`` on one seed even though the optimum lives there).

This version keeps that geometry and acquisition unchanged and adds:

* low-discrepancy (Halton) coverage probes early in the run, mapped through
  each parameter's declared warp (log / logit / linear) so the probes spread
  over the whole space instead of the raw hyper-box.  Front-loading the probes
  is deliberate: on these tasks a good configuration found at the first
  post-initialization trial is worth far more than the same configuration found
  near the budget limit, because the score averages the running improvement;
* a wider multi-start optimisation of the acquisition surface;
* a repair for the reference's duplicate fallback: when the acquisition
  optimiser can only return an already observed point, the next configuration
  comes from the best unseen member of a structured pool scored by the same
  posterior.

Any failure falls back to the reference behaviour, and spaces containing
categorical or string parameters keep the reference one-hot encoder.
"""

from __future__ import annotations

import itertools
import json
import math
import random

from solver_support import (
    CategoricalParam,
    FloatParam,
    GpEiAlgorithm,
    IntParam,
    StringParam,
    TrialSuggestion,
)

_COVERAGE_STRIDE = 3
_COVERAGE_LIMIT = 18
_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _identity(config):
    return json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _halton(index, base):
    result = 0.0
    fraction = 1.0
    value = int(index)
    while value > 0:
        fraction /= base
        result += fraction * (value % base)
        value //= base
    return result


def _logit(value):
    return math.log(value / (1.0 - value))


def _expit(value):
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    scaled = math.exp(value)
    return scaled / (1.0 + scaled)


def _map_unit(param, unit):
    """Map a unit-interval value to the parameter domain using its warp."""

    if isinstance(param, FloatParam):
        low, high = float(param.low), float(param.high)
        if bool(getattr(param, "log", False)) and low > 0.0:
            return math.exp(math.log(low) + unit * (math.log(high) - math.log(low)))
        if 0.0 < low and high < 1.0:
            return _expit(_logit(low) + unit * (_logit(high) - _logit(low)))
        return low + unit * (high - low)
    if isinstance(param, IntParam):
        low, high = int(param.low), int(param.high)
        if bool(getattr(param, "log", False)) and low > 0:
            raw = math.exp(math.log(low) + unit * (math.log(high) - math.log(low)))
        else:
            raw = low + unit * (high - low)
        return int(round(min(max(raw, low), high)))
    if isinstance(param, CategoricalParam):
        choices = list(param.choices)
        position = min(int(unit * len(choices)), len(choices) - 1)
        return choices[position]
    return None


def _extremes(param):
    if isinstance(param, (FloatParam, IntParam)):
        return [param.low, param.high]
    if isinstance(param, CategoricalParam):
        return list(param.choices)
    return []


def _perturb(param, value, rng, sigma):
    if isinstance(param, FloatParam):
        if bool(getattr(param, "log", False)) and float(value) > 0.0:
            candidate = float(value) * math.exp(sigma * rng.gauss(0.0, 1.0))
        else:
            candidate = float(value) + sigma * (float(param.high) - float(param.low)) * rng.gauss(0.0, 1.0)
        return min(max(candidate, float(param.low)), float(param.high))
    if isinstance(param, IntParam):
        span = float(param.high) - float(param.low)
        candidate = float(value) + sigma * span * rng.gauss(0.0, 1.0)
        return int(round(min(max(candidate, float(param.low)), float(param.high))))
    return value


class CoverageAugmentedReference(GpEiAlgorithm):
    """Reference GP-EI geometry with scheduled coverage probes."""

    @property
    def name(self) -> str:
        return "reference_gp_ei_coverage"

    def setup(self, task_spec, seed: int = 0, **kwargs) -> None:
        super().setup(task_spec, seed, **kwargs)
        if self._fixed_initialization is not None:
            self._init_length = len(self._fixed_initialization.configurations)
        else:
            self._init_length = int(self.startup_trials)

    # -------------------------------------------------------------- coverage
    def _coverage_config(self, index):
        space = self.require_search_space()
        params = list(space)
        for step in range(64):
            probe = index + step
            candidate = {}
            usable = True
            for position, param in enumerate(params):
                base = _PRIMES[position % len(_PRIMES)]
                value = _map_unit(param, _halton(probe, base))
                if value is None:
                    usable = False
                    break
                candidate[param.name] = value
            if not usable:
                return None
            try:
                coerced = space.coerce_config(candidate, use_defaults=False)
            except Exception:
                continue
            if _identity(coerced) not in self._seen_config_ids:
                return coerced
        return None

    def _coverage_ask(self):
        offset = len(self._history) - self._init_length
        if offset < 0 or offset % _COVERAGE_STRIDE != 0 or offset > _COVERAGE_LIMIT:
            return None
        if len(self._successful_history()) < 2:
            return None
        index = offset // _COVERAGE_STRIDE + 1
        try:
            config = self._coverage_config(index)
        except Exception:
            return None
        if config is None:
            return None
        self._seen_config_ids.add(_identity(config))
        return TrialSuggestion(
            config=config,
            metadata={
                "gp_ei_phase": "coverage_probe",
                "gp_ei_backend": "botorch",
                "gp_ei_training_points": len(self._successful_history()),
                "gp_ei_coverage_index": index,
                "gp_ei_coverage_offset": offset,
            },
        )

    # ------------------------------------------------------------------ pool
    def _structured_configs(self):
        space = self.require_search_space()
        params = list(space)
        incumbent = dict(self._best.config) if self._best is not None else space.defaults()
        points = []
        known = set()

        def add(candidate):
            try:
                coerced = space.coerce_config(candidate, use_defaults=False)
            except Exception:
                return
            key = _identity(coerced)
            if key in self._seen_config_ids or key in known:
                return
            known.add(key)
            points.append(coerced)

        numeric = all(isinstance(p, (FloatParam, IntParam)) for p in params)
        if numeric and 3 ** len(params) <= 2000:
            for combo in itertools.product((1, 2, 3), repeat=len(params)):
                candidate = dict(incumbent)
                for param, choice in zip(params, combo):
                    if choice == 2:
                        candidate[param.name] = param.low
                    elif choice == 3:
                        candidate[param.name] = param.high
                add(candidate)
        else:
            for param in params:
                for value in _extremes(param):
                    candidate = dict(incumbent)
                    candidate[param.name] = value
                    add(candidate)
        rng = random.Random(self._stable_int("pool_local", len(self._history)))
        for _ in range(64):
            candidate = {p.name: _perturb(p, incumbent[p.name], rng, 0.2) for p in params}
            add(candidate)
        return points

    def _best_structured(self):
        configs = self._structured_configs()
        if not configs:
            return None
        scored = self.evaluate_virtual_configs(configs, include_acquisition=True)
        best_score = None
        best_config = None
        for row in scored:
            score = row.get("acquisition_score")
            if score is None:
                continue
            if best_score is None or float(score) > best_score:
                best_score = float(score)
                best_config = row["config"]
        return best_config

    # ------------------------------------------------------------------- ask
    def ask(self) -> TrialSuggestion:
        if len(self._history) >= self._init_length:
            probe = self._coverage_ask()
            if probe is not None:
                return probe
        base = super().ask()
        if base.metadata.get("gp_ei_phase") != "acquisition_duplicate_fallback":
            return base
        try:
            config = self._best_structured()
        except Exception:
            return base
        if config is None:
            return base
        self._seen_config_ids.add(_identity(config))
        metadata = dict(base.metadata)
        metadata["gp_ei_phase"] = "structured_acquisition"
        return TrialSuggestion(config=config, metadata=metadata)


def create_optimizer():
    return CoverageAugmentedReference(
        pool_size=None,
        startup_trials=2,
        xi=0.0,
        alpha=1e-06,
        n_restarts_optimizer=0,
        acqf_num_restarts=16,
        max_acqf_attempts=8,
        candidate_attempt_multiplier=20,
        acquisition="ei",
        acquisition_beta=2.0,
        device="cpu",
        kernel="default",
        input_scaling="train_mean_std",
    )
