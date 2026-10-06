"""Shared construction policy for comparable single-objective baselines.

The standalone benchmark runner and optimizer-as-tool bridge both use this
module. Keeping the defaults here prevents a tool backend from silently
becoming a second implementation of an algorithm with the same name.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..core.algo import Algorithm


COMPARABLE_BASELINE_BACKENDS = (
    "random",
    "sobol",
    "local_perturbation",
    "gp_ei",
    "tpe",
    "cma_es",
    "turbo",
)

_ALIASES = {
    "random_search": "random",
    "sobol_search": "sobol",
    "optuna_tpe": "tpe",
    "pycma": "cma_es",
    "botorch_turbo": "turbo",
    "gpei": "gp_ei",
    "gp_bo": "gp_ei",
}

# These are the defaults used by bbo.run.run_single_experiment. Model
# candidate budgets remain None here so the task-owned benchmark protocol
# resolves the shared dimension-scaled value for both the runner and tools.
COMPARABLE_BASELINE_DEFAULTS: dict[str, dict[str, Any]] = {
    "random": {},
    "sobol": {"scramble": True},
    "local_perturbation": {"jitter_fraction": 0.1},
    "gp_ei": {
        "pool_size": None,
        "startup_trials": 2,
        "xi": 0.0,
        "alpha": 1e-6,
        "n_restarts_optimizer": 0,
        "acquisition": "ei",
        "device": "cpu",
    },
    "tpe": {},
    "cma_es": {"sigma_fraction": 0.18, "popsize": 2},
    "turbo": {"startup_trials": 5, "n_candidates": None, "device": "cpu"},
}

_REGISTRY_NAMES = {
    "random": "random",
    "sobol": "sobol",
    "local_perturbation": "local_perturbation",
    "gp_ei": "gp_ei",
    "tpe": "optuna_tpe",
    "cma_es": "cma_es",
    "turbo": "turbo",
}


def normalize_comparable_backend(raw: str) -> str:
    """Return the canonical portfolio name for a baseline registry alias."""

    normalized = str(raw).strip().lower().replace("-", "_")
    normalized = _ALIASES.get(normalized, normalized)
    if normalized not in COMPARABLE_BASELINE_DEFAULTS:
        raise ValueError(f"Unknown comparable baseline backend {raw!r}.")
    return normalized


def comparable_baseline_kwargs(
    backend: str,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve one backend's canonical defaults plus explicit runner controls."""

    normalized = normalize_comparable_backend(backend)
    kwargs = dict(COMPARABLE_BASELINE_DEFAULTS[normalized])
    if overrides:
        kwargs.update(dict(overrides))
    return kwargs


def create_comparable_baseline(
    backend: str,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> Algorithm:
    """Instantiate the exact registered baseline used by optimizer tools."""

    normalized = normalize_comparable_backend(backend)
    # Import lazily because the registry also imports the Agent tool package.
    # Runtime lookup makes the registry the single source of implementation
    # identity instead of maintaining a second class table that can drift.
    from .registry import ALGORITHM_REGISTRY

    factory = ALGORITHM_REGISTRY[_REGISTRY_NAMES[normalized]].factory
    return factory(
        **comparable_baseline_kwargs(normalized, overrides=overrides)
    )


__all__ = [
    "COMPARABLE_BASELINE_BACKENDS",
    "COMPARABLE_BASELINE_DEFAULTS",
    "comparable_baseline_kwargs",
    "create_comparable_baseline",
    "normalize_comparable_backend",
]
