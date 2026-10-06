"""Optimizer-as-tool suggestions that never call the benchmark evaluator."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ....core import (
    CategoricalParam,
    FloatParam,
    IntParam,
    ObjectiveDirection,
    ObjectiveSpec,
    SearchSpace,
    StringParam,
    TaskSpec,
    TrialObservation,
    TrialStatus,
    TrialSuggestion,
)
from ...baseline_factory import (
    COMPARABLE_BASELINE_BACKENDS,
    comparable_baseline_kwargs,
    create_comparable_baseline,
    normalize_comparable_backend,
)
from ..serialization import dump_json, stable_config_identity
from ..optimizer_backend import StatefulOptimizerBackend
from .base import BaseBBOTool
from .context import BBOToolContext


OPTIMIZER_BACKENDS = COMPARABLE_BASELINE_BACKENDS
GP_ACQUISITIONS = ("ei", "logei", "noisy_logei", "pi", "ucb", "sobol")
OPTIMIZER_DECISION_TOOLS = frozenset(
    {"optimizer_suggest", "optimizer_score", "optimizer_portfolio_suggest"}
)
OPTIMIZER_ACTION_TOOLS = frozenset(
    {
        "assess_backend_suitability",
        "optimizer_suggest",
        "optimizer_recommend_backends",
        "optimizer_portfolio_suggest",
        "optimizer_predict",
        "optimizer_score",
        "optimizer_diagnostics",
        "optimizer_status",
        "optimizer_set_backend",
        "optimizer_set_bounds",
        "optimizer_set_acquisition",
        "optimizer_reset_policy",
    }
)


def execute_via_optimizer_backend(
    *,
    action: str,
    task_spec: TaskSpec,
    history: Iterable[TrialObservation],
    allowlist: Iterable[str],
    state_path: Path,
    seed: int,
    incumbent: dict[str, Any] | None,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Route an agent tool action through the evaluator-isolated backend."""

    backend = StatefulOptimizerBackend(allowlist=allowlist, state_path=state_path)
    return backend.execute(
        action,
        task_spec=task_spec,
        history=history,
        seed=seed,
        incumbent=incumbent,
        arguments=arguments,
    )



class OptimizerSuggestTool(BaseBBOTool):
    """Return one unevaluated suggestion from an allowlisted optimizer."""

    name = "optimizer_suggest"
    description = (
        "Return one candidate suggested by an allowlisted optimizer backend using only "
        "the current evaluated history. Call-specific bounds or around arguments are "
        "one-off probes and never change persistent policy. Around defaults to the "
        "incumbent and uses a radius expressed as a fraction of each domain width. "
        "This never calls the evaluator or consumes trial budget."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "backend": {"type": "string", "enum": list(OPTIMIZER_BACKENDS)},
            "q": {"type": "integer", "minimum": 1, "maximum": 3, "default": 1},
            "bounds": {"type": "object"},
            "around": {
                "type": "object",
                "properties": {
                    "center": {"type": "object"},
                    "radius": {
                        "type": "number",
                        "exclusiveMinimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.1,
                    },
                    "vary": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "radii": {"type": "object"},
                    "fix": {"type": "object"},
                    "choices": {"type": "object"},
                },
                "additionalProperties": False,
            },
            "options": {"type": "object"},
            "seed": {"type": "integer"},
        },
    }

    async def execute(
        self,
        context: BBOToolContext,
        backend: str | None = None,
        q: int = 1,
        bounds: dict[str, Any] | None = None,
        around: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        seed: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        policy = context.task_spec.metadata.get("optimizer_tool_policy")
        policy = dict(policy) if isinstance(policy, dict) else {}
        if policy.get("minimal_suggest") is True:
            if backend is None:
                raise ValueError(
                    "This condition requires optimizer_suggest(backend=...) with an explicit backend."
                )
            if (
                int(q) != 1
                or bounds is not None
                or around is not None
                or options not in (None, {})
                or seed is not None
            ):
                raise ValueError(
                    "This condition permits only optimizer_suggest(backend=...) with q=1."
                )
        arguments = {
            "backend": backend,
            "q": q,
            "bounds": bounds,
            "options": options,
        }
        if around is not None:
            arguments["around"] = around
        return execute_via_optimizer_backend(
            action=self.name,
            task_spec=context.task_spec,
            history=context.history,
            allowlist=context.optimizer_backend_allowlist,
            state_path=context.state_dir / "optimizer_tool_state.json",
            seed=context.seed if seed is None else int(seed),
            incumbent=None if context.incumbent is None else context.incumbent.config,
            arguments=arguments,
        )


class _OptimizerActionTool(BaseBBOTool):
    """Base class for stateful optimizer control calls."""

    async def execute(self, context: BBOToolContext, **kwargs: Any) -> dict[str, Any]:
        return execute_via_optimizer_backend(
            action=self.name,
            task_spec=context.task_spec,
            history=context.history,
            allowlist=context.optimizer_backend_allowlist,
            state_path=context.state_dir / "optimizer_tool_state.json",
            seed=context.seed,
            incumbent=None if context.incumbent is None else context.incumbent.config,
            arguments=kwargs,
        )



class AssessBackendSuitabilityTool(_OptimizerActionTool):
    """Return decision-neutral evidence for the allowlisted GP/TPE backends."""

    name = "assess_backend_suitability"
    description = (
        "Measure backend-specific evidence from evaluated history without recommending "
        "or selecting a backend. GP evidence reports cross-validated predictive fit and "
        "ranking accuracy; TPE evidence reports elite/non-elite parameter separability. "
        "This never proposes a candidate, calls the evaluator, or consumes trial budget."
    )
    parameters_schema = {"type": "object", "properties": {}}


class OptimizerRecommendBackendsTool(_OptimizerActionTool):
    name = "optimizer_recommend_backends"
    description = (
        "Return an explainable, non-binding ranking of enabled baseline backends "
        "from the current phase, coverage, stagnation, space type, and audited "
        "backend credit. The agent may ignore the recommendation."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "k": {
                "type": "integer",
                "minimum": 1,
                "maximum": len(OPTIMIZER_BACKENDS),
                "default": 3,
            }
        },
    }


class OptimizerPortfolioSuggestTool(_OptimizerActionTool):
    name = "optimizer_portfolio_suggest"
    description = (
        "Return a comparable menu from multiple registered baseline optimizers "
        "using the same real history, active bounds, objective, and benchmark "
        "candidate-budget policy. The menu never calls the evaluator."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "backends": {
                "type": "array",
                "items": {"type": "string", "enum": list(OPTIMIZER_BACKENDS)},
                "minItems": 1,
                "maxItems": len(OPTIMIZER_BACKENDS),
            },
            "q_per_backend": {
                "type": "integer",
                "minimum": 1,
                "maximum": 2,
                "default": 1,
            },
            "bounds": {"type": "object"},
        },
    }


class OptimizerPredictTool(_OptimizerActionTool):
    name = "optimizer_predict"
    description = "Predict primary-objective mean and uncertainty for 1-32 virtual configs."
    parameters_schema = {
        "type": "object",
        "properties": {
            "configs": {"type": "array", "items": {"type": "object"}, "maxItems": 32},
        },
        "required": ["configs"],
    }


class OptimizerScoreTool(_OptimizerActionTool):
    name = "optimizer_score"
    description = (
        "Score 1-32 agent-created virtual configs with the active GP acquisition. "
        "A successful call is a candidate-decision action."
    )
    parameters_schema = OptimizerPredictTool.parameters_schema


class OptimizerDiagnosticsTool(_OptimizerActionTool):
    name = "optimizer_diagnostics"
    description = "Return optimizer/surrogate health, data sufficiency, and active-policy diagnostics."
    parameters_schema = {"type": "object", "properties": {}}


class OptimizerStatusTool(_OptimizerActionTool):
    name = "optimizer_status"
    description = "Return persistent backend, active sub-bounds, acquisition, allowlist, and immutable objective."
    parameters_schema = {"type": "object", "properties": {}}


class OptimizerSetBackendTool(_OptimizerActionTool):
    name = "optimizer_set_backend"
    description = "Persist the backend used by later optimizer suggestions."
    parameters_schema = {
        "type": "object",
        "properties": {"backend": {"type": "string", "enum": list(OPTIMIZER_BACKENDS)}},
        "required": ["backend"],
    }


class OptimizerSetBoundsTool(_OptimizerActionTool):
    name = "optimizer_set_bounds"
    description = "Persist numeric search sub-bounds after evidence or a strong prior supports making the region the working domain; use optimizer_suggest(bounds=...) for one-off probes."
    parameters_schema = {
        "type": "object",
        "properties": {"bounds": {"type": "object"}, "evidence": {"type": "object", "properties": {"basis": {"type": "string", "enum": ["strong_prior", "trial_evidence", "surrogate_diagnostics"]}, "reason": {"type": "string"}, "reference_trials": {"type": "array", "items": {"type": "integer"}}, "diagnostic_call_id": {"type": "string"}}, "required": ["basis", "reason"]}},
        "required": ["bounds"],
    }


class OptimizerSetAcquisitionTool(_OptimizerActionTool):
    name = "optimizer_set_acquisition"
    description = (
        "Persist the GP acquisition policy: EI, LogEI, noisy LogEI, PI, UCB, or "
        "Sobol. Non-GP backends reject it."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": list(GP_ACQUISITIONS)},
            "parameters": {"type": "object"},
        },
        "required": ["name"],
    }


class OptimizerResetPolicyTool(_OptimizerActionTool):
    name = "optimizer_reset_policy"
    description = "Reset persistent backend, sub-bounds, and acquisition to run defaults."
    parameters_schema = {"type": "object", "properties": {}}


def create_optimizer_tools() -> list[BaseBBOTool]:
    """Return the complete single-objective Agentic Search control surface."""

    return [
        OptimizerSuggestTool(),
        OptimizerRecommendBackendsTool(),
        AssessBackendSuitabilityTool(),
        OptimizerPortfolioSuggestTool(),
        OptimizerPredictTool(),
        OptimizerScoreTool(),
        OptimizerDiagnosticsTool(),
        OptimizerStatusTool(),
        OptimizerSetBackendTool(),
        OptimizerSetBoundsTool(),
        OptimizerSetAcquisitionTool(),
        OptimizerResetPolicyTool(),
    ]


def execute_optimizer_action(
    *,
    action: str,
    task_spec: TaskSpec,
    history: Iterable[TrialObservation],
    allowlist: Iterable[str],
    state_path: Path,
    seed: int,
    incumbent: dict[str, Any] | None,
    arguments: dict[str, Any],
    gp_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one optimizer action using agent-visible single-objective state."""

    if len(task_spec.objectives) != 1:
        raise ValueError("Agentic optimizer tools support single-objective tasks only.")
    allowed = tuple(dict.fromkeys(normalize_backend(item) for item in allowlist))
    observations = list(history)
    state = _load_optimizer_state(state_path, task_spec=task_spec)

    if action == "assess_backend_suitability":
        return _assess_backend_suitability(
            task_spec=task_spec,
            observations=observations,
            allowed=allowed,
            seed=int(seed),
        )
    if action == "optimizer_reset_policy":
        state = _default_optimizer_state(task_spec)
        _save_optimizer_state(state_path, state)
        return _optimizer_status(task_spec, observations, allowed, state)
    if action == "optimizer_set_backend":
        state["backend"] = _require_allowed_backend(arguments.get("backend"), allowed)
        _save_optimizer_state(state_path, state)
        return _optimizer_status(task_spec, observations, allowed, state)
    if action == "optimizer_set_bounds":
        state["bounds"] = _validate_bounds(task_spec.search_space, arguments.get("bounds"))
        _save_optimizer_state(state_path, state)
        return _optimizer_status(task_spec, observations, allowed, state)
    if action == "optimizer_set_acquisition":
        if state.get("backend") not in {None, "gp_ei"}:
            raise ValueError("Acquisition control is only valid for the gp_ei backend.")
        state["acquisition"] = _normalize_acquisition(
            arguments.get("name"), arguments.get("parameters")
        )
        _save_optimizer_state(state_path, state)
        return _optimizer_status(task_spec, observations, allowed, state)
    if action == "optimizer_status":
        return _optimizer_status(task_spec, observations, allowed, state)
    if action == "optimizer_diagnostics":
        payload = _optimizer_status(task_spec, observations, allowed, state)
        active_bounds = dict(state.get("bounds") or {})
        bounded_spec, _ = _bounded_problem(task_spec, observations, active_bounds)
        acquisition = _normalize_acquisition(
            (state.get("acquisition") or {}).get("name", "logei"),
            (state.get("acquisition") or {}).get("parameters"),
        )
        parameters = dict(acquisition.get("parameters") or {})
        algorithm = create_comparable_baseline(
            "gp_ei",
            overrides={
                **(gp_overrides or {}),
                "acquisition": acquisition["name"],
                "xi": float(parameters.get("xi", 0.0)),
                "acquisition_beta": float(parameters.get("beta", 2.0)),
            },
        )
        algorithm.setup(task_spec, seed=int(seed))
        algorithm.replay(observations)
        payload["diagnostics"] = algorithm.diagnose()
        payload["diagnostics"].update(
            unique_configs=len(
                {stable_config_identity(item.suggestion.config) for item in observations}
            ),
            full_history_size=len(observations),
            bounded_history_size=len(observations),
            candidate_region=_search_space_region(bounded_spec.search_space),
        )
        return payload
    if action == "optimizer_recommend_backends":
        return _recommend_backend_payload(
            task_spec,
            observations,
            allowed,
            k=min(max(int(arguments.get("k", 3)), 1), len(allowed)),
        )
    if action == "optimizer_portfolio_suggest":
        return _portfolio_suggest(
            task_spec=task_spec,
            observations=observations,
            allowed=allowed,
            state=state,
            seed=int(seed),
            incumbent=incumbent,
            arguments=arguments,
        )
    if action in {"optimizer_predict", "optimizer_score"}:
        active_bounds = dict(state.get("bounds") or {})
        configs = _coerce_virtual_configs(
            task_spec.search_space,
            arguments.get("configs"),
            {},
        )
        acquisition = _normalize_acquisition(
            (state.get("acquisition") or {}).get("name", "logei"),
            (state.get("acquisition") or {}).get("parameters"),
        )
        bounded_spec, _ = _bounded_problem(
            task_spec, observations, active_bounds
        )
        parameters = dict(acquisition.get("parameters") or {})
        overrides = {
            **(gp_overrides or {}),
            "acquisition": acquisition["name"],
            "xi": float(parameters.get("xi", 0.0)),
            "acquisition_beta": float(parameters.get("beta", 2.0)),
        }
        algorithm = create_comparable_baseline("gp_ei", overrides=overrides)
        algorithm.setup(task_spec, seed=int(seed))
        algorithm.replay(observations)
        predictions = algorithm.evaluate_virtual_configs(
            configs,
            include_acquisition=action == "optimizer_score",
        )
        return {
            "action": action,
            "backend": "gp_ei",
            "acquisition": acquisition,
            "predictions": predictions,
            "implementation": "shared_baseline_registry",
            "algorithm_class": (
                f"{type(algorithm).__module__}.{type(algorithm).__name__}"
            ),
            "history_size": len(observations),
            "backend_history_size": len(observations),
            "candidate_region": _search_space_region(bounded_spec.search_space),
            "budget_consumed": False,
            "evaluator_called": False,
        }
    if action != "optimizer_suggest":
        raise ValueError(f"Unknown optimizer action {action!r}.")

    backend = _require_allowed_backend(
        arguments.get("backend")
        or state.get("backend")
        or (allowed[0] if allowed else None),
        allowed,
    )
    raw_bounds = arguments.get("bounds")
    raw_around = arguments.get("around")
    if raw_bounds is not None and raw_around is not None:
        raise ValueError("optimizer_suggest accepts either bounds or around, not both.")
    bounds = (
        _validate_bounds(task_spec.search_space, raw_bounds)
        if raw_bounds is not None
        else dict(state.get("bounds") or {})
    )
    options = dict(arguments.get("options") or {})
    unknown_options = sorted(
        set(options) - {"acquisition", "acquisition_parameters"}
    )
    if unknown_options:
        raise ValueError(
            "Optimizer tools do not permit baseline hyperparameter overrides; "
            f"unsupported options={unknown_options!r}."
        )
    acquisition = _normalize_acquisition(
        options.get(
            "acquisition",
            (state.get("acquisition") or {}).get("name", "logei"),
        ),
        options.get(
            "acquisition_parameters",
            (state.get("acquisition") or {}).get("parameters"),
        ),
    )
    if backend != "gp_ei" and (
        "acquisition" in options or "acquisition_parameters" in options
    ):
        raise ValueError(
            f"Backend {backend!r} does not support acquisition control; use gp_ei."
        )
    q = min(max(int(arguments.get("q", 1)), 1), 3)
    bounded_spec, bounded_history = _bounded_problem(task_spec, observations, bounds)
    candidate_spec = bounded_spec
    candidate_history = bounded_history
    around_payload = None
    if raw_around is not None:
        candidate_spec, candidate_history, around_payload = _around_problem(
            bounded_spec,
            observations,
            incumbent=incumbent,
            raw=raw_around,
        )
    menu: list[dict[str, Any]] = []
    menu_ids: set[str] = set()
    started = time.monotonic()
    for index in range(q):
        suggestion = suggest_candidate(
            task_spec=candidate_spec,
            history=candidate_history,
            backend=backend,
            allowlist=allowed,
            seed=int(seed) + index * 1009,
            incumbent=None,
            excluded=menu_ids,
            acquisition=acquisition,
            options=options,
            training_task_spec=task_spec if backend == "gp_ei" else None,
            training_history=observations if backend == "gp_ei" else None,
            gp_overrides=gp_overrides,
            candidate_search_space=(
                candidate_spec.search_space if backend == "gp_ei" else None
            ),
        )
        menu_ids.add(str(suggestion["identity"]))
        menu.append(
            {
                "candidate": suggestion["candidate"],
                "identity": suggestion["identity"],
                "duplicate": suggestion["duplicate"],
                "menu_index": index,
                "suggestion_metadata": suggestion["suggestion_metadata"],
            }
        )
    result = {
        "backend": backend,
        "allowed_backends": list(allowed),
        "q": q,
        "candidates": menu,
        "active_bounds": bounds,
        "active_region": _search_space_region(candidate_spec.search_space),
        "around": around_payload,
        "bounds_scope": (
            "one_off_around"
            if raw_around is not None
            else "one_off"
            if raw_bounds is not None
            else "persistent"
        ),
        "persistent_bounds": dict(state.get("bounds") or {}),
        "acquisition": acquisition if backend == "gp_ei" else None,
        "history_size": len(observations),
        "backend_history_size": (
            len(observations) if backend == "gp_ei" else len(candidate_history)
        ),
        "duration_ms": round((time.monotonic() - started) * 1000.0, 3),
        "budget_consumed": False,
        "evaluator_called": False,
    }
    if menu:
        result.update(
            {
                "candidate": menu[0]["candidate"],
                "identity": menu[0]["identity"],
                "duplicate": menu[0]["duplicate"],
                "suggestion_metadata": menu[0]["suggestion_metadata"],
            }
        )
    return result


def _assess_backend_suitability(
    *,
    task_spec: TaskSpec,
    observations: list[TrialObservation],
    allowed: tuple[str, ...],
    seed: int,
    gamma: float = 0.25,
) -> dict[str, Any]:
    """Compute GP predictive and TPE quantile-structure evidence only."""

    primary = task_spec.primary_objective
    scored = [
        item
        for item in observations
        if item.success and primary.name in item.objectives
    ]
    payload: dict[str, Any] = {
        "action": "assess_backend_suitability",
        "decision_neutral": True,
        "successful_history_size": len(scored),
        "assessed_backends": [
            backend for backend in allowed if backend in {"gp_ei", "tpe"}
        ],
    }

    if "gp_ei" in allowed:
        compatible, reason = _backend_compatibility(task_spec.search_space, "gp_ei")
        if not compatible:
            payload["gp_ei"] = {
                "cv_r2": None,
                "ranking_accuracy": None,
                "status": "unsupported",
                "reason": reason,
            }
        else:
            algorithm = create_comparable_baseline(
                "gp_ei", overrides={"acquisition": "ei"}
            )
            algorithm.setup(task_spec, seed=seed)
            algorithm.replay(scored)
            diagnostic = algorithm.cross_validate_landscape(max_folds=3)
            payload["gp_ei"] = {
                "cv_r2": diagnostic.get("global_gp_cv_r2"),
                "ranking_accuracy": diagnostic.get(
                    "global_gp_ranking_accuracy"
                ),
                "cv_folds": int(diagnostic.get("folds", 0)),
                "minimum_sample_count": int(
                    diagnostic.get("minimum_sample_count", 0)
                ),
                "status": diagnostic.get("status", "unknown"),
            }

    if "tpe" in allowed:
        payload["tpe"] = _tpe_suitability_evidence(
            task_spec=task_spec,
            observations=scored,
            gamma=gamma,
        )

    payload.update(
        implementation="shared_baseline_registry_and_read_only_diagnostics",
        budget_consumed=False,
        evaluator_called=False,
    )
    return payload


def _tpe_suitability_evidence(
    *,
    task_spec: TaskSpec,
    observations: list[TrialObservation],
    gamma: float,
) -> dict[str, Any]:
    primary = task_spec.primary_objective
    sample_count = len(observations)
    minimum_sample_count = 8
    base: dict[str, Any] = {
        "elite_separability": None,
        "parameter_separability": {},
        "gamma": float(gamma),
        "sample_count": sample_count,
        "elite_count": 0,
        "non_elite_count": 0,
        "status": "insufficient_data",
    }
    if sample_count < minimum_sample_count:
        base["minimum_sample_count"] = minimum_sample_count
        return base

    reverse = primary.direction == ObjectiveDirection.MAXIMIZE
    ordered = sorted(
        observations,
        key=lambda item: float(item.objectives[primary.name]),
        reverse=reverse,
    )
    elite_count = min(
        max(2, int(math.ceil(float(gamma) * sample_count))),
        sample_count - 2,
    )
    elite = ordered[:elite_count]
    non_elite = ordered[elite_count:]
    parameter_separability: dict[str, float] = {}
    parameter_methods: dict[str, str] = {}
    for parameter in task_spec.search_space:
        elite_values = [item.suggestion.config[parameter.name] for item in elite]
        non_elite_values = [
            item.suggestion.config[parameter.name] for item in non_elite
        ]
        if isinstance(parameter, (FloatParam, IntParam)):
            statistic = _empirical_ks_statistic(elite_values, non_elite_values)
            method = "ks"
        elif isinstance(parameter, (CategoricalParam, StringParam)):
            statistic = _total_variation_distance(elite_values, non_elite_values)
            method = "total_variation"
        else:  # pragma: no cover - all current ParameterSpec variants are covered.
            continue
        parameter_separability[parameter.name] = round(float(statistic), 12)
        parameter_methods[parameter.name] = method

    scores = list(parameter_separability.values())
    base.update(
        elite_separability=(
            None if not scores else round(float(np.mean(scores)), 12)
        ),
        parameter_separability=parameter_separability,
        parameter_methods=parameter_methods,
        elite_count=len(elite),
        non_elite_count=len(non_elite),
        status="usable" if scores else "unsupported",
    )
    return base


def _empirical_ks_statistic(
    left: Iterable[float | int], right: Iterable[float | int]
) -> float:
    left_values = np.sort(np.asarray(list(left), dtype=float))
    right_values = np.sort(np.asarray(list(right), dtype=float))
    if left_values.size == 0 or right_values.size == 0:
        return 0.0
    support = np.unique(np.concatenate((left_values, right_values)))
    left_cdf = np.searchsorted(left_values, support, side="right") / left_values.size
    right_cdf = np.searchsorted(right_values, support, side="right") / right_values.size
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _total_variation_distance(left: Iterable[Any], right: Iterable[Any]) -> float:
    left_values = [
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        for value in left
    ]
    right_values = [
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        for value in right
    ]
    if not left_values or not right_values:
        return 0.0
    support = set(left_values) | set(right_values)
    return 0.5 * sum(
        abs(
            left_values.count(value) / len(left_values)
            - right_values.count(value) / len(right_values)
        )
        for value in support
    )


def _portfolio_suggest(
    *,
    task_spec: TaskSpec,
    observations: list[TrialObservation],
    allowed: tuple[str, ...],
    state: dict[str, Any],
    seed: int,
    incumbent: dict[str, Any] | None,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    recommendation = _recommend_backend_payload(
        task_spec, observations, allowed, k=min(3, len(allowed))
    )
    raw_backends = arguments.get("backends")
    if raw_backends is None:
        selected = [
            str(item["backend"]) for item in recommendation["recommended"]
        ]
    else:
        if not isinstance(raw_backends, (list, tuple)) or not raw_backends:
            raise TypeError("backends must be a non-empty list when provided.")
        selected = list(
            dict.fromkeys(
                _require_allowed_backend(item, allowed)
                for item in raw_backends
            )
        )
    q_per_backend = min(max(int(arguments.get("q_per_backend", 1)), 1), 2)
    raw_bounds = arguments.get("bounds")
    bounds = (
        _validate_bounds(task_spec.search_space, raw_bounds)
        if raw_bounds is not None
        else dict(state.get("bounds") or {})
    )
    bounded_spec, bounded_history = _bounded_problem(
        task_spec, observations, bounds
    )
    bounded_anchor = _project_config_to_space(
        bounded_spec.search_space, incumbent
    )
    acquisition = _normalize_acquisition(
        (state.get("acquisition") or {}).get("name", "logei"),
        (state.get("acquisition") or {}).get("parameters"),
    )
    menu: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    excluded: set[str] = set()
    started = time.monotonic()
    for backend in selected:
        compatible, reason = _backend_compatibility(
            bounded_spec.search_space, backend
        )
        if not compatible:
            errors.append({"backend": backend, "error": reason})
            continue
        for candidate_index in range(q_per_backend):
            try:
                suggestion = suggest_candidate(
                    task_spec=bounded_spec,
                    history=bounded_history,
                    backend=backend,
                    allowlist=allowed,
                    seed=int(seed) + candidate_index * 1009,
                    incumbent=bounded_anchor,
                    excluded=excluded,
                    acquisition=acquisition,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "backend": backend,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                break
            identity = str(suggestion["identity"])
            excluded.add(identity)
            menu.append(
                {
                    "portfolio_candidate_id": f"{backend}:{identity}",
                    "backend": backend,
                    "backend_candidate_index": candidate_index,
                    "candidate": suggestion["candidate"],
                    "identity": identity,
                    "duplicate": suggestion["duplicate"],
                    "comparison": _candidate_geometry(
                        bounded_spec.search_space,
                        suggestion["candidate"],
                        bounded_history,
                        bounded_anchor,
                    ),
                    "suggestion_metadata": suggestion["suggestion_metadata"],
                }
            )
    return {
        "action": "optimizer_portfolio_suggest",
        "selection_policy": "agent_decides",
        "auto_selected": False,
        "requested_backends": selected,
        "q_per_backend": q_per_backend,
        "candidates": menu,
        "backend_errors": errors,
        "recommendation": recommendation,
        "backend_credit": _backend_credit(task_spec, observations, allowed),
        "active_bounds": bounds,
        "bounds_scope": "one_off" if raw_bounds is not None else "persistent",
        "persistent_bounds": dict(state.get("bounds") or {}),
        "history_size": len(observations),
        "backend_history_size": len(bounded_history),
        "duration_ms": round((time.monotonic() - started) * 1000.0, 3),
        "budget_consumed": False,
        "evaluator_called": False,
    }


def _recommend_backend_payload(
    task_spec: TaskSpec,
    history: list[TrialObservation],
    allowed: tuple[str, ...],
    *,
    k: int,
) -> dict[str, Any]:
    primary = task_spec.primary_objective
    scored = [
        item
        for item in history
        if item.success and primary.name in item.objectives
    ]
    dimension = len(task_spec.search_space)
    utilities = [
        (
            -float(item.objectives[primary.name])
            if primary.direction == ObjectiveDirection.MINIMIZE
            else float(item.objectives[primary.name])
        )
        for item in scored
    ]
    best = -math.inf
    last_improvement = -1
    for index, value in enumerate(utilities):
        if value > best:
            best = value
            last_improvement = index
    stagnation = 0 if not utilities else len(utilities) - 1 - last_improvement
    early = len(scored) < max(10, 2 * dimension)
    has_categorical = any(
        isinstance(param, CategoricalParam) for param in task_spec.search_space
    )
    credits = _backend_credit(task_spec, history, allowed)
    base_scores = {
        "random": 0.20,
        "sobol": 0.35,
        "local_perturbation": 0.25,
        "gp_ei": 0.45,
        "tpe": 0.45,
        "cma_es": 0.25,
        "turbo": 0.30,
    }
    rankings: list[dict[str, Any]] = []
    for backend in allowed:
        compatible, compatibility_reason = _backend_compatibility(
            task_spec.search_space, backend
        )
        score = base_scores.get(backend, 0.0)
        reasons: list[str] = []
        if early:
            early_bonus = {"sobol": 0.50, "random": 0.25, "tpe": 0.20}
            if backend in early_bonus:
                score += early_bonus[backend]
                reasons.append("early phase favors broad or robust coverage")
        else:
            model_bonus = {"gp_ei": 0.40, "tpe": 0.30}
            if backend in model_bonus:
                score += model_bonus[backend]
                reasons.append("history is large enough for model-based search")
        if stagnation >= max(5, dimension // 2):
            stagnation_bonus = {
                "local_perturbation": 0.45,
                "turbo": 0.40,
                "cma_es": 0.35,
                "sobol": 0.15,
            }
            if backend in stagnation_bonus:
                score += stagnation_bonus[backend]
                reasons.append("stagnation favors refinement or renewed coverage")
        if len(scored) >= max(20, 2 * dimension):
            mature_bonus = {"turbo": 0.25, "cma_es": 0.20}
            if backend in mature_bonus:
                score += mature_bonus[backend]
                reasons.append("mature history supports a local search phase")
        if has_categorical and backend == "tpe":
            score += 0.35
            reasons.append("mixed or categorical search space")
        credit = credits.get(backend, {})
        attributed = int(credit.get("attributed_trials", 0))
        improvements = int(credit.get("improvements", 0))
        if attributed:
            score += 0.40 * improvements / attributed
            reasons.append(
                f"audited credit: {improvements}/{attributed} improvements"
            )
        if not compatible:
            score = -math.inf
            reasons = [compatibility_reason]
        elif not reasons:
            reasons.append("general-purpose portfolio option")
        rankings.append(
            {
                "backend": backend,
                "score": None if not math.isfinite(score) else round(score, 6),
                "compatible": compatible,
                "reasons": reasons,
            }
        )
    rankings.sort(
        key=lambda item: (
            item["score"] is None,
            -float(item["score"] or 0.0),
            item["backend"],
        )
    )
    recommended = [item for item in rankings if item["compatible"]][
        : min(max(k, 1), len(rankings))
    ]
    return {
        "action": "optimizer_recommend_backends",
        "advisory_only": True,
        "phase": "early" if early else "model_based",
        "successful_history_size": len(scored),
        "dimension": dimension,
        "stagnation_trials": stagnation,
        "recommended": recommended,
        "ranking": rankings,
        "budget_consumed": False,
        "evaluator_called": False,
    }


def _backend_credit(
    task_spec: TaskSpec,
    history: list[TrialObservation],
    allowed: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    credit = {
        backend: {
            "attributed_trials": 0,
            "adopted_trials": 0,
            "refined_trials": 0,
            "improvements": 0,
            "cumulative_utility_improvement": 0.0,
        }
        for backend in allowed
    }
    primary = task_spec.primary_objective
    best = -math.inf
    for observation in history:
        if not observation.success or primary.name not in observation.objectives:
            continue
        utility = float(observation.objectives[primary.name])
        if primary.direction == ObjectiveDirection.MINIMIZE:
            utility = -utility
        previous_best = best
        best = max(best, utility)
        action = observation.suggestion.metadata.get("search_action")
        optimizer = action.get("optimizer") if isinstance(action, dict) else None
        if not isinstance(optimizer, dict):
            continue
        try:
            backend = normalize_backend(str(optimizer.get("backend")))
        except (TypeError, ValueError):
            continue
        if backend not in credit:
            continue
        relationship = str(optimizer.get("relationship", "")).strip().lower()
        item = credit[backend]
        item["attributed_trials"] += 1
        if relationship == "adopt":
            item["adopted_trials"] += 1
        elif relationship == "refine":
            item["refined_trials"] += 1
        if previous_best > -math.inf and utility > previous_best:
            item["improvements"] += 1
            item["cumulative_utility_improvement"] += utility - previous_best
    for item in credit.values():
        item["cumulative_utility_improvement"] = round(
            float(item["cumulative_utility_improvement"]), 12
        )
    return credit


def _backend_compatibility(
    space: SearchSpace, backend: str
) -> tuple[bool, str]:
    has_string = any(isinstance(param, StringParam) for param in space)
    has_categorical = any(
        isinstance(param, CategoricalParam) for param in space
    )
    if has_string and backend != "random":
        return False, "open string parameters are unsupported by this baseline"
    if has_categorical and backend in {"sobol", "turbo"}:
        return False, "this registered baseline is numeric-only"
    return True, "supported"


def _candidate_geometry(
    space: SearchSpace,
    candidate: dict[str, Any],
    history: list[TrialObservation],
    incumbent: dict[str, Any] | None,
) -> dict[str, float | None]:
    vector = np.asarray(_encode_config(space, candidate), dtype=float)
    incumbent_distance = None
    if incumbent is not None:
        incumbent_vector = np.asarray(
            _encode_config(space, incumbent), dtype=float
        )
        incumbent_distance = float(np.linalg.norm(vector - incumbent_vector))
    nearest_distance = None
    if history:
        history_vectors = np.asarray(
            [
                _encode_config(space, item.suggestion.config)
                for item in history
            ],
            dtype=float,
        )
        nearest_distance = float(
            np.min(np.linalg.norm(history_vectors - vector, axis=1))
        )
    return {
        "normalized_distance_to_incumbent": incumbent_distance,
        "nearest_normalized_history_distance": nearest_distance,
    }


def suggest_candidate(
    *,
    task_spec: TaskSpec,
    history: Iterable[TrialObservation],
    backend: str,
    allowlist: Iterable[str],
    seed: int,
    incumbent: dict[str, Any] | None = None,
    excluded: set[str] | None = None,
    acquisition: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    training_task_spec: TaskSpec | None = None,
    training_history: Iterable[TrialObservation] | None = None,
    candidate_search_space: SearchSpace | None = None,
    gp_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate one candidate without evaluating it."""

    normalized = normalize_backend(backend)
    allowed = tuple(normalize_backend(item) for item in allowlist)
    if normalized not in allowed:
        raise PermissionError(
            f"Optimizer backend {normalized!r} is not enabled for this condition; "
            f"allowed={list(allowed)!r}."
        )
    candidate_observations = list(history)
    observations = (
        list(training_history)
        if training_history is not None
        else candidate_observations
    )
    started = time.monotonic()
    prepared_spec = training_task_spec or task_spec
    overrides: dict[str, Any] = {}
    if normalized == "gp_ei":
        active_acquisition = acquisition or {"name": "logei", "parameters": {}}
        params = dict(active_acquisition.get("parameters") or {})
        overrides = {
            **(gp_overrides or {}),
            "acquisition": str(active_acquisition.get("name", "logei")),
            "xi": float(params.get("xi", 0.0)),
            "acquisition_beta": float(params.get("beta", 2.0)),
        }
    algorithm = create_comparable_baseline(normalized, overrides=overrides)
    algorithm.setup(prepared_spec, seed=int(seed))
    if normalized == "gp_ei" and candidate_search_space is not None:
        set_candidate_search_space = getattr(
            algorithm, "set_candidate_search_space", None
        )
        if not callable(set_candidate_search_space):
            raise TypeError(
                "The registered gp_ei backend cannot restrict candidate generation."
            )
        set_candidate_search_space(candidate_search_space)
    _restore_baseline_algorithm(algorithm, normalized, observations)
    suggestion = algorithm.ask()
    suggestion.metadata.update(
        {
            "optimizer_tool_backend": normalized,
            "optimizer_tool_implementation": "shared_baseline_registry",
            "optimizer_tool_algorithm_class": (
                f"{type(algorithm).__module__}.{type(algorithm).__name__}"
            ),
            "optimizer_tool_baseline_kwargs": comparable_baseline_kwargs(
                normalized, overrides=overrides
            ),
        }
    )
    config = prepared_spec.search_space.coerce_config(
        suggestion.config, use_defaults=False
    )
    if candidate_search_space is not None:
        candidate_search_space.validate_config(config)
    seen = {stable_config_identity(item.suggestion.config) for item in observations}
    seen.update(excluded or ())
    identity = stable_config_identity(config)
    duration_ms = round((time.monotonic() - started) * 1000.0, 3)
    return {
        "backend": normalized,
        "allowed_backends": list(allowed),
        "candidate": config,
        "identity": identity,
        "duplicate": identity in seen,
        "history_size": len(observations),
        "successful_history_size": sum(1 for item in observations if item.success),
        "suggestion_metadata": dict(suggestion.metadata),
        "duration_ms": duration_ms,
        "budget_consumed": False,
        "evaluator_called": False,
    }


def suggest_from_workspace(
    *,
    workspace_dir: Path,
    config: dict[str, Any],
    backend: str,
    seed: int | None = None,
) -> dict[str, Any]:
    """Rebuild the agent-visible task and history from workspace files."""

    space_payload = _read_json(workspace_dir / "space.json")
    objective_payload = _read_json(workspace_dir / "objective.json")
    search_space = _search_space_from_schema(space_payload.get("parameters", []))
    objective_name = str(objective_payload["name"])
    direction = ObjectiveDirection(str(objective_payload["direction"]))
    task_spec = TaskSpec(
        name=str(config.get("optimizer_agent_task_id") or "agent_visible_task"),
        search_space=search_space,
        objectives=(ObjectiveSpec(objective_name, direction),),
        max_evaluations=max(1, int(config.get("optimizer_max_evaluations") or 1)),
        metadata=dict(config.get("optimizer_task_metadata") or {}),
    )
    history = _history_from_workspace(workspace_dir / "history.jsonl")
    incumbent_payload = _read_json(workspace_dir / "incumbent.json")
    incumbent = incumbent_payload.get("config")
    return suggest_candidate(
        task_spec=task_spec,
        history=history,
        backend=backend,
        allowlist=config.get("optimizer_backend_allowlist") or (),
        seed=int(config.get("seed", 0) if seed is None else seed),
        incumbent=incumbent if isinstance(incumbent, dict) else None,
    )


def _restore_baseline_algorithm(
    algorithm: Any,
    backend: str,
    observations: list[TrialObservation],
) -> None:
    """Restore a registered baseline from the Agent-visible real history."""

    if backend in {"random", "sobol"}:
        for observation in observations:
            expected = algorithm.ask()
            algorithm.tell(
                TrialObservation(
                    suggestion=TrialSuggestion(
                        config=dict(expected.config),
                        trial_id=observation.suggestion.trial_id,
                        budget=observation.suggestion.budget,
                        metadata=dict(expected.metadata),
                    ),
                    status=observation.status,
                    objectives=dict(observation.objectives),
                    metrics=dict(observation.metrics),
                    elapsed_seconds=observation.elapsed_seconds,
                    error_type=observation.error_type,
                    error_message=observation.error_message,
                    timestamp=observation.timestamp,
                    metadata=dict(observation.metadata),
                )
            )
        return
    if backend == "tpe":
        for observation in observations:
            algorithm.seed(observation)
        return
    if backend == "cma_es":
        for observation in observations:
            algorithm.seed(observation)
        return
    algorithm.replay(observations)


def execute_from_workspace(
    *,
    workspace_dir: Path,
    config: dict[str, Any],
    action: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Execute a stateful optimizer action from agent-visible workspace files."""

    space_payload = _read_json(workspace_dir / "space.json")
    objective_payload = _read_json(workspace_dir / "objective.json")
    task_spec = TaskSpec(
        name=str(config.get("optimizer_agent_task_id") or "agent_visible_task"),
        search_space=_search_space_from_schema(space_payload.get("parameters", [])),
        objectives=(
            ObjectiveSpec(
                str(objective_payload["name"]),
                ObjectiveDirection(str(objective_payload["direction"])),
            ),
        ),
        max_evaluations=max(1, int(config.get("optimizer_max_evaluations") or 1)),
        metadata=dict(config.get("optimizer_task_metadata") or {}),
    )
    history = _history_from_workspace(workspace_dir / "history.jsonl")
    incumbent_payload = _read_json(workspace_dir / "incumbent.json")
    incumbent = incumbent_payload.get("config")
    return execute_optimizer_action(
        action=action,
        task_spec=task_spec,
        history=history,
        allowlist=config.get("optimizer_backend_allowlist") or (),
        state_path=Path(
            config.get("optimizer_state_path")
            or workspace_dir / "optimizer_tool_state.json"
        ),
        seed=int(config.get("seed", 0)),
        incumbent=incumbent if isinstance(incumbent, dict) else None,
        arguments=arguments,
    )


def _default_optimizer_state(task_spec: TaskSpec | None = None) -> dict[str, Any]:
    acquisition_name = "logei"
    if task_spec is not None:
        policy = task_spec.metadata.get("optimizer_tool_policy")
        policy = dict(policy) if isinstance(policy, dict) else {}
        configured_acquisition = policy.get("default_acquisition")
        if configured_acquisition is None:
            # Older minimal-suggest conditions recorded the fixed GP
            # acquisition under this key. Treat it as the initial state so
            # the shared suggest implementation does not need an arm-specific
            # runtime override.
            configured_acquisition = policy.get("gp_acquisition")
        if configured_acquisition is not None:
            acquisition_name = str(configured_acquisition)
        acquisition_name = _normalize_acquisition(acquisition_name, None)["name"]
    return {
        "schema_version": 1,
        "backend": None,
        "bounds": {},
        "acquisition": {"name": acquisition_name, "parameters": {}},
    }


def _load_optimizer_state(
    path: Path, *, task_spec: TaskSpec | None = None
) -> dict[str, Any]:
    default = _default_optimizer_state(task_spec)
    if not path.exists():
        return default
    data = _read_json(path)
    return {**default, **data}


def _save_optimizer_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(path, state)


def _optimizer_status(
    task_spec: TaskSpec,
    history: list[TrialObservation],
    allowed: tuple[str, ...],
    state: dict[str, Any],
) -> dict[str, Any]:
    objective = task_spec.primary_objective
    return {
        "single_objective": True,
        "objective": {
            "name": objective.name,
            "direction": objective.direction.value,
            "mutable": False,
        },
        "allowed_backends": list(allowed),
        "persistent_backend": state.get("backend"),
        "active_bounds": dict(state.get("bounds") or {}),
        "acquisition": dict(state.get("acquisition") or {}),
        "history_size": len(history),
        "budget_consumed": False,
        "evaluator_called": False,
    }


def _require_allowed_backend(raw: Any, allowed: tuple[str, ...]) -> str:
    if raw is None or not str(raw).strip():
        raise ValueError(
            "backend is required until a persistent optimizer backend has been selected."
        )
    backend = normalize_backend(str(raw))
    if backend not in allowed:
        raise PermissionError(
            f"Optimizer backend {backend!r} is not enabled; allowed={list(allowed)!r}."
        )
    return backend


def _normalize_acquisition(name: Any, parameters: Any) -> dict[str, Any]:
    normalized = str(name or "logei").strip().lower().replace("-", "_")
    aliases = {
        "expectedimprovement": "ei",
        "logexpectedimprovement": "logei",
        "noisylogei": "noisy_logei",
        "noisy_logexpectedimprovement": "noisy_logei",
        "noisylogexpectedimprovement": "noisy_logei",
        "probabilityofimprovement": "pi",
        "upperconfidencebound": "ucb",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in GP_ACQUISITIONS:
        raise ValueError(
            f"Unknown GP acquisition {name!r}; choose one of {list(GP_ACQUISITIONS)!r}."
        )
    params = dict(parameters or {})
    allowed = (
        {"beta"}
        if normalized == "ucb"
        else {"xi"}
        if normalized in {"ei", "logei", "pi"}
        else set()
    )
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(
            f"Unsupported acquisition parameters for {normalized}: {unknown!r}."
        )
    return {
        "name": normalized,
        "parameters": {key: float(value) for key, value in params.items()},
    }


def _validate_bounds(
    space: SearchSpace, raw: Any
) -> dict[str, list[float | int]]:
    if raw in (None, {}):
        return {}
    if not isinstance(raw, dict):
        raise TypeError(
            "bounds must map numeric parameter names to [low, high]."
        )
    by_name = {param.name: param for param in space}
    result: dict[str, list[float | int]] = {}
    for raw_name, interval in raw.items():
        name = str(raw_name)
        param = by_name.get(name)
        if not isinstance(param, (FloatParam, IntParam)):
            raise ValueError(
                f"Bounds may only control numeric parameters; got {name!r}."
            )
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError(f"Bound for {name!r} must be [low, high].")
        low = float(interval[0])
        high = float(interval[1])
        if (
            low > high
            or low < float(param.low)
            or high > float(param.high)
        ):
            raise ValueError(
                f"Bound for {name!r} must be ordered and inside "
                f"[{param.low}, {param.high}]."
            )
        if isinstance(param, IntParam):
            if int(low) != low or int(high) != high:
                raise ValueError(
                    f"Integer parameter {name!r} requires integer sub-bounds."
                )
            result[name] = [int(low), int(high)]
        else:
            result[name] = [low, high]
    return result


def _bounded_problem(
    task_spec: TaskSpec,
    history: list[TrialObservation],
    bounds: dict[str, Any],
) -> tuple[TaskSpec, list[TrialObservation]]:
    if not bounds:
        return task_spec, history
    parameters = []
    for param in task_spec.search_space:
        if param.name not in bounds:
            parameters.append(param)
            continue
        low, high = bounds[param.name]
        default = min(max(param.effective_default(), low), high)
        if isinstance(param, FloatParam):
            parameters.append(
                FloatParam(
                    name=param.name,
                    low=float(low),
                    high=float(high),
                    log=param.log,
                    default=float(default),
                )
            )
        elif isinstance(param, IntParam):
            parameters.append(
                IntParam(
                    name=param.name,
                    low=int(low),
                    high=int(high),
                    log=param.log,
                    default=int(default),
                )
            )
    metadata = dict(task_spec.metadata)
    protocol = metadata.get("benchmark_protocol")
    if isinstance(protocol, dict):
        bounded_protocol = dict(protocol)
        # The original fixed prefix belongs to the full domain and may contain
        # points outside this Agent-selected subproblem. Candidate-budget and
        # other comparability policy remain intact.
        bounded_protocol.pop("initialization", None)
        metadata["benchmark_protocol"] = bounded_protocol
    bounded = TaskSpec(
        name=task_spec.name,
        search_space=SearchSpace(parameters),
        objectives=task_spec.objectives,
        max_evaluations=task_spec.max_evaluations,
        default_budget=task_spec.default_budget,
        budget_range=task_spec.budget_range,
        supports_budget=task_spec.supports_budget,
        description_ref=task_spec.description_ref,
        metadata=metadata,
    )
    usable = [
        item
        for item in history
        if _config_in_bounds(item.suggestion.config, bounds)
    ]
    return bounded, usable


def _around_problem(
    task_spec: TaskSpec,
    history: list[TrialObservation],
    *,
    incumbent: dict[str, Any] | None,
    raw: Any,
) -> tuple[TaskSpec, list[TrialObservation], dict[str, Any]]:
    if not isinstance(raw, dict):
        raise TypeError("around must be an object.")
    unknown_fields = sorted(
        set(raw) - {"center", "radius", "radii", "fix", "choices", "vary"}
    )
    if unknown_fields:
        raise ValueError(f"Unknown around fields: {unknown_fields!r}.")
    center_raw = raw.get("center", incumbent)
    if not isinstance(center_raw, dict):
        raise ValueError(
            "around requires a complete center config when no incumbent exists."
        )
    space = task_spec.search_space
    center = space.coerce_config(center_raw, use_defaults=False)
    radius = _around_radius(raw.get("radius", 0.1), field="radius")
    radii = dict(raw.get("radii") or {})
    fixed = dict(raw.get("fix") or {})
    choices = dict(raw.get("choices") or {})
    names = set(space.names())
    vary = raw.get("vary")
    if vary is not None:
        if not isinstance(vary, list) or not vary or not set(vary) <= names:
            raise ValueError("around.vary must name at least one known parameter.")
        if set(vary) & set(fixed):
            raise ValueError("around.vary and around.fix must be disjoint.")
        for name in names - set(vary):
            fixed.setdefault(name, center[name])
    for label, mapping in (("radii", radii), ("fix", fixed), ("choices", choices)):
        extra = sorted(set(mapping) - names)
        if extra:
            raise ValueError(f"around.{label} references unknown parameters: {extra!r}.")
    overlap = sorted(set(fixed) & set(choices))
    if overlap:
        raise ValueError(
            f"around parameters cannot appear in both fix and choices: {overlap!r}."
        )

    parameters = []
    for param in space:
        if param.name in fixed:
            value = param.coerce(fixed[param.name])
            if isinstance(param, FloatParam):
                parameters.append(
                    FloatParam(
                        name=param.name,
                        low=float(value),
                        high=float(value),
                        log=param.log,
                        default=float(value),
                    )
                )
            elif isinstance(param, IntParam):
                parameters.append(
                    IntParam(
                        name=param.name,
                        low=int(value),
                        high=int(value),
                        log=param.log,
                        default=int(value),
                    )
                )
            elif isinstance(param, CategoricalParam):
                parameters.append(
                    CategoricalParam(
                        name=param.name,
                        choices=(value,),
                        default=value,
                    )
                )
            else:
                raise TypeError(
                    f"around does not support parameter type {type(param).__name__}."
                )
            continue

        if isinstance(param, (FloatParam, IntParam)):
            local_radius = _around_radius(
                radii.get(param.name, radius),
                field=f"radii.{param.name}",
            )
            value = min(max(float(center[param.name]), float(param.low)), float(param.high))
            if param.log:
                log_low = math.log(float(param.low))
                log_high = math.log(float(param.high))
                half_width = local_radius * (log_high - log_low)
                low = max(float(param.low), math.exp(math.log(value) - half_width))
                high = min(float(param.high), math.exp(math.log(value) + half_width))
            else:
                half_width = local_radius * (float(param.high) - float(param.low))
                low = max(float(param.low), value - half_width)
                high = min(float(param.high), value + half_width)
            if isinstance(param, IntParam):
                int_low = max(param.low, int(math.floor(low)))
                int_high = min(param.high, int(math.ceil(high)))
                default = min(max(int(round(value)), int_low), int_high)
                parameters.append(
                    IntParam(
                        name=param.name,
                        low=int_low,
                        high=int_high,
                        log=param.log,
                        default=default,
                    )
                )
            else:
                default = min(max(float(value), float(low)), float(high))
                parameters.append(
                    FloatParam(
                        name=param.name,
                        low=float(low),
                        high=float(high),
                        log=param.log,
                        default=default,
                    )
                )
            continue

        if isinstance(param, CategoricalParam):
            raw_choices = choices.get(param.name, [center[param.name]])
            if not isinstance(raw_choices, (list, tuple)) or not raw_choices:
                raise ValueError(
                    f"around.choices.{param.name} must be a non-empty list."
                )
            allowed = tuple(param.coerce(value) for value in raw_choices)
            allowed = tuple(dict.fromkeys(allowed))
            default = center[param.name] if center[param.name] in allowed else allowed[0]
            parameters.append(
                CategoricalParam(
                    name=param.name,
                    choices=allowed,
                    default=default,
                )
            )
            continue
        raise TypeError(
            f"around does not support parameter type {type(param).__name__}."
        )

    candidate_space = SearchSpace(parameters)
    around_spec = _task_spec_with_search_space(task_spec, candidate_space)
    usable = [
        item
        for item in history
        if _config_in_search_space(candidate_space, item.suggestion.config)
    ]
    payload = {
        "center": center,
        "radius": radius,
        "vary": vary,
        "radii": {key: float(value) for key, value in radii.items()},
        "fix": fixed,
        "choices": choices,
        "resolved_region": _search_space_region(candidate_space),
        "persistent": False,
    }
    return around_spec, usable, payload


def _around_radius(value: Any, *, field: str) -> float:
    try:
        radius = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"around.{field} must be numeric.") from exc
    if not math.isfinite(radius) or radius <= 0.0 or radius > 1.0:
        raise ValueError(f"around.{field} must be in (0, 1].")
    return radius


def _task_spec_with_search_space(
    task_spec: TaskSpec, search_space: SearchSpace
) -> TaskSpec:
    metadata = dict(task_spec.metadata)
    protocol = metadata.get("benchmark_protocol")
    if isinstance(protocol, dict):
        candidate_protocol = dict(protocol)
        candidate_protocol.pop("initialization", None)
        metadata["benchmark_protocol"] = candidate_protocol
    return TaskSpec(
        name=task_spec.name,
        search_space=search_space,
        objectives=task_spec.objectives,
        max_evaluations=task_spec.max_evaluations,
        default_budget=task_spec.default_budget,
        budget_range=task_spec.budget_range,
        supports_budget=task_spec.supports_budget,
        description_ref=task_spec.description_ref,
        metadata=metadata,
    )


def _config_in_search_space(space: SearchSpace, config: dict[str, Any]) -> bool:
    try:
        space.validate_config(config)
    except (TypeError, ValueError):
        return False
    return True


def _search_space_region(space: SearchSpace) -> dict[str, Any]:
    region: dict[str, Any] = {}
    for param in space:
        if isinstance(param, (FloatParam, IntParam)):
            region[param.name] = {
                "type": "float" if isinstance(param, FloatParam) else "int",
                "bounds": [param.low, param.high],
                "log": param.log,
            }
        elif isinstance(param, CategoricalParam):
            region[param.name] = {
                "type": "categorical",
                "choices": list(param.choices),
            }
    return region


def _project_config_to_space(
    space: SearchSpace, config: dict[str, Any] | None
) -> dict[str, Any] | None:
    if config is None:
        return None
    projected: dict[str, Any] = {}
    for param in space:
        value = config.get(param.name, param.effective_default())
        if isinstance(param, FloatParam):
            projected[param.name] = min(max(float(value), float(param.low)), float(param.high))
        elif isinstance(param, IntParam):
            projected[param.name] = min(max(int(value), int(param.low)), int(param.high))
        elif isinstance(param, CategoricalParam):
            projected[param.name] = value if value in param.choices else param.effective_default()
        else:
            projected[param.name] = value
    return space.coerce_config(projected, use_defaults=False)


def _config_in_bounds(config: dict[str, Any], bounds: dict[str, Any]) -> bool:
    return all(
        float(interval[0]) <= float(config[name]) <= float(interval[1])
        for name, interval in bounds.items()
    )


def _coerce_virtual_configs(
    space: SearchSpace, raw: Any, bounds: dict[str, Any]
) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw or len(raw) > 32:
        raise ValueError("configs must contain 1-32 complete config objects.")
    configs = [
        space.coerce_config(config, use_defaults=False) for config in raw
    ]
    outside = [
        index
        for index, config in enumerate(configs)
        if not _config_in_bounds(config, bounds)
    ]
    if outside:
        raise ValueError(
            f"Virtual configs outside active bounds at indices {outside}."
        )
    return configs


def _encode_config(space: SearchSpace, config: dict[str, Any]) -> list[float]:
    encoded: list[float] = []
    for param in space:
        value = config[param.name]
        if isinstance(param, (FloatParam, IntParam)):
            span = max(float(param.high) - float(param.low), 1e-12)
            encoded.append((float(value) - float(param.low)) / span)
        elif isinstance(param, CategoricalParam):
            encoded.append(
                param.choices.index(value) / max(len(param.choices) - 1, 1)
            )
        else:
            encoded.append((abs(hash(str(value))) % 10000) / 9999.0)
    return encoded


def normalize_backend(raw: str) -> str:
    return normalize_comparable_backend(raw)


def _search_space_from_schema(raw_parameters: Any) -> SearchSpace:
    if not isinstance(raw_parameters, list) or not raw_parameters:
        raise ValueError("space.json must contain a non-empty parameters list.")
    parameters = []
    for raw in raw_parameters:
        if not isinstance(raw, dict):
            raise TypeError("Each parameter schema must be an object.")
        common = {"name": str(raw["name"]), "default": raw.get("default")}
        kind = raw.get("type")
        if kind == "float":
            parameters.append(
                FloatParam(
                    **common,
                    low=float(raw["low"]),
                    high=float(raw["high"]),
                    log=bool(raw.get("log", False)),
                )
            )
        elif kind == "int":
            parameters.append(
                IntParam(
                    **common,
                    low=int(raw["low"]),
                    high=int(raw["high"]),
                    log=bool(raw.get("log", False)),
                )
            )
        elif kind == "categorical":
            parameters.append(CategoricalParam(**common, choices=tuple(raw["choices"])))
        elif kind == "string":
            parameters.append(
                StringParam(
                    **common,
                    min_length=int(raw.get("min_length", 0)),
                    max_length=raw.get("max_length"),
                    pattern=raw.get("pattern"),
                )
            )
        else:
            raise ValueError(f"Unsupported parameter type {kind!r}.")
    return SearchSpace(parameters)


def _history_from_workspace(path: Path) -> list[TrialObservation]:
    observations: list[TrialObservation] = []
    if not path.exists():
        return observations
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        observations.append(
            TrialObservation(
                suggestion=TrialSuggestion(
                    config=dict(raw.get("config") or {}),
                    trial_id=raw.get("trial_id"),
                    budget=raw.get("budget"),
                    metadata=dict(raw.get("suggestion_metadata") or {}),
                ),
                status=TrialStatus(str(raw.get("status", "success"))),
                objectives={
                    key: float(value)
                    for key, value in dict(raw.get("objectives") or {}).items()
                },
                metrics=dict(raw.get("metrics") or {}),
                error_type=raw.get("error_type"),
                error_message=raw.get("error_message"),
                metadata=dict(raw.get("metadata") or {}),
            )
        )
    return observations


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return data


__all__ = [
    "OPTIMIZER_BACKENDS",
    "OptimizerSuggestTool",
    "normalize_backend",
    "suggest_candidate",
    "suggest_from_workspace",
]
