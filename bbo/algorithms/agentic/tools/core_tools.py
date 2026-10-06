"""Core BBO tools for task context, history, validation, sampling, and memory."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ....core import (
    CategoricalParam,
    FloatParam,
    IntParam,
    ObjectiveDirection,
    SearchSpace,
    StringParam,
    TrialObservation,
    search_space_to_schema,
)
from ..serialization import stable_config_identity
from .base import BaseBBOTool
from .context import BBOToolContext


AGENT_METADATA_DENYLIST = frozenset(
    {
        "benchmark",
        "bbob_function_id",
        "benchmark_protocol",
        "bbob_instance_id",
        "bbob_problem_id",
        "function_name",
        "known_optimum",
        "known_optima",
        "problem_key",
        "true_task_name",
        "task_name",
        "task_id",
        "benchmark_suite",
        "dataset",
        "model",
        "target_smiles",
        "description_paths",
        "description_dir",
        "description_directory",
        "workspace_seed_files",
        "provenance",
        "family",
        "real_world_domain",
        "source_path",
        "checkpoint_path",
    }
)
AGENT_METRIC_DENYLIST = frozenset({"regret", "log10_regret", "distance_to_known_optimum"})
AGENT_VISIBLE_FLOAT_DECIMALS = 4


def agent_visible_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return task metadata that is safe to expose to optimization agents."""

    sanitized = sanitize_agent_context_payload(metadata)
    return sanitized if isinstance(sanitized, dict) else {}


def sanitize_agent_context_payload(value: Any) -> Any:
    """Recursively remove identity/provenance fields from agent-visible context."""

    if isinstance(value, dict):
        return {
            key: sanitize_agent_context_payload(item)
            for key, item in value.items()
            if str(key).lower() not in AGENT_METADATA_DENYLIST
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_agent_context_payload(item) for item in value]
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str) and os.path.isabs(value):
        return Path(value).name
    return agent_visible_payload(value)


def agent_visible_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return trial metrics that do not leak reference solution information."""

    return agent_visible_payload({key: value for key, value in metrics.items() if key not in AGENT_METRIC_DENYLIST})


def agent_visible_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a compact display/config payload for agent-facing context."""

    return agent_visible_payload(config)


def agent_visible_payload(value: Any) -> Any:
    """Round finite floats in agent-facing JSON so candidate strings stay readable."""

    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            return numeric
        rounded = round(numeric, AGENT_VISIBLE_FLOAT_DECIMALS)
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, np.ndarray):
        return agent_visible_payload(value.tolist())
    if isinstance(value, dict):
        return {key: agent_visible_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [agent_visible_payload(item) for item in value]
    return value


class GetTaskContextTool(BaseBBOTool):
    name = "get_task_context"
    description = "Return task documentation, manifest, objective metadata, and benchmark constraints."
    parameters_schema = {
        "type": "object",
        "properties": {
            "sections": {"type": "array", "items": {"type": "string"}},
            "max_chars_per_section": {"type": "integer", "minimum": 200, "default": 4000},
            "include_manifest": {"type": "boolean", "default": True},
        },
    }

    async def execute(
        self,
        context: BBOToolContext,
        sections: list[str] | None = None,
        max_chars_per_section: int = 4000,
        include_manifest: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        available_sections = context.agent_description_sections or context.description.section_map
        wanted = {section for section in sections or available_sections}
        docs = {
            name: _truncate(text, int(max_chars_per_section))
            for name, text in available_sections.items()
            if name in wanted
        }
        return {
            "task_id": context.agent_task_id or context.task_spec.name,
            "objectives": [
                {"name": objective.name, "direction": objective.direction.value}
                for objective in context.task_spec.objectives
            ],
            "max_evaluations": context.task_spec.max_evaluations,
            "metadata": agent_visible_metadata(context.task_spec.metadata),
            "sections": docs,
            "manifest": (
                context.agent_manifest_payload
                if include_manifest and context.agent_manifest_payload is not None
                else context.manifest.to_dict() if include_manifest else None
            ),
        }


class GetSearchSpaceTool(BaseBBOTool):
    name = "get_search_space"
    description = "Return the exact BBO search-space schema, defaults, and parameter ordering."
    parameters_schema = {"type": "object", "properties": {}}

    async def execute(self, context: BBOToolContext, **_: Any) -> dict[str, Any]:
        return {
            "parameters": search_space_schema(context.task_spec.search_space),
            "defaults": agent_visible_config(context.task_spec.search_space.defaults()),
            "dimension": len(context.task_spec.search_space),
        }


class GetTrialHistoryTool(BaseBBOTool):
    name = "get_trial_history"
    description = "Return evaluated BBO trials without consuming objective budget."
    parameters_schema = {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["recent", "best", "all"], "default": "recent"},
            "limit": {"type": "integer", "minimum": 1, "default": 20},
            "offset": {"type": "integer", "minimum": 0, "default": 0},
        },
    }

    async def execute(
        self,
        context: BBOToolContext,
        mode: str = "recent",
        limit: int = 20,
        offset: int = 0,
        **_: Any,
    ) -> dict[str, Any]:
        observations = list(context.history)
        if mode == "best":
            observations = _sort_by_primary(observations, context)
        elif mode == "recent":
            observations = observations[::-1]
        elif mode != "all":
            raise ValueError("mode must be one of recent, best, all.")
        limit = max(1, int(limit))
        offset = max(0, int(offset))
        page = observations[offset : offset + limit]
        return {
            "mode": mode,
            "total": len(context.history),
            "offset": offset,
            "limit": limit,
            "trials": [_observation_summary(observation) for observation in page],
        }


class GetIncumbentTool(BaseBBOTool):
    name = "get_incumbent"
    description = "Return the current best known BBO configuration and objectives."
    parameters_schema = {"type": "object", "properties": {}}

    async def execute(self, context: BBOToolContext, **_: Any) -> dict[str, Any]:
        incumbent = context.incumbent
        if incumbent is None:
            return {"incumbent": None}
        return {
            "incumbent": {
                "config": agent_visible_config(incumbent.config),
                "score": agent_visible_payload(incumbent.score),
                "objectives": agent_visible_payload(incumbent.objectives),
                "trial_id": incumbent.trial_id,
                "metadata": incumbent.metadata,
            }
        }


class ValidateCandidatesTool(BaseBBOTool):
    name = "validate_candidates"
    description = "Validate candidate configurations against the BBO search space and duplicate history."
    parameters_schema = {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Candidate objects, either raw configs or objects with a `config` field.",
            }
        },
        "required": ["candidates"],
    }

    async def execute(self, context: BBOToolContext, candidates: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        if not isinstance(candidates, list):
            raise TypeError("candidates must be a list.")
        seen_history = {
            stable_config_identity(observation.suggestion.config)
            for observation in context.history
        }
        seen_payload: set[str] = set()
        valid: list[dict[str, Any]] = []
        invalid: list[dict[str, Any]] = []
        for index, item in enumerate(candidates):
            if not isinstance(item, dict):
                invalid.append({"index": index, "error": "candidate is not an object"})
                continue
            raw = item.get("config", item)
            if not isinstance(raw, dict):
                invalid.append({"index": index, "error": "`config` is not an object"})
                continue
            try:
                config = context.task_spec.search_space.coerce_config(raw, use_defaults=False)
            except Exception as exc:
                invalid.append({"index": index, "error": str(exc)})
                continue
            identity = stable_config_identity(config)
            duplicate = identity in seen_history or identity in seen_payload
            seen_payload.add(identity)
            valid.append({"index": index, "config": agent_visible_config(config), "duplicate": duplicate, "identity": identity})
        return {"valid": valid, "invalid": invalid, "valid_count": len(valid), "invalid_count": len(invalid)}


class ValidateCandidateTool(BaseBBOTool):
    name = "validate_candidate"
    description = (
        "Validate one evaluator-facing BBO candidate and report exact schema, bounds, constraint, "
        "duplicate, and local repair information without modifying the candidate."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "candidate": {"type": "object", "description": "Raw config or an object with a `config` field."},
            "too_similar_threshold": {"type": "number", "minimum": 0.0},
        },
        "required": ["candidate"],
    }

    async def execute(
        self,
        context: BBOToolContext,
        candidate: dict[str, Any],
        too_similar_threshold: float | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        return _validate_candidate_report(context, candidate, too_similar_threshold=too_similar_threshold)


class SampleCandidatesTool(BaseBBOTool):
    name = "sample_candidates"
    description = "Sample valid BBO candidates without evaluating them."
    parameters_schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "minimum": 1, "maximum": 128, "default": 4},
            "seed": {"type": "integer"},
            "strategy": {"type": "string", "enum": ["random", "around_incumbent"], "default": "random"},
            "jitter_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.1},
        },
    }

    async def execute(
        self,
        context: BBOToolContext,
        n: int = 4,
        seed: int | None = None,
        strategy: str = "random",
        jitter_fraction: float = 0.1,
        **_: Any,
    ) -> dict[str, Any]:
        rng = random.Random(context.seed if seed is None else int(seed))
        target = min(max(1, int(n)), 128)
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        seen_history = {
            stable_config_identity(observation.suggestion.config)
            for observation in context.history
        }
        for _attempt in range(target * 200):
            if strategy == "around_incumbent" and context.incumbent is not None:
                config = _sample_around(context.task_spec.search_space, context.incumbent.config, rng, float(jitter_fraction))
            elif strategy == "random":
                config = _sample_search_space(context.task_spec.search_space, rng)
            else:
                raise ValueError("strategy must be random or around_incumbent.")
            identity = stable_config_identity(config)
            if identity in seen or identity in seen_history:
                continue
            seen.add(identity)
            candidates.append({"config": agent_visible_config(config), "identity": identity})
            if len(candidates) >= target:
                break
        return {"strategy": strategy, "candidates": candidates, "count": len(candidates)}


class GetHistoryOverviewTool(BaseBBOTool):
    name = "get_history_overview"
    description = (
        "Return a compact BBO history overview: incumbent, best-so-far progression, recent objectives, "
        "budget use, invalid trials, and recent skill/search-action metadata."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "recent_limit": {"type": "integer", "minimum": 1, "default": 8},
            "progression_limit": {"type": "integer", "minimum": 1, "default": 40},
        },
    }

    async def execute(
        self,
        context: BBOToolContext,
        recent_limit: int = 8,
        progression_limit: int = 40,
        **_: Any,
    ) -> dict[str, Any]:
        primary = context.task_spec.primary_objective.name
        direction = context.task_spec.primary_objective.direction
        scored = [obs for obs in context.history if _primary_score(obs, primary) is not None]
        best_so_far: list[dict[str, Any]] = []
        best_score: float | None = None
        last_best_trial_id: int | None = None
        for obs in scored:
            score = _primary_score(obs, primary)
            assert score is not None
            if best_score is None or _is_better(score, best_score, direction):
                best_score = score
                last_best_trial_id = obs.suggestion.trial_id
            best_so_far.append(
                {
                    "trial_id": obs.suggestion.trial_id,
                    "score": agent_visible_payload(score),
                    "best_so_far": agent_visible_payload(best_score),
                }
            )
        recent = list(context.history)[-max(1, int(recent_limit)) :]
        return {
            "primary_objective": primary,
            "direction": direction.value,
            "evaluated_count": len(context.history),
            "used_budget": len(context.history),
            "remaining_budget": None
            if context.task_spec.max_evaluations is None
            else max(0, int(context.task_spec.max_evaluations) - len(context.history)),
            "invalid_candidate_count": sum(1 for obs in context.history if obs.status.value == "invalid"),
            "incumbent": None
            if context.incumbent is None
            else {
                "trial_id": context.incumbent.trial_id,
                "score": agent_visible_payload(context.incumbent.score),
                "config": agent_visible_config(context.incumbent.config),
            },
            "best_so_far_progression": best_so_far[-max(1, int(progression_limit)) :],
            "last_best_trial_id": last_best_trial_id,
            "recent_objectives": [
                {
                    "trial_id": obs.suggestion.trial_id,
                    "status": obs.status.value,
                    "objective": agent_visible_payload(_primary_score(obs, primary)),
                }
                for obs in recent
            ],
            "recent_search_actions": [_search_action_summary(obs) for obs in recent],
        }


class SummarizeObjectiveMetricsTool(BaseBBOTool):
    name = "summarize_objective_metrics"
    description = (
        "Summarize objective progress and agent-visible numeric metrics from evaluated history "
        "without consuming evaluator budget."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "recent_limit": {"type": "integer", "minimum": 1, "default": 8},
            "progression_limit": {"type": "integer", "minimum": 1, "default": 40},
        },
    }

    async def execute(
        self,
        context: BBOToolContext,
        recent_limit: int = 8,
        progression_limit: int = 40,
        **_: Any,
    ) -> dict[str, Any]:
        return _objective_metrics_summary(
            context.history,
            primary=context.task_spec.primary_objective.name,
            direction=context.task_spec.primary_objective.direction,
            recent_limit=max(1, int(recent_limit)),
            progression_limit=max(1, int(progression_limit)),
        )


class CompareTrialsTool(BaseBBOTool):
    name = "compare_trials"
    description = (
        "Precisely compare two or more evaluated trials by objective, changed variables, unchanged variables, "
        "legality/status, and stored search-action metadata."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "trial_ids": {
                "type": "array",
                "items": {"type": ["integer", "string"]},
                "minItems": 2,
            }
        },
        "required": ["trial_ids"],
    }

    async def execute(self, context: BBOToolContext, trial_ids: list[int | str], **_: Any) -> dict[str, Any]:
        if not isinstance(trial_ids, list) or len(trial_ids) < 2:
            raise ValueError("trial_ids must contain at least two trial identifiers.")
        trials = [_resolve_trial(context.history, item) for item in trial_ids]
        primary = context.task_spec.primary_objective.name
        base = trials[0]
        comparisons = []
        for other in trials[1:]:
            base_score = _primary_score(base, primary)
            other_score = _primary_score(other, primary)
            comparisons.append(
                {
                    "from_trial_id": base.suggestion.trial_id,
                    "to_trial_id": other.suggestion.trial_id,
                    "objective": {
                        "name": primary,
                        "from": agent_visible_payload(base_score),
                        "to": agent_visible_payload(other_score),
                        "delta": None
                        if base_score is None or other_score is None
                        else agent_visible_payload(other_score - base_score),
                        "direction": context.task_spec.primary_objective.direction.value,
                    },
                    "changed_variables": _changed_variables(
                        context.task_spec.search_space,
                        base.suggestion.config,
                        other.suggestion.config,
                    ),
                    "unchanged_variables": _unchanged_variables(base.suggestion.config, other.suggestion.config),
                    "status": {"from": base.status.value, "to": other.status.value},
                    "action_metadata": {
                        "from": _search_action_summary(base),
                        "to": _search_action_summary(other),
                    },
                }
            )
        return {"trial_ids": [_trial_label(obs) for obs in trials], "comparisons": comparisons}


class FindNearestTrialsTool(BaseBBOTool):
    name = "find_nearest_trials"
    description = (
        "Find nearest evaluated trials to a target candidate or trial using search-space-aware normalized distances."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "target": {
                "description": "A trial identifier, raw config, or object with a `config` field.",
            },
            "k": {"type": "integer", "minimum": 1, "default": 5},
        },
        "required": ["target"],
    }

    async def execute(self, context: BBOToolContext, target: Any, k: int = 5, **_: Any) -> dict[str, Any]:
        target_config, target_trial_id = _target_config(context, target)
        rows = []
        limitations: list[str] = []
        for obs in context.history:
            if target_trial_id is not None and obs.suggestion.trial_id == target_trial_id:
                continue
            distance, details, notes = _config_distance(
                context.task_spec.search_space,
                target_config,
                obs.suggestion.config,
            )
            limitations.extend(note for note in notes if note not in limitations)
            rows.append(
                {
                    "trial_id": obs.suggestion.trial_id,
                    "distance": agent_visible_payload(distance),
                    "objective": agent_visible_payload(_primary_score(obs, context.task_spec.primary_objective.name)),
                    "status": obs.status.value,
                    "major_differences": details[:8],
                }
            )
        rows.sort(key=lambda item: (float("inf") if item["distance"] is None else float(item["distance"]), str(item["trial_id"])))
        return {
            "target_trial_id": target_trial_id,
            "neighbors": rows[: max(1, int(k))],
            "limitations": limitations,
        }


class EstimateLocalEffectsTool(BaseBBOTool):
    name = "estimate_local_effects"
    description = (
        "Estimate observational local variable effects around a reference trial or region using comparable history pairs; "
        "reports supporting and contradicting trial comparisons without causal confidence scores."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "reference": {"description": "Trial identifier, raw config, or object with a `config` field."},
            "variables": {"type": "array", "items": {"type": "string"}},
            "local_radius": {"type": "number", "minimum": 0.0, "default": 0.35},
        },
        "required": ["reference"],
    }

    async def execute(
        self,
        context: BBOToolContext,
        reference: Any,
        variables: list[str] | None = None,
        local_radius: float = 0.35,
        **_: Any,
    ) -> dict[str, Any]:
        reference_config, reference_trial_id = _target_config(context, reference)
        primary = context.task_spec.primary_objective.name
        direction = context.task_spec.primary_objective.direction
        names = variables or context.task_spec.search_space.names()
        scored = [obs for obs in context.history if _primary_score(obs, primary) is not None]
        results: dict[str, Any] = {}
        for name in names:
            if not context.task_spec.search_space.contains(name):
                results[name] = {"error": "unknown variable"}
                continue
            supporting: list[str] = []
            contradicting: list[str] = []
            comparable: list[str] = []
            for a_index, a in enumerate(scored):
                for b in scored[a_index + 1 :]:
                    changed = [item["name"] for item in _changed_variables(context.task_spec.search_space, a.suggestion.config, b.suggestion.config)]
                    if name not in changed:
                        continue
                    other_changed = [item for item in changed if item != name]
                    if other_changed:
                        other_a = dict(a.suggestion.config)
                        other_b = dict(b.suggestion.config)
                        other_a[name] = reference_config.get(name)
                        other_b[name] = reference_config.get(name)
                        other_distance, _, _ = _config_distance(context.task_spec.search_space, other_a, other_b)
                        if other_distance is None or other_distance > float(local_radius):
                            continue
                    dist_a, _, _ = _config_distance(context.task_spec.search_space, reference_config, a.suggestion.config)
                    dist_b, _, _ = _config_distance(context.task_spec.search_space, reference_config, b.suggestion.config)
                    if (dist_a is not None and dist_a > float(local_radius)) or (dist_b is not None and dist_b > float(local_radius)):
                        continue
                    score_a = _primary_score(a, primary)
                    score_b = _primary_score(b, primary)
                    if score_a is None or score_b is None or score_a == score_b:
                        continue
                    label = f"{_trial_label(a)}->{_trial_label(b)}"
                    comparable.append(label)
                    improved = _is_better(score_b, score_a, direction)
                    if improved:
                        supporting.append(label)
                    else:
                        contradicting.append(label)
            results[name] = {
                "supporting_local_comparisons": len(supporting),
                "contradicting_local_comparisons": len(contradicting),
                "supporting_trials": supporting[:20],
                "contradicting_trials": contradicting[:20],
                "applicable_local_radius": float(local_radius),
                "status": "insufficient comparable evidence" if not comparable else "observational comparisons only",
            }
        return {
            "reference_trial_id": reference_trial_id,
            "primary_objective": primary,
            "direction": direction.value,
            "note": "These are observational comparisons, not causal claims.",
            "effects": results,
        }


class MeasureSearchCoverageTool(BaseBBOTool):
    name = "measure_search_coverage"
    description = (
        "Measure trajectory-level search coverage: visited numeric ranges, unvisited categories, recent distances, "
        "underexplored regions, collapse toward one area, and repeated variable modifications."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "recent_limit": {"type": "integer", "minimum": 2, "default": 8},
        },
    }

    async def execute(self, context: BBOToolContext, recent_limit: int = 8, **_: Any) -> dict[str, Any]:
        observations = list(context.history)
        recent = observations[-max(2, int(recent_limit)) :]
        numeric_coverage: dict[str, Any] = {}
        categorical_coverage: dict[str, Any] = {}
        underexplored: list[str] = []
        for param in context.task_spec.search_space:
            values = [obs.suggestion.config.get(param.name) for obs in observations if param.name in obs.suggestion.config]
            if isinstance(param, (FloatParam, IntParam)):
                if not values:
                    numeric_coverage[param.name] = {"visited": False, "unvisited_ranges": [[param.low, param.high]]}
                    underexplored.append(param.name)
                    continue
                low = float(param.low)
                high = float(param.high)
                span = max(high - low, 1e-12)
                seen_min = min(float(value) for value in values)
                seen_max = max(float(value) for value in values)
                frac = max(0.0, min(1.0, (seen_max - seen_min) / span))
                gaps = []
                if seen_min > low:
                    gaps.append([agent_visible_payload(low), agent_visible_payload(seen_min)])
                if seen_max < high:
                    gaps.append([agent_visible_payload(seen_max), agent_visible_payload(high)])
                if frac < 0.35:
                    underexplored.append(param.name)
                numeric_coverage[param.name] = {
                    "visited_min": agent_visible_payload(seen_min),
                    "visited_max": agent_visible_payload(seen_max),
                    "domain": [agent_visible_payload(low), agent_visible_payload(high)],
                    "visited_fraction": agent_visible_payload(frac),
                    "unvisited_edge_ranges": gaps,
                }
            elif isinstance(param, CategoricalParam):
                visited = {value for value in values}
                missing = [choice for choice in param.choices if choice not in visited]
                if missing:
                    underexplored.append(param.name)
                categorical_coverage[param.name] = {
                    "visited_categories": list(visited),
                    "unvisited_categories": missing,
                }
        pairwise = _average_pairwise_distance(context.task_spec.search_space, [obs.suggestion.config for obs in recent])
        incumbent_distance = None
        if context.incumbent is not None and recent:
            distances = [
                _config_distance(context.task_spec.search_space, obs.suggestion.config, context.incumbent.config)[0]
                for obs in recent
            ]
            finite = [value for value in distances if value is not None]
            incumbent_distance = None if not finite else sum(finite) / len(finite)
        repeated_variables = _recent_repeated_variables(context.task_spec.search_space, recent)
        return {
            "evaluated_count": len(observations),
            "numeric_coverage": numeric_coverage,
            "categorical_coverage": categorical_coverage,
            "recent_average_pairwise_distance": agent_visible_payload(pairwise),
            "recent_average_distance_to_incumbent": agent_visible_payload(incumbent_distance),
            "underexplored_variables_or_regions": underexplored,
            "trajectory_collapse_warning": bool(pairwise is not None and pairwise < 0.08 and len(recent) >= 4),
            "repeated_modified_variables": repeated_variables,
        }


class GetRecentSearchActionsTool(BaseBBOTool):
    name = "get_recent_search_actions"
    description = (
        "Return recent stored search actions, including skill, parent/reference trials, modified variables, intent, "
        "distances, best refresh status, tested hypothesis, and repair metadata."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "default": 10},
        },
    }

    async def execute(self, context: BBOToolContext, limit: int = 10, **_: Any) -> dict[str, Any]:
        primary = context.task_spec.primary_objective.name
        best_score: float | None = None
        rows: list[dict[str, Any]] = []
        for obs in context.history:
            score = _primary_score(obs, primary)
            refreshed = False
            if score is not None and (best_score is None or _is_better(score, best_score, context.task_spec.primary_objective.direction)):
                refreshed = True
                best_score = score
            action = _search_action_summary(obs)
            rows.append(
                {
                    "trial_id": obs.suggestion.trial_id,
                    "search_intent": action.get("search_intent"),
                    "skill": action.get("skill"),
                    "parent_trials": action.get("parent_trials", []),
                    "reference_trials": action.get("reference_trials", []),
                    "modified_variables": action.get("modified_variables", []),
                    "distance_to_incumbent": None
                    if context.incumbent is None
                    else agent_visible_payload(
                        _config_distance(context.task_spec.search_space, obs.suggestion.config, context.incumbent.config)[0]
                    ),
                    "refreshed_best": refreshed,
                    "hypothesis": action.get("hypothesis"),
                    "repaired": action.get("repaired", False),
                    "repair": action.get("repair"),
                }
            )
        return {"actions": rows[-max(1, int(limit)) :], "count": min(len(rows), max(1, int(limit)))}


class FitAndCheckSurrogateTool(BaseBBOTool):
    name = "fit_and_check_surrogate"
    description = (
        "Fit simple deterministic surrogate models and report validation evidence; returns usable_signal=false "
        "when history is too short, representation is unsuitable, or validation quality is poor."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "min_observations": {"type": "integer", "minimum": 3, "default": 6},
        },
    }

    async def execute(self, context: BBOToolContext, min_observations: int = 6, **_: Any) -> dict[str, Any]:
        return _fit_surrogate(context, min_observations=max(3, int(min_observations)))


class ScoreVirtualCandidatesTool(BaseBBOTool):
    name = "score_virtual_candidates"
    description = (
        "Score virtual candidates with a previously validated surrogate without consuming evaluator budget; "
        "also reports validation precheck and distance to evaluated history."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "model_id": {"type": "string"},
            "candidates": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["model_id", "candidates"],
    }

    async def execute(self, context: BBOToolContext, model_id: str, candidates: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        return _score_virtual_candidates(context, model_id=model_id, candidates=candidates)


class AnalyzeHistoryTool(BaseBBOTool):
    name = "analyze_history"
    description = "Compute lightweight BBO history statistics for agent reasoning."
    parameters_schema = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "default": 100},
        },
    }

    async def execute(self, context: BBOToolContext, limit: int = 100, **_: Any) -> dict[str, Any]:
        observations = [obs for obs in context.history if obs.success]
        if limit > 0:
            observations = observations[-int(limit) :]
        primary = context.task_spec.primary_objective.name
        scored = [obs for obs in observations if primary in obs.objectives]
        if not scored:
            return {"history_size": len(context.history), "success_count": 0, "primary_objective": primary}
        scores = np.asarray([float(obs.objectives[primary]) for obs in scored], dtype=float)
        direction = context.task_spec.primary_objective.direction
        best_index = int(np.argmin(scores) if direction == ObjectiveDirection.MINIMIZE else np.argmax(scores))
        analysis: dict[str, Any] = {
            "history_size": len(context.history),
            "success_count": len(scored),
            "primary_objective": primary,
            "direction": direction.value,
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "score_mean": float(np.mean(scores)),
            "best_trial": _observation_summary(scored[best_index]),
            "numeric_correlations": {},
            "categorical_groups": {},
        }
        for param in context.task_spec.search_space:
            values = [obs.suggestion.config.get(param.name) for obs in scored]
            if isinstance(param, (FloatParam, IntParam)):
                xs = np.asarray([float(value) for value in values], dtype=float)
                if len(xs) > 1 and float(np.std(xs)) > 0.0 and float(np.std(scores)) > 0.0:
                    analysis["numeric_correlations"][param.name] = float(np.corrcoef(xs, scores)[0, 1])
            elif isinstance(param, CategoricalParam):
                groups: dict[str, list[float]] = {}
                for value, score in zip(values, scores, strict=True):
                    groups.setdefault(str(value), []).append(float(score))
                analysis["categorical_groups"][param.name] = {
                    key: {"count": len(vals), "mean": float(np.mean(vals))}
                    for key, vals in groups.items()
                }
        return agent_visible_payload(analysis)


class _AdvancedAnalysisTool(BaseBBOTool):
    """Shared implementation for focused, evaluator-free history diagnostics."""

    diagnostic: str = ""

    async def execute(
        self,
        context: BBOToolContext,
        limit: int = 200,
        top_fraction: float = 0.25,
        top_k: int = 8,
        bins: int = 5,
        **_: Any,
    ) -> dict[str, Any]:
        return _advanced_history_diagnostic(
            context,
            self.diagnostic,
            limit=max(1, int(limit)),
            top_fraction=min(max(float(top_fraction), 0.05), 0.5),
            top_k=min(max(int(top_k), 1), 32),
            bins=min(max(int(bins), 2), 20),
        )


class ProfileHistoryQualityTool(_AdvancedAnalysisTool):
    name = "profile_history_quality"
    diagnostic = name
    description = "Audit history completeness, status balance, duplicate configs, finite scores, and usable sample count."
    parameters_schema = {
        "type": "object",
        "properties": {"limit": {"type": "integer", "minimum": 1, "default": 200}},
    }


class AnalyzeConvergenceTool(_AdvancedAnalysisTool):
    name = "analyze_convergence"
    diagnostic = name
    description = "Quantify best-so-far improvement, recent slope, improvement frequency, and plateau length."
    parameters_schema = ProfileHistoryQualityTool.parameters_schema


class RankParameterImportanceTool(_AdvancedAnalysisTool):
    name = "rank_parameter_importance"
    diagnostic = name
    description = "Rank numeric and categorical parameters by univariate association with the primary objective."
    parameters_schema = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "default": 200},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 32, "default": 8},
        },
    }


class AnalyzeParameterInteractionsTool(_AdvancedAnalysisTool):
    name = "analyze_parameter_interactions"
    diagnostic = name
    description = "Screen pairwise numeric parameter interactions by objective association and report data limitations."
    parameters_schema = RankParameterImportanceTool.parameters_schema


class LocatePromisingRegionsTool(_AdvancedAnalysisTool):
    name = "locate_promising_regions"
    diagnostic = name
    description = "Summarize numeric intervals and categorical modes occupied by the best observed fraction."
    parameters_schema = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "default": 200},
            "top_fraction": {"type": "number", "minimum": 0.05, "maximum": 0.5, "default": 0.25},
        },
    }


class LocateUnderexploredRegionsTool(_AdvancedAnalysisTool):
    name = "locate_underexplored_regions"
    diagnostic = name
    description = "Locate low-occupancy numeric bins and categorical values without proposing or evaluating candidates."
    parameters_schema = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "default": 200},
            "bins": {"type": "integer", "minimum": 2, "maximum": 20, "default": 5},
        },
    }


class RecommendSearchRegionsTool(_AdvancedAnalysisTool):
    name = "recommend_search_regions"
    diagnostic = name
    description = (
        "Rank at most three actionable marginal regions for exploitation, "
        "exploration, or a balanced search decision without proposing a joint candidate."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "default": 200},
            "bins": {"type": "integer", "minimum": 2, "maximum": 20, "default": 5},
            "mode": {
                "type": "string",
                "enum": ["auto", "exploit", "explore", "balanced"],
                "default": "auto",
            },
        },
    }

    async def execute(
        self,
        context: BBOToolContext,
        limit: int = 200,
        bins: int = 5,
        mode: str = "auto",
        **_: Any,
    ) -> dict[str, Any]:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"auto", "exploit", "explore", "balanced"}:
            raise ValueError("mode must be auto, exploit, explore, or balanced.")
        return _advanced_history_diagnostic(
            context,
            self.diagnostic,
            limit=max(1, int(limit)),
            top_fraction=0.25,
            top_k=8,
            bins=min(max(int(bins), 2), 20),
            mode=normalized_mode,
        )


class AnalyzeSearchStrategyTool(BaseBBOTool):
    name = "analyze_search_strategy"
    description = (
        "Infer actionable landscape structure from evaluated history and return a conservative "
        "joint search subspace, parameter biases, and a downstream optimizer policy. The tool "
        "keeps the original domain when evidence is insufficient."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "default": 200},
            "elite_fraction": {"type": "number", "minimum": 0.1, "maximum": 0.5, "default": 0.3},
            "min_width_fraction": {"type": "number", "minimum": 0.1, "maximum": 1.0, "default": 0.35},
            "max_cv_folds": {"type": "integer", "minimum": 2, "maximum": 5, "default": 5},
        },
    }

    async def execute(
        self,
        context: BBOToolContext,
        limit: int = 200,
        elite_fraction: float = 0.3,
        min_width_fraction: float = 0.35,
        max_cv_folds: int = 5,
        **_: Any,
    ) -> dict[str, Any]:
        return _analyze_search_strategy(
            context,
            limit=max(1, int(limit)),
            elite_fraction=min(max(float(elite_fraction), 0.1), 0.5),
            min_width_fraction=min(max(float(min_width_fraction), 0.1), 1.0),
            max_cv_folds=min(max(int(max_cv_folds), 2), 5),
        )


def _advanced_history_diagnostic(
    context: BBOToolContext,
    diagnostic: str,
    *,
    limit: int,
    top_fraction: float,
    top_k: int,
    bins: int,
    mode: str = "auto",
) -> dict[str, Any]:
    observations = list(context.history)[-limit:]
    primary = context.task_spec.primary_objective.name
    direction = context.task_spec.primary_objective.direction
    scored = [
        item
        for item in observations
        if item.success
        and primary in item.objectives
        and math.isfinite(float(item.objectives[primary]))
    ]
    scores = np.asarray(
        [float(item.objectives[primary]) for item in scored], dtype=float
    )
    base = {
        "diagnostic": diagnostic,
        "history_size": len(context.history),
        "inspected_count": len(observations),
        "usable_count": len(scored),
        "primary_objective": primary,
        "direction": direction.value,
        "budget_consumed": False,
    }
    if diagnostic == "profile_history_quality":
        identities = [
            stable_config_identity(item.suggestion.config)
            for item in observations
        ]
        base.update(
            {
                "status_counts": {
                    status: sum(
                        1 for item in observations if item.status.value == status
                    )
                    for status in ("success", "failed", "invalid")
                },
                "missing_primary_count": sum(
                    1 for item in observations if primary not in item.objectives
                ),
                "nonfinite_primary_count": sum(
                    1
                    for item in observations
                    if primary in item.objectives
                    and not math.isfinite(float(item.objectives[primary]))
                ),
                "duplicate_config_count": len(identities) - len(set(identities)),
                "usable_fraction": (
                    len(scored) / len(observations) if observations else 0.0
                ),
            }
        )
        return agent_visible_payload(base)
    if not scored:
        base["message"] = "No usable primary-objective observations."
        return base
    utilities = -scores if direction == ObjectiveDirection.MINIMIZE else scores
    if diagnostic == "analyze_convergence":
        best = np.maximum.accumulate(utilities)
        improvements = np.diff(best) > 1e-12
        plateau = 0
        for improved in improvements[::-1]:
            if improved:
                break
            plateau += 1
        recent = best[-min(10, len(best)) :]
        slope = (
            float(np.polyfit(np.arange(len(recent)), recent, 1)[0])
            if len(recent) >= 2
            else 0.0
        )
        base.update(
            {
                "best_so_far_utility": best.tolist(),
                "total_improvements": int(np.sum(improvements)),
                "recent_utility_slope": slope,
                "plateau_rounds": plateau,
                "stagnating": plateau >= max(5, len(best) // 5),
            }
        )
        return agent_visible_payload(base)
    if diagnostic in {
        "rank_parameter_importance",
        "analyze_parameter_interactions",
    }:
        numeric: dict[str, np.ndarray] = {}
        importance: list[dict[str, Any]] = []
        for param in context.task_spec.search_space:
            values = [item.suggestion.config[param.name] for item in scored]
            if isinstance(param, (FloatParam, IntParam)):
                xs = np.asarray(values, dtype=float)
                numeric[param.name] = xs
                association = (
                    0.0
                    if len(xs) < 2
                    or float(np.std(xs)) == 0.0
                    or float(np.std(utilities)) == 0.0
                    else abs(float(np.corrcoef(xs, utilities)[0, 1]))
                )
            elif isinstance(param, CategoricalParam):
                means = []
                for choice in param.choices:
                    selected = [
                        float(utility)
                        for value, utility in zip(values, utilities, strict=True)
                        if value == choice
                    ]
                    if selected:
                        means.append(float(np.mean(selected)))
                association = (
                    0.0
                    if len(means) < 2
                    else float(np.std(means))
                    / max(float(np.std(utilities)), 1e-12)
                )
            else:
                continue
            importance.append(
                {"parameter": param.name, "association": association}
            )
        importance.sort(key=lambda item: item["association"], reverse=True)
        if diagnostic == "rank_parameter_importance":
            base["ranking"] = importance[:top_k]
            base["method"] = "absolute Pearson / normalized category-mean spread"
            return agent_visible_payload(base)
        interactions = []
        names = sorted(numeric)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                product = (
                    (numeric[left] - np.mean(numeric[left]))
                    * (numeric[right] - np.mean(numeric[right]))
                )
                association = (
                    0.0
                    if len(product) < 3
                    or float(np.std(product)) == 0.0
                    or float(np.std(utilities)) == 0.0
                    else abs(float(np.corrcoef(product, utilities)[0, 1]))
                )
                interactions.append(
                    {
                        "parameters": [left, right],
                        "association": association,
                    }
                )
        interactions.sort(
            key=lambda item: item["association"], reverse=True
        )
        base["ranking"] = interactions[:top_k]
        base["method"] = "centered-product association screen"
        base["caution"] = "Screening evidence, not a causal interaction estimate."
        return agent_visible_payload(base)
    if diagnostic == "locate_promising_regions":
        count = max(1, int(math.ceil(len(scored) * top_fraction)))
        order = np.argsort(utilities)[::-1][:count]
        regions: dict[str, Any] = {}
        for param in context.task_spec.search_space:
            values = [
                scored[int(index)].suggestion.config[param.name]
                for index in order
            ]
            if isinstance(param, (FloatParam, IntParam)):
                numeric_values = np.asarray(values, dtype=float)
                regions[param.name] = {
                    "low": float(np.min(numeric_values)),
                    "high": float(np.max(numeric_values)),
                    "median": float(np.median(numeric_values)),
                }
            elif isinstance(param, CategoricalParam):
                counts = {
                    str(choice): values.count(choice)
                    for choice in param.choices
                    if values.count(choice)
                }
                regions[param.name] = {
                    "counts": counts,
                    "mode": max(counts, key=counts.get) if counts else None,
                }
        base.update(
            {
                "top_fraction": top_fraction,
                "top_count": count,
                "regions": regions,
            }
        )
        return agent_visible_payload(base)
    if diagnostic == "locate_underexplored_regions":
        regions = {}
        for param in context.task_spec.search_space:
            values = [item.suggestion.config[param.name] for item in scored]
            if isinstance(param, (FloatParam, IntParam)):
                edges = np.linspace(float(param.low), float(param.high), bins + 1)
                counts, _ = np.histogram(np.asarray(values, dtype=float), edges)
                minimum = int(np.min(counts))
                regions[param.name] = [
                    {
                        "low": float(edges[index]),
                        "high": float(edges[index + 1]),
                        "count": int(count),
                    }
                    for index, count in enumerate(counts)
                    if int(count) == minimum
                ]
            elif isinstance(param, CategoricalParam):
                counts = {
                    str(choice): values.count(choice)
                    for choice in param.choices
                }
                minimum = min(counts.values()) if counts else 0
                regions[param.name] = {
                    key: count
                    for key, count in counts.items()
                    if count == minimum
                }
        base.update({"bins": bins, "underexplored_regions": regions})
        return agent_visible_payload(base)
    if diagnostic == "recommend_search_regions":
        best: list[float] = []
        running = -math.inf
        for utility in utilities:
            running = max(running, float(utility))
            best.append(running)
        improvements = np.diff(np.asarray(best, dtype=float)) > 1e-12
        plateau = 0
        for improved in improvements[::-1]:
            if bool(improved):
                break
            plateau += 1
        recent_improvements = int(np.sum(improvements[-10:]))
        sparse_threshold = max(10, 2 * len(context.task_spec.search_space))
        if mode == "auto":
            if len(scored) < sparse_threshold:
                selected_mode = "explore"
                mode_reason = "usable history is sparse relative to search-space size"
            elif plateau >= max(5, len(best) // 5):
                selected_mode = "explore"
                mode_reason = "best-so-far history is stagnant"
            elif recent_improvements >= 2:
                selected_mode = "exploit"
                mode_reason = "recent observations repeatedly improved the incumbent"
            else:
                selected_mode = "balanced"
                mode_reason = "history supports neither pure exploitation nor pure exploration"
        else:
            selected_mode = mode
            mode_reason = "mode explicitly requested by the agent"

        recommendations: dict[str, Any] = {}
        for param in context.task_spec.search_space:
            values = [item.suggestion.config[param.name] for item in scored]
            cells: list[dict[str, Any]] = []
            if isinstance(param, (FloatParam, IntParam)):
                edges = np.linspace(float(param.low), float(param.high), bins + 1)
                assignments = np.clip(
                    np.digitize(np.asarray(values, dtype=float), edges[1:-1]),
                    0,
                    bins - 1,
                )
                for index in range(bins):
                    selected = [
                        float(utility)
                        for utility, assignment in zip(utilities, assignments, strict=True)
                        if int(assignment) == index
                    ]
                    cells.append(
                        {
                            "region": {
                                "low": float(edges[index]),
                                "high": float(edges[index + 1]),
                            },
                            "count": len(selected),
                            "mean_utility": None if not selected else float(np.mean(selected)),
                        }
                    )
            elif isinstance(param, CategoricalParam):
                for choice in param.choices:
                    selected = [
                        float(utility)
                        for value, utility in zip(values, utilities, strict=True)
                        if value == choice
                    ]
                    cells.append(
                        {
                            "region": {"value": choice},
                            "count": len(selected),
                            "mean_utility": None if not selected else float(np.mean(selected)),
                        }
                    )
            if not cells:
                continue
            observed_means = [
                float(cell["mean_utility"])
                for cell in cells
                if cell["mean_utility"] is not None
            ]
            utility_low = min(observed_means) if observed_means else 0.0
            raw_span = max(observed_means) - utility_low if observed_means else 0.0
            maximum_count = max(int(cell["count"]) for cell in cells)
            for cell in cells:
                exploit_score = (
                    0.0
                    if cell["mean_utility"] is None
                    else (
                        0.5
                        if raw_span <= 1e-12
                        else (float(cell["mean_utility"]) - utility_low) / raw_span
                    )
                )
                explore_score = 1.0 - int(cell["count"]) / max(maximum_count, 1)
                cell["exploit_score"] = exploit_score
                cell["explore_score"] = explore_score
                cell["balanced_score"] = 0.7 * exploit_score + 0.3 * explore_score
            score_key = f"{selected_mode}_score"
            ranked = sorted(cells, key=lambda cell: float(cell[score_key]), reverse=True)
            recommendations[param.name] = {
                "selected": ranked[0],
                "alternatives": ranked[1:3],
            }
        max_actionable_parameters = min(3, len(recommendations))
        parameter_ranking = []
        score_key = f"{selected_mode}_score"
        for name, recommendation in recommendations.items():
            selected = recommendation["selected"]
            alternatives = recommendation["alternatives"]
            next_score = max(
                (float(item[score_key]) for item in alternatives),
                default=0.0,
            )
            parameter_ranking.append(
                {
                    "parameter": name,
                    "priority_score": float(selected[score_key]),
                    "balanced_support_score": float(selected["balanced_score"]),
                    "score_margin": float(selected[score_key]) - next_score,
                }
            )
        parameter_ranking.sort(
            key=lambda item: (
                float(item["priority_score"]),
                float(item["balanced_support_score"]),
                float(item["score_margin"]),
                str(item["parameter"]),
            ),
            reverse=True,
        )
        active_names = [
            str(item["parameter"])
            for item in parameter_ranking[:max_actionable_parameters]
        ]
        actionable_regions = {
            name: recommendations[name] for name in active_names
        }
        context_only_parameters = [
            str(item["parameter"])
            for item in parameter_ranking[max_actionable_parameters:]
        ]
        base.update(
            {
                "requested_mode": mode,
                "recommended_mode": selected_mode,
                "mode_reason": mode_reason,
                "bins": bins,
                "plateau_rounds": plateau,
                "recent_improvements": recent_improvements,
                "max_actionable_parameters": max_actionable_parameters,
                "actionable_regions": actionable_regions,
                "context_only_parameters": context_only_parameters,
                "parameter_priority_ranking": parameter_ranking,
                "joint_action_policy": (
                    "Change only actionable-region parameters from the incumbent unless a "
                    "successful interaction analysis, joint candidate score, or optimizer "
                    "decision supports a wider move."
                ),
                "method": "per-parameter utility and occupancy ranking with a top-3 action cap",
                "caution": (
                    "Marginal evidence only. Context-only parameters are not recommendations; "
                    "never concatenate all marginal regions into one joint candidate."
                ),
            }
        )
        return agent_visible_payload(base)
    raise ValueError(f"Unknown advanced history diagnostic {diagnostic!r}.")


def _quadratic_features(values: np.ndarray) -> np.ndarray:
    columns = [values]
    columns.append(values * values)
    interactions = [
        (values[:, left] * values[:, right]).reshape(-1, 1)
        for left in range(values.shape[1])
        for right in range(left + 1, values.shape[1])
    ]
    if interactions:
        columns.append(np.column_stack(interactions))
    return np.column_stack(columns)


def _cv_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denominator <= 1e-12:
        return None
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denominator


def _cross_validate_landscape_models(
    context: BBOToolContext,
    scored: list[TrialObservation],
    utilities: np.ndarray,
    *,
    optimizer_bounds: dict[str, list[float | int]],
    max_folds: int,
) -> dict[str, Any]:
    feature_spec, limitations = _surrogate_feature_spec(context.task_spec.search_space)
    minimum = max(8, len(feature_spec) + 3)
    result: dict[str, Any] = {
        "status": "insufficient_data" if len(scored) < minimum else "pending",
        "sample_count": len(scored), "minimum_sample_count": minimum,
        "folds": 0, "models": {}, "local_scope_count": 0,
        "global_gp_on_local_r2": None, "local_gp_r2": None,
        "local_gp_gain_over_global": None,
        "shape_hypothesis": "unavailable", "limitations": limitations,
        "gp_backend": "botorch.SingleTaskGP",
    }
    if limitations or len(scored) < minimum:
        return result

    x_values = np.asarray([_encode_config_from_spec(item.suggestion.config, feature_spec) for item in scored], dtype=float)
    y_values = np.asarray(utilities, dtype=float)
    fold_count = min(max_folds, len(scored))
    rng = np.random.default_rng(int(context.seed))
    all_indices = np.arange(len(scored))
    folds = np.array_split(rng.permutation(all_indices), fold_count)
    linear_predictions = np.empty_like(y_values)
    quadratic_predictions = np.empty_like(y_values)
    for held_out in folds:
        train = np.setdiff1d(all_indices, held_out)
        linear = _fit_ridge(x_values[train], y_values[train], ridge=1e-4)
        linear_predictions[held_out] = _predict_ridge(x_values[held_out], linear)
        train_quadratic = _quadratic_features(x_values[train])
        quadratic = _fit_ridge(train_quadratic, y_values[train], ridge=1e-3)
        quadratic_predictions[held_out] = _predict_ridge(_quadratic_features(x_values[held_out]), quadratic)

    from ...baseline_factory import create_comparable_baseline
    gp = create_comparable_baseline("gp_ei", overrides={"acquisition": "logei"})
    gp.setup(context.task_spec, seed=int(context.seed))
    gp.replay(scored)
    gp_validation = gp.cross_validate_landscape(local_bounds=optimizer_bounds, max_folds=max_folds)
    result["models"] = {
        "linear": {"cv_r2": _cv_r2(y_values, linear_predictions), "ranking_accuracy": _pairwise_ranking_accuracy(y_values, linear_predictions)},
        "quadratic": {"cv_r2": _cv_r2(y_values, quadratic_predictions), "ranking_accuracy": _pairwise_ranking_accuracy(y_values, quadratic_predictions)},
        "global_gp": {"cv_r2": gp_validation.get("global_gp_cv_r2"), "ranking_accuracy": gp_validation.get("global_gp_ranking_accuracy")},
    }
    result.update(
        status=gp_validation.get("status", "fit_failed"), folds=fold_count,
        local_scope_count=gp_validation.get("local_scope_count", 0),
        global_gp_on_local_r2=gp_validation.get("global_gp_on_local_r2"),
        local_gp_r2=gp_validation.get("local_gp_r2"),
        local_gp_gain_over_global=gp_validation.get("local_gp_gain_over_global"),
        limitations=[*limitations, *list(gp_validation.get("limitations") or [])],
    )
    linear_r2 = result["models"]["linear"]["cv_r2"]
    quadratic_r2 = result["models"]["quadratic"]["cv_r2"]
    gp_r2 = result["models"]["global_gp"]["cv_r2"]
    if isinstance(gp_r2, (int, float)) and isinstance(quadratic_r2, (int, float)) and gp_r2 >= quadratic_r2 + 0.1:
        result["shape_hypothesis"] = "coupled_or_multimodal"
    elif isinstance(quadratic_r2, (int, float)) and isinstance(linear_r2, (int, float)) and quadratic_r2 >= linear_r2 + 0.1:
        result["shape_hypothesis"] = "curved_or_unimodal"
    else:
        result["shape_hypothesis"] = "approximately_monotonic_or_unresolved"
    return agent_visible_payload(result)


def _analyze_search_strategy(
    context: BBOToolContext,
    *,
    limit: int,
    elite_fraction: float,
    min_width_fraction: float,
    max_cv_folds: int,
) -> dict[str, Any]:
    primary = context.task_spec.primary_objective.name
    direction = context.task_spec.primary_objective.direction
    inspected = list(context.history)[-limit:]
    scored = [
        item for item in inspected
        if item.success and primary in item.objectives
        and math.isfinite(float(item.objectives[primary]))
    ]
    dimension = len(context.task_spec.search_space)
    minimum_for_structure = max(8, 2 * dimension)
    base: dict[str, Any] = {
        "history_size": len(context.history),
        "usable_count": len(scored),
        "minimum_for_structure": minimum_for_structure,
        "evidence_sufficient": len(scored) >= minimum_for_structure,
        "primary_objective": primary,
        "direction": direction.value,
        "budget_consumed": False,
    }
    if not scored:
        base.update(
            landscape={"status": "no_usable_observations"},
            bias={"status": "none"},
            recommended_subspace={"apply": False, "reason": "no usable observations", "optimizer_bounds": {}},
            downstream_policy={"mode": "explore", "acquisition": "ucb", "reason": "no usable observations"},
        )
        return base

    scores = np.asarray([float(item.objectives[primary]) for item in scored], dtype=float)
    utilities = -scores if direction == ObjectiveDirection.MINIMIZE else scores
    order = np.argsort(utilities)[::-1]
    elite_count = max(2, int(math.ceil(len(scored) * elite_fraction)))
    elite_indices = order[:elite_count]
    parameter_signals: list[dict[str, Any]] = []
    optimizer_bounds: dict[str, list[float | int]] = {}
    categorical_bias: dict[str, Any] = {}

    for param in context.task_spec.search_space:
        values = [item.suggestion.config[param.name] for item in scored]
        elite_values = [values[int(index)] for index in elite_indices]
        if isinstance(param, (FloatParam, IntParam)):
            xs = np.asarray(values, dtype=float)
            ranks = np.argsort(np.argsort(xs)).astype(float)
            utility_ranks = np.argsort(np.argsort(utilities)).astype(float)
            association = (
                0.0 if np.std(ranks) == 0 or np.std(utility_ranks) == 0
                else float(np.corrcoef(ranks, utility_ranks)[0, 1])
            )
            span = max(float(param.high) - float(param.low), 1e-12)
            elite_numeric = np.asarray(elite_values, dtype=float)
            raw_low = float(np.quantile(elite_numeric, 0.1))
            raw_high = float(np.quantile(elite_numeric, 0.9))
            target_width = max(raw_high - raw_low, min_width_fraction * span)
            center = float(np.median(elite_numeric))
            low = max(float(param.low), center - target_width / 2.0)
            high = min(float(param.high), center + target_width / 2.0)
            if high - low < target_width:
                if low <= float(param.low) + 1e-12:
                    high = min(float(param.high), low + target_width)
                else:
                    low = max(float(param.low), high - target_width)
            if isinstance(param, IntParam):
                bound = [int(math.floor(low)), int(math.ceil(high))]
            else:
                bound = [low, high]
            optimizer_bounds[param.name] = bound
            parameter_signals.append(
                {
                    "parameter": param.name,
                    "kind": "numeric",
                    "rank_association": association,
                    "strength": abs(association),
                    "preferred_direction": "higher" if association > 0.15 else "lower" if association < -0.15 else "unclear",
                    "elite_median": center,
                    "proposed_bound": bound,
                }
            )
        elif isinstance(param, CategoricalParam):
            all_counts = {str(choice): values.count(choice) for choice in param.choices}
            elite_counts = {str(choice): elite_values.count(choice) for choice in param.choices}
            best_choice = max(elite_counts, key=elite_counts.get) if elite_counts else None
            elite_share = 0.0 if best_choice is None else elite_counts[best_choice] / elite_count
            background_share = 0.0 if best_choice is None else all_counts[best_choice] / len(values)
            enrichment = elite_share - background_share
            categorical_bias[param.name] = {
                "preferred_values": [] if best_choice is None else [best_choice],
                "elite_share": elite_share,
                "background_share": background_share,
                "enrichment": enrichment,
                "apply_as_hard_restriction": bool(len(scored) >= minimum_for_structure and elite_counts.get(best_choice, 0) >= 2 and enrichment >= 0.2),
            }
            parameter_signals.append(
                {"parameter": param.name, "kind": "categorical", "strength": max(enrichment, 0.0), "preferred_value": best_choice}
            )

    parameter_signals.sort(key=lambda item: float(item["strength"]), reverse=True)
    active_parameters = [item["parameter"] for item in parameter_signals[: min(3, len(parameter_signals))]]
    subspace_sample_supported = len(scored) >= minimum_for_structure and elite_count >= 3
    candidate_bounds = dict(optimizer_bounds) if subspace_sample_supported else {}

    running_best = np.maximum.accumulate(utilities)
    improvements = np.diff(running_best) > 1e-12
    plateau = 0
    for improved in improvements[::-1]:
        if improved:
            break
        plateau += 1
    model_validation = _cross_validate_landscape_models(
        context,
        scored,
        utilities,
        optimizer_bounds=candidate_bounds,
        max_folds=max_cv_folds,
    )
    local_r2 = model_validation.get("local_gp_r2")
    local_gain = model_validation.get("local_gp_gain_over_global")
    subspace_cv_supported = bool(
        isinstance(local_r2, (int, float))
        and isinstance(local_gain, (int, float))
        and local_r2 > 0.0
        and local_gain >= 0.1
    )
    apply_subspace = subspace_sample_supported and subspace_cv_supported
    optimizer_bounds = candidate_bounds if apply_subspace else {}
    concentrated = sum(float(item["strength"]) for item in parameter_signals[:3]) >= 1.0
    if len(scored) < minimum_for_structure or plateau >= max(5, len(scored) // 5):
        mode, acquisition, strategy = "explore", "ucb", "global_bo"
        reason = "history is sparse or stagnant; preserve uncertainty-driven exploration"
    elif isinstance(local_gain, (int, float)) and local_gain >= 0.1:
        mode, acquisition, strategy = "exploit", "logei", "local_bo"
        reason = "local GP cross-validation improves on the global GP by at least 0.1 R2"
    elif concentrated:
        mode, acquisition, strategy = "exploit", "logei", "focused_global_bo"
        reason = "repeatable parameter bias is concentrated in a small active set"
    else:
        mode, acquisition, strategy = "balanced", "logei", "global_bo"
        reason = "evidence supports refinement but not aggressive restriction"

    base.update(
        landscape={
            "status": "screened" if subspace_sample_supported else "tentative",
            "parameter_signals": parameter_signals,
            "active_parameters": active_parameters,
            "effective_dimension_hypothesis": len([item for item in parameter_signals if float(item["strength"]) >= 0.2]),
            "plateau_rounds": plateau,
            "model_validation": model_validation,
            "shape_hypothesis": model_validation.get("shape_hypothesis", "unavailable"),
            "caution": "Associations describe observed history and may reflect adaptive-sampling bias; they are hypotheses, not causal effects.",
        },
        bias={
            "anchor": None if context.incumbent is None else agent_visible_config(context.incumbent.config),
            "active_parameters": active_parameters,
            "categorical": categorical_bias,
            "status": "supported" if subspace_sample_supported else "tentative",
        },
        recommended_subspace={
            "apply": apply_subspace,
            "reason": (
                "local GP cross-validation supports the candidate subspace"
                if apply_subspace
                else "candidate subspace is not validated; keep the original domain"
                if subspace_sample_supported
                else "insufficient evidence; keep the original domain"
            ),
            "optimizer_bounds": optimizer_bounds,
            "candidate_bounds": candidate_bounds,
            "sample_supported": subspace_sample_supported,
            "cv_supported": subspace_cv_supported,
            "elite_fraction": elite_fraction,
            "elite_count": elite_count,
            "min_width_fraction": min_width_fraction,
            "reversible": True,
        },
        downstream_policy={"mode": mode, "acquisition": acquisition, "strategy": strategy, "reason": reason},
    )
    return agent_visible_payload(base)

class RenderSearchDiagnosticsTool(BaseBBOTool):
    """Render an offline JSON/SVG search diagnostic artifact."""

    name = "render_search_diagnostics"
    description = "Render convergence and parameter-importance diagnostics to run-local JSON and SVG artifacts."
    parameters_schema = {
        "type": "object",
        "properties": {"title": {"type": "string", "default": "Search diagnostics"}},
    }

    async def execute(
        self,
        context: BBOToolContext,
        title: str = "Search diagnostics",
        **_: Any,
    ) -> dict[str, Any]:
        convergence = _advanced_history_diagnostic(
            context, "analyze_convergence", limit=500, top_fraction=0.25, top_k=8, bins=5
        )
        importance = _advanced_history_diagnostic(
            context, "rank_parameter_importance", limit=500, top_fraction=0.25, top_k=8, bins=5
        )
        artifacts_dir = context.workspace_dir / "analysis_artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        stem = f"search_diagnostics_{len(context.history):04d}"
        json_path = artifacts_dir / f"{stem}.json"
        svg_path = artifacts_dir / f"{stem}.svg"
        payload = {
            "title": str(title)[:200],
            "history_size": len(context.history),
            "convergence": convergence,
            "parameter_importance": importance,
            "budget_consumed": False,
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        progression = list(convergence.get("best_so_far_utility") or [])
        points = _svg_progression_points(progression)
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="300" viewBox="0 0 720 300">'
            '<rect width="720" height="300" fill="white"/>'
            f'<text x="24" y="30" font-size="18">{_xml_escape(str(title)[:100])}</text>'
            '<line x1="40" y1="260" x2="700" y2="260" stroke="#777"/>'
            '<line x1="40" y1="50" x2="40" y2="260" stroke="#777"/>'
            f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="3"/>'
            '<text x="580" y="286" font-size="12">optimization round</text>'
            "</svg>\n"
        )
        svg_path.write_text(svg, encoding="utf-8")
        return {
            "rendered": True,
            "json_path": str(json_path.relative_to(context.workspace_dir)),
            "svg_path": str(svg_path.relative_to(context.workspace_dir)),
            "history_size": len(context.history),
            "budget_consumed": False,
        }


def _svg_progression_points(values: list[Any]) -> str:
    if not values:
        return ""
    numeric = [float(value) for value in values]
    low = min(numeric)
    span = max(max(numeric) - low, 1e-12)
    denominator = max(len(numeric) - 1, 1)
    return " ".join(
        f"{40 + 660 * index / denominator:.2f},{260 - 210 * (value - low) / span:.2f}"
        for index, value in enumerate(numeric)
    )


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


class MemoryReadTool(BaseBBOTool):
    name = "memory_read"
    description = "Read append-only BBO agent memory records."
    parameters_schema = {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": "integer", "minimum": 1, "default": 20},
        },
    }

    async def execute(
        self,
        context: BBOToolContext,
        kind: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
        **_: Any,
    ) -> dict[str, Any]:
        if context.memory_store is None:
            return {"enabled": False, "records": []}
        records = context.memory_store.read(kind=kind, tags=tags, limit=int(limit))
        return {"enabled": True, "records": records, "count": len(records)}


class MemoryWriteTool(BaseBBOTool):
    name = "memory_write"
    description = "Append hypotheses, priors, failure notes, or strategy notes to BBO memory."
    parameters_schema = {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "content": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "source_call_id": {"type": "string"},
            "trial_range": {"type": "array", "items": {"type": "integer"}},
            "metadata": {"type": "object"},
        },
        "required": ["kind", "content"],
    }

    async def execute(
        self,
        context: BBOToolContext,
        kind: str,
        content: str,
        tags: list[str] | None = None,
        source_call_id: str | None = None,
        trial_range: list[int] | None = None,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if context.memory_store is None:
            return {"enabled": False, "written": False}
        record = context.memory_store.append(
            kind=kind,
            content=content,
            tags=tags or (),
            source_call_id=source_call_id,
            trial_range=trial_range,
            metadata=metadata,
        )
        return {"enabled": True, "written": True, "record": record}


def create_core_BBO_tools(*, enable_memory: bool = True) -> list[BaseBBOTool]:
    tools: list[BaseBBOTool] = [
        GetTaskContextTool(),
        GetSearchSpaceTool(),
        GetTrialHistoryTool(),
        GetIncumbentTool(),
        GetHistoryOverviewTool(),
        CompareTrialsTool(),
        FindNearestTrialsTool(),
        EstimateLocalEffectsTool(),
        MeasureSearchCoverageTool(),
        SummarizeObjectiveMetricsTool(),
        FitAndCheckSurrogateTool(),
        ScoreVirtualCandidatesTool(),
        ValidateCandidateTool(),
        ValidateCandidatesTool(),
        GetRecentSearchActionsTool(),
        SampleCandidatesTool(),
        AnalyzeHistoryTool(),
        ProfileHistoryQualityTool(),
        AnalyzeConvergenceTool(),
        RankParameterImportanceTool(),
        AnalyzeParameterInteractionsTool(),
        LocatePromisingRegionsTool(),
        LocateUnderexploredRegionsTool(),
        RecommendSearchRegionsTool(),
        AnalyzeSearchStrategyTool(),
        RenderSearchDiagnosticsTool(),
    ]
    if enable_memory:
        tools.extend([MemoryReadTool(), MemoryWriteTool()])
    return tools


def search_space_schema(search_space: SearchSpace) -> list[dict[str, Any]]:
    return search_space_to_schema(search_space)


def _sort_by_primary(observations: list[TrialObservation], context: BBOToolContext) -> list[TrialObservation]:
    primary = context.task_spec.primary_objective.name
    direction = context.task_spec.primary_objective.direction
    scored = [obs for obs in observations if obs.success and primary in obs.objectives]
    reverse = direction == ObjectiveDirection.MAXIMIZE
    return sorted(scored, key=lambda obs: float(obs.objectives[primary]), reverse=reverse)


def _objective_metrics_summary(
    observations: list[TrialObservation],
    *,
    primary: str,
    direction: ObjectiveDirection,
    recent_limit: int,
    progression_limit: int,
) -> dict[str, Any]:
    scored: list[tuple[TrialObservation, float]] = []
    best_so_far: list[dict[str, Any]] = []
    best_observation: TrialObservation | None = None
    best_score: float | None = None
    for observation in observations:
        score = _primary_score(observation, primary)
        if score is None:
            continue
        scored.append((observation, score))
        if best_score is None or _is_better(score, best_score, direction):
            best_score = score
            best_observation = observation
        best_so_far.append(
            {
                "trial_id": observation.suggestion.trial_id,
                "score": agent_visible_payload(score),
                "best_so_far": agent_visible_payload(best_score),
            }
        )
    recent_scored = scored[-recent_limit:]
    scores = [score for _, score in scored]
    recent_scores = [score for _, score in recent_scored]
    metric_values: dict[str, list[float]] = {}
    metric_last: dict[str, float] = {}
    for observation, _score in scored:
        visible_metrics = agent_visible_metrics(observation.metrics)
        if not isinstance(visible_metrics, dict):
            continue
        for key, value in visible_metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                continue
            metric_values.setdefault(str(key), []).append(numeric)
            metric_last[str(key)] = numeric
    numeric_metrics = {
        key: {
            "count": len(values),
            "min": agent_visible_payload(min(values)),
            "max": agent_visible_payload(max(values)),
            "mean": agent_visible_payload(sum(values) / len(values)),
            "last": agent_visible_payload(metric_last[key]),
        }
        for key, values in sorted(metric_values.items())
        if values
    }
    current_observation, current_score = scored[-1] if scored else (None, None)
    previous_score = scored[-2][1] if len(scored) >= 2 else None
    recent_delta = None if previous_score is None or current_score is None else current_score - previous_score
    improvement_count = 0
    for (_left_obs, left), (_right_obs, right) in zip(recent_scored, recent_scored[1:], strict=False):
        if _is_better(right, left, direction):
            improvement_count += 1
    objective_summary = {
        "count": len(scores),
        "min": None if not scores else agent_visible_payload(min(scores)),
        "max": None if not scores else agent_visible_payload(max(scores)),
        "mean": None if not scores else agent_visible_payload(sum(scores) / len(scores)),
        "best": agent_visible_payload(best_score),
        "current": agent_visible_payload(current_score),
        "current_delta_from_best": None
        if current_score is None or best_score is None
        else agent_visible_payload(current_score - best_score),
        "recent_delta": agent_visible_payload(recent_delta),
        "recent_improvement_count": improvement_count,
    }
    return {
        "primary_objective": primary,
        "direction": direction.value,
        "evaluated_count": len(observations),
        "success_count": len(scored),
        "failed_count": sum(1 for obs in observations if obs.status.value == "failed"),
        "invalid_count": sum(1 for obs in observations if obs.status.value == "invalid"),
        "objective": objective_summary,
        "best_trial": None if best_observation is None else _observation_summary(best_observation),
        "current_trial": None if current_observation is None else _observation_summary(current_observation),
        "best_so_far_progression": best_so_far[-progression_limit:],
        "recent_objectives": [
            {
                "trial_id": observation.suggestion.trial_id,
                "objective": agent_visible_payload(score),
            }
            for observation, score in recent_scored
        ],
        "numeric_metrics": numeric_metrics,
        "budget_consumed": False,
    }


def _primary_score(observation: TrialObservation, primary: str) -> float | None:
    if not observation.success or primary not in observation.objectives:
        return None
    return float(observation.objectives[primary])


def _is_better(score: float, reference: float, direction: ObjectiveDirection) -> bool:
    if direction == ObjectiveDirection.MAXIMIZE:
        return score > reference
    return score < reference


def _trial_label(observation: TrialObservation) -> str:
    trial_id = observation.suggestion.trial_id
    return f"trial_{trial_id}" if trial_id is not None else "trial_unknown"


def _parse_trial_identifier(identifier: int | str) -> int:
    if isinstance(identifier, int):
        return identifier
    text = str(identifier).strip()
    if text.startswith("trial_"):
        text = text.removeprefix("trial_")
    return int(text)


def _resolve_trial(history: list[TrialObservation], identifier: int | str) -> TrialObservation:
    trial_id = _parse_trial_identifier(identifier)
    for observation in history:
        if observation.suggestion.trial_id == trial_id:
            return observation
    raise ValueError(f"Unknown trial identifier: {identifier!r}")


def _target_config(context: BBOToolContext, target: Any) -> tuple[dict[str, Any], int | None]:
    if isinstance(target, (int, str)) and not (isinstance(target, str) and target.strip().startswith("{")):
        observation = _resolve_trial(context.history, target)
        return dict(observation.suggestion.config), observation.suggestion.trial_id
    if isinstance(target, dict):
        raw = target.get("config", target)
        if not isinstance(raw, dict):
            raise ValueError("target `config` must be an object.")
        return context.task_spec.search_space.coerce_config(raw, use_defaults=False), None
    raise ValueError("target must be a trial identifier, raw config, or object with a `config` field.")


def _changed_variables(search_space: SearchSpace, left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for param in search_space:
        old = left.get(param.name)
        new = right.get(param.name)
        if old == new:
            continue
        item: dict[str, Any] = {
            "name": param.name,
            "old": agent_visible_payload(old),
            "new": agent_visible_payload(new),
        }
        if isinstance(param, (FloatParam, IntParam)):
            span = max(float(param.high) - float(param.low), 1e-12)
            try:
                item["normalized_delta"] = agent_visible_payload((float(new) - float(old)) / span)
            except Exception:
                pass
        changes.append(item)
    return changes


def _unchanged_variables(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    return sorted(name for name in set(left) & set(right) if left.get(name) == right.get(name))


def _search_action_summary(observation: TrialObservation) -> dict[str, Any]:
    metadata = dict(observation.suggestion.metadata or {})
    action = metadata.get("search_action")
    payload: dict[str, Any] = dict(action) if isinstance(action, dict) else {}
    for key in (
        "skill",
        "primary_skill",
        "search_intent",
        "parent_trials",
        "reference_trials",
        "modified_variables",
        "hypothesis",
        "change_summary",
        "expected_evidence",
        "repaired",
        "repair",
    ):
        if key in metadata and key not in payload:
            payload[key] = metadata[key]
    if "skill" not in payload and "primary_skill" in payload:
        payload["skill"] = payload["primary_skill"]
    if "search_intent" not in payload:
        payload["search_intent"] = metadata.get("agent_search_intent") or metadata.get("agent_source") or "unknown"
    return agent_visible_payload(payload)


def _config_distance(
    search_space: SearchSpace,
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[float | None, list[dict[str, Any]], list[str]]:
    contributions: list[float] = []
    details: list[dict[str, Any]] = []
    limitations: list[str] = []
    for param in search_space:
        if param.name not in left or param.name not in right:
            limitations.append(f"Parameter `{param.name}` missing from one config.")
            continue
        a = left[param.name]
        b = right[param.name]
        if isinstance(param, (FloatParam, IntParam)):
            span = max(float(param.high) - float(param.low), 1e-12)
            contribution = abs(float(a) - float(b)) / span
        elif isinstance(param, CategoricalParam):
            contribution = 0.0 if a == b else 1.0
        elif isinstance(param, StringParam):
            contribution = 0.0 if a == b else 1.0
            note = f"Open string parameter `{param.name}` uses exact-match distance only; semantic distance is undefined."
            if note not in limitations:
                limitations.append(note)
        else:
            limitations.append(f"Unsupported parameter type for `{param.name}`.")
            continue
        contributions.append(float(contribution))
        if contribution > 0:
            details.append(
                {
                    "name": param.name,
                    "left": agent_visible_payload(a),
                    "right": agent_visible_payload(b),
                    "distance_contribution": agent_visible_payload(contribution),
                }
            )
    if not contributions:
        return None, details, limitations
    details.sort(key=lambda item: float(item.get("distance_contribution", 0.0)), reverse=True)
    distance = math.sqrt(sum(value * value for value in contributions) / len(contributions))
    return float(distance), details, limitations


def _average_pairwise_distance(search_space: SearchSpace, configs: list[dict[str, Any]]) -> float | None:
    distances: list[float] = []
    for left_index, left in enumerate(configs):
        for right in configs[left_index + 1 :]:
            distance, _, _ = _config_distance(search_space, left, right)
            if distance is not None:
                distances.append(distance)
    if not distances:
        return None
    return sum(distances) / len(distances)


def _recent_repeated_variables(search_space: SearchSpace, observations: list[TrialObservation]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for left, right in zip(observations, observations[1:]):
        for change in _changed_variables(search_space, left.suggestion.config, right.suggestion.config):
            counts[change["name"]] = counts.get(change["name"], 0) + 1
    threshold = max(2, len(observations) // 2)
    return {
        "counts": counts,
        "repeated": sorted(name for name, count in counts.items() if count >= threshold),
    }


def _validate_candidate_report(
    context: BBOToolContext,
    candidate: dict[str, Any],
    *,
    too_similar_threshold: float | None = None,
) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        return {
            "valid": False,
            "violations": [{"field": None, "rule": "candidate_object", "message": "candidate is not an object", "repairable": False}],
            "repairable": False,
            "recommended_minimal_repairs": [],
        }
    raw = candidate.get("config", candidate)
    if not isinstance(raw, dict):
        return {
            "valid": False,
            "violations": [{"field": "config", "rule": "object", "message": "`config` is not an object", "repairable": False}],
            "repairable": False,
            "recommended_minimal_repairs": [],
        }
    expected = set(context.task_spec.search_space.names())
    provided = set(raw)
    violations: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    for name in sorted(expected - provided):
        violations.append({"field": name, "rule": "required", "message": "missing required parameter", "repairable": False})
    for name in sorted(provided - expected):
        violations.append({"field": name, "rule": "unexpected", "message": "unexpected extra parameter", "repairable": True})
        repairs.append({"field": name, "operation": "remove"})
    coerced: dict[str, Any] = {}
    for param in context.task_spec.search_space:
        if param.name not in raw:
            continue
        value = raw[param.name]
        try:
            coerced[param.name] = param.coerce(value)
        except Exception as exc:
            repairable = isinstance(param, (FloatParam, IntParam)) and not isinstance(value, bool)
            repair: dict[str, Any] | None = None
            if repairable:
                try:
                    numeric = float(value)
                    clipped = min(max(numeric, float(param.low)), float(param.high))
                    repaired_value: Any = int(round(clipped)) if isinstance(param, IntParam) else clipped
                    repaired_value = param.coerce(repaired_value)
                    repair = {"field": param.name, "operation": "clip_to_bounds", "value": repaired_value}
                except Exception:
                    repairable = False
            violations.append({"field": param.name, "rule": "type_or_bounds", "message": str(exc), "repairable": repairable})
            if repair is not None:
                repairs.append(repair)
    identity: str | None = None
    duplicate = False
    nearest_distance: float | None = None
    too_similar = False
    if not violations:
        identity = stable_config_identity(coerced)
        history_ids = {stable_config_identity(observation.suggestion.config) for observation in context.history}
        duplicate = identity in history_ids
        if duplicate:
            violations.append({"field": None, "rule": "duplicate", "message": "candidate exactly matches evaluated history", "repairable": False})
        if context.history:
            distances = [
                _config_distance(context.task_spec.search_space, coerced, observation.suggestion.config)[0]
                for observation in context.history
            ]
            finite = [distance for distance in distances if distance is not None]
            nearest_distance = min(finite) if finite else None
            if too_similar_threshold is not None and nearest_distance is not None and nearest_distance < float(too_similar_threshold):
                too_similar = True
                violations.append(
                    {
                        "field": None,
                        "rule": "too_similar_to_recent_or_history",
                        "message": f"nearest evaluated distance {nearest_distance:.4g} is below threshold",
                        "repairable": False,
                    }
                )
    return {
        "valid": not violations,
        "config": agent_visible_config(coerced) if coerced and not (expected - provided) else None,
        "identity": identity,
        "violations": violations,
        "duplicate": duplicate,
        "too_similar": too_similar,
        "nearest_evaluated_distance": agent_visible_payload(nearest_distance),
        "repairable": bool(violations) and all(bool(item.get("repairable")) for item in violations),
        "recommended_minimal_repairs": agent_visible_payload(repairs),
    }


def _encode_config_from_spec(config: dict[str, Any], feature_spec: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for item in feature_spec:
        name = item["name"]
        kind = item["kind"]
        if kind == "numeric":
            low = float(item["low"])
            high = float(item["high"])
            span = max(high - low, 1e-12)
            values.append((float(config[name]) - low) / span)
        elif kind == "categorical_onehot":
            values.append(1.0 if config.get(name) == item["choice"] else 0.0)
        else:
            raise ValueError(f"Unsupported surrogate feature kind: {kind}")
    return values


def _surrogate_feature_spec(search_space: SearchSpace) -> tuple[list[dict[str, Any]], list[str]]:
    spec: list[dict[str, Any]] = []
    limitations: list[str] = []
    for param in search_space:
        if isinstance(param, (FloatParam, IntParam)):
            spec.append({"kind": "numeric", "name": param.name, "low": float(param.low), "high": float(param.high)})
        elif isinstance(param, CategoricalParam):
            for choice in param.choices:
                spec.append({"kind": "categorical_onehot", "name": param.name, "choice": choice})
        elif isinstance(param, StringParam):
            limitations.append(f"String parameter `{param.name}` is not encoded for this simple surrogate.")
        else:
            limitations.append(f"Unsupported parameter `{param.name}` for surrogate encoding.")
    return spec, limitations


def _fit_ridge(X: np.ndarray, y: np.ndarray, *, ridge: float = 1e-6) -> np.ndarray:
    design = np.column_stack([np.ones(X.shape[0]), X])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.pinv(design.T @ design + penalty) @ design.T @ y


def _predict_ridge(X: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(X.shape[0]), X])
    return design @ coefficients


def _pairwise_ranking_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    correct = 0
    total = 0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            truth = float(y_true[i] - y_true[j])
            pred = float(y_pred[i] - y_pred[j])
            if abs(truth) <= 1e-12:
                continue
            total += 1
            if truth * pred > 0:
                correct += 1
    if total == 0:
        return None
    return correct / total


def _fit_surrogate(context: BBOToolContext, *, min_observations: int) -> dict[str, Any]:
    primary = context.task_spec.primary_objective.name
    scored = [obs for obs in context.history if _primary_score(obs, primary) is not None]
    feature_spec, limitations = _surrogate_feature_spec(context.task_spec.search_space)
    if limitations:
        return {
            "usable_signal": False,
            "warnings": limitations,
            "models": [],
            "selected_model_id": None,
        }
    required = max(min_observations, min(12, len(feature_spec) + 2))
    if len(scored) < required:
        return {
            "usable_signal": False,
            "warnings": [f"Need at least {required} successful observations for surrogate validation, got {len(scored)}."],
            "models": [],
            "selected_model_id": None,
        }
    X = np.asarray([_encode_config_from_spec(obs.suggestion.config, feature_spec) for obs in scored], dtype=float)
    y = np.asarray([float(obs.objectives[primary]) for obs in scored], dtype=float)
    if float(np.std(y)) <= 1e-12:
        return {
            "usable_signal": False,
            "warnings": ["Objective has no measurable variation in the available history."],
            "models": [],
            "selected_model_id": None,
        }
    preds: list[float] = []
    for index in range(len(scored)):
        train_mask = np.ones(len(scored), dtype=bool)
        train_mask[index] = False
        coefficients = _fit_ridge(X[train_mask], y[train_mask])
        preds.append(float(_predict_ridge(X[index : index + 1], coefficients)[0]))
    pred_array = np.asarray(preds, dtype=float)
    mae = float(np.mean(np.abs(pred_array - y)))
    baseline = float(np.mean(np.abs(np.mean(y) - y)))
    ranking = _pairwise_ranking_accuracy(y, pred_array)
    coefficients = _fit_ridge(X, y)
    fitted = _predict_ridge(X, coefficients)
    residual_rmse = float(np.sqrt(np.mean((fitted - y) ** 2)))
    usable = bool(baseline > 0 and mae < baseline * 0.95 and ranking is not None and ranking >= 0.6)
    model_id_source = {
        "task": context.task_spec.name,
        "primary": primary,
        "history": [
            {
                "trial_id": obs.suggestion.trial_id,
                "config": obs.suggestion.config,
                "objective": obs.objectives.get(primary),
            }
            for obs in scored
        ],
        "model": "ridge_linear",
    }
    model_id = "surrogate_" + hashlib.sha256(json.dumps(model_id_source, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
    model_payload = {
        "model_id": model_id,
        "model_type": "ridge_linear",
        "feature_representation": feature_spec,
        "objective": primary,
        "direction": context.task_spec.primary_objective.direction.value,
        "coefficients": [float(value) for value in coefficients],
        "validation": {
            "method": "leave_one_out",
            "mae": mae,
            "baseline_mae": baseline,
            "pairwise_ranking_accuracy": ranking,
            "residual_rmse": residual_rmse,
        },
        "usable_signal": usable,
        "n_observations": len(scored),
    }
    model_dir = context.state_dir / "surrogate_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / f"{model_id}.json").write_text(json.dumps(agent_visible_payload(model_payload), sort_keys=True, indent=2), encoding="utf-8")
    return {
        "usable_signal": usable,
        "selected_model_id": model_id if usable else None,
        "models": [agent_visible_payload(model_payload)],
        "warnings": [] if usable else ["Surrogate validation did not beat the baseline clearly enough for proposal use."],
    }


def _score_virtual_candidates(context: BBOToolContext, *, model_id: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(candidates, list):
        raise TypeError("candidates must be a list.")
    model_path = context.state_dir / "surrogate_models" / f"{model_id}.json"
    if not model_path.exists():
        return {"usable_signal": False, "error": f"Unknown surrogate model_id `{model_id}`.", "scores": []}
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if not model.get("usable_signal"):
        return {"usable_signal": False, "error": "Surrogate model did not pass validation.", "scores": []}
    feature_spec = list(model["feature_representation"])
    coefficients = np.asarray(model["coefficients"], dtype=float)
    scores = []
    for index, item in enumerate(candidates):
        validation = _validate_candidate_report(context, item)
        config = validation.get("config")
        if not validation.get("valid") or not isinstance(config, dict):
            scores.append({"index": index, "validation": validation, "scored": False})
            continue
        vector = np.asarray([_encode_config_from_spec(config, feature_spec)], dtype=float)
        predicted = float(_predict_ridge(vector, coefficients)[0])
        distances = [
            _config_distance(context.task_spec.search_space, config, observation.suggestion.config)[0]
            for observation in context.history
        ]
        finite = [distance for distance in distances if distance is not None]
        scores.append(
            {
                "index": index,
                "config": agent_visible_config(config),
                "predicted_objective": agent_visible_payload(predicted),
                "uncertainty": agent_visible_payload((model.get("validation") or {}).get("residual_rmse")),
                "validation": validation,
                "nearest_evaluated_distance": agent_visible_payload(min(finite) if finite else None),
                "budget_consumed": False,
                "scored": True,
            }
        )
    return {"usable_signal": True, "model_id": model_id, "scores": scores, "budget_consumed": False}


def _observation_summary(observation: TrialObservation) -> dict[str, Any]:
    return {
        "trial_id": observation.suggestion.trial_id,
        "config": agent_visible_config(observation.suggestion.config),
        "budget": agent_visible_payload(observation.suggestion.budget),
        "status": observation.status.value,
        "objectives": agent_visible_payload(observation.objectives),
        "metrics": agent_visible_metrics(observation.metrics),
        "elapsed_seconds": agent_visible_payload(observation.elapsed_seconds),
        "error_type": observation.error_type,
        "error_message": observation.error_message,
        "timestamp": agent_visible_payload(observation.timestamp),
        "metadata": agent_visible_metadata(observation.metadata),
        "suggestion_metadata": agent_visible_payload(observation.suggestion.metadata),
        "search_action": agent_visible_payload(_search_action_summary(observation)),
    }


def _sample_around(
    search_space: SearchSpace,
    incumbent: dict[str, Any],
    rng: random.Random,
    jitter_fraction: float,
) -> dict[str, Any]:
    config: dict[str, Any] = {}
    fraction = min(max(float(jitter_fraction), 0.0), 1.0)
    for param in search_space:
        current = incumbent.get(param.name, param.effective_default())
        if isinstance(param, FloatParam):
            span = float(param.high) - float(param.low)
            value = float(current) + rng.uniform(-span * fraction, span * fraction)
            config[param.name] = param.coerce(min(max(value, param.low), param.high))
        elif isinstance(param, IntParam):
            span = int(param.high) - int(param.low)
            step = max(1, int(round(span * fraction)))
            value = int(current) + rng.randint(-step, step)
            config[param.name] = param.coerce(min(max(value, param.low), param.high))
        elif isinstance(param, CategoricalParam):
            config[param.name] = current if rng.random() > fraction else param.sample(rng)
        elif isinstance(param, StringParam):
            config[param.name] = current if rng.random() > fraction else _sample_string_param(param, rng)
        else:
            raise TypeError(f"Unsupported parameter type: {type(param).__name__}")
    return search_space.coerce_config(config, use_defaults=False)


def _sample_search_space(search_space: SearchSpace, rng: random.Random) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for param in search_space:
        if isinstance(param, StringParam):
            config[param.name] = _sample_string_param(param, rng)
        else:
            config[param.name] = param.sample(rng)
    return search_space.coerce_config(config, use_defaults=False)


def _sample_string_param(param: StringParam, rng: random.Random) -> str:
    if param.name.lower() in {"smiles", "smile"}:
        pool = _load_smiles_pool()
        if pool:
            return param.coerce(rng.choice(pool))
    try:
        return param.effective_default()
    except Exception as exc:
        raise TypeError(
            f"StringParam `{param.name}` does not define a generic sampler; "
            "set BBO_SMILES_POOL_PATH for SMILES tasks."
        ) from exc


def _load_smiles_pool() -> list[str]:
    raw_path = os.environ.get("BBO_SMILES_POOL_PATH", "").strip()
    if not raw_path:
        return []
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"BBO_SMILES_POOL_PATH does not exist: {path}")
    values: list[str] = []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                value = row.get("smile") or row.get("smiles") or row.get("SMILES")
                if value and value.strip():
                    values.append(value.strip())
    else:
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            if text.startswith("{"):
                try:
                    row = json.loads(text)
                except json.JSONDecodeError:
                    row = None
                if isinstance(row, dict):
                    text = str(row.get("smiles") or row.get("smile") or row.get("SMILES") or "").strip()
            if text:
                values.append(text)
    return values


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


__all__ = [
    "AnalyzeHistoryTool",
    "CompareTrialsTool",
    "EstimateLocalEffectsTool",
    "FindNearestTrialsTool",
    "FitAndCheckSurrogateTool",
    "GetHistoryOverviewTool",
    "GetIncumbentTool",
    "GetRecentSearchActionsTool",
    "GetSearchSpaceTool",
    "GetTaskContextTool",
    "GetTrialHistoryTool",
    "MeasureSearchCoverageTool",
    "MemoryReadTool",
    "MemoryWriteTool",
    "SampleCandidatesTool",
    "ScoreVirtualCandidatesTool",
    "ValidateCandidateTool",
    "ValidateCandidatesTool",
    "create_core_BBO_tools",
    "search_space_schema",
]
