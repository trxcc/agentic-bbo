"""Canonical tool service shared by function-calling and workspace transports."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from ....core import (
    BBOBenchmarkManifest, CategoricalParam, FloatParam, Incumbent, IntParam,
    ObjectiveDirection, ObjectiveSpec, SearchSpace, StringParam, TaskDescriptionBundle,
    TaskSpec, TrialObservation, TrialStatus, TrialSuggestion,
)
from .context import BBOToolContext
from .core_tools import create_core_BBO_tools
from .registry import BBOToolRegistry


CANONICAL_WORKSPACE_TOOLS = frozenset({
    "get_search_space", "get_trial_history", "get_incumbent", "get_history_overview",
    "compare_trials", "find_nearest_trials", "estimate_local_effects",
    "measure_search_coverage", "summarize_objective_metrics", "fit_and_check_surrogate",
    "score_virtual_candidates", "validate_candidate", "validate_candidates",
    "get_recent_search_actions", "sample_candidates", "analyze_history",
    "profile_history_quality", "analyze_convergence", "rank_parameter_importance",
    "analyze_parameter_interactions", "locate_promising_regions",
    "locate_underexplored_regions", "recommend_search_regions", "analyze_search_strategy",
})


async def execute_tool(context: BBOToolContext, tool_name: str, arguments: dict[str, Any]) -> Any:
    registry = BBOToolRegistry(create_core_BBO_tools(enable_memory=False))
    return await registry.execute_payload(tool_name, arguments, context)


def execute_workspace_tool(config: dict[str, Any], tool_name: str, arguments: dict[str, Any]) -> Any:
    return asyncio.run(execute_tool(load_workspace_context(config), tool_name, arguments))


def load_workspace_context(config: dict[str, Any]) -> BBOToolContext:
    workspace = Path(str(config["workspace_dir"]))
    parameters = _read_json(workspace / "space.json").get("parameters", [])
    objective = _read_json(workspace / "objective.json")
    manifest_data = _read_json(workspace / "manifest.json")
    search_space = SearchSpace(tuple(_parameter(item) for item in parameters))
    objectives = tuple(
        ObjectiveSpec(str(item["name"]), ObjectiveDirection(str(item["direction"])))
        for item in objective.get("all_objectives", [objective])
    )
    max_evaluations = int(config.get("optimizer_max_evaluations") or manifest_data.get("max_evaluations") or 1)
    task_spec = TaskSpec(
        name=str(config.get("optimizer_agent_task_id") or manifest_data.get("task_id") or "workspace_task"),
        search_space=search_space, objectives=objectives, max_evaluations=max_evaluations,
        metadata=dict(config.get("optimizer_task_metadata") or {}),
    )
    history = [_observation(item) for item in _read_jsonl(workspace / "history.jsonl")]
    incumbent_data = _read_json(workspace / "incumbent.json")
    incumbent = None
    if isinstance(incumbent_data.get("config"), dict):
        incumbent = Incumbent(
            config=dict(incumbent_data["config"]), score=incumbent_data.get("score"),
            objectives=dict(incumbent_data.get("objectives") or {}),
            trial_id=incumbent_data.get("trial_id"),
        )
    return BBOToolContext(
        task_spec=task_spec, description=TaskDescriptionBundle.empty(task_id=task_spec.name),
        manifest=BBOBenchmarkManifest.from_dict(manifest_data), workspace_dir=workspace,
        state_dir=Path(str(config.get("state_dir") or workspace)), history=history,
        incumbent=incumbent, seed=int(config.get("seed", 0)),
        optimizer_backend_allowlist=tuple(config.get("optimizer_backend_allowlist") or ()),
    )


def _parameter(raw: dict[str, Any]):
    common = {"name": str(raw["name"]), "default": raw.get("default")}
    kind = raw.get("type")
    if kind == "float":
        return FloatParam(**common, low=float(raw["low"]), high=float(raw["high"]), log=bool(raw.get("log")))
    if kind == "int":
        return IntParam(**common, low=int(raw["low"]), high=int(raw["high"]), log=bool(raw.get("log")))
    if kind == "categorical":
        return CategoricalParam(**common, choices=tuple(raw.get("choices") or ()))
    if kind == "string":
        return StringParam(**common, min_length=int(raw.get("min_length", 0)),
                           max_length=raw.get("max_length"), pattern=raw.get("pattern"))
    raise ValueError(f"Unsupported workspace parameter type {kind!r}.")


def _observation(raw: dict[str, Any]) -> TrialObservation:
    metadata = dict(raw.get("suggestion_metadata") or {})
    if raw.get("search_action") and "search_action" not in metadata:
        metadata["search_action"] = raw["search_action"]
    return TrialObservation(
        suggestion=TrialSuggestion(config=dict(raw.get("config") or {}), trial_id=raw.get("trial_id"),
                                   budget=raw.get("budget"), metadata=metadata),
        status=TrialStatus(str(raw.get("status", "failed"))),
        objectives={str(k): float(v) for k, v in (raw.get("objectives") or {}).items()},
        metrics=dict(raw.get("metrics") or {}), elapsed_seconds=raw.get("elapsed_seconds"),
        timestamp=float(raw.get("timestamp") or 0.0), metadata=dict(raw.get("metadata") or {}),
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


__all__ = ["CANONICAL_WORKSPACE_TOOLS", "execute_tool", "execute_workspace_tool", "load_workspace_context"]
