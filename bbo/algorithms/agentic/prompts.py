"""Prompt builders for the Pablo Planner/Explorer/Worker roles."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ...core import CategoricalParam, FloatParam, IntParam, SearchSpace, TaskDescriptionBundle, TaskSpec


@dataclass(frozen=True)
class PromptBundle:
    role: str
    system: str
    user: str
    context: dict[str, Any] = field(default_factory=dict)


def summarize_search_space(search_space: SearchSpace, *, max_choices: int = 16) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for param in search_space:
        if isinstance(param, FloatParam):
            summary.append(
                {
                    "name": param.name,
                    "type": "float",
                    "low": float(param.low),
                    "high": float(param.high),
                    "default": param.effective_default(),
                }
            )
        elif isinstance(param, IntParam):
            summary.append(
                {
                    "name": param.name,
                    "type": "int",
                    "low": int(param.low),
                    "high": int(param.high),
                    "default": param.effective_default(),
                }
            )
        elif isinstance(param, CategoricalParam):
            choices = list(param.choices)
            summary.append(
                {
                    "name": param.name,
                    "type": "categorical",
                    "default": param.effective_default(),
                    "choice_count": len(choices),
                    "choices_preview": choices[:max_choices],
                    "choices_truncated": len(choices) > max_choices,
                }
            )
        else:
            summary.append({"name": param.name, "type": type(param).__name__})
    return summary


def summarize_description(bundle: TaskDescriptionBundle, *, max_chars_per_section: int = 360) -> dict[str, str]:
    summary: dict[str, str] = {}
    for kind, text in bundle.section_map.items():
        compact = " ".join(text.split())
        if len(compact) > max_chars_per_section:
            compact = compact[: max_chars_per_section - 3] + "..."
        summary[kind] = compact
    return summary


def build_explorer_prompt(
    *,
    task_spec: TaskSpec,
    description: TaskDescriptionBundle,
    c_global: list[dict[str, Any]],
    best_objective: float | None,
) -> PromptBundle:
    search_space_summary = summarize_search_space(task_spec.search_space)
    description_summary = summarize_description(description)
    system = (
        "You are the Explorer role in Pablo. Read the global candidate context and propose only JSON. "
        "Return exactly one object with key `candidates`. Each candidate must be a full benchmark configuration."
    )
    user = json.dumps(
        {
            "task_name": task_spec.name,
            "objective": task_spec.primary_objective.name,
            "objective_direction": task_spec.primary_objective.direction.value,
            "best_objective": best_objective,
            "task_description_summary": description_summary,
            "search_space": search_space_summary,
            "c_global": c_global,
        },
        indent=2,
        sort_keys=True,
    )
    return PromptBundle(
        role="explorer",
        system=system,
        user=user,
        context={
            "task_spec": task_spec,
            "search_space": task_spec.search_space,
            "c_global": c_global,
            "best_objective": best_objective,
            "description_summary": description_summary,
        },
    )


def build_planner_prompt(
    *,
    task_spec: TaskSpec,
    description: TaskDescriptionBundle,
    c_global: list[dict[str, Any]],
    performance_stats: dict[str, Any],
    existing_tasks_summary: list[dict[str, Any]],
) -> PromptBundle:
    description_summary = summarize_description(description)
    system = (
        "You are the Planner role in Pablo. Read only the global context and task registry summary. "
        "Return one JSON object whose keys are task names and whose values are task texts."
    )
    user = json.dumps(
        {
            "task_name": task_spec.name,
            "objective": task_spec.primary_objective.name,
            "objective_direction": task_spec.primary_objective.direction.value,
            "task_description_summary": description_summary,
            "performance_stats": performance_stats,
            "existing_tasks_summary": existing_tasks_summary,
            "c_global": c_global,
        },
        indent=2,
        sort_keys=True,
    )
    return PromptBundle(
        role="planner",
        system=system,
        user=user,
        context={
            "task_spec": task_spec,
            "search_space": task_spec.search_space,
            "description_summary": description_summary,
            "performance_stats": performance_stats,
            "existing_tasks_summary": existing_tasks_summary,
            "c_global": c_global,
        },
    )


def build_worker_prompt(
    *,
    task_spec: TaskSpec,
    planner_task_name: str,
    planner_task_text: str,
    current_seed: dict[str, Any],
) -> PromptBundle:
    search_space_summary = summarize_search_space(task_spec.search_space)
    domain_prefix = (
        "You are the Worker role in Pablo. Generate benchmark candidate configurations that satisfy the declared "
        "search space exactly. Search-space schema: "
        + json.dumps(search_space_summary, sort_keys=True)
        + "\nPlanner task: "
        + planner_task_text.strip()
    )
    domain_suffix = (
        "\nReturn raw JSON only with exactly one top-level key named `candidates`. "
        "Do not include markdown, comments, prose, or partial configurations."
    )
    system = domain_prefix + domain_suffix
    user = json.dumps({"current_seed": current_seed}, indent=2, sort_keys=True)
    return PromptBundle(
        role="worker",
        system=system,
        user=user,
        context={
            "task_spec": task_spec,
            "search_space": task_spec.search_space,
            "planner_task_name": planner_task_name,
            "planner_task_text": planner_task_text,
            "current_seed": current_seed,
        },
    )


__all__ = [
    "PromptBundle",
    "build_explorer_prompt",
    "build_planner_prompt",
    "build_worker_prompt",
    "summarize_description",
    "summarize_search_space",
]
