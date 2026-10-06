"""General coding-agent optimizer for black-box optimization tasks."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import random
import re
import shutil
import sys
import textwrap
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...core import (
    Incumbent,
    ObjectiveDirection,
    SearchSpace,
    load_BBO_manifest,
    TaskDescriptionBundle,
    TaskSpec,
    TrialObservation,
    TrialSuggestion,
)
from ...core.algo import Algorithm
from ..benchmark_protocol import FixedInitializationProtocol, resolve_fixed_initialization
from .general_agent_engines import (
    AgentResult,
    AgentWorkCopy,
    GeneralAgentEngine,
    create_general_agent_engine,
    normalize_agent_framework,
)
from .serialization import append_jsonl, dump_json, stable_config_identity, to_jsonable
from .tools import (
    BBOMemoryStore,
    BBOToolCallLogger,
    BBOToolContext,
    BBOToolRegistry,
    BBOWebSourceLogger,
    CodeInterpreterTool,
    DisabledBBOCodeBackend,
    FetchURLTool,
    DockerBBOCodeBackend,
    MockBBOCodeBackend,
    OPTIMIZER_ACTION_TOOLS,
    OPTIMIZER_DECISION_TOOLS,
    SandboxFusionBBOCodeBackend,
    WebSearchTool,
    create_BBO_web_search_provider,
    create_optimizer_tools,
    create_core_BBO_tools,
)
from .tools.core_tools import agent_visible_config, agent_visible_metadata, agent_visible_metrics, agent_visible_payload
from .agent_candidate import GeneralAgentValidationError, ParsedAgentCandidate, _paired_xy_parameter_count, _retry_feedback_block, parse_agent_candidate_payload, search_space_schema

BBO_NANOBOT_SKILL_NAMES = (
    "initialize-search",
    "refine-incumbent",
    "follow-promising-direction",
    "isolate-variable-effect",
    "probe-variable-interaction",
    "recombine-complementary-elites",
    "escape-search-stagnation",
    "surrogate-guided-proposal",
    "repair-invalid-candidate",
    "distill-search-memory",
)
NANOBOT_BUILTIN_SKILL_NAMES = (
    "clawhub",
    "cron",
    "github",
    "memory",
    "my",
    "skill-creator",
    "summarize",
    "tmux",
    "weather",
)
AGENT_SEARCH_INTENTS = frozenset(
    {
        "initialization",
        "exploitation",
        "directional_extrapolation",
        "hypothesis_test",
        "interaction_test",
        "recombination",
        "exploration",
        "stagnation_recovery",
        "surrogate_proposal",
        "repair",
    }
)
SKILL_TO_SEARCH_INTENT = {
    "initialize-search": "initialization",
    "refine-incumbent": "exploitation",
    "follow-promising-direction": "directional_extrapolation",
    "isolate-variable-effect": "hypothesis_test",
    "probe-variable-interaction": "interaction_test",
    "recombine-complementary-elites": "recombination",
    "escape-search-stagnation": "stagnation_recovery",
    "surrogate-guided-proposal": "surrogate_proposal",
    "repair-invalid-candidate": "repair",
}
SKILL_EVIDENCE_TOOL_GROUPS: dict[str, tuple[tuple[str, ...], ...]] = {
    "initialize-search": (
        ("get_history_overview", "summarize_objective_metrics"),
        ("measure_search_coverage",),
        ("validate_candidate", "validate_candidates"),
    ),
    "refine-incumbent": (
        ("get_history_overview", "summarize_objective_metrics"),
        ("find_nearest_trials",),
        ("estimate_local_effects",),
        ("validate_candidate", "validate_candidates"),
    ),
    "follow-promising-direction": (
        ("compare_trials",),
        ("estimate_local_effects",),
        ("find_nearest_trials",),
        ("validate_candidate", "validate_candidates"),
    ),
    "isolate-variable-effect": (
        ("compare_trials",),
        ("find_nearest_trials", "estimate_local_effects"),
        ("validate_candidate", "validate_candidates"),
    ),
    "probe-variable-interaction": (
        ("compare_trials",),
        ("estimate_local_effects", "find_nearest_trials"),
        ("validate_candidate", "validate_candidates"),
    ),
    "recombine-complementary-elites": (
        ("get_trial_history",),
        ("compare_trials",),
        ("validate_candidate", "validate_candidates"),
    ),
    "escape-search-stagnation": (
        ("get_history_overview", "summarize_objective_metrics"),
        ("measure_search_coverage",),
        ("get_recent_search_actions",),
        ("find_nearest_trials",),
        ("validate_candidate", "validate_candidates"),
    ),
    "surrogate-guided-proposal": (
        ("fit_and_check_surrogate",),
        ("score_virtual_candidates",),
        ("validate_candidate", "validate_candidates"),
    ),
    "repair-invalid-candidate": (
        ("validate_candidate", "validate_candidates"),
    ),
}
NON_PROPOSAL_BBO_SKILLS = frozenset({"distill-search-memory"})
BBO_NUMERIC_EVIDENCE_TOOLS = frozenset(
    {
        "assess_backend_suitability",
        "summarize_objective_metrics",
        "get_history_overview",
        "compare_trials",
        "find_nearest_trials",
        "estimate_local_effects",
        "measure_search_coverage",
        "fit_and_check_surrogate",
        "score_virtual_candidates",
        "analyze_history",
        "profile_history_quality",
        "analyze_convergence",
        "rank_parameter_importance",
        "analyze_parameter_interactions",
        "locate_promising_regions",
        "locate_underexplored_regions",
        "recommend_search_regions",
        "analyze_search_strategy",
    }
)
BBO_REGION_EVIDENCE_TOOLS = frozenset({"analyze_search_strategy"})

BBO_REGION_JOINT_SUPPORT_TOOLS = frozenset(
    {
        "analyze_parameter_interactions",
        "analyze_search_strategy",
        "score_virtual_candidates",
        "optimizer_suggest",
        "optimizer_portfolio_suggest",
        "optimizer_score",
    }
)
MAX_UNSUPPORTED_MARGINAL_REGION_CHANGES = 3
BBO_WORKSPACE_TOOL_NAMES = frozenset(
    {
        "get_task_context",
        "get_manifest",
        "get_search_space",
        "get_objective",
        "get_tool_specs",
        "get_trial_history",
        "get_incumbent",
        "get_history_overview",
        "compare_trials",
        "find_nearest_trials",
        "estimate_local_effects",
        "measure_search_coverage",
        "summarize_objective_metrics",
        "fit_and_check_surrogate",
        "profile_history_quality",
        "analyze_convergence",
        "rank_parameter_importance",
        "analyze_parameter_interactions",
        "locate_promising_regions",
        "locate_underexplored_regions",
        "recommend_search_regions",
        "analyze_search_strategy",
        *OPTIMIZER_ACTION_TOOLS,
        "render_search_diagnostics",
        "code_interpreter",
        "score_virtual_candidates",
        "validate_candidate",
        "validate_candidates",
        "get_recent_search_actions",
        "sample_candidates",
        "analyze_history",
        "memory_read",
        "memory_write",
    }
)
BBO_PYTHON_API_TOOL_ALIASES = {
    "task_context": "get_task_context",
    "manifest": "get_manifest",
    "search_space": "get_search_space",
    "objective": "get_objective",
    "tool_specs": "get_tool_specs",
    "history": "get_trial_history",
    "incumbent": "get_incumbent",
    "history_overview": "get_history_overview",
    "recent_search_actions": "get_recent_search_actions",
    "sample": "sample_candidates",
    "validate": "validate_candidates",
}
_NANOBOT_SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
def _search_action_metadata(
    raw_metadata: Mapping[str, Any] | None,
    *,
    call_id: str | None = None,
    candidate_index: int | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    metadata = dict(raw_metadata or {})
    action = metadata.get("search_action")
    if isinstance(action, Mapping):
        normalized_action = dict(action)
    else:
        normalized_action = {}
    skill = metadata.get("skill") or metadata.get("primary_skill") or normalized_action.get("skill")
    if skill:
        normalized_action["skill"] = str(skill)
    intent = (
        metadata.get("search_intent")
        or metadata.get("agent_search_intent")
        or normalized_action.get("search_intent")
        or (SKILL_TO_SEARCH_INTENT.get(str(skill)) if skill else None)
    )
    if not intent:
        if source == "initial_random":
            intent = "initialization"
        elif source == "fallback_random":
            intent = "exploration"
        else:
            intent = "exploration"
    intent = str(intent).strip().lower().replace("-", "_")
    if intent not in AGENT_SEARCH_INTENTS:
        intent = "exploration"
    normalized_action["search_intent"] = intent
    for key in (
        "parent_trials",
        "reference_trials",
        "modified_variables",
        "hypothesis",
        "change_summary",
        "expected_evidence",
        "repaired",
        "repair",
        "original_search_intent",
    ):
        if key in metadata and key not in normalized_action:
            normalized_action[key] = metadata[key]
    if call_id is not None:
        normalized_action["agent_call_id"] = call_id
    if candidate_index is not None:
        normalized_action["agent_candidate_index"] = candidate_index
    normalized = dict(metadata)
    normalized["search_intent"] = intent
    normalized["agent_search_intent"] = intent
    normalized["search_action"] = normalized_action
    return normalized


def _declared_agent_skill_names(candidates: list[ParsedAgentCandidate]) -> set[str]:
    names: set[str] = set()
    for candidate in candidates:
        metadata = dict(candidate.metadata or {})
        action = metadata.get("search_action")
        raw_skill = None
        if isinstance(action, Mapping):
            raw_skill = action.get("skill")
        raw_skill = metadata.get("skill") or metadata.get("primary_skill") or raw_skill
        if raw_skill is None:
            continue
        skill = str(raw_skill).strip()
        if not skill or skill.lower() in {"null", "none"}:
            continue
        if _NANOBOT_SKILL_NAME_RE.fullmatch(skill):
            names.add(skill)
    return names


def _nanobot_read_skill_names_for_call(log_dir: Path, call_id: str) -> set[str]:
    call_dir = log_dir / call_id
    if not call_dir.exists():
        return set()
    read: set[str] = set()
    for path in sorted(call_dir.glob("*_agent-end.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        messages = payload.get("messages")
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                skill = _skill_name_from_nanobot_read_file_call(tool_call)
                if skill:
                    read.add(skill)
    return read


def _bbo_workspace_tool_names_for_call(tool_calls_path: Path, call_id: str) -> set[str]:
    if not tool_calls_path.exists():
        return set()
    names: set[str] = set()
    for line in tool_calls_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("agent_call_id") != call_id and record.get("call_id") != call_id:
            continue
        if record.get("success") is False:
            continue
        name = record.get("tool_name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _bbo_tool_names_from_nanobot_session(log_dir: Path, call_id: str) -> set[str]:
    """Best-effort recovery for older workspace logs that lack agent_call_id."""

    call_dir = log_dir / call_id
    if not call_dir.exists():
        return set()
    names: set[str] = set()
    for path in sorted(call_dir.glob("*_agent-end.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        messages = payload.get("messages")
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                function = tool_call.get("function") if isinstance(tool_call, Mapping) else None
                if not isinstance(function, Mapping) or function.get("name") != "exec":
                    continue
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                if not isinstance(arguments, Mapping):
                    continue
                command = arguments.get("command")
                if isinstance(command, str):
                    names.update(_bbo_tool_names_from_command(command))
    return names


def _bbo_tool_names_from_command(command: str) -> set[str]:
    names: set[str] = set()
    for tool_name in BBO_WORKSPACE_TOOL_NAMES:
        if re.search(rf"\b{re.escape(tool_name)}\s*\(", command):
            names.add(tool_name)
    for method_name, tool_name in BBO_PYTHON_API_TOOL_ALIASES.items():
        if re.search(rf"\b{re.escape(method_name)}\s*\(", command):
            names.add(tool_name)
    return names


def _build_skill_usage_audit(
    *,
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
    read_skills: set[str],
    used_tools: set[str],
    history: list[TrialObservation],
    incumbent: Incumbent | None,
) -> dict[str, Any]:
    declared_skill = _declared_skill_name_from_metadata(metadata)
    candidate_pattern, pattern_evidence = _candidate_pattern(config=config, metadata=metadata, history=history, incumbent=incumbent)
    evidence_skills = set(read_skills)
    if declared_skill:
        evidence_skills.add(declared_skill)
    inferred_skill = _infer_skill_usage(
        declared_skill=declared_skill,
        read_skills=read_skills,
        used_tools=used_tools,
        candidate_pattern=candidate_pattern,
    )
    if inferred_skill:
        evidence_skills.add(inferred_skill)
    evidence = {
        skill: _skill_evidence_payload(skill=skill, read_skills=read_skills, used_tools=used_tools)
        for skill in sorted(evidence_skills)
    }
    return {
        "declared_skill": declared_skill,
        "read_skills": sorted(read_skills),
        "bbo_tools": sorted(used_tools),
        "candidate_pattern": candidate_pattern,
        "candidate_pattern_evidence": pattern_evidence,
        "inferred_skill": inferred_skill,
        "skill_evidence": evidence,
        "compliance": _skill_audit_compliance(
            declared_skill=declared_skill,
            read_skills=read_skills,
            inferred_skill=inferred_skill,
            evidence=evidence,
            used_tools=used_tools,
        ),
    }


def _declared_skill_name_from_metadata(metadata: Mapping[str, Any]) -> str | None:
    action = metadata.get("search_action")
    raw_skill = None
    if isinstance(action, Mapping):
        raw_skill = action.get("skill")
    raw_skill = metadata.get("skill") or metadata.get("primary_skill") or raw_skill
    if raw_skill is None:
        return None
    skill = str(raw_skill).strip()
    if not skill or skill.lower() in {"null", "none"}:
        return None
    if not _NANOBOT_SKILL_NAME_RE.fullmatch(skill):
        return None
    return skill


def _skill_evidence_payload(*, skill: str, read_skills: set[str], used_tools: set[str]) -> dict[str, Any]:
    required_groups = SKILL_EVIDENCE_TOOL_GROUPS.get(skill, ())
    satisfied = [group for group in required_groups if any(tool in used_tools for tool in group)]
    missing = [group for group in required_groups if not any(tool in used_tools for tool in group)]
    return {
        "read_skill_md": skill in read_skills,
        "proposal_allowed": skill not in NON_PROPOSAL_BBO_SKILLS,
        "required_tool_groups": [list(group) for group in required_groups],
        "satisfied_tool_groups": [list(group) for group in satisfied],
        "missing_tool_groups": [list(group) for group in missing],
    }


def _infer_skill_usage(
    *,
    declared_skill: str | None,
    read_skills: set[str],
    used_tools: set[str],
    candidate_pattern: str,
) -> str | None:
    if declared_skill and declared_skill not in NON_PROPOSAL_BBO_SKILLS:
        return declared_skill
    if not read_skills:
        return None
    scored: list[tuple[float, str]] = []
    for skill in read_skills:
        if skill in NON_PROPOSAL_BBO_SKILLS:
            continue
        score = _skill_evidence_score(skill, used_tools)
        if skill == "isolate-variable-effect" and candidate_pattern in {
            "single_variable_change_from_parent",
            "single_variable_change_from_incumbent",
        }:
            score += 2.0
        elif skill == "refine-incumbent" and candidate_pattern in {
            "single_variable_change_from_incumbent",
            "local_refinement_from_incumbent",
        }:
            score += 1.5
        elif skill == "follow-promising-direction" and "compare_trials" in used_tools:
            score += 1.0
        elif skill == "probe-variable-interaction" and candidate_pattern == "multi_variable_change_from_parent":
            score += 1.0
        elif skill == "recombine-complementary-elites" and candidate_pattern == "multi_parent_candidate":
            score += 1.5
        elif skill == "surrogate-guided-proposal" and "score_virtual_candidates" in used_tools:
            score += 2.0
        elif skill == "initialize-search" and (
            "sample_candidates" in used_tools or "measure_search_coverage" in used_tools
        ):
            score += 1.0
        if score >= 2.0:
            scored.append((score, skill))
    if not scored:
        return None
    return sorted(scored, key=lambda item: (-item[0], item[1]))[0][1]


def _skill_evidence_score(skill: str, used_tools: set[str]) -> float:
    groups = SKILL_EVIDENCE_TOOL_GROUPS.get(skill)
    if not groups:
        return 0.0
    return float(sum(1 for group in groups if any(tool in used_tools for tool in group)))


def _skill_audit_compliance(
    *,
    declared_skill: str | None,
    read_skills: set[str],
    inferred_skill: str | None,
    evidence: Mapping[str, Any],
    used_tools: set[str],
) -> str:
    if declared_skill:
        declared_evidence = evidence.get(declared_skill)
        if not isinstance(declared_evidence, Mapping):
            return "declared_unknown_skill"
        if not declared_evidence.get("read_skill_md"):
            return "declared_without_read"
        if declared_evidence.get("proposal_allowed") is False:
            return "declared_non_proposal_skill"
        if declared_evidence.get("missing_tool_groups"):
            return "declared_without_required_evidence"
        return "declared_and_supported"
    if inferred_skill:
        return "read_and_inferred_but_not_declared"
    if read_skills:
        return "read_without_inferred_skill"
    if any(tool in BBO_NUMERIC_EVIDENCE_TOOLS for tool in used_tools):
        return "tools_without_skill"
    return "no_skill_signal"


def _candidate_pattern(
    *,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    history: list[TrialObservation],
    incumbent: Incumbent | None,
) -> tuple[str, dict[str, Any]]:
    action = metadata.get("search_action")
    parent_ids: list[int] = []
    if isinstance(action, Mapping):
        parent_ids = [_trial_id_as_int(value) for value in action.get("parent_trials") or []]
        parent_ids = [value for value in parent_ids if value is not None]
    if len(parent_ids) > 1:
        return "multi_parent_candidate", {"parent_trials": parent_ids}
    history_by_id = {observation.suggestion.trial_id: observation for observation in history}
    if len(parent_ids) == 1:
        parent_id = parent_ids[0]
        parent = history_by_id.get(parent_id)
        if parent is not None:
            modified = _modified_config_keys(parent.suggestion.config, config)
            if len(modified) == 1:
                pattern = (
                    "single_variable_change_from_incumbent"
                    if incumbent is not None and incumbent.trial_id == parent_id
                    else "single_variable_change_from_parent"
                )
            elif 1 < len(modified) <= max(2, len(config) // 3):
                pattern = (
                    "local_refinement_from_incumbent"
                    if incumbent is not None and incumbent.trial_id == parent_id
                    else "local_refinement_from_parent"
                )
            else:
                pattern = "multi_variable_change_from_parent"
            return (
                pattern,
                {
                    "parent_trial_id": parent_id,
                    "modified_variables": modified,
                    "n_modified_variables": len(modified),
                },
            )
    if incumbent is not None:
        modified = _modified_config_keys(incumbent.config, config)
        if len(modified) == 1:
            return (
                "single_variable_change_from_incumbent",
                {
                    "parent_trial_id": incumbent.trial_id,
                    "modified_variables": modified,
                    "n_modified_variables": len(modified),
                },
            )
        if 1 < len(modified) <= max(2, len(config) // 3):
            return (
                "local_refinement_from_incumbent",
                {
                    "parent_trial_id": incumbent.trial_id,
                    "modified_variables": modified,
                    "n_modified_variables": len(modified),
                },
            )
    return "direct_candidate", {}


def _modified_config_keys(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    keys = sorted(set(left) | set(right))
    return [key for key in keys if left.get(key) != right.get(key)]


def _trial_id_as_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("trial_"):
            stripped = stripped.removeprefix("trial_")
        if stripped.isdigit():
            return int(stripped)
    return None


def _format_tool_group(group: tuple[str, ...]) -> str:
    if len(group) == 1:
        return f"`{group[0]}`"
    return "one of " + "/".join(f"`{tool}`" for tool in group)


def _skill_name_from_nanobot_read_file_call(tool_call: object) -> str | None:
    if not isinstance(tool_call, Mapping):
        return None
    function = tool_call.get("function")
    if not isinstance(function, Mapping) or function.get("name") != "read_file":
        return None
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, Mapping):
        return None
    raw_path = arguments.get("path")
    if not isinstance(raw_path, str):
        return None
    normalized = raw_path.replace("\\", "/").strip()
    parts = [part for part in normalized.split("/") if part]
    if len(parts) < 3:
        return None
    for index in range(len(parts) - 2):
        if parts[index] == "skills" and parts[index + 2] == "SKILL.md":
            skill = parts[index + 1]
            if _NANOBOT_SKILL_NAME_RE.fullmatch(skill):
                return skill
    return None
