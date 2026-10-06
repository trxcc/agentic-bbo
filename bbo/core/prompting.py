"""Prompt compilation utilities for general black-box optimization tasks."""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from .description import TaskDescriptionBundle
from .space import CategoricalParam, FloatParam, IntParam, ParameterSpec, SearchSpace, StringParam
from .task import ObjectiveDirection, TaskSpec
from .trial import TrialObservation

PromptKind = Literal["candidate_generation", "score_prediction"]
HistorySort = Literal["best", "recent", "original"]
SearchSpaceBlockFormat = Literal["markdown", "json"]

DEFAULT_CONTEXT_SECTIONS = (
    "background",
    "goal",
    "constraints",
    "prior_knowledge",
)


class PromptOutputValidationError(ValueError):
    """Raised when a model response does not satisfy a prompt output contract."""


@dataclass(frozen=True)
class SearchSpaceRenderPolicy:
    """Controls how search-space information is rendered into prompts."""

    parameter_order: tuple[str, ...] | None = None
    block_format: SearchSpaceBlockFormat = "markdown"


@dataclass(frozen=True)
class TaskContextRenderPolicy:
    """Controls which task-description sections are surfaced in prompts."""

    sections: tuple[str, ...] = DEFAULT_CONTEXT_SECTIONS
    max_chars_per_section: int | None = 4000


@dataclass(frozen=True)
class HistoryRenderPolicy:
    """Controls how observed trials are rendered into prompts."""

    max_trials: int | None = None
    sort: HistorySort = "best"
    include_failed: bool = False


@dataclass(frozen=True)
class OutputContract:
    """JSON response contract requested from an LLM."""

    kind: PromptKind
    allow_config_wrappers: bool = False


@dataclass(frozen=True)
class PromptContract:
    """End-to-end prompt compilation contract for one LLM call."""

    kind: PromptKind
    search_space: SearchSpaceRenderPolicy = field(default_factory=SearchSpaceRenderPolicy)
    task_context: TaskContextRenderPolicy = field(default_factory=TaskContextRenderPolicy)
    history: HistoryRenderPolicy = field(default_factory=HistoryRenderPolicy)
    output: OutputContract | None = None

    def __post_init__(self) -> None:
        if self.output is not None and self.output.kind != self.kind:
            raise ValueError("PromptContract.output.kind must match PromptContract.kind.")


@dataclass(frozen=True)
class PromptContext:
    """Compiled prompt fields shared by candidate-generation and score-prediction templates."""

    task_name: str
    objective_name: str
    objective_direction: str
    better_relation: str
    max_evaluations: int
    num_observed_trials: int
    best_objective_value: float | None
    search_space_block: str
    task_context_block: str
    history_block: str
    candidate_config: str | None = None


def default_candidate_generation_contract(
    *,
    max_history_trials: int | None = None,
    parameter_order: Sequence[str] | None = None,
    max_chars_per_section: int | None = 4000,
) -> PromptContract:
    return PromptContract(
        kind="candidate_generation",
        search_space=SearchSpaceRenderPolicy(
            parameter_order=None if parameter_order is None else tuple(parameter_order),
        ),
        task_context=TaskContextRenderPolicy(max_chars_per_section=max_chars_per_section),
        history=HistoryRenderPolicy(max_trials=max_history_trials, sort="best"),
        output=OutputContract(kind="candidate_generation"),
    )


def default_score_prediction_contract(
    *,
    max_history_trials: int | None = None,
    parameter_order: Sequence[str] | None = None,
    max_chars_per_section: int | None = 4000,
) -> PromptContract:
    return PromptContract(
        kind="score_prediction",
        search_space=SearchSpaceRenderPolicy(
            parameter_order=None if parameter_order is None else tuple(parameter_order),
        ),
        task_context=TaskContextRenderPolicy(max_chars_per_section=max_chars_per_section),
        history=HistoryRenderPolicy(max_trials=max_history_trials, sort="best"),
        output=OutputContract(kind="score_prediction"),
    )


def build_prompt_context(
    *,
    task_spec: TaskSpec,
    description: TaskDescriptionBundle,
    history: Sequence[TrialObservation],
    contract: PromptContract,
    candidate_config: Mapping[str, Any] | None = None,
) -> PromptContext:
    primary = task_spec.primary_objective
    direction = primary.direction
    return PromptContext(
        task_name=_prompt_task_name(task_spec),
        objective_name=primary.name,
        objective_direction=direction.value,
        better_relation=_better_relation(direction),
        max_evaluations=task_spec.max_evaluations,
        num_observed_trials=len(history),
        best_objective_value=best_primary_objective(history, task_spec),
        search_space_block=render_search_space_block(task_spec.search_space, contract.search_space),
        task_context_block=render_task_context_block(description, contract.task_context),
        history_block=render_history_block(history, task_spec, contract.history, contract.search_space),
        candidate_config=(
            None
            if candidate_config is None
            else serialize_config(
                task_spec.search_space,
                dict(candidate_config),
                parameter_order=contract.search_space.parameter_order,
            )
        ),
    )


def _prompt_task_name(task_spec: TaskSpec) -> str:
    display_name = task_spec.metadata.get("display_name")
    if isinstance(display_name, str) and display_name.strip():
        return display_name.strip()
    return task_spec.name


def compile_candidate_generation_prompt(context: PromptContext, *, num_candidates: int) -> str:
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive.")
    return f"""You are helping with a general black-box optimization task.

## Optimization Objective

Task name: {context.task_name}

Primary objective: {context.objective_name}
Optimization direction: {context.objective_direction}
Better objective values are: {context.better_relation}

Evaluation budget: {context.max_evaluations}
Current number of observed trials: {context.num_observed_trials}
Current best objective value: {_json_scalar(context.best_objective_value)}

You may only use the observed trial history and the task information provided in this prompt. Do not assume gradients, closed-form equations, hidden simulators, or unobserved evaluations.

## Search Space

Each candidate must be a complete configuration in the following search space:

{context.search_space_block}

Search-space requirements:
- Use exactly the parameter names listed above.
- Respect all parameter types, bounds, categorical choices, and validity constraints.
- Return full configurations, not partial updates.
- Do not repeat any previously observed configuration.
- Avoid invalid values, out-of-bound values, and malformed configurations.
- For continuous parameters, avoid unnecessary rounding unless the parameter is explicitly integer-valued.
- Do not simply copy the default configuration unless the observed history strongly supports it.

## Task Context

The following task description provides task background, goal, constraints, and useful prior knowledge.

{context.task_context_block}

## Observed Trials

The following trials have already been evaluated.
They are sorted from best to worst according to the primary objective.

{context.history_block}

## Candidate Generation Task

Propose {num_candidates} new candidate configurations that are likely to improve over the current best objective value.

When proposing candidates, balance:
- Exploitation: refine patterns that appear strong in the observed trials.
- Exploration: try diverse regions that are still plausible under the task context and search-space constraints.
- Validity: every candidate must satisfy the search-space schema.

Do not request additional evaluations yourself. Only return candidate configurations.

## Output Format

Return only a valid JSON object with exactly one top-level key:

{{
  "candidates": [
    {{ ... complete configuration ... }}
  ]
}}

Output requirements:
- Return exactly {num_candidates} candidates.
- Each candidate must be a JSON object.
- Each candidate must contain all required parameters.
- Do not include markdown.
- Do not include comments.
- Do not include explanations.
- Do not include any text outside the JSON object."""


def compile_score_prediction_prompt(context: PromptContext) -> str:
    if context.candidate_config is None:
        raise ValueError("Score-prediction prompts require candidate_config in PromptContext.")
    return f"""You are acting as a surrogate model for a general black-box optimization task.

## Optimization Objective

Task name: {context.task_name}

Primary objective: {context.objective_name}
Optimization direction: {context.objective_direction}
Better objective values are: {context.better_relation}

Your task is to predict the objective value of a candidate configuration using only the task information and observed trials provided below.

## Search Space

The candidate belongs to the following search space:

{context.search_space_block}

## Task Context

{context.task_context_block}

## Observed Trials

The following trials have already been evaluated.
They are sorted from best to worst according to the primary objective.

{context.history_block}

## Candidate Configuration

Predict the objective value for the following candidate configuration:

{context.candidate_config}

## Prediction Task

Estimate the objective value that would be returned by the black-box evaluator for this candidate.

Use the observed trials as in-context examples. Pay attention to:
- The objective direction.
- Similar configurations in the history.
- Patterns between parameter values and objective values.
- Domain priors and constraints in the task context.

## Output Format

Return only a valid JSON object:

{{
  "predicted_objective": <number>
}}

Output requirements:
- The value must be a finite number.
- Do not include uncertainty unless explicitly requested.
- Do not include markdown.
- Do not include explanations.
- Do not include any text outside the JSON object."""


def search_space_to_schema(search_space: SearchSpace) -> list[dict[str, Any]]:
    """Serialize a SearchSpace into a JSON-friendly parameter schema."""

    return [_parameter_to_schema(param) for param in search_space]


def candidate_response_json_schema(search_space: SearchSpace, *, num_candidates: int) -> dict[str, Any]:
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive.")
    return {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": candidate_config_json_schema(search_space),
                "minItems": int(num_candidates),
                "maxItems": int(num_candidates),
            }
        },
        "required": ["candidates"],
        "additionalProperties": False,
    }


def candidate_config_json_schema(search_space: SearchSpace) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param in search_space:
        required.append(param.name)
        properties[param.name] = _parameter_to_json_schema(param)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def score_prediction_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "predicted_objective": {"type": "number"},
        },
        "required": ["predicted_objective"],
        "additionalProperties": False,
    }


def render_search_space_block(
    search_space: SearchSpace,
    policy: SearchSpaceRenderPolicy | None = None,
) -> str:
    policy = policy or SearchSpaceRenderPolicy()
    parameter_order = _parameter_order(search_space, policy.parameter_order)
    schema_by_name = {item["name"]: item for item in search_space_to_schema(search_space)}
    ordered_schema = [schema_by_name[name] for name in parameter_order]
    if policy.block_format == "json":
        return json.dumps({"parameters": ordered_schema}, indent=2, sort_keys=True)

    lines: list[str] = []
    for item in ordered_schema:
        typ = item["type"]
        if typ in {"float", "int"}:
            line = f"- {item['name']}: type={typ}; bounds=[{item['low']}, {item['high']}]"
            if item.get("log"):
                line += "; scale=log"
            if "default" in item:
                line += f"; default={_json_scalar(item['default'])}"
            lines.append(line)
            continue
        if typ == "categorical":
            line = f"- {item['name']}: type=categorical; choices={json.dumps(item['choices'], ensure_ascii=False, separators=(',', ':'))}"
            if "default" in item:
                line += f"; default={_json_scalar(item['default'])}"
            lines.append(line)
            continue
        if typ == "string":
            fields = [f"- {item['name']}: type=string"]
            if "min_length" in item:
                fields.append(f"min_length={item['min_length']}")
            if "max_length" in item:
                fields.append(f"max_length={item['max_length']}")
            if "pattern" in item:
                fields.append(f"pattern={json.dumps(item['pattern'], ensure_ascii=False)}")
            if "default" in item:
                fields.append(f"default={_json_scalar(item['default'])}")
            lines.append("; ".join(fields))
            continue
        lines.append(f"- {item['name']}: type={typ}")
    return "\n".join(lines)


def render_task_context_block(
    description: TaskDescriptionBundle,
    policy: TaskContextRenderPolicy | None = None,
) -> str:
    policy = policy or TaskContextRenderPolicy()
    if description.is_empty:
        return "No structured task context was supplied."

    docs_by_kind = {doc.kind: doc for doc in description.all_docs}
    sections: list[str] = []
    for kind in policy.sections:
        doc = docs_by_kind.get(kind)
        if doc is None:
            continue
        content = _strip_markdown_heading(doc.content).strip()
        if not content:
            continue
        if policy.max_chars_per_section is not None:
            content = _truncate(content, policy.max_chars_per_section)
        sections.append(f"### {doc.title}\n\n{content}")
    if sections:
        return "\n\n".join(sections)
    return "No structured task context was supplied."


def render_history_block(
    history: Sequence[TrialObservation],
    task_spec: TaskSpec,
    history_policy: HistoryRenderPolicy | None = None,
    search_space_policy: SearchSpaceRenderPolicy | None = None,
) -> str:
    history_policy = history_policy or HistoryRenderPolicy()
    search_space_policy = search_space_policy or SearchSpaceRenderPolicy()
    selected = _select_history(history, task_spec, history_policy)
    if not selected:
        return "No observed trials are available."

    primary = task_spec.primary_objective.name
    rows: list[dict[str, Any]] = []
    for observation in selected:
        row: dict[str, Any] = {
            "trial_id": observation.suggestion.trial_id,
            "status": observation.status.value,
            "config": _ordered_config(
                task_spec.search_space,
                observation.suggestion.config,
                search_space_policy.parameter_order,
            ),
        }
        if primary in observation.objectives:
            row["objective"] = _format_scalar(float(observation.objectives[primary]))
        if observation.error_type:
            row["error_type"] = observation.error_type
        if observation.error_message:
            row["error_message"] = observation.error_message
        rows.append(row)
    return json.dumps(rows, indent=2, sort_keys=True)


def best_primary_objective(history: Sequence[TrialObservation], task_spec: TaskSpec) -> float | None:
    primary = task_spec.primary_objective
    values = [
        float(observation.objectives[primary.name])
        for observation in history
        if observation.success and primary.name in observation.objectives
    ]
    if not values:
        return None
    return min(values) if primary.direction == ObjectiveDirection.MINIMIZE else max(values)


def serialize_config(
    search_space: SearchSpace,
    config: Mapping[str, Any],
    *,
    parameter_order: Sequence[str] | None = None,
) -> str:
    normalized = search_space.coerce_config(dict(config), use_defaults=False)
    ordered = _ordered_config(search_space, normalized, parameter_order)
    return json.dumps(ordered, separators=(",", ":"))


def format_candidate_response(
    search_space: SearchSpace,
    candidates: Sequence[Mapping[str, Any]],
    *,
    parameter_order: Sequence[str] | None = None,
) -> str:
    payload = {
        "candidates": [
            _ordered_config(search_space, search_space.coerce_config(dict(candidate), use_defaults=False), parameter_order)
            for candidate in candidates
        ]
    }
    return json.dumps(payload, separators=(",", ":"))


def format_score_prediction_response(value: float) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("predicted objective must be finite.")
    return json.dumps({"predicted_objective": numeric}, separators=(",", ":"))


def parse_candidate_response(
    raw_text: str,
    search_space: SearchSpace,
    *,
    expected_count: int | None = None,
    strict: bool = True,
) -> list[dict[str, Any]]:
    raw_items = _candidate_items_from_response(raw_text, allow_bare_config=not strict)
    if not raw_items:
        raise PromptOutputValidationError("Response does not contain a `candidates` array or candidate blocks.")

    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    errors: list[str] = []
    for index, item in enumerate(raw_items):
        mapping = _candidate_mapping(item)
        if mapping is None:
            message = f"Candidate {index} is not a JSON object."
            if strict:
                raise PromptOutputValidationError(message)
            errors.append(message)
            continue
        try:
            config = search_space.coerce_config(mapping, use_defaults=False)
        except Exception as exc:
            message = f"Candidate {index} is invalid: {exc}"
            if strict:
                raise PromptOutputValidationError(message) from exc
            errors.append(message)
            continue
        identity = json.dumps(config, sort_keys=True, separators=(",", ":"))
        if identity in seen:
            continue
        seen.add(identity)
        parsed.append(config)

    if not parsed:
        detail = "; ".join(errors) if errors else "No valid unique candidates were returned."
        raise PromptOutputValidationError(detail)
    if expected_count is not None and len(parsed) != expected_count:
        raise PromptOutputValidationError(f"Expected {expected_count} candidates, got {len(parsed)}.")
    return parsed


def parse_score_prediction_response(raw_text: str, *, strict: bool = True) -> float:
    text = raw_text.strip()
    payload = _parse_jsonish_mapping(text)
    if payload is None:
        payload = _extract_json_mapping_with_key(text, "predicted_objective")
    if payload is not None and "predicted_objective" in payload:
        value = payload["predicted_objective"]
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise PromptOutputValidationError("`predicted_objective` must be a number.") from exc
        if not math.isfinite(numeric):
            raise PromptOutputValidationError("`predicted_objective` must be finite.")
        return numeric

    blocks = re.findall(
        r"<score>\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*</score>",
        text,
        flags=re.IGNORECASE,
    )
    if not blocks and not strict:
        blocks = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text)
    if blocks:
        numeric = float(blocks[0])
        if math.isfinite(numeric):
            return numeric
    raise PromptOutputValidationError("Response does not contain a finite predicted objective.")


def _candidate_items_from_response(raw_text: str, *, allow_bare_config: bool = False) -> list[Any]:
    text = raw_text.strip()
    payload = _parse_jsonish_mapping(text)
    if payload is None:
        payload = _extract_json_mapping_with_key(text, "candidates")
    if payload is not None and "candidates" in payload:
        candidates = payload["candidates"]
        if isinstance(candidates, list):
            return list(candidates)
        raise PromptOutputValidationError("`candidates` must be a JSON array.")
    if allow_bare_config and payload is not None:
        return [payload]

    blocks = re.findall(r"<candidate>\s*(.*?)\s*</candidate>", text, flags=re.IGNORECASE | re.DOTALL)
    if not blocks:
        return []
    items: list[Any] = []
    for block in blocks:
        value = _parse_jsonish_value(block.strip())
        if value is not None:
            items.append(value)
    return items


def _candidate_mapping(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    item = dict(value)
    if "config" in item and isinstance(item["config"], Mapping):
        return dict(item["config"])
    return item


def _select_history(
    history: Sequence[TrialObservation],
    task_spec: TaskSpec,
    policy: HistoryRenderPolicy,
) -> list[TrialObservation]:
    primary = task_spec.primary_objective
    if policy.include_failed:
        observations = list(history)
    else:
        observations = [
            observation
            for observation in history
            if observation.success and primary.name in observation.objectives
        ]

    if policy.sort == "best":
        scored = [
            observation
            for observation in observations
            if observation.success and primary.name in observation.objectives
        ]
        reverse = primary.direction == ObjectiveDirection.MAXIMIZE
        observations = sorted(scored, key=lambda observation: float(observation.objectives[primary.name]), reverse=reverse)
    elif policy.sort == "recent":
        observations = list(reversed(observations))
    elif policy.sort == "original":
        observations = list(observations)
    else:
        raise ValueError(f"Unknown history sort policy `{policy.sort}`.")

    if policy.max_trials is not None:
        observations = observations[: max(0, int(policy.max_trials))]
    return observations


def _parameter_order(search_space: SearchSpace, parameter_order: Sequence[str] | None) -> list[str]:
    if parameter_order is None:
        return search_space.names()
    order = list(parameter_order)
    expected = set(search_space.names())
    if set(order) != expected:
        raise ValueError("parameter_order must contain exactly the search-space parameter names.")
    return order


def _ordered_config(
    search_space: SearchSpace,
    config: Mapping[str, Any],
    parameter_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    normalized = search_space.coerce_config(dict(config), use_defaults=False)
    order = _parameter_order(search_space, parameter_order)
    return {name: _format_scalar(normalized[name]) for name in order}


def _parameter_to_schema(param: ParameterSpec) -> dict[str, Any]:
    default = _safe_default(param)
    if isinstance(param, FloatParam):
        schema: dict[str, Any] = {
            "name": param.name,
            "type": "float",
            "low": float(param.low),
            "high": float(param.high),
            "log": bool(param.log),
        }
    elif isinstance(param, IntParam):
        schema = {
            "name": param.name,
            "type": "int",
            "low": int(param.low),
            "high": int(param.high),
            "log": bool(param.log),
        }
    elif isinstance(param, CategoricalParam):
        schema = {
            "name": param.name,
            "type": "categorical",
            "choices": list(param.choices),
        }
    elif isinstance(param, StringParam):
        schema = {
            "name": param.name,
            "type": "string",
            "min_length": int(param.min_length),
        }
        if param.max_length is not None:
            schema["max_length"] = int(param.max_length)
        if param.pattern is not None:
            schema["pattern"] = param.pattern.pattern if hasattr(param.pattern, "pattern") else str(param.pattern)
    else:
        raise TypeError(f"Unsupported parameter type for prompt schema: {type(param).__name__}")
    if default is not _MISSING:
        schema["default"] = default
    return schema


def _parameter_to_json_schema(param: ParameterSpec) -> dict[str, Any]:
    if isinstance(param, FloatParam):
        return {"type": "number", "minimum": float(param.low), "maximum": float(param.high)}
    if isinstance(param, IntParam):
        return {"type": "integer", "minimum": int(param.low), "maximum": int(param.high)}
    if isinstance(param, CategoricalParam):
        return {"enum": list(param.choices)}
    if isinstance(param, StringParam):
        schema: dict[str, Any] = {"type": "string", "minLength": int(param.min_length)}
        if param.max_length is not None:
            schema["maxLength"] = int(param.max_length)
        if param.pattern is not None:
            schema["pattern"] = param.pattern.pattern if hasattr(param.pattern, "pattern") else str(param.pattern)
        return schema
    raise TypeError(f"Unsupported parameter type for candidate JSON schema: {type(param).__name__}")


class _Missing:
    pass


_MISSING = _Missing()


def _safe_default(param: ParameterSpec) -> Any:
    try:
        return param.effective_default()
    except Exception:
        return _MISSING


def _better_relation(direction: ObjectiveDirection) -> str:
    return "lower" if direction == ObjectiveDirection.MINIMIZE else "higher"


def _format_scalar(value: Any) -> Any:
    if isinstance(value, float):
        return float(f"{value:.12g}")
    return value


def _json_scalar(value: Any) -> str:
    return json.dumps(_format_scalar(value), ensure_ascii=False)


def _strip_markdown_heading(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        return "\n".join(lines[1:]).strip()
    return text.strip()


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _parse_jsonish_mapping(text: str) -> dict[str, Any] | None:
    value = _parse_jsonish_value(text)
    return dict(value) if isinstance(value, Mapping) else None


def _parse_jsonish_value(text: str) -> Any:
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except Exception:
            continue
    return None


def _extract_json_mapping_with_key(text: str, key: str) -> dict[str, Any] | None:
    if not text or text.startswith("```"):
        return None
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and key in payload:
            return dict(payload)
    return None


__all__ = [
    "DEFAULT_CONTEXT_SECTIONS",
    "HistoryRenderPolicy",
    "OutputContract",
    "PromptContext",
    "PromptContract",
    "PromptOutputValidationError",
    "SearchSpaceRenderPolicy",
    "TaskContextRenderPolicy",
    "best_primary_objective",
    "build_prompt_context",
    "candidate_config_json_schema",
    "candidate_response_json_schema",
    "compile_candidate_generation_prompt",
    "compile_score_prediction_prompt",
    "default_candidate_generation_contract",
    "default_score_prediction_contract",
    "format_candidate_response",
    "format_score_prediction_response",
    "parse_candidate_response",
    "parse_score_prediction_response",
    "render_history_block",
    "render_search_space_block",
    "render_task_context_block",
    "score_prediction_json_schema",
    "search_space_to_schema",
    "serialize_config",
]
