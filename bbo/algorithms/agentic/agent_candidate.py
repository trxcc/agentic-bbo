"""Candidate parsing and normalization for general-agent optimizers."""

from __future__ import annotations

import html
import json
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Mapping

from ...core import SearchSpace, search_space_to_schema
from .validation import CandidateValidationError, parse_json_object
from .serialization import stable_config_identity


class GeneralAgentValidationError(ValueError):
    """Raised when a general agent returns an invalid candidate payload."""



@dataclass(frozen=True)
class ParsedAgentCandidate:
    """One parsed candidate from a raw agent response."""

    config: dict[str, Any]
    candidate_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _retry_feedback_block(last_error: str | None, *, mention_tool_calls: bool = True) -> str:
    if not last_error:
        return ""
    error_text = " ".join(str(last_error).split())
    if len(error_text) > 800:
        error_text = error_text[:800].rstrip() + "..."
    tool_call_line = (
        "- Do not return tool-call XML such as `<tool_call>` or `<function=...>`."
        if mention_tool_calls
        else "- Do not return markup wrappers or intermediate action requests."
    )
    return textwrap.dedent(
        f"""
        Previous attempt failed and was not accepted:
        {error_text}

        Retry correction:
        - Correct the issue above in this attempt.
        - Rewrite `final_candidate.json` with exactly the required raw JSON object.
        - Verify it with `python -m json.tool final_candidate.json` before finishing.
        - Return the same raw JSON object as the final answer.
        {tool_call_line}
        - Do not return an error message, prose-only analysis, markdown, or a partial configuration.
        """
    ).rstrip()


def parse_agent_candidate_payload(raw_text: str, search_space: SearchSpace) -> list[ParsedAgentCandidate]:
    try:
        payload = parse_json_object(raw_text)
    except CandidateValidationError as exc:
        extracted = (
            _extract_candidates_json_object(raw_text)
            or _extract_bbo_text_tool_call_candidates(raw_text)
            or _extract_named_numeric_candidate_payload(raw_text, search_space)
        )
        if extracted is None:
            raise GeneralAgentValidationError(str(exc)) from exc
        payload = extracted
    if set(payload) != {"candidates"}:
        raise GeneralAgentValidationError("Agent response must contain exactly one top-level key: `candidates`.")
    raw_candidates = payload["candidates"]
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise GeneralAgentValidationError("Agent response must provide a non-empty `candidates` list.")

    parsed: list[ParsedAgentCandidate] = []
    seen: set[str] = set()
    errors: list[str] = []
    for index, item in enumerate(raw_candidates):
        if not isinstance(item, Mapping):
            single_param_name = _single_parameter_name(search_space)
            if single_param_name is None:
                errors.append(f"Candidate {index} is not a JSON object.")
                continue
            item_dict = {single_param_name: item}
            metadata = {}
            config_mapping = item_dict
        else:
            item_dict = dict(item)
            metadata: dict[str, Any] = {}
            if "config" in item_dict:
                raw_config = item_dict.pop("config")
                metadata = dict(item_dict)
                if not isinstance(raw_config, Mapping):
                    errors.append(f"Candidate {index} `config` is not a JSON object.")
                    continue
                config_mapping = dict(raw_config)
            else:
                config_mapping = item_dict
                if "x" in item_dict and "y" in item_dict:
                    metadata = {key: value for key, value in item_dict.items() if key not in {"x", "y"}}
        compact_config = _compact_xy_candidate_to_config(config_mapping, search_space)
        if compact_config is not None:
            config_mapping = compact_config
        try:
            config = search_space.coerce_config(config_mapping, use_defaults=False)
        except Exception as exc:
            errors.append(f"Candidate {index} is invalid: {exc}")
            continue
        identity = stable_config_identity(config)
        if identity in seen:
            continue
        seen.add(identity)
        parsed.append(ParsedAgentCandidate(config=config, candidate_index=index, metadata=metadata))
    if not parsed:
        detail = "; ".join(errors) if errors else "Agent response did not contain any valid unique configurations."
        raise GeneralAgentValidationError(detail)
    return parsed


def _extract_candidates_json_object(raw_text: str) -> dict[str, Any] | None:
    text = raw_text.strip()
    if not text:
        return None
    text = _strip_markdown_json_fence(text)
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "candidates" in payload:
            return payload
    for object_text in _balanced_json_object_texts(text):
        payload = _loads_lenient_json_object(object_text)
        if isinstance(payload, dict) and "candidates" in payload:
            return payload
    return None


def _extract_bbo_text_tool_call_candidates(raw_text: str) -> dict[str, Any] | None:
    text = html.unescape(raw_text.strip())
    if not text:
        return None
    direct_candidates = _extract_inline_bbo_validate_calls(text)
    if direct_candidates:
        return {"candidates": direct_candidates}
    tool_call_re = re.compile(r"<tool_call>(.*?)</tool_call>", re.IGNORECASE | re.DOTALL)
    function_re = re.compile(r"<function=([A-Za-z_][\w.-]*)>(.*?)</function>", re.IGNORECASE | re.DOTALL)
    parameter_re = re.compile(r"<parameter=([A-Za-z_][\w.-]*)>(.*?)</parameter>", re.IGNORECASE | re.DOTALL)
    candidate_actions = {
        "",
        "validate",
        "validate_candidate",
        "validate_candidates",
        "submit_candidate",
        "suggest_candidate",
        "propose_candidate",
    }
    candidates: list[Any] = []
    for block_match in tool_call_re.finditer(text):
        function_match = function_re.search(block_match.group(1))
        if not function_match:
            continue
        function_name = function_match.group(1).strip().lower().replace("-", "_")
        body = function_match.group(2).strip()
        params = {
            match.group(1).strip().lower().replace("-", "_"): html.unescape(match.group(2)).strip()
            for match in parameter_re.finditer(body)
        }
        action = str(params.get("action") or "").strip().lower().replace("-", "_")
        if function_name == "bbo":
            if action not in candidate_actions:
                continue
        elif function_name not in candidate_actions:
            continue
        if "candidates" in params:
            decoded = _loads_lenient_json_value(params["candidates"])
            if isinstance(decoded, dict) and "candidates" in decoded:
                raw_candidates = decoded.get("candidates")
                if isinstance(raw_candidates, list):
                    candidates.extend(raw_candidates)
            elif isinstance(decoded, list):
                candidates.extend(decoded)
            elif isinstance(decoded, dict):
                candidates.append(decoded)
            continue
        for key in ("candidate", "config"):
            if key not in params:
                continue
            decoded = _loads_lenient_json_value(params[key])
            if isinstance(decoded, dict) and "candidates" in decoded:
                raw_candidates = decoded.get("candidates")
                if isinstance(raw_candidates, list):
                    candidates.extend(raw_candidates)
            elif isinstance(decoded, list):
                candidates.extend(decoded)
            elif isinstance(decoded, dict):
                candidates.append(decoded)
            break
        else:
            for action_name in sorted(candidate_actions - {""}):
                if action_name not in params:
                    continue
                decoded = _loads_lenient_json_value(params[action_name])
                if isinstance(decoded, dict) and "candidates" in decoded:
                    raw_candidates = decoded.get("candidates")
                    if isinstance(raw_candidates, list):
                        candidates.extend(raw_candidates)
                elif isinstance(decoded, list):
                    candidates.extend(decoded)
                elif isinstance(decoded, dict):
                    candidates.append(decoded)
                elif isinstance(decoded, str) and decoded.strip():
                    candidates.append(decoded.strip())
                break
            if "smiles" in params:
                candidates.append({"smiles": params["smiles"]})
    if not candidates:
        return None
    return {"candidates": candidates}


def _extract_inline_bbo_validate_calls(text: str) -> list[Any]:
    calls: list[Any] = []
    pattern = re.compile(
        r"BBO(?:\(\))?\.(?:validate_candidate|validate_candidates)\s*\(",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        start = match.end()
        end = _matching_paren_index(text, start - 1)
        if end is None:
            continue
        raw_args = text[start:end].strip()
        if not raw_args:
            continue
        decoded = _loads_lenient_json_value(raw_args)
        if isinstance(decoded, dict) and "candidates" in decoded:
            raw_candidates = decoded.get("candidates")
            if isinstance(raw_candidates, list):
                calls.extend(raw_candidates)
        elif isinstance(decoded, list):
            calls.extend(decoded)
        elif isinstance(decoded, dict):
            if isinstance(decoded.get("candidate"), Mapping):
                calls.append(dict(decoded["candidate"]))
            elif isinstance(decoded.get("config"), Mapping):
                calls.append(dict(decoded["config"]))
            else:
                calls.append(decoded)
        elif isinstance(decoded, str) and decoded.strip():
            calls.append(decoded.strip())
    return calls


def _matching_paren_index(text: str, open_index: int) -> int | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(open_index, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _strip_markdown_json_fence(text: str) -> str:
    lines = text.splitlines()
    if not lines or not lines[0].lstrip().startswith("```"):
        return text
    fence_label = lines[0].strip()[3:].strip().lower()
    if fence_label and fence_label not in {"json", "javascript", "js"}:
        return text
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return "\n".join(lines[1:]).strip()


def _balanced_json_object_texts(text: str) -> list[str]:
    objects: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if start is None:
            if char != "{":
                continue
            start = index
            depth = 1
            in_string = False
            escaped = False
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                objects.append(text[start : index + 1])
                start = None
    return objects


def _loads_lenient_json_object(object_text: str) -> dict[str, Any] | None:
    candidates = [object_text, _escape_control_chars_in_strings(object_text)]
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    try:
        from json_repair import loads as repair_loads  # type: ignore
    except ImportError:
        return None
    try:
        payload = repair_loads(object_text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _loads_lenient_json_value(raw_text: str) -> Any:
    text = _strip_markdown_json_fence(raw_text.strip())
    for candidate in (text, _escape_control_chars_in_strings(text)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    object_payload = _loads_lenient_json_object(text)
    if object_payload is not None:
        return object_payload
    argument_pair_payload = _loads_lenient_key_value_argument_pair(text)
    if argument_pair_payload is not None:
        return argument_pair_payload
    key_value_payload = _loads_lenient_quoted_key_value_fragment(text)
    if key_value_payload is not None:
        return key_value_payload
    return text


def _loads_lenient_key_value_argument_pair(text: str) -> dict[str, Any] | None:
    stripped = text.strip().rstrip(",")
    for candidate in (stripped, _escape_control_chars_in_strings(stripped)):
        try:
            payload = json.loads(f"[{candidate}]")
        except json.JSONDecodeError:
            continue
        if (
            isinstance(payload, list)
            and len(payload) == 2
            and isinstance(payload[0], str)
            and re.fullmatch(r"[A-Za-z_][\w.-]*", payload[0])
        ):
            return {payload[0]: payload[1]}
    match = re.fullmatch(r"""(["'])([A-Za-z_][\w.-]*)\1\s*,\s*(["'])(.*)\3""", stripped, re.DOTALL)
    if match:
        return {match.group(2): match.group(4)}
    return None


def _loads_lenient_quoted_key_value_fragment(text: str) -> dict[str, Any] | None:
    stripped = text.strip().rstrip(",")
    match = re.fullmatch(r"""(["'])([A-Za-z_][\w.-]*)\1\s*:\s*(.+)""", stripped, re.DOTALL)
    if not match:
        return None
    key = match.group(2)
    value_text = match.group(3).strip().rstrip(",")
    if not value_text:
        return None
    for candidate in (value_text, _escape_control_chars_in_strings(value_text)):
        try:
            return {key: json.loads(candidate)}
        except json.JSONDecodeError:
            continue
    quote = value_text[0] if value_text else ""
    if quote in {"'", '"'} and len(value_text) >= 2 and value_text[-1] == quote:
        return {key: value_text[1:-1]}
    return {key: value_text}


def _extract_named_numeric_candidate_payload(raw_text: str, search_space: SearchSpace) -> dict[str, Any] | None:
    text = html.unescape(raw_text)
    names = search_space.names()
    if not names:
        return None
    lowered = text.lower()
    keyword_positions = [lowered.rfind(keyword) for keyword in ("candidate", "propose", "suggest")]
    start = max(keyword_positions)
    scan_text = text[start:] if start >= 0 else text
    config: dict[str, Any] = {}
    for name in names:
        param = search_space[name]
        if getattr(param, "low", None) is None or getattr(param, "high", None) is None:
            return None
        pattern = re.compile(
            rf"(?<![\w.-]){re.escape(name)}\s*[:=]\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        )
        matches = list(pattern.finditer(scan_text))
        if not matches:
            return None
        config[name] = float(matches[-1].group(1))
    return {"candidates": [{"config": config}]}


def _single_parameter_name(search_space: SearchSpace) -> str | None:
    names = search_space.names()
    return names[0] if len(names) == 1 else None


def _escape_control_chars_in_strings(text: str) -> str:
    result: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                result.append(char)
                escaped = False
            elif char == "\\":
                result.append(char)
                escaped = True
            elif char == '"':
                result.append(char)
                in_string = False
            elif char == "\n":
                result.append("\\n")
            elif char == "\r":
                result.append("\\r")
            elif char == "\t":
                result.append("\\t")
            else:
                result.append(char)
            continue
        result.append(char)
        if char == '"':
            in_string = True
    return "".join(result)


def search_space_schema(search_space: SearchSpace) -> list[dict[str, Any]]:
    return search_space_to_schema(search_space)


def _paired_xy_parameter_count(search_space: SearchSpace) -> int:
    names = set(search_space.names())
    count = 0
    while f"x_{count}" in names and f"y_{count}" in names:
        count += 1
    if count <= 0:
        return 0
    expected = {f"x_{index}" for index in range(count)} | {f"y_{index}" for index in range(count)}
    return count if expected.issubset(names) else 0


def _compact_xy_candidate_to_config(candidate: Mapping[str, Any], search_space: SearchSpace) -> dict[str, Any] | None:
    n_pairs = _paired_xy_parameter_count(search_space)
    if n_pairs <= 0 or "x" not in candidate or "y" not in candidate:
        return None
    xs = candidate.get("x")
    ys = candidate.get("y")
    if not isinstance(xs, list) or not isinstance(ys, list):
        return None
    min_compact_length = max(1, n_pairs - 2)
    if len(xs) < min_compact_length or len(ys) < min_compact_length:
        return None
    xs = xs[:n_pairs]
    ys = ys[:n_pairs]
    if len(xs) < n_pairs:
        xs.extend(_default_xy_values(search_space, "x", start=len(xs), stop=n_pairs))
    if len(ys) < n_pairs:
        ys.extend(_default_xy_values(search_space, "y", start=len(ys), stop=n_pairs))
    config: dict[str, Any] = {}
    for index, value in enumerate(xs):
        name = f"x_{index}"
        config[name] = _clip_numeric_to_bounds(search_space, name, value)
    for index, value in enumerate(ys):
        name = f"y_{index}"
        config[name] = _clip_numeric_to_bounds(search_space, name, value)
    return config


def _clip_numeric_to_bounds(search_space: SearchSpace, name: str, value: Any) -> Any:
    param = search_space[name]
    low = getattr(param, "low", None)
    high = getattr(param, "high", None)
    if low is None or high is None or isinstance(value, bool):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    return min(max(numeric, float(low)), float(high))


def _default_xy_values(search_space: SearchSpace, axis: str, *, start: int, stop: int) -> list[Any]:
    values: list[Any] = []
    for index in range(start, stop):
        values.append(search_space[f"{axis}_{index}"].effective_default())
    return values
