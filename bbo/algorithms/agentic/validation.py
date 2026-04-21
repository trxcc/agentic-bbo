"""Strict JSON validation for Pablo role outputs."""

from __future__ import annotations

import json
from typing import Any, Mapping

from ...core import SearchSpace
from .serialization import stable_config_identity


class PabloValidationError(ValueError):
    """Raised when a provider response violates the Pablo JSON contract."""


def parse_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if not text:
        raise PabloValidationError("Provider response is empty.")
    if text.startswith("```"):
        raise PabloValidationError("Provider response must be raw JSON, not markdown-wrapped JSON.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PabloValidationError(f"Provider response is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise PabloValidationError("Provider response must be a JSON object.")
    return payload


def validate_planner_tasks(raw_text: str) -> dict[str, str]:
    payload = parse_json_object(raw_text)
    validated: dict[str, str] = {}
    for raw_name, raw_text_value in payload.items():
        name = str(raw_name).strip()
        text = str(raw_text_value).strip()
        if not name or not text:
            raise PabloValidationError("Planner task names and texts must be non-empty strings.")
        validated[name] = text
    if not validated:
        raise PabloValidationError("Planner must return at least one task.")
    return validated


def validate_candidate_payload(raw_text: str, search_space: SearchSpace) -> list[dict[str, Any]]:
    payload = parse_json_object(raw_text)
    if set(payload) != {"candidates"}:
        raise PabloValidationError("Candidate response must contain exactly one top-level key: `candidates`.")
    candidates = payload["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise PabloValidationError("Candidate response must provide a non-empty `candidates` list.")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in candidates:
        if not isinstance(item, Mapping):
            raise PabloValidationError("Each candidate must be a JSON object.")
        candidate_mapping = dict(item)
        if "config" in candidate_mapping and isinstance(candidate_mapping["config"], Mapping):
            candidate_mapping = dict(candidate_mapping["config"])
        config = search_space.coerce_config(candidate_mapping, use_defaults=False)
        identity = stable_config_identity(config)
        if identity in seen:
            continue
        seen.add(identity)
        normalized.append(config)
    if not normalized:
        raise PabloValidationError("Candidate response did not contain any valid unique configurations.")
    return normalized


__all__ = ["PabloValidationError", "parse_json_object", "validate_candidate_payload", "validate_planner_tasks"]
