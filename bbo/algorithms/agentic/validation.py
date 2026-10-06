"""Strict object parsing for model candidate responses."""
import json


class CandidateValidationError(ValueError):
    pass


def parse_json_object(raw_text):
    text = raw_text.strip()
    if not text:
        raise CandidateValidationError('Provider response is empty.')
    if text.startswith('```'):
        raise CandidateValidationError('Provider response must be raw JSON, not markdown-wrapped JSON.')
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CandidateValidationError(f'Provider response is not valid JSON: {exc}') from exc
    if not isinstance(payload, dict):
        raise CandidateValidationError('Provider response must be a JSON object.')
    return payload
