"""Reviewable text-only proposal contract; no model or evaluator is invoked here."""
from __future__ import annotations
import copy
import json
import math

INSTRUCTIONS = '''You are optimizing a black-box objective within a fixed evaluation budget.
Use only the supplied task facts, parameter definitions and host-evaluated observations.
You have no tools: no shell, code execution, files, browsing, external services,
optimizer interfaces or tool-based submission. Do not request or simulate tool calls.
Reason over the provided information and return exactly one JSON object:
{"config": {"parameter_name": value}}
Include every required parameter with its original name and legal value. Do not
include commentary, Markdown fences, multiple candidates or claimed objective values.
Only the host evaluates the configuration and provides its actual result next round.
The conversation persists across rounds. Use observations to assess the supplied
prior; no fixed search strategy or change of strategy is required each round.
Do not reconstruct the hidden evaluator or claim unevaluated objective scores.
'''


def first_message(shared):
    return dict(task=shared['visible_task_id'],objective=shared['objective'],
                evaluation_budget=shared['evaluation_budget'],
                task_facts=shared['task_facts'],parameter_definitions=shared['parameter_definitions'],
                initial_history=shared['initial_history'],round=1,
                remaining_evaluations=shared['evaluation_budget']['new_evaluations'],
                submission='Return exactly one JSON object {"config":{...}} with all parameters.')


def strip_tools(provider_request):
    request=copy.deepcopy(provider_request)
    for key in ('tools','tool_choice','parallel_tool_calls'):request.pop(key,None)
    return request


def reject_tool_response(response):
    if response.get('tool_calls') or response.get('function_call'):
        raise ValueError('Text-only condition rejects every tool/function call')


def parse_candidate(text,space):
    value=json.loads(text)
    if not isinstance(value,dict) or set(value)!={'config'} or not isinstance(value['config'],dict):
        raise ValueError('Expected exactly one object with a config field')
    candidate=value['config']
    if set(candidate)!=set(space.names()):raise ValueError('Missing or additional parameter')
    if any(isinstance(v,(int,float)) and not isinstance(v,bool) and not math.isfinite(v) for v in candidate.values()):
        raise ValueError('Nonfinite value')
    space.validate_config(candidate)
    return candidate
