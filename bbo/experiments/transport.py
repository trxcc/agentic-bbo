"""Explicit reasoning settings and usage auditing for the native bridge."""
from contextlib import contextmanager
from copy import deepcopy
import json
import threading
import time
import uuid

from .io import save, append


def apply_effort(request, profile):
    value = deepcopy(request)
    for key in ('reasoning', 'thinking', 'reasoning_effort', 'chat_template_kwargs', 'separate_reasoning'):
        value.pop(key, None)
    value['reasoning_effort'] = profile['reasoning_effort']
    if profile.get('explicit_thinking_enabled'):
        value['thinking'] = {'type': 'enabled'}
    return value


@contextmanager
def audited_transport(out, profile):
    # Run each trajectory in its own process; restore hooks on every exit path.
    from bbo.algorithms.agentic import codex_responses_compat as compat
    from bbo.algorithms.agentic.summary_checkpoint import finalize_wire_request
    original_translate = compat.responses_to_sglang_chat_request
    original_stream = compat.chat_to_responses_sse
    local = threading.local()

    def translate(body, *, dialect):
        native = json.loads(body)
        value = finalize_wire_request(native, json.loads(original_translate(body, dialect=dialect)),
                                      {'exec_command', 'write_stdin'})
        value = apply_effort(value, profile)
        if value['model'] != profile['model']:
            raise ValueError('Requested model changed during translation')
        local.request_id = uuid.uuid4().hex
        save(out / 'wire_audit' / (local.request_id + '.request.json'), value)
        append(out / 'requests.jsonl', dict(request_id=local.request_id, timestamp=time.time(),
            model=value['model'], requested_effort=value['reasoning_effort'], thinking=value.get('thinking')))
        return json.dumps(value, separators=(',', ':')).encode()

    def stream(events, *, request, **kwargs):
        identifier = local.request_id
        upstream = kwargs.pop('audit', None)
        models = set()
        def observed():
            for event in events:
                chunk = compat._sse_payload(event)
                if isinstance(chunk, dict):
                    if chunk.get('model'):
                        model = chunk['model']
                        if model not in profile.get('approved_response_model_ids', [profile['model']]):
                            raise ValueError('Unexpected response model: ' + str(model))
                        models.add(model)
                    if chunk.get('usage'):
                        append(out / 'provider_usage_chunks.jsonl', dict(request_id=identifier,
                            timestamp=time.time(), model=chunk.get('model'), usage=chunk['usage']))
                yield event
        def record(event):
            if upstream:
                upstream(event)
            if event.get('event') == 'upstream_stream_end':
                append(out / 'provider_usage.jsonl', dict(request_id=identifier, timestamp=time.time(),
                    returned_models=sorted(models), requested_effort=profile['reasoning_effort'], **event))
        yield from original_stream(observed(), request=request, audit=record, **kwargs)

    compat.responses_to_sglang_chat_request = translate
    compat.chat_to_responses_sse = stream
    try:
        yield
    finally:
        compat.responses_to_sglang_chat_request = original_translate
        compat.chat_to_responses_sse = original_stream
