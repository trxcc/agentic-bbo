from __future__ import annotations

import asyncio
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import sys
import threading
import time
from urllib.parse import urlsplit

import pytest

from bbo.algorithms.agentic import AgentResult, MockAgentEngine, CodexBBOAlgorithm
from bbo.algorithms.agentic.codex_responses_compat import (
    SGLangResponsesCompatibilityProxy, chat_to_responses_sse,
)
from bbo.algorithms.agentic import runtime_reliability as reliability
from bbo.algorithms.agentic.general_agent_engines import _HOST_TOOL_CLIENT
from conftest import create_agent_test_task


@pytest.mark.parametrize("recover", [True, False])
def test_proxy_retries_same_request_and_caps_upstream_budget(tmp_path, monkeypatch, recover):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            status = 200 if recover and len(requests) > 1 else 429
            body = b'{"ok":true}' if status == 200 else b'{"error":"rate limited"}'
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Retry-After', '0')
            self.end_headers()
            self.wfile.write(body)

    monkeypatch.setattr(reliability, 'HTTP_RETRIES', 2)
    monkeypatch.setattr(reliability, 'pace_request', lambda: None)
    monkeypatch.setattr(reliability, 'retry_delay', lambda *_: 0)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    audit = tmp_path/'audit.jsonl'
    proxy = SGLangResponsesCompatibilityProxy(
        f'http://127.0.0.1:{server.server_port}/v1', dialect='deepseek',
        reliable_runtime=True, audit_path=audit, deadline=time.monotonic()+10,
    )
    proxy.start()
    address = urlsplit(proxy.base_url)
    body = json.dumps({'model':'test-kimi', 'input':'read workspace', 'max_output_tokens':123,
                       'tools':[{'type':'function','name':'shell','parameters':{'type':'object'}}]})

    def request():
        connection = http.client.HTTPConnection(address.hostname, address.port, timeout=5)
        connection.request('POST', '/v1/responses', body, {'Content-Type':'application/json'})
        response = connection.getresponse()
        status = response.status
        response.read()
        connection.close()
        return status

    try:
        assert request() == (200 if recover else 429)
        count = 2 if recover else 3
        assert len(requests) == count
        assert all(r == requests[0] for r in requests)
        assert 'max_tokens' not in requests[0]
        assert requests[0]['tools'][0]['function']['name'] == 'shell'
        if not recover:
            assert request() == 429
            assert len(requests) == count  # Native CLI retry cannot restart budget.
        events = [json.loads(line) for line in audit.read_text().splitlines()]
        assert any(e['event'] == 'http_retry' for e in events)
        assert any(e['event'] == 'request_schema' for e in events)
    finally:
        proxy.close()
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_truncated_sse_cannot_become_successful_completion():
    event = b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
    with pytest.raises(ConnectionError, match='without'):
        list(chat_to_responses_sse([event], request={}, require_done=True))
    completed = b''.join(chat_to_responses_sse([event, b'data: [DONE]\n\n'], request={}, require_done=True))
    assert b'response.completed' in completed


def test_retry_after_overrides_exponential_backoff(monkeypatch):
    monkeypatch.setattr(reliability.random, 'uniform', lambda *_: 0)
    assert reliability.retry_delay(4, '17') == 17
    assert reliability.retry_delay(0, None) == 5
    assert reliability.retry_delay(4, None) == 60


def test_operational_failure_stops_without_candidate_correction(tmp_path):
    class FailedEngine(MockAgentEngine):
        async def run_agent(self, *args, **kwargs):
            self.calls += 1
            return AgentResult(status='failed', answer='', error='HTTP 429')

    engine = FailedEngine()
    task = create_agent_test_task(max_evaluations=4, seed=7)
    algorithm = CodexBBOAlgorithm(engine=engine, run_dir=tmp_path/'run',
                                   reliable_runtime=True, max_retries=5, allow_fallback=False)
    algorithm.setup(task.spec, seed=7, task_description=task.get_description())
    with pytest.raises(reliability.AgentTransportError, match='429'):
        algorithm.ask()
    assert engine.calls == 1
    records = [json.loads(line) for line in Path(algorithm.artifact_paths['agent_calls_jsonl']).read_text().splitlines()]
    assert len(records) == 1 and records[0]['status'] == 'failed'


def test_native_policy_failure_is_not_mislabeled_as_provider_outage(tmp_path):
    from bbo.algorithms.agentic.native_round_guard import AgentRoundPolicyError
    class Exhausted(MockAgentEngine):
        async def run_agent(self, *args, **kwargs):
            self.calls += 1
            return AgentResult(status='failed',answer='',error='local guard exhausted',llm_log={
                'nativeRoundGuard':{'failure':{'error_type':'NativeBudgetExceeded','message':'Native budget exhausted'}}})
    engine=Exhausted()
    task=create_agent_test_task(max_evaluations=4,seed=7)
    algorithm=CodexBBOAlgorithm(engine=engine,run_dir=tmp_path/'run',reliable_runtime=True,max_retries=5)
    algorithm.setup(task.spec,seed=7,task_description=task.get_description())
    with pytest.raises(AgentRoundPolicyError,match='Native budget'):
        algorithm.ask()
    assert engine.calls==1


def test_candidate_correction_remains_honest_and_resume_policy_is_fixed(tmp_path):
    task = create_agent_test_task(max_evaluations=4, seed=7)
    algorithm = CodexBBOAlgorithm(engine=MockAgentEngine(), run_dir=tmp_path/'run', reliable_runtime=True)
    algorithm.setup(task.spec, seed=7, task_description=task.get_description())
    correction = algorithm._retry_instruction(last_error='bad JSON', round_call_ids=[], call_id='call1')
    assert 'remain available' in correction
    assert 'Do not assume a valid candidate already exists' in correction
    changed = CodexBBOAlgorithm(engine=MockAgentEngine(), run_dir=tmp_path/'run', resume=True)
    with pytest.raises(ValueError, match='reliability policy'):
        changed.setup(task.spec, seed=7, task_description=task.get_description())


def test_tool_client_replaces_inherited_30_second_timeout_and_records_receipt(tmp_path, monkeypatch):
    class Client:
        timeout = 30
        request = None

        def sendall(self, body):
            self.request = json.loads(body)

        def settimeout(self, timeout):
            self.timeout = timeout

        def recv(self, size):
            # Simulate a GP result that needs 31 seconds: legacy recv would fail.
            if self.timeout < 31:
                raise TimeoutError('GP outlasted socket timeout')
            return json.dumps({'ok':True,'output':{'x':0.5},'request_id':self.request['request_id']}).encode()+b'\n'

    client = Client()
    monkeypatch.setattr(socket, 'create_connection', lambda *args, **kwargs: client)
    monkeypatch.setenv('BBO_HOST_TOOL_SOCKET', 'tcp://127.0.0.1:1234')
    monkeypatch.setenv('BBO_HOST_TOOL_DEADLINE', str(time.monotonic()+90))
    monkeypatch.setenv('BBO_TOOL_RECEIPT_PATH', 'receipt.jsonl')
    monkeypatch.setattr(sys, 'argv', ['bbo_tool.py','optimizer_suggest','{}'])
    exec(compile(_HOST_TOOL_CLIENT, 'bbo_tool.py', 'exec'), {'__file__':str(tmp_path/'bbo_tool.py')})
    receipt = json.loads((tmp_path/'receipt.jsonl').read_text())
    assert receipt['event'] == 'client_received'
    assert receipt['request_id'] == client.request['request_id']
    assert receipt['ok'] is True


def test_resumed_candidate_correction_keeps_native_shell(tmp_path, monkeypatch):
    from bbo.algorithms.agentic import general_agent_engines as engines
    executable = tmp_path/'codex'
    executable.touch()
    workspace, state = tmp_path/'workspace', tmp_path/'state'
    workspace.mkdir()
    state.mkdir()
    captured = []

    class Process:
        returncode = 0

        async def communicate(self):
            return b'{"type":"item.completed","item":{"type":"agent_message","text":"{}"}}', b''

    async def create(*cmd, **kwargs):
        captured.extend(cmd)
        return Process()

    monkeypatch.setattr(engines.asyncio, 'create_subprocess_exec', create)
    result = asyncio.run(engines.CodexEngine().run_agent(
        'thread1', 'continue', engines.AgentWorkCopy(
            state_dir=state, config_path=state/'config.toml', project_root=workspace,
            workspace_root=workspace, extra={'codex_config':{'executable':str(executable),'reliable_runtime':True}},
        ), timeout=1, final_instruction='Correct the candidate JSON.',
    ))
    assert result.status == 'success'
    assert 'resume' in captured
    assert 'shell_tool' not in captured


@pytest.mark.parametrize('finish_reason', ['length', 'content_filter'])
def test_done_marker_does_not_hide_provider_truncation(finish_reason):
    events = [b'data: ' + json.dumps({'choices': [{'delta': {'content': '{"x":1}'},
              'finish_reason': finish_reason}]}).encode() + b'\n\n', b'data: [DONE]\n\n']
    audit = []
    with pytest.raises(ConnectionError, match=finish_reason):
        list(chat_to_responses_sse(events, request={}, require_done=True, audit=audit.append))
    end = audit[-1]
    assert end['done_received'] and end['finish_reason'] == finish_reason
    assert end['text_chars'] == 7
    assert 'content' not in end  # Audit contains counts, not model text.


def test_normal_tool_stream_preserves_calls_and_audits_finish_and_usage():
    chunks = [
        {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'call1',
          'function': {'name': 'exec_command', 'arguments': '{"cmd":"true"}'}}]}, 'finish_reason': None}]},
        {'choices': [{'delta': {}, 'finish_reason': 'tool_calls'}]},
        {'choices': [], 'usage': {'prompt_tokens': 100, 'completion_tokens': 20,
         'prompt_tokens_details': {'cached_tokens': 80}}},
    ]
    events = [b'data: ' + json.dumps(c).encode() + b'\n\n' for c in chunks] + [b'data: [DONE]\n\n']
    audit = []
    result = b''.join(chat_to_responses_sse(events, request={}, require_done=True, audit=audit.append))
    assert b'response.completed' in result and b'exec_command' in result
    assert audit[-1]['finish_reason'] == 'tool_calls'
    assert audit[-1]['usage']['cached_input_tokens'] == 80
    assert audit[-1]['tool_calls'] == 1


def test_broken_stream_audits_partial_usage_without_success():
    def source():
        yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
        raise TimeoutError('upstream read')
    audit = []
    with pytest.raises(TimeoutError):
        list(chat_to_responses_sse(source(), request={}, require_done=True, audit=audit.append))
    assert audit[-1]['done_received'] is False
    assert audit[-1]['text_chars'] == 7


def test_native_timeout_preserves_stdout_events_session_and_usage(tmp_path, monkeypatch):
    from bbo.algorithms.agentic import general_agent_engines as engines
    executable = tmp_path/'codex'
    executable.touch()
    workspace, state = tmp_path/'workspace', tmp_path/'state'
    workspace.mkdir()
    state.mkdir()
    original_create = asyncio.create_subprocess_exec
    events = [
        {'type': 'thread.started', 'thread_id': 'retained-thread'},
        {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'partial work'}},
        {'type': 'turn.completed', 'usage': {'input_tokens': 123, 'output_tokens': 45}},
    ]
    script = 'import time\n' + '\n'.join(f'print({json.dumps(e)!r}, flush=True)' for e in events) + '\ntime.sleep(30)'
    async def create(*cmd, **kwargs):
        return await original_create(sys.executable, '-u', '-c', script, **kwargs)
    monkeypatch.setattr(engines.asyncio, 'create_subprocess_exec', create)
    result = asyncio.run(engines.CodexEngine().run_agent(
        '', 'offline test', engines.AgentWorkCopy(
            state_dir=state, config_path=state/'config.toml', project_root=workspace,
            workspace_root=workspace, extra={'codex_config': {'executable': str(executable), 'reliable_runtime': True}},
        ), timeout=0.5,
    ))
    assert result.status == 'timeout'
    assert result.raw == events
    assert result.llm_log['sessionId'] == 'retained-thread'
    assert result.llm_log['usage']['input_tokens'] == 123


def test_unlimited_agent_config_keeps_none_and_rejects_nonpositive_limits(tmp_path):
    task = create_agent_test_task(max_evaluations=4, seed=7)
    algorithm = CodexBBOAlgorithm(engine=MockAgentEngine(), run_dir=tmp_path/'run', timeout_seconds=None)
    algorithm.setup(task.spec, seed=7, task_description=task.get_description())
    assert algorithm.config.timeout_seconds is None
    for invalid in (0, -1):
        with pytest.raises(ValueError, match='positive or None'):
            CodexBBOAlgorithm(engine=MockAgentEngine(), timeout_seconds=invalid)


def test_unlimited_tool_client_clears_connection_timeout(tmp_path, monkeypatch):
    class Client:
        timeout = 30
        def settimeout(self, timeout):
            self.timeout = timeout
        def sendall(self, body):
            self.request = json.loads(body)
        def recv(self, size):
            assert self.timeout is None
            return json.dumps({'ok': True, 'output': {}, 'request_id': self.request['request_id']}).encode()+b'\n'
    monkeypatch.setattr(socket, 'create_connection', lambda *a, **k: Client())
    monkeypatch.setenv('BBO_HOST_TOOL_SOCKET', 'tcp://127.0.0.1:1234')
    monkeypatch.setenv('BBO_HOST_TOOL_UNLIMITED', '1')
    monkeypatch.setenv('BBO_HOST_TOOL_DEADLINE', '0')
    monkeypatch.delenv('BBO_TOOL_RECEIPT_PATH', raising=False)
    monkeypatch.setattr(sys, 'argv', ['bbo_tool.py', 'optimizer_suggest', '{}'])
    exec(compile(_HOST_TOOL_CLIENT, 'bbo_tool.py', 'exec'), {'__file__': str(tmp_path/'bbo_tool.py')})


def test_unlimited_host_tools_do_not_restore_900_or_180_second_deadline(tmp_path, monkeypatch):
    from bbo.algorithms.agentic import general_agent_engines as engines
    captured = {}
    real_server = engines._start_host_tool_server
    def server(**kwargs):
        captured.update(kwargs)
        return real_server(**kwargs)
    class Engine(engines.CodexEngine):
        async def run_agent(self, *args, **kwargs):
            captured['inner_timeout'] = kwargs['timeout']
            captured['env'] = kwargs['extra_env']
            return AgentResult(status='success', answer='{}')
    async def tool(*args):
        return '{}'
    monkeypatch.setattr(engines, '_start_host_tool_server', server)
    work = engines.AgentWorkCopy(state_dir=tmp_path/'state', config_path=tmp_path/'config.toml',
        project_root=tmp_path, workspace_root=tmp_path,
        extra={'codex_config': {'reliable_runtime': True}})
    asyncio.run(Engine()._run_with_host_tools('', 'test', work, agent_id='test', timeout=None,
        extra_env=None, tools=[{'type': 'function', 'function': {'name': 'optimizer_suggest'}}],
        tool_executor=tool, max_tool_calls=64, final_instruction=None))
    assert captured['deadline'] is None and captured['unlimited_wait'] is True
    assert captured['inner_timeout'] is None
    assert captured['env']['BBO_HOST_TOOL_DEADLINE'] == '0'
    assert captured['env']['BBO_HOST_TOOL_UNLIMITED'] == '1'


def test_real_stream_progress_reaches_client_before_response_finishes():
    events = [b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
              b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
              b'data: [DONE]\n\n']
    consumed = []
    def source():
        for event in events:
            consumed.append(event)
            yield event
    stream = iter(chat_to_responses_sse(source(), request={}, require_done=True,
                                       progress_interval_seconds=0))
    emitted = [next(stream)]
    assert b'response.created' in emitted[0] and consumed == []
    emitted.append(next(stream))
    assert b'response.in_progress' in emitted[1] and len(consumed) == 1
    emitted.extend(stream)
    decoded = [json.loads(next(l[6:] for l in e.decode().splitlines() if l.startswith('data: '))) for e in emitted]
    assert [e['sequence_number'] for e in decoded] == list(range(len(decoded)))
    assert len({e['response']['id'] for e in decoded if 'response' in e}) == 1
    assert sum(e['type'] == 'response.created' for e in decoded) == 1
    assert decoded[-1]['type'] == 'response.completed'


def test_native_unlimited_wait_uses_no_outer_timer(tmp_path, monkeypatch):
    from bbo.algorithms.agentic import general_agent_engines as engines
    executable = tmp_path/'codex'
    executable.touch()
    (tmp_path/'state').mkdir()
    original_create = asyncio.create_subprocess_exec
    event = json.dumps({'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'finished'}})
    script = f'import time; time.sleep(0.1); print({event!r}, flush=True)'
    async def create(*cmd, **kwargs):
        return await original_create(sys.executable, '-u', '-c', script, **kwargs)
    async def forbidden_timer(*args, **kwargs):
        pytest.fail('Unlimited invocation unexpectedly installed an outer timer')
    monkeypatch.setattr(engines.asyncio, 'create_subprocess_exec', create)
    monkeypatch.setattr(engines.asyncio, 'wait_for', forbidden_timer)
    result = asyncio.run(engines.CodexEngine().run_agent('', 'offline test', engines.AgentWorkCopy(
        state_dir=tmp_path/'state', config_path=tmp_path/'state/config.toml', project_root=tmp_path,
        workspace_root=tmp_path, extra={'codex_config': {'executable': str(executable), 'reliable_runtime': True}},
    ), timeout=None))
    assert result.status == 'success' and result.answer == 'finished'
