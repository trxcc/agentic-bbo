"""Unlimited call counts keep the tool gateway and submission checks active."""
import asyncio
import json
import socket
from types import SimpleNamespace

import pytest

from bbo.algorithms.agentic import general_agent_engines as engines
from bbo.algorithms.agentic.native_round_guard import NativeRoundGuard, InvalidNativeResponse


def test_unlimited_guard_allows_many_shell_calls_and_polls_but_requires_submission():
    guard = NativeRoundGuard(native_tool_limit=0)
    request = {'tools': [{'type': 'function', 'name': 'exec_command'},
                         {'type': 'function', 'name': 'write_stdin'}]}
    for i in range(1000):
        prepared = guard.prepare(request)
        assert 'no tool-call count limit' in prepared['instructions']
        guard.validate(SimpleNamespace(text_parts=[], tool_calls={str(i): {
            'name': 'exec_command' if i == 0 else 'write_stdin'}}), compact=False)
    assert guard.used == 1000 and guard.receipt is None
    with pytest.raises(InvalidNativeResponse, match='No candidate'):
        guard.validate(SimpleNamespace(text_parts=['Done'], tool_calls={}), compact=False)
    with pytest.raises(InvalidNativeResponse, match='Only exec_command'):
        guard.validate(SimpleNamespace(text_parts=[], tool_calls={'x': {'name': 'web_search'}}), compact=False)
    guard.observe_tool_result('submit_candidate', {'ok': True, 'result': {
        'status': 'accepted', 'terminal': True, 'submission_id': 's1'}})
    assert guard.receipt['submission_id'] == 's1'


def test_unlimited_host_gateway_allows_more_than_64_io_calls():
    async def scenario():
        called = []
        async def execute(name, arguments, call_id):
            called.append(arguments['i'])
            return json.dumps({'ok': True, 'result': arguments['i']})
        server, thread = engines._start_host_tool_server(
            allowed={'get_trial_history'}, executor=execute,
            loop=asyncio.get_running_loop(), max_calls=0)
        def requests():
            responses = []
            for i in range(70):
                with socket.create_connection(server.server_address, timeout=5) as client:
                    client.sendall(json.dumps({'name': 'get_trial_history', 'arguments': {'i': i}}).encode() + b'\n')
                    response = client.makefile('rb').readline()
                    responses.append(json.loads(response))
            return responses
        try:
            responses = await asyncio.to_thread(requests)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        assert called == list(range(70))
        assert all(r['ok'] and json.loads(r['output'])['ok'] for r in responses)
    asyncio.run(scenario())


def test_zero_budget_still_creates_cli_and_submission_guard(tmp_path):
    captured = {}
    class Engine(engines.CodexEngine):
        async def run_agent(self, session, prompt, work, **kwargs):
            guard = work.extra['codex_config']['round_guard']
            captured['limit'] = guard.limit
            captured['cli'] = (tmp_path / 'bbo_tool.py').is_file()
            return engines.AgentResult(status='success', answer='ok', llm_log={})
    async def execute(*args):
        return '{}'
    work = engines.AgentWorkCopy(state_dir=tmp_path/'state', config_path=tmp_path/'config.toml',
        project_root=tmp_path, workspace_root=tmp_path,
        extra={'codex_config': {'reliable_runtime': True, 'native_round_guard': True}})
    result = asyncio.run(Engine()._run_with_host_tools('', 'test', work,
        agent_id='test', timeout=None, extra_env=None,
        tools=[{'type': 'function', 'function': {'name': 'submit_candidate'}}],
        tool_executor=execute, max_tool_calls=0, final_instruction=None))
    assert result.status == 'success' and captured == {'limit': 0, 'cli': True}
    assert result.llm_log['nativeRoundGuard']['native_tool_limit'] == 0


@pytest.mark.parametrize('limit', [-1, True, None, 1.5])
def test_guard_rejects_invalid_limits(limit):
    with pytest.raises(ValueError, match='non-negative'):
        NativeRoundGuard(limit)
