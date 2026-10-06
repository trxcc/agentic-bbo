"""Opt-in real native CLI/container smoke with a local fake model, never a paid API."""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import shutil
import threading
import time

import pytest


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get('BBO_DOCKER_TEST_IMAGE'), reason='Set BBO_DOCKER_TEST_IMAGE for real Docker smoke')
def test_native_container_submission(tmp_path, monkeypatch):
    from bbo.experiments.tasks import PaperTask
    from bbo.experiments.agent import build_agent
    from bbo.experiments.transport import audited_transport
    task=PaperTask('bbob_f15_d10')
    candidate=task.spec.search_space.defaults()
    calls=[]
    command='python3 bbo_tool.py submit_candidate '+repr(json.dumps({'config':candidate}))
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append(body)
            assert body['reasoning_effort']=='max'
            names=[t['function']['name'] for t in body.get('tools',[])]
            name=next(n for n in names if n.split('__')[-1]=='exec_command')
            chunk=dict(id='test-completion',object='chat.completion.chunk',created=int(time.time()),model='gpt-6-astra',
                choices=[dict(index=0,delta=dict(role='assistant',tool_calls=[dict(index=0,id='test-call',type='function',
                    function=dict(name=name,arguments=json.dumps(dict(cmd=command,yield_time_ms=1000,max_output_tokens=1000))))]),
                    finish_reason='tool_calls')],usage=dict(prompt_tokens=12,completion_tokens=8,total_tokens=20))
            raw=('data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n\n').encode()
            self.send_response(200);self.send_header('Content-Type','text/event-stream')
            self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    profile=dict(model='gpt-6-astra',api_base=f'http://127.0.0.1:{server.server_port}/v1',
                 api_key_env='BBO_TEST_KEY',reasoning_effort='max')
    monkeypatch.setenv('BBO_TEST_KEY','fake-test-only-key')
    agent=build_agent(task,tmp_path,profile,docker_image=os.environ['BBO_DOCKER_TEST_IMAGE'],
                      executable=os.environ.get('BBO_CODEX_EXECUTABLE') or shutil.which('codex'))
    try:
        agent.setup(task.spec,seed=2,task_description=task.get_description())
        agent.replay(task.prefix)
        with audited_transport(tmp_path,profile):
            suggestion=agent.ask()
        assert suggestion.config==candidate
        assert len(calls)==1
        launch=json.loads((agent._state_dir/'container_launches.jsonl').read_text().splitlines()[-1])
        assert launch['cpus']==32 and launch['network']=='none'
        mounts={m['destination'] for m in launch['mounts']}
        assert mounts=={'/workspace','/state','/opt/native-agent','/opt/bbo-entry.py','/run/bbo-model.sock','/run/bbo-tool.sock'}
        assert all('fake-test-only-key' not in p.read_text(errors='replace') for p in tmp_path.rglob('*.json'))
    finally:
        server.shutdown();server.server_close();thread.join()


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get('BBO_DOCKER_TEST_IMAGE'), reason='Opt-in native CLI smoke')
def test_direct_native_session_has_no_tools(tmp_path, monkeypatch):
    from bbo.experiments.tasks import PaperTask
    from bbo.experiments.direct import DirectAgent
    task=PaperTask('bbob_f15_d10',suite='main')
    candidate=task.spec.search_space.defaults()
    seen=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append(request)
            assert not any(k in request for k in ('tools','tool_choice','parallel_tool_calls'))
            assert request['reasoning_effort']=='high'
            chunk=dict(id='direct-test',object='chat.completion.chunk',created=int(time.time()),model='deepseek-v4-flash',
                choices=[dict(index=0,delta=dict(role='assistant',content=json.dumps({'config':candidate})),finish_reason='stop')],
                usage=dict(prompt_tokens=10,completion_tokens=20,total_tokens=30))
            raw=('data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n\n').encode()
            self.send_response(200);self.send_header('Content-Type','text/event-stream')
            self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    profile=dict(model='deepseek-v4-flash',api_base=f'http://127.0.0.1:{server.server_port}/v1',
                 api_key_env='BBO_TEST_KEY',reasoning_effort='high',explicit_thinking_enabled=True)
    monkeypatch.setenv('BBO_TEST_KEY','fake-test-only-key')
    agent=DirectAgent(task,tmp_path,profile,executable=shutil.which('codex'))
    try:
        agent.setup(task.spec,seed=2)
        agent.replay(task.prefix)
        result=agent.ask()
        assert result.config==candidate and len(seen)==1
    finally:
        agent.close();server.shutdown();server.server_close();thread.join()
