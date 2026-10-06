"""Actual Codex CLI owns turns, MCP execution, sessions, and compaction.

The existing Responses adapter only translates wire formats. This workflow adds
an explicit model-visible tool allowlist and the original DeepSeek settings.
There is deliberately no Python loop over model messages or tool calls here.
"""
import copy
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .io import save as atomic_json, read as read_json, canonical, append
from .tasks import ASSETS as HERE
from .transport import apply_effort
from bbo.algorithms.agentic import codex_responses_compat as compat
from bbo.algorithms.agentic import runtime_reliability

DISABLED = ['apps','browser_use','computer_use','goals','image_generation',
            'multi_agent','plugins','plugin_sharing','remote_plugin','skill_search',
            'shell_tool','unified_exec','shell_snapshot','code_mode','code_mode_host',
            'in_app_browser','hooks','memories','view_image','tool_suggest',
            'sleep_tool','workspace_dependencies']

class NativeRuntime:
    def __init__(self, out, prompt_file, tool_specs, action, *, profile, executable='codex', compact_limit=900000):
        self.executable = executable
        self.out = Path(out)
        self.home = self.out/'codex_home'
        self.workspace = self.out/'workspace'
        self.home.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(exist_ok=True)
        assert tool_specs == [], "STRICT_NO_TOOLS permits no tools"
        self.specs = tool_specs
        self.action = action
        self.active_round = None
        self.counter = 0
        self.lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if body['method'] == 'tools/list':
                    result = {'tools':[dict(name=s['function']['name'], description=s['function']['description'],
                        inputSchema=s['function']['parameters']) for s in owner.specs]}
                elif body['method'] == 'tools/call':
                    with owner.lock:
                        try:
                            value = owner.action(body['params']['name'], body['params'].get('arguments',{}))
                            result = {'content':[{'type':'text','text':canonical(value)}], 'isError':False}
                        except Exception as exc:
                            result = {'content':[{'type':'text','text':canonical({'error':type(exc).__name__, 'message':str(exc)})}], 'isError':True}
                elif body['method'] in ('resources/list','resources/templates/list'):
                    result = {'resources':[]} if body['method']=='resources/list' else {'resourceTemplates':[]}
                else:
                    result = {}
                data = canonical(result).encode()
                self.send_response(200);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)

        self.server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
        self.original_converter = compat.responses_to_sglang_chat_request
        self.original_accumulator = compat.ChatCompletionAccumulator
        allowed = {s['function']['name'] for s in tool_specs}

        class ExactNamesAccumulator(self.original_accumulator):
            def events(acc, **kwargs):
                if acc.tool_calls:
                    raise ValueError("STRICT_NO_TOOLS rejects all provider tool calls, including unadvertised native tools")
                # MCP namespaces contain '__'; undo only the actual namespace
                # separator using the original Codex schema, never string guesses.
                mapping = {}
                for item in acc.request.get('tools',[]):
                    if item.get('type')=='namespace':
                        for nested in item.get('tools',[]):
                            qualified=item['name']+'.'+nested['name']
                            mapping[qualified.replace('.','__')]=qualified
                    elif item.get('type')=='function':
                        mapping[item['name'].replace('.','__')]=item['name']
                for call in acc.tool_calls.values():
                    if call['name'] not in mapping:
                        raise ValueError('Provider returned an unregistered native tool name')
                    call['name']=mapping[call['name']]
                original_dialect=acc.dialect
                acc.dialect='sglang'  # Names are already decoded exactly above.
                try: return super().events(**kwargs)
                finally: acc.dialect=original_dialect

        compat.ChatCompletionAccumulator = ExactNamesAccumulator

        def translate(body, *, dialect='deepseek'):
            original = json.loads(body)
            request = json.loads(self.original_converter(body, dialect=dialect))
            # Restrict model visibility; Codex still owns the actual MCP call loop.
            offered = request.get('tools',[])
            def permitted(name):
                return any(name in (f'mcp__bbo__{tool}', f'functions__mcp__bbo__{tool}') for tool in allowed)
            selected = [tool for tool in offered if permitted(tool['function']['name'])]
            if selected:
                request['tools'] = selected
                request['tool_choice'] = 'auto'
                request['parallel_tool_calls'] = False
            else:
                for key in ('tools','tool_choice','parallel_tool_calls'):
                    request.pop(key,None)
            request.pop('max_tokens',None)
            request = apply_effort(request, profile)
            for message in request['messages']:
                for call in message.get('tool_calls',[]):
                    call['function']['name']=call['function']['name'].replace('.','__')
            self.counter += 1
            target = self.active_round or self.out/'startup'
            target.mkdir(parents=True,exist_ok=True)
            atomic_json(target/f'provider_request_{self.counter:04d}.json',request)
            atomic_json(target/f'codex_request_{self.counter:04d}.json',original)
            atomic_json(target/f'tool_visibility_{self.counter:04d}.json',dict(
                native=[x['function']['name'] for x in offered],
                visible=[x['function']['name'] for x in selected],
                removed=[x['function']['name'] for x in offered if x not in selected]))
            # Full-tool startup must actually contain the registered MCP tools.
            if allowed and offered and not selected and len(original.get('input',[])) < 10:
                raise RuntimeError('Codex MCP tools absent or unexpected naming; refusing paid request')
            return canonical(request).encode()

        compat.responses_to_sglang_chat_request = translate
        self.original_stream = compat.chat_to_responses_sse
        def audited_stream(events, *, request, **kwargs):
            number = self.counter
            previous = kwargs.pop('audit', None)
            def record(event):
                if previous: previous(event)
                if event.get('event') == 'upstream_stream_end':
                    append(self.out/'provider_usage.jsonl',dict(request_number=number,timestamp=time.time(),**event))
            yield from self.original_stream(events,request=request,audit=record,**kwargs)
        compat.chat_to_responses_sse = audited_stream
        self.original_http_retries = runtime_reliability.HTTP_RETRIES
        runtime_reliability.HTTP_RETRIES = 3
        self.proxy = compat.SGLangResponsesCompatibilityProxy(profile['api_base'],
            dialect='chat_completions', reliable_runtime=True, upstream_timeout_seconds=1200,
            audit_path=self.out/'transport_audit.jsonl')
        self.proxy.start()
        catalog = copy.deepcopy(read_json(HERE/'direct_model_catalog.json'))
        selected = catalog['models'][0]
        selected['slug'] = selected['display_name'] = profile['model']
        selected['default_reasoning_level'] = profile['reasoning_effort']
        if profile['reasoning_effort'] not in {x['effort'] for x in selected['supported_reasoning_levels']}:
            selected['supported_reasoning_levels'].append(dict(effort=profile['reasoning_effort'], description='Explicit experiment setting'))
        atomic_json(self.home/'model_catalog.json', catalog)
        cfg = [f'model = {json.dumps(profile["model"])}', 'model_provider = "benchmark_deepseek"',
            'approval_policy = "never"', 'sandbox_mode = "read-only"', 'web_search = "disabled"',
            'project_doc_max_bytes = 0', 'project_root_markers = []',
            'model_context_window = 1000000', f'model_auto_compact_token_limit = {compact_limit}',
            'model_reasoning_effort = '+json.dumps(profile['reasoning_effort']),
            f'model_catalog_json = {json.dumps(str(self.home/"model_catalog.json"))}',
            f'model_instructions_file = {json.dumps(str(prompt_file))}',
            '[features]', *[f'{name} = false' for name in DISABLED],
            'skip_host_skill_discovery = true',
            '[model_providers.benchmark_deepseek]', 'name = "DeepSeek via native Codex Responses bridge"',
            f'base_url = {json.dumps(self.proxy.base_url)}', 'wire_api = "responses"',
            'env_key = '+json.dumps(profile['api_key_env']), 'request_max_retries = 0', 'stream_max_retries = 0',
            'stream_idle_timeout_ms = 1200000']
        if tool_specs:
            cfg += ['[mcp_servers.bbo]', 'command = "/usr/bin/python3"',
                'args = '+json.dumps(['-P',str(HERE/'mcp_stdio.py'),f'http://127.0.0.1:{self.server.server_port}']),
                'startup_timeout_sec = 30', 'tool_timeout_sec = 1200', 'required = true']
            for spec in tool_specs:
                cfg += [f"[mcp_servers.bbo.tools.{spec['function']['name']}]", 'approval_mode = "approve"']
        # Disable the user's host skill catalog without changing user settings.
        for skill in sorted((Path.home()/'.agents/skills').glob('*/SKILL.md')):
            cfg += ['[[skills.config]]',f'path = {json.dumps(str(skill.parent))}','enabled = false',
                    '[[skills.config]]',f'path = {json.dumps(str(skill))}','enabled = false']
        for name in ['imagegen','openai-docs','plugin-creator','skill-creator','skill-installer']:
            folder=self.home/'skills/.system'/name
            for path in (folder,folder/'SKILL.md'):
                cfg += ['[[skills.config]]',f'path = {json.dumps(str(path))}','enabled = false']
        (self.home/'config.toml').write_text('\n'.join(cfg)+'\n')
        self.env = {k:v for k,v in os.environ.items() if k in {'PATH','LANG','LC_ALL','HOME','TMPDIR',profile['api_key_env']}}
        self.env['CODEX_HOME'] = str(self.home)
        # Real provider credentials are inherited only by the Codex process.
        if profile['api_key_env'] != 'OPENAI_API_KEY':
            self.env.pop('OPENAI_API_KEY',None)
        for key in ('CODEX_THREAD_ID','CODEX_SESSION_ID'):self.env.pop(key,None)
        self.session_path = self.out/'native_session.json'

    def run(self, message, round_dir):
        self.active_round = Path(round_dir)
        self.active_round.mkdir(parents=True,exist_ok=True)
        final_path = self.active_round/'final.txt'
        marker = self.active_round/'native_invocation.json'
        if marker.exists():
            raise RuntimeError('Unresolved native turn exists: inspect journal before retry')
        command = [self.executable,'exec','--skip-git-repo-check','--json','--color','never',
                   '--output-last-message',str(final_path)]
        if self.session_path.exists():
            command = [self.executable,'exec','resume','--skip-git-repo-check','--json',
                       '--output-last-message',str(final_path),read_json(self.session_path)['thread_id'],'-']
        else:
            command += ['-']
        atomic_json(marker,dict(command=command,cwd=str(self.workspace),started_at=time.time(),
            executable_version=subprocess.check_output([self.executable,'--version'],text=True).strip(),
            existing_session=read_json(self.session_path) if self.session_path.exists() else None))
        (self.active_round/'input.txt').write_text(message)
        with (self.active_round/'events.jsonl').open('w') as events, (self.active_round/'stderr.log').open('w') as stderr:
            process = subprocess.run(command,input=message,text=True,cwd=self.workspace,env=self.env,
                                     stdout=events,stderr=stderr)
        thread_ids=[]
        for line in (self.active_round/'events.jsonl').read_text().splitlines():
            try: event=json.loads(line)
            except json.JSONDecodeError: continue
            if event.get('type')=='thread.started':thread_ids.append(event['thread_id'])
        if thread_ids:
            thread_id=thread_ids[-1]
            if self.session_path.exists():
                assert read_json(self.session_path)['thread_id']==thread_id,'Codex failed to resume the same session'
            atomic_json(self.session_path,dict(engine='codex',thread_id=thread_id))
        atomic_json(self.active_round/'native_exit.json',dict(returncode=process.returncode,thread_ids=thread_ids))
        if process.returncode or not final_path.exists() or not self.session_path.exists():
            raise RuntimeError(f'Native Codex failed ({process.returncode}); see {self.active_round}')
        return final_path.read_text()

    def close(self):
        self.proxy.close()
        self.server.shutdown();self.server.server_close()
        compat.responses_to_sglang_chat_request = self.original_converter
        compat.ChatCompletionAccumulator = self.original_accumulator
        runtime_reliability.HTTP_RETRIES = self.original_http_retries
        compat.chat_to_responses_sse = self.original_stream
