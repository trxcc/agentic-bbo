"""Local Chat adapter to the same gateway's validated native Gemini endpoint.

The optimizer/harness keeps its existing tool protocol. Native HIGH and complete
Gemini thought signatures are recorded and replayed without inventing signatures.
"""

from copy import deepcopy
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import http.client
import json
import threading
import time
import uuid
from urllib.parse import urlsplit


def native_request(chat,call_cache,text_cache):
    system=[];contents=[];call_names={}
    def add(role,parts):
        if not parts:return
        if contents and contents[-1]['role']==role:contents[-1]['parts'].extend(parts)
        else:contents.append(dict(role=role,parts=parts))
    for message in chat['messages']:
        role=message['role'];text=message.get('content') or ''
        if not isinstance(text,str):raise ValueError('Only the existing text-only frontier transport is supported')
        if role in ('system','developer'):
            if text:system.append(text)
        elif role=='user':add('user',[{'text':text}] if text else [])
        elif role=='assistant':
            calls=message.get('tool_calls') or []
            if calls:
                for call in calls:call_names[call['id']]=call['function']['name']
                cached=[call_cache.get(call['id']) for call in calls]
                if any(item is None for item in cached):raise ValueError('Missing native Gemini thought-signature replay record')
                parts=deepcopy(cached[0]['parts'])
                expected=[p['functionCall']['name'] for p in parts if 'functionCall' in p]
                assert expected==[c['function']['name'] for c in calls]
                add('model',parts)
            elif text:
                add('model',deepcopy(text_cache.get(text,[{'text':text}])))
        elif role=='tool':
            identifier=message['tool_call_id']
            if identifier not in call_names:raise ValueError('Tool result has no preceding function call')
            function=dict(name=call_names[identifier],response={'result':text})
            native_id=call_cache.get(identifier,{}).get('native_id')
            if native_id:function['id']=native_id
            add('user',[{'functionResponse':function}])
        else:raise ValueError('Unexpected chat role: '+role)
    result=dict(contents=contents,generationConfig={'thinkingConfig':{'thinkingLevel':chat['reasoning_effort'].upper()}})
    if system:result['systemInstruction']={'parts':[{'text':'\n\n'.join(system)}]}
    if chat.get('temperature') is not None:result['generationConfig']['temperature']=chat['temperature']
    if chat.get('top_p') is not None:result['generationConfig']['topP']=chat['top_p']
    declarations=[]
    for item in chat.get('tools') or []:
        function=item['function']
        declarations.append(dict(name=function['name'],description=function.get('description',''),
                                 parametersJsonSchema=function.get('parameters',{'type':'object','properties':{}})))
    if declarations:
        result['tools']=[{'functionDeclarations':declarations}]
        choice=chat.get('tool_choice','auto')
        mode={'auto':'AUTO','required':'ANY','none':'NONE'}.get(choice) if isinstance(choice,str) else None
        if not mode:raise ValueError('Unsupported tool-choice override')
        result['toolConfig']={'functionCallingConfig':{'mode':mode}}
    return result


def chat_response(native,profile,call_cache,text_cache):
    version=native.get('modelVersion')
    if version not in profile['approved_response_model_ids']:raise ValueError('Unexpected native Gemini model: '+str(version))
    candidates=native.get('candidates') or []
    if len(candidates)!=1:raise ValueError('Expected exactly one native Gemini candidate')
    candidate=candidates[0];parts=candidate.get('content',{}).get('parts',[])
    text=''.join(p.get('text','') for p in parts if not p.get('thought'))
    reasoning=''.join(p.get('text','') for p in parts if p.get('thought'))
    calls=[]
    for part in parts:
        if 'functionCall' not in part:continue
        call=part['functionCall'];identifier='call_'+uuid.uuid4().hex
        calls.append(dict(id=identifier,type='function',function=dict(name=call['name'],arguments=json.dumps(call.get('args',{}),separators=(',',':')))))
        call_cache[identifier]=dict(parts=deepcopy(parts),native_id=call.get('id'))
    if text:text_cache[text]=deepcopy(parts)
    message=dict(role='assistant',content=text or None)
    if calls:message['tool_calls']=calls
    if reasoning:message['reasoning_content']=reasoning
    counters=native.get('usageMetadata') or {}
    if 'promptTokenCount' not in counters:raise ValueError('Native Gemini token usage missing')
    thought=int(counters.get('thoughtsTokenCount',0));visible=int(counters.get('candidatesTokenCount',0))
    usage=dict(prompt_tokens=int(counters['promptTokenCount']),completion_tokens=visible+thought,
               total_tokens=int(counters['promptTokenCount'])+visible+thought,
               prompt_tokens_details={'cached_tokens':int(counters.get('cachedContentTokenCount',0))},
               completion_tokens_details={'reasoning_tokens':thought})
    finish='tool_calls' if calls else 'length' if candidate.get('finishReason')=='MAX_TOKENS' else 'stop'
    return dict(id='chatcmpl_'+uuid.uuid4().hex,object='chat.completion',created=int(time.time()),model=version,
                choices=[dict(index=0,message=message,finish_reason=finish)],usage=usage)


class Gateway:
    def __init__(self,profile,key,out,allow_invalid=False):
        self.profile=profile;self.key=key;self.out=out;self.allow_invalid=allow_invalid
        self.call_cache={};self.text_cache={};self.lock=threading.Lock()
        from .io import save,append,read
        self.save=save;self.append=append
        cache_path=out/'native_signature_cache.json'
        if cache_path.exists():
            cache=read(cache_path)
            self.call_cache=cache['calls'];self.text_cache=cache['text']
        gateway=self
        class Handler(BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'
            def do_POST(self):
                raw=self.rfile.read(int(self.headers.get('Content-Length','0')))
                identifier=uuid.uuid4().hex;connection=None
                try:
                    chat=json.loads(raw)
                    assert chat['model']==profile['model']
                    if not gateway.allow_invalid:assert chat['reasoning_effort']=='high'
                    with gateway.lock:native=native_request(chat,gateway.call_cache,gateway.text_cache)
                    gateway.save(out/'native_wire'/(identifier+'.request.json'),native)
                    gateway.append(out/'native_requests.jsonl',dict(request_id=identifier,timestamp=time.time(),model=profile['model'],
                        thinking_level=native['generationConfig']['thinkingConfig']['thinkingLevel'],
                        thought_signature_parts=sum('thoughtSignature' in p for c in native['contents'] for p in c['parts'])))
                    upstream=urlsplit(profile['api_base'])
                    connection=http.client.HTTPSConnection(upstream.hostname,timeout=1200)
                    connection.request('POST','/v1beta/models/'+profile['model']+':generateContent',json.dumps(native).encode(),
                        {'Authorization':'Bearer '+key,'x-goog-api-key':key,'Content-Type':'application/json'})
                    response=connection.getresponse();payload=response.read().replace(key.encode(),b'[REDACTED]')
                    (out/'native_wire'/(identifier+'.response.json')).write_bytes(payload)
                    if response.status>=400:
                        self.send_response(response.status);self.send_header('Content-Type','application/json')
                        self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload);return
                    original=json.loads(payload)
                    with gateway.lock:
                        result=chat_response(original,profile,gateway.call_cache,gateway.text_cache)
                        gateway.save(out/'native_signature_cache.json',dict(calls=gateway.call_cache,text=gateway.text_cache))
                    gateway.append(out/'native_usage.jsonl',dict(request_id=identifier,model=original.get('modelVersion'),usageMetadata=original.get('usageMetadata')))
                    if chat.get('stream'):
                        choice=result['choices'][0];delta=deepcopy(choice['message'])
                        for i,call in enumerate(delta.get('tool_calls',[])):call['index']=i
                        chunk={k:result[k] for k in ('id','created','model')}
                        chunk.update(object='chat.completion.chunk',choices=[dict(index=0,delta=delta,finish_reason=choice['finish_reason'])],usage=result['usage'])
                        body=('data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n\n').encode()
                        content_type='text/event-stream'
                    else:body=json.dumps(result).encode();content_type='application/json'
                    self.send_response(200);self.send_header('Content-Type',content_type)
                    self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
                except (BrokenPipeError,ConnectionResetError):pass
                except Exception as exc:
                    gateway.save(out/'native_wire'/(identifier+'.adapter_error.json'),dict(type=type(exc).__name__,message=str(exc).replace(key,'[REDACTED]')))
                    body=json.dumps({'error':{'message':str(exc).replace(key,'[REDACTED]'),'type':'native_gemini_adapter_error'}}).encode()
                    self.send_response(502);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)))
                    self.end_headers();self.wfile.write(body)
                finally:
                    if connection:connection.close()
            def log_message(self,*args):pass
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler);self.server.daemon_threads=True
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
    @property
    def base_url(self):return 'http://127.0.0.1:'+str(self.server.server_port)+'/v1'
    def start(self):self.thread.start()
    def close(self):self.server.shutdown();self.server.server_close();self.thread.join(timeout=5)
