"""End-to-end HTTP tests of native BBO semantics, without paid model calls."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import json
import threading
from urllib.parse import urlsplit

import pytest

from bbo.algorithms.agentic.codex_responses_compat import SGLangResponsesCompatibilityProxy
from bbo.algorithms.agentic.native_round_guard import NativeRoundGuard, is_compaction
from bbo.algorithms.agentic import runtime_reliability
from bbo.algorithms.agentic.submission_audit import unique_submission_receipts


def stream(text=None, *, tools=False, done=True):
    delta = {"content": text} if text is not None else {}
    if tools:
        delta["tool_calls"] = [{"index":0,"id":"call_1","function":{
            "name":"exec_command","arguments":'{"cmd":"python3 bbo_tool.py submit_candidate"}'}}]
    payload = {"choices":[{"delta":delta,"finish_reason":"tool_calls" if tools else "stop"}],
               "usage":{"prompt_tokens":30,"completion_tokens":10}}
    return b"data: "+json.dumps(payload).encode()+b"\n\n"+(b"data: [DONE]\n\n" if done else b"")


def request(*, compact=False):
    return {"model":"test", "stream":True,
            "instructions":"Keep task boundaries.\nUse the apply_patch tool to edit files.",
            "input":[{"type":"message","role":"user","content":
                      "You are doing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary."
                      if compact else "Choose one next candidate and submit it."}],
            "tools":[] if compact else [
                {"type":"function","name":"exec_command","parameters":{"type":"object"}},
                {"type":"function","name":"write_stdin","parameters":{"type":"object"}},
                {"type":"function","name":"apply_patch","parameters":{"type":"object"}}]}


@contextmanager
def proxy_fixture(tmp_path, monkeypatch, replies, *, limit=64):
    recorded = []
    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            recorded.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            answer = replies[min(len(recorded)-1, len(replies)-1)]
            status, body = answer if isinstance(answer, tuple) else (200, answer)
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length",str(len(body)))
            self.send_header("Retry-After","0")
            self.end_headers()
            self.wfile.write(body)

    monkeypatch.setattr(runtime_reliability, "pace_request", lambda: None)
    monkeypatch.setattr(runtime_reliability, "retry_delay", lambda *_: 0)
    upstream = ThreadingHTTPServer(("127.0.0.1",0), Upstream)
    thread = threading.Thread(target=upstream.serve_forever,daemon=True)
    thread.start()
    guard = NativeRoundGuard(limit)
    audit = tmp_path / "audit.jsonl"
    proxy = SGLangResponsesCompatibilityProxy(f"http://127.0.0.1:{upstream.server_port}/v1",
        dialect="chat_completions", reliable_runtime=True, audit_path=audit, round_guard=guard)
    proxy.start()
    url = urlsplit(proxy.base_url)

    def send(payload):
        connection = http.client.HTTPConnection(url.hostname,url.port,timeout=5)
        connection.request("POST","/v1/responses",json.dumps(payload),{"Content-Type":"application/json"})
        response = connection.getresponse()
        result = response.status,response.read()
        connection.close()
        return result

    try:
        yield guard, recorded, send, audit
    finally:
        proxy.close()
        upstream.shutdown()
        upstream.server_close()
        thread.join(2)


def accept(guard):
    guard.observe_tool_result("submit_candidate", json.dumps({"ok":True,"result":{
        "status":"accepted","terminal":True,"submission_id":"s_123"}}))


SUMMARY = ("Task: minimize value over ten bounded parameters.\n"
           "History: twenty observations are available in history.jsonl; no new trial is complete.\n"
           "Candidate: no candidate has been submitted yet. The next action is to read scratch/candidate.json, "
           "validate the bounds, and call submit_candidate with that path.\n")


def test_missing_submit_is_corrected_without_ending_native_turn(tmp_path, monkeypatch):
    with proxy_fixture(tmp_path,monkeypatch,[stream("Submitting it."),stream(tools=True)]) as (guard,seen,send,audit):
        status,body=send(request())
        assert status==200 and body.count(b"event: response.completed\n")==1
        assert body.count(b"event: response.created\n")==1
        assert b"Submitting it." not in body
        assert len(seen)==2 and guard.used==1
        assert all(r["tool_choice"]=="required" for r in seen)
        assert {t['function']['name'] for t in seen[0]['tools']}=={'exec_command','write_stdin'}
        assert 'Use the apply_patch tool' not in seen[0]['messages'][0]['content']
        assert seen[1]['messages'][-2]['content']=='Submitting it.'
        events=[json.loads(l) for l in audit.read_text().splitlines()]
        assert sum(e['event']=='upstream_stream_end' for e in events)==2
        assert sum(e['event']=='native_response_rejected' for e in events)==1


def test_host_receipt_ends_round_without_another_provider_request(tmp_path,monkeypatch):
    with proxy_fixture(tmp_path,monkeypatch,[stream(tools=True)],limit=1) as (guard,seen,send,audit):
        assert send(request())[0]==200
        accept(guard)
        status,body=send(request())
        assert status==200 and b'Submission accepted. Round complete.' in body
        assert len(seen)==1
        events=[json.loads(l) for l in audit.read_text().splitlines()]
        assert any(e['event']=='submission_acknowledged_by_host' and e['provider_tokens']==0 for e in events)


def test_plain_text_claim_cannot_spoof_authoritative_receipt(tmp_path,monkeypatch):
    with proxy_fixture(tmp_path,monkeypatch,[stream('Submission accepted.'),stream(tools=True)]) as (guard,seen,send,_):
        send(request())
        assert guard.receipt is None and len(seen)==2


def test_bad_compaction_is_retried_before_any_completed_summary(tmp_path,monkeypatch):
    with proxy_fixture(tmp_path,monkeypatch,[stream("I'll write a concise handoff summary."),stream(SUMMARY)]) as (_,seen,send,_):
        status,body=send(request(compact=True))
        assert status==200 and body.count(b'event: response.completed\n')==1
        assert b"I'll write" not in body
        assert len(seen)==2 and all('tool_choice' not in r for r in seen)
        assert 'actual handoff summary' in seen[-1]['messages'][-1]['content']


def test_repeated_bad_compaction_fails_closed_and_caps_retries(tmp_path,monkeypatch):
    with proxy_fixture(tmp_path,monkeypatch,[stream('I will summarize.')]) as (_,seen,send,_):
        _,body=send(request(compact=True))
        assert b'event: response.completed\n' not in body and len(seen)==3
        assert send(request(compact=True))[0]==502
        assert len(seen)==3


def test_compaction_after_submission_preserves_terminal_state(tmp_path,monkeypatch):
    with proxy_fixture(tmp_path,monkeypatch,[stream(SUMMARY)]) as (guard,seen,send,_):
        accept(guard)
        _,body=send(request(compact=True))
        assert b'Host-confirmed state' in body and b's_123' in body
        _,body=send(request())
        assert b'Round complete' in body and len(seen)==1


def test_incomplete_stream_retries_without_releasing_partial_tool_calls(tmp_path,monkeypatch):
    with proxy_fixture(tmp_path,monkeypatch,[stream(tools=True,done=False),stream(tools=True)]) as (guard,seen,send,_):
        _,body=send(request())
        assert len(seen)==2 and guard.used==1
        assert body.count(b'event: response.output_item.done\n')==1
        assert body.count(b'event: response.completed\n')==1


def test_http_retry_and_native_budget_cannot_restart_upstream_allowance(tmp_path,monkeypatch):
    with proxy_fixture(tmp_path,monkeypatch,[(429,b'busy'),stream(tools=True)],limit=1) as (_,seen,send,_):
        assert send(request())[0]==200 and len(seen)==2
        status,body=send(request())
        assert status==502 and b'budget exhausted' in body and len(seen)==2
        assert send(request())[0]==502 and len(seen)==2


def test_required_selector_fallback_retains_semantic_submission_validation(tmp_path,monkeypatch):
    with proxy_fixture(tmp_path,monkeypatch,[(400,b'{"error":"tool_choice required is not supported"}'),
                                           stream('Submitting it.'),stream(tools=True)]) as (guard,seen,send,_):
        assert send(request())[0]==200
        assert len(seen)==3 and not guard.required_tool_choice
        assert seen[0]['tool_choice']=='required'
        assert seen[1]['tool_choice']==seen[2]['tool_choice']=='auto'
        assert guard.used==1 and guard.receipt is None


@pytest.mark.parametrize('status,expected_requests',[(400,1),(502,3)])
def test_guarded_upstream_failure_budget_survives_native_retries(tmp_path,monkeypatch,status,expected_requests):
    monkeypatch.setattr(runtime_reliability,'HTTP_RETRIES',2)
    with proxy_fixture(tmp_path,monkeypatch,[(status,b'upstream failure')]) as (_,seen,send,_):
        assert send(request())[0]==502
        assert len(seen)==expected_requests
        assert send(request())[0]==502
        assert len(seen)==expected_requests


def test_receipt_deduplication_rejects_identity_conflicts():
    def call(candidate):
        return {'tool_name':'submit_candidate','success':True,'result_preview':json.dumps({
            'ok':True,'result':{'status':'accepted','submission_id':'s_1','candidate_id':candidate,'context_version':'v1'}})}
    assert len(unique_submission_receipts([call('c1'),call('c1')]))==1
    with pytest.raises(ValueError,match='Conflicting'):
        unique_submission_receipts([call('c1'),call('c2')])
