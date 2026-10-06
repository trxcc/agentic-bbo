"""Host-owned guards for native BBO rounds, independent of model-generated text."""
from __future__ import annotations

import copy
import hashlib
import http.client
import json
import re
import threading
import time
import uuid

from . import runtime_reliability as reliability

VERSION = "native_round_guard_v1"
NATIVE_TOOLS = frozenset({"exec_command", "write_stdin"})
MAX_RESPONSE_CORRECTIONS = 2
CAPABILITIES = """Available native tools: exec_command and write_stdin only.
Use bash or Python to create and edit workspace scratch files. There is no
advertised apply_patch tool. Do not rely on apply_patch or jq shell commands;
use Python for JSON. Declared BBO tools use the supplied bbo_tool.py CLI.
Only an accepted host submission receipt completes the optimization round.
Do not finish with a promise to submit. Use the actual submit_candidate CLI.
"""


class InvalidNativeResponse(ValueError):
    def __init__(self, reason, answer):
        super().__init__(reason)
        self.answer = answer


class NativeBudgetExceeded(ValueError):
    pass


class AgentRoundPolicyError(RuntimeError):
    """A local round/summary policy failed, distinct from an upstream outage."""


def is_compaction(request):
    items = request.get("input", [])
    if not isinstance(items, list):
        return False
    for item in reversed(items):
        if isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "user":
            content = item.get("content", "")
            if isinstance(content, list):
                content = "\n".join(part.get("text", "") for part in content
                    if isinstance(part, dict) and part.get("type") in {"text", "input_text"})
            return isinstance(content, str) and bool(re.match(
                r"\A\s*You are (?:performing|doing) a CONTEXT CHECKPOINT COMPACTION\b", content, re.I))
    return False


class NativeRoundGuard:
    def __init__(self, native_tool_limit=64):
        # Zero disables the call-count cap while retaining submission validation.
        if type(native_tool_limit) is not int or native_tool_limit < 0:
            raise ValueError("native_tool_limit must be a non-negative integer (0 means unlimited)")
        self.limit = native_tool_limit
        self.used = 0
        self.receipt = None
        self.required_tool_choice = True
        self.failure = None
        self.lock = threading.Lock()

    def observe_tool_result(self, name, output):
        if name != "submit_candidate":
            return
        payload = json.loads(output) if isinstance(output, str) else output
        receipt = payload.get("result", {}) if isinstance(payload, dict) else {}
        if (isinstance(payload, dict) and payload.get("ok") is True and receipt.get("status") == "accepted"
                and receipt.get("terminal") is True and receipt.get("submission_id")):
            with self.lock:
                self.receipt = dict(receipt)

    def prepare(self, request):
        request = copy.deepcopy(request)
        compact = is_compaction(request)
        if self.limit > 0 and self.used >= self.limit and self.receipt is None:
            raise NativeBudgetExceeded(f"Native tool-call budget exhausted ({self.limit}); no candidate submitted.")
        # Replace unsupported edit-tool instructions without changing task data.
        def clean(text):
            return "\n".join(line for line in text.splitlines() if "apply_patch" not in line)
        request["instructions"] = clean(str(request.get("instructions") or "")) + "\n\n" + CAPABILITIES
        for item in request.get("input", []) if isinstance(request.get("input"), list) else []:
            if item.get("role") in {"system", "developer"}:
                if isinstance(item.get("content"), str):
                    item["content"] = clean(item["content"])
                elif isinstance(item.get("content"), list):
                    for part in item["content"]:
                        if "text" in part:
                            part["text"] = clean(part["text"])
        if not compact:
            filtered = []
            for tool in request.get("tools", []):
                if tool.get("type") == "function" and tool.get("name") in NATIVE_TOOLS:
                    filtered.append(tool)
                elif tool.get("type") == "namespace":
                    nested = [t for t in tool.get("tools", []) if t.get("name") in NATIVE_TOOLS]
                    if nested:
                        filtered.append({**tool, "tools": nested})
            request["tools"] = filtered
            if self.receipt is None:
                if not filtered:
                    raise ValueError("BBO round requires advertised native shell tools")
                request["tool_choice"] = "required" if self.required_tool_choice else "auto"
                if self.limit > 0:
                    request["instructions"] += f"\nNative tool calls remaining this invocation: {self.limit-self.used}. Reserve a call to submit."
                else:
                    request["instructions"] += "\nThere is no tool-call count limit for this invocation. Submit one candidate when ready."
        if compact:
            from .summary_checkpoint import MARKER, DIRECTIVE
            request["instructions"] = request["instructions"].replace("\n\n" + CAPABILITIES, "") + "\n\n" + MARKER + "\n" + DIRECTIVE
            for item in reversed(request.get("input", [])):
                if item.get("type") == "message" and item.get("role") == "user":
                    if isinstance(item.get("content"), str):
                        item["content"] += "\n\n" + DIRECTIVE
                    elif isinstance(item.get("content"), list):
                        item["content"].append({"type": "input_text", "text": DIRECTIVE})
                    break
            request["tools"] = []
            request["tool_choice"] = "none"
        return request

    def validate(self, accumulator, *, compact):
        text = "".join(accumulator.text_parts).strip()
        if compact:
            lower = text.lower()
            useful = (len(text) >= 200 and not accumulator.tool_calls
                      and any(w in lower for w in ("task", "objective"))
                      and any(w in lower for w in ("history", "trial", "observation"))
                      and any(w in lower for w in ("submit", "candidate", "pending")))
            if not useful:
                raise InvalidNativeResponse("Compaction returned no substantive task/state handoff", text)
            if self.receipt is not None:
                accumulator.text_parts.append(
                    "\n\nHost-confirmed state: submission " + self.receipt["submission_id"] +
                    " has already been accepted. This round is complete. Do not run tools or resubmit. "
                    "Wait for the next optimization round and its host observation.")
            return
        if not accumulator.tool_calls:
            raise InvalidNativeResponse("No candidate was submitted; a text promise does not execute the CLI", text)
        if any(call["name"].split("__")[-1] not in NATIVE_TOOLS for call in accumulator.tool_calls.values()):
            raise InvalidNativeResponse("Only exec_command and write_stdin are available native tools", text)
        with self.lock:
            if self.limit > 0 and self.used + len(accumulator.tool_calls) > self.limit:
                raise NativeBudgetExceeded(f"Native tool-call budget exceeded ({self.limit})")
            self.used += len(accumulator.tool_calls)


def relay_guarded(handler, body):
    """Keep corrections in the same native request/container; execute no partial calls."""
    from . import codex_responses_compat as compat

    server, guard = handler.server, handler.server.round_guard
    original = json.loads(body)
    compact = is_compaction(original)
    response_id = "resp_" + uuid.uuid4().hex
    headers_sent, created, sequence = False, False, 0

    def audit(event, **fields):
        reliability.audit_event(server.audit_path, {"event": event, **fields})

    def write(event):
        nonlocal headers_sent, created, sequence
        if not headers_sent:
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.send_header("Cache-Control", "no-cache")
            handler.send_header("Connection", "close")
            handler.end_headers()
            headers_sent = True
        payload = compat._sse_payload(event)
        if payload.get("type") == "response.created":
            if created:
                return
            created = True
        payload["sequence_number"] = sequence
        sequence += 1
        handler.wfile.write(compat._format_response_event(payload))
        handler.wfile.flush()

    connection = None
    try:
        request = guard.prepare(original)
        if guard.receipt is not None and not compact:
            accumulator = compat.ChatCompletionAccumulator(request, dialect=server.dialect)
            accumulator.text_parts = ["Submission accepted. Round complete."]
            accumulator.finish_reason = "stop"
            for event in accumulator.events(response_id=response_id):
                write(event)
            audit("submission_acknowledged_by_host", submission_id=guard.receipt["submission_id"],
                  provider_request=False, provider_tokens=0)
            return
        corrections, failures = 0, 0
        while True:
            request_id = uuid.uuid4().hex
            translated = json.loads(compat.responses_to_sglang_chat_request(
                json.dumps(request).encode(), dialect=server.dialect))
            if server.reliable_runtime:
                translated.pop("max_tokens", None)
            audit("request_schema", request_id=request_id, model=translated.get("model"),
                  tool_names=[t["function"]["name"] for t in translated.get("tools", [])],
                  tools_sha256=hashlib.sha256(compat._json_bytes({"tools": translated.get("tools", [])})).hexdigest(),
                  max_tokens_omitted="max_tokens" not in translated,
                  tool_choice=translated.get("tool_choice"), compaction=compact,
                  correction_index=corrections, native_calls_used=guard.used)
            connection_cls = http.client.HTTPSConnection if server.upstream.scheme == "https" else http.client.HTTPConnection
            timeout = server.upstream_timeout_seconds
            if server.deadline is not None:
                timeout = min(timeout, server.deadline-time.monotonic())
                if timeout <= 0:
                    raise TimeoutError("Agent deadline exceeded")
            connection = connection_cls(server.upstream.hostname, server.upstream.port, timeout=timeout)
            headers = {k:v for k,v in handler.headers.items()
                       if k.lower() not in compat._HOP_BY_HOP_HEADERS and k.lower() != "host"}
            headers["Host"] = server.upstream.netloc
            base = server.upstream.path.rstrip("/")
            path = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
            try:
                retry_after = None
                if server.reliable_runtime:
                    reliability.pace_request()
                connection.request("POST", path, compat._json_bytes(translated), headers)
                response = connection.getresponse()
                audit("http_response", request_id=request_id, attempt=failures, status=response.status)
                if response.status == 400 and translated.get("tool_choice") == "required":
                    detail = response.read().decode(errors="replace").lower()
                    if "tool_choice" in detail and any(word in detail for word in ("unsupported", "not support", "invalid")):
                        # Some compatible providers lack the required selector.
                        # Semantic validation remains mandatory; only the wire hint changes.
                        guard.required_tool_choice = False
                        request["tool_choice"] = "auto"
                        audit("required_tool_choice_unsupported", request_id=request_id)
                        continue
                if response.status >= 400:
                    if response.status not in reliability.RETRYABLE_STATUSES:
                        raise ValueError(f"Non-retryable upstream HTTP {response.status}")
                    retry_after = response.getheader("Retry-After")
                    raise ConnectionError(f"Upstream HTTP {response.status}")
                retry_after = None
                if "text/event-stream" not in response.getheader("Content-Type", ""):
                    raise ConnectionError("Expected a complete upstream SSE response")
                for event in compat.chat_to_responses_sse(
                    compat._iter_sse_events(response), request=request, dialect=server.dialect,
                    require_done=True, progress_interval_seconds=15.0, response_id=response_id,
                    validate_final=lambda acc: guard.validate(acc, compact=compact),
                    audit=lambda event: reliability.audit_event(server.audit_path, {**event,"request_id":request_id}),
                ):
                    write(event)
                return
            except InvalidNativeResponse as exc:
                audit("native_response_rejected", request_id=request_id, reason=str(exc),
                      answer_chars=len(exc.answer), correction_index=corrections, compaction=compact)
                if corrections >= MAX_RESPONSE_CORRECTIONS:
                    raise
                corrections += 1
                instruction = (
                    "Provide the actual handoff summary now, not a promise. Include the task/objective, "
                    "observed history, current candidate/submission status, useful scratch files, and next action."
                    if compact else
                    "You ended with text without submitting. Call exec_command to execute the supplied "
                    "bbo_tool.py submit_candidate CLI with your chosen config or workspace JSON path. "
                    "If more evidence is essential, use the available tools. Do not merely say you will submit."
                )
                items = request.get("input", [])
                if isinstance(items, str):
                    items = [{"type":"message","role":"user","content":items}]
                request["input"] = [*items,
                    {"type":"message","role":"assistant","content":exc.answer[-4000:]},
                    {"type":"message","role":"user","content":instruction}]
            except (OSError, http.client.HTTPException) as exc:
                if failures >= (reliability.HTTP_RETRIES if server.reliable_runtime else 0):
                    raise
                delay = reliability.retry_delay(failures, retry_after)
                failures += 1
                audit("http_retry", request_id=request_id, attempt=failures, error_type=type(exc).__name__, delay_seconds=delay)
                if server.deadline is not None and time.monotonic()+delay >= server.deadline:
                    raise TimeoutError("Retry would exceed agent deadline")
                time.sleep(delay)
            finally:
                connection.close()
                connection = None
    except (BrokenPipeError, ConnectionResetError):
        pass
    except Exception as exc:
        kind = "native_budget_exhausted" if isinstance(exc, NativeBudgetExceeded) else "native_round_failed"
        guard.failure = {"kind": kind, "error_type": type(exc).__name__, "message": str(exc)}
        audit(kind, error_type=type(exc).__name__, message=str(exc))
        error = {"error":{"type":kind,"message":str(exc)}}
        server.terminal_response = (502, "application/json", compat._json_bytes(error))
        if not headers_sent:
            handler._send_terminal_response()
        # No response.completed is emitted: native history cannot accept a bad summary.
    finally:
        if connection is not None:
            connection.close()
        handler.close_connection = True
