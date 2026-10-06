"""Opt-in transport controls for audited, serial native-harness campaigns."""
from __future__ import annotations

from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import random
import threading
import time
from typing import Any


RUNTIME_VERSION = "reliable_native_v1"
HTTP_RETRIES = 5
MIN_REQUEST_INTERVAL = 2.0
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
_request_lock = threading.Lock()
_last_request = 0.0
_audit_lock = threading.Lock()


class AgentTransportError(RuntimeError):
    """Stop a run after transport recovery is exhausted; never correct a candidate."""


def retry_delay(attempt: int, retry_after: str | None) -> float:
    """Honor Retry-After; otherwise use bounded exponential backoff plus jitter."""
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                return max(0.0, parsedate_to_datetime(retry_after).timestamp() - time.time())
            except (ValueError, TypeError, OverflowError):
                pass
    return min(60.0, 5.0 * 2 ** attempt) + random.uniform(0.0, 1.0)


def pace_request() -> None:
    global _last_request
    with _request_lock:
        delay = MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request)
        if delay > 0:
            time.sleep(delay)
        _last_request = time.monotonic()


def audit_event(path: str | Path | None, event: dict[str, Any]) -> None:
    """Log metadata only: never headers, keys, request bodies or model output."""
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _audit_lock, target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": time.time(), **event}, sort_keys=True) + "\n")
