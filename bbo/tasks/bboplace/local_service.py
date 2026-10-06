"""Host-only HTTP service for frozen, geometry-repaired placement tasks."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .repair_backend import PROTOCOL, RepairService

BUNDLE_ROOT = Path(__file__).resolve().parent / "assets" / "repair_bundles"


class BBOPlaceLocalBridge:
    def __init__(self, *, repair_bundles: Path = BUNDLE_ROOT, audit_log: Path | None = None):
        self.repair_service = RepairService(repair_bundles)
        self.audit_log = audit_log
        self._lock = threading.Lock()

    def evaluate_payload(self, payload: dict) -> dict:
        with self._lock:
            started = time.perf_counter()
            result = self.repair_service.evaluate_payload(payload)
            if self.audit_log is not None:
                self.audit_log.parent.mkdir(parents=True, exist_ok=True)
                with self.audit_log.open("a") as stream:
                    stream.write(json.dumps(dict(timestamp=time.time(), request=payload,
                        response=result, objective_evaluations=1,
                        elapsed_seconds=time.perf_counter() - started), allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            return result


class _Handler(BaseHTTPRequestHandler):
    bridge: BBOPlaceLocalBridge

    def do_GET(self):
        if self.path.rstrip("/") != "/health":
            return self._send_json(404, {"status": "error", "message": "Unknown route"})
        self._send_json(200, dict(status="ok", protocol=PROTOCOL,
            bundles={f"{key[0]}__s{key[1]}": ev.sha256
                     for key, ev in self.bridge.repair_service.evaluators.items()}))

    def do_POST(self):
        if self.path.rstrip("/") != "/evaluate":
            return self._send_json(404, {"status": "error", "message": "Unknown route"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1_048_576:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Request must be a JSON object")
            response = self.bridge.evaluate_payload(payload)
        except (ValueError, TypeError, KeyError) as exc:
            return self._send_json(400, dict(status="error", message=str(exc)))
        self._send_json(200, response)

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, code, payload):
        raw = json.dumps(payload, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8070)
    parser.add_argument("--repair-bundles", type=Path, default=BUNDLE_ROOT)
    parser.add_argument("--audit-log", type=Path)
    args = parser.parse_args(argv)
    bridge = BBOPlaceLocalBridge(repair_bundles=args.repair_bundles, audit_log=args.audit_log)
    handler = type("PlacementHandler", (_Handler,), {"bridge": bridge})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Geometry repair service listening on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
