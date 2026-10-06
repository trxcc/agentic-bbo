"""Experimental network-disabled Codex container with narrow host gateways.

Only workspace files, agent-owned session state, executable files and two Unix
sockets cross the boundary. Evaluator state and provider secrets stay on host.
"""
from __future__ import annotations

import http.client
import json
import math
import os
from pathlib import Path
import shutil
import socketserver
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlsplit
import uuid


DEFAULT_ISOLATED_DOCKER_CPUS = 16.0


def validate_docker_cpus(value: float) -> float:
    """Validate the CPU quota before passing it to Docker."""
    if isinstance(value, bool):
        raise ValueError("docker_cpus must be finite and positive")
    cpus = float(value)
    if not math.isfinite(cpus) or cpus <= 0:
        raise ValueError("docker_cpus must be finite and positive")
    return cpus


class _UnixHTTPServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


class ModelGateway:
    """Accept only Responses POSTs and inject credentials for one fixed target."""

    def __init__(self, socket_path: Path, target: str, api_key: str, timeout: float):
        upstream = urlsplit(target)
        if upstream.scheme != "http" or upstream.hostname != "127.0.0.1":
            raise ValueError("The gateway requires a host loopback compatibility proxy")

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                if self.path != "/v1/responses":
                    self.send_error(403, "Only /v1/responses is available")
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    size = 0
                if self.headers.get("Transfer-Encoding") or not 0 < size <= 16 * 1024 * 1024:
                    self.send_error(413, "A bounded Content-Length is required")
                    return
                self.connection.settimeout(timeout)
                body = self.rfile.read(size)
                conn = http.client.HTTPConnection(upstream.hostname, upstream.port, timeout=timeout)
                started = False
                try:
                    # Never forward agent-selected routing or authorization headers.
                    conn.request("POST", "/v1/responses", body=body, headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    })
                    response = conn.getresponse()
                    self.send_response(response.status)
                    self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    started = True
                    while chunk := response.read1(65536):
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (OSError, http.client.HTTPException):
                    if not started:
                        self.send_error(502, "Model gateway unavailable")
                finally:
                    conn.close()
                    self.close_connection = True

        self.server = _UnixHTTPServer(str(socket_path), Handler)
        socket_path.chmod(0o600)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def native_codex_binary(executable: str) -> Path:
    """Resolve an ELF binary without exposing the host npm tree to the agent."""
    path = Path(executable).resolve()
    if path.suffix == ".js":
        candidates = list(path.parent.parent.glob("**/vendor/*/bin/codex"))
        if len(candidates) != 1:
            raise ValueError("Set agent executable to the native Codex ELF binary")
        path = candidates[0]
    with path.open("rb") as handle:
        if handle.read(4) != b"\x7fELF":
            raise ValueError("isolated_docker requires a native Codex ELF executable")
    return path


class IsolatedDockerRuntime:
    """Own one ephemeral container and model socket; retain only Codex sessions."""

    model_base_url = "http://127.0.0.1:38080/v1"

    def __init__(self, *, state_dir: Path, config_path: Path, target: str,
                 api_key: str, timeout: float):
        self.name = "bbo-agent-" + uuid.uuid4().hex
        # Never mount state_dir: it also contains authoritative submission receipts.
        self.agent_state = state_dir / "isolated_codex"
        if self.agent_state.is_symlink():
            raise ValueError("Agent session directory must not be a symlink")
        self.agent_state.mkdir(parents=True, exist_ok=True)
        (self.agent_state / "home").mkdir(exist_ok=True)
        temporary_config = self.agent_state / ("config-" + uuid.uuid4().hex + ".tmp")
        temporary_config.write_bytes(config_path.read_bytes())
        temporary_config.replace(self.agent_state / "config.toml")
        self.audit_path = state_dir / "container_launches.jsonl"
        self.temporary = tempfile.TemporaryDirectory(prefix="bbo-model-")
        self.socket = Path(self.temporary.name) / "model.sock"
        try:
            self.gateway = ModelGateway(self.socket, target, api_key, timeout)
        except Exception:
            self.temporary.cleanup()
            raise

    def command(self, command: list[str], *, workspace: Path, executable: Path,
                image: str, env: dict[str, str], tool_socket: Path | None,
                cpus: float = DEFAULT_ISOLATED_DOCKER_CPUS) -> list[str]:
        cpus = validate_docker_cpus(cpus)
        docker = shutil.which("docker")
        if not docker:
            raise RuntimeError("Docker unavailable; refusing host fallback")
        if self.agent_state.parent.resolve().is_relative_to(workspace.resolve()):
            raise ValueError("Host state must be outside the mounted workspace")
        mounts = [
            (workspace.resolve(), "/workspace", False),
            (self.agent_state.resolve(), "/state", False),
            (executable.resolve(), "/opt/native-agent", True),
            (Path(__file__).with_name("isolated_docker_entry.py").resolve(), "/opt/bbo-entry.py", True),
            (self.socket, "/run/bbo-model.sock", True),
        ]
        if tool_socket is not None:
            mounts.append((tool_socket, "/run/bbo-tool.sock", True))
        result = [docker, "run", "--rm", "--init", "--name", self.name,
                  "--network", "none", "--read-only", "--user", f"{os.getuid()}:{os.getgid()}",
                  "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                  "--pids-limit", "256", "--memory", "4g", "--cpus", format(cpus, "g"),
                  "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=512m", "--workdir", "/workspace"]
        for source, destination, readonly in mounts:
            result += ["--mount", f"type=bind,src={source},dst={destination}" + (",readonly" if readonly else "")]
        for key, value in sorted(env.items()):
            result += ["--env", f"{key}={value}"]
        with self.audit_path.open("a") as handle:
            handle.write(json.dumps({"container": self.name, "image": image,
                "network": "none", "cpus": cpus, "memory": "4g", "pids_limit": 256,
                "mounts": [{"source": str(s), "destination": d,
                "readonly": ro} for s, d, ro in mounts]}) + "\n")
        return [*result, "--entrypoint", "python3", image, "/opt/bbo-entry.py", *command]

    def close(self):
        # Killing the Docker client does not reliably kill its container.
        try:
            subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, timeout=15)
        finally:
            self.gateway.close()
            self.temporary.cleanup()
