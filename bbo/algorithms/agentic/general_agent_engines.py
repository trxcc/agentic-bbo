"""Native Codex execution and host-mediated benchmark tools."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import shutil
import signal
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal


BBOToolExecutor = Callable[[str, dict[str, Any], str | None], Awaitable[str]]


def _first_optimizer_backend(tools: list[dict[str, Any]]) -> str | None:
    """Return the first advertised optimizer backend for a CLI example."""

    for spec in tools:
        function = spec.get("function") if isinstance(spec, dict) else None
        if not isinstance(function, dict) or function.get("name") != "optimizer_suggest":
            continue
        parameters = function.get("parameters")
        properties = parameters.get("properties") if isinstance(parameters, dict) else None
        backend = properties.get("backend") if isinstance(properties, dict) else None
        choices = backend.get("enum") if isinstance(backend, dict) else None
        if isinstance(choices, list) and choices:
            return str(choices[0])
    return None


class _HostToolTCPServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


class _HostToolUnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


def _start_host_tool_server(
    *,
    allowed: set[str],
    executor: BBOToolExecutor,
    loop: asyncio.AbstractEventLoop,
    max_calls: int,
    deadline: float | None = None,
    delivery_log: Path | None = None,
    unlimited_wait: bool = False,
    unix_path: Path | None = None,
) -> tuple[_HostToolTCPServer | _HostToolUnixServer, threading.Thread]:
    """Expose allowlisted BBO tools over a private Unix socket or loopback port."""
    counter = {"used": 0}
    lock = threading.Lock()

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            from .runtime_reliability import audit_event

            request_id = None
            try:
                request = json.loads(self.rfile.readline(1024 * 1024))
                name, arguments = request.get("name"), request.get("arguments", {})
                call_id = request.get("call_id")
                request_id = request.get("request_id")
                if name not in allowed or not isinstance(arguments, dict):
                    raise ValueError("Tool is unavailable or arguments are invalid.")
                with lock:
                    if max_calls > 0 and counter["used"] >= max_calls:
                        raise ValueError(f"Exceeded max BBO tool calls ({max_calls}).")
                    counter["used"] += 1
                future = asyncio.run_coroutine_threadsafe(
                    executor(
                        str(name), arguments, None if call_id is None else str(call_id)
                    ),
                    loop,
                )
                wait_seconds = (None if unlimited_wait else 180) if deadline is None else max(0.01, deadline - time.monotonic())
                payload = {"ok": True, "output": future.result(timeout=wait_seconds)}
                audit_event(delivery_log, {"event": "backend_completed", "request_id": request_id, "call_id": call_id, "tool_name": name})
            except Exception as exc:
                audit_event(delivery_log, {"event": "backend_failed", "request_id": request_id, "error_type": type(exc).__name__})
                payload = {
                    "ok": False,
                    "error": type(exc).__name__,
                    "message": str(exc),
                }
            payload["request_id"] = request_id
            try:
                self.wfile.write(json.dumps(payload, ensure_ascii=False).encode() + b"\n")
                self.wfile.flush()
                audit_event(delivery_log, {"event": "response_written", "request_id": request_id, "ok": payload["ok"]})
            except (BrokenPipeError, ConnectionResetError):
                audit_event(delivery_log, {"event": "response_write_failed", "request_id": request_id})

    if unix_path is None:
        server = _HostToolTCPServer(("127.0.0.1", 0), Handler)
    else:
        server = _HostToolUnixServer(str(unix_path), Handler)
        unix_path.chmod(0o600)
    thread = threading.Thread(
        target=server.serve_forever, daemon=True, name="bbo-host-tool-socket"
    )
    thread.start()
    return server, thread


_HOST_TOOL_CLIENT = """#!/usr/bin/env python3
import argparse, json, os, socket, time, uuid
parser = argparse.ArgumentParser()
parser.add_argument("tool_name")
parser.add_argument("arguments", nargs="?", default="{}")
args = parser.parse_args()
request = {"name": args.tool_name, "arguments": json.loads(args.arguments), "call_id": os.environ.get("BBO_AGENT_CALL_ID"), "request_id": uuid.uuid4().hex}
deadline = float(os.environ.get("BBO_HOST_TOOL_DEADLINE", "0"))
endpoint = os.environ.get("BBO_HOST_TOOL_SOCKET", "")
if endpoint.startswith("tcp://"):
    host, port = endpoint[6:].rsplit(":", 1)
    client = socket.create_connection((host, int(port)), timeout=30)
else:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(endpoint or "/run/bbo-tool.sock")
if os.environ.get("BBO_HOST_TOOL_UNLIMITED") == "1":
    client.settimeout(None)
client.sendall(json.dumps(request).encode() + b"\\n")
data = b""
while not data.endswith(b"\\n"):
    if deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0: raise TimeoutError("Host tool exceeded agent deadline")
        client.settimeout(remaining)
    chunk = client.recv(65536)
    if not chunk: break
    data += chunk
response = json.loads(data)
receipt_path = os.environ.get("BBO_TOOL_RECEIPT_PATH")
if receipt_path:
    if not os.path.isabs(receipt_path):
        receipt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), receipt_path)
    with open(receipt_path, "a") as handle:
        handle.write(json.dumps({"timestamp": time.time(), "event": "client_received", "request_id": request["request_id"], "call_id": request["call_id"], "tool_name": args.tool_name, "ok": response.get("ok", False)}) + "\\n")
if not response.get("ok"):
    print(json.dumps(response, sort_keys=True)); raise SystemExit(2)
output = response.get("output", "")
print(output if isinstance(output, str) else json.dumps(output, sort_keys=True))
result = output
if isinstance(output, str):
    try:
        result = json.loads(output)
    except json.JSONDecodeError:
        pass
if isinstance(result, dict) and result.get("ok") is False:
    raise SystemExit(2)
"""


@dataclass
class AgentResult:
    """Result of one external agent invocation."""

    status: Literal["success", "failed", "timeout"]
    answer: str
    error: str | None = None
    returncode: int | None = None
    raw: Any = None
    llm_log: dict[str, Any] | None = None


@dataclass
class AgentWorkCopy:
    """Workspace and framework state handed to one agent engine."""

    state_dir: Path
    config_path: Path | None
    project_root: Path
    workspace_root: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _native_black_box_error(config: dict[str, Any], *, framework: str) -> str | None:
    """Return a fail-closed diagnostic when strong native isolation is unavailable."""

    if not config.get("black_box_required"):
        return None
    if str(config.get("tool_mode") or "no_tool") == "workspace_json":
        return (
            "Strict native black-box mode rejects `workspace_json`; workspace bridge "
            "tools expose optimizer-side runtime paths. Use `no_tool` or the sealed "
            "host-mediated `function_calling` bridge."
        )
    if framework == "Claude Code":
        return (
            "Strict Claude Code black-box mode is unavailable with the current in-process "
            "Agent SDK transport; refusing to expose the host filesystem. Use a sealed "
            "out-of-process Claude runner before collecting formal results."
        )
    if not sys.platform.startswith("linux"):
        return (
            f"Strict {framework} black-box mode requires Linux Docker support "
            "in this runtime."
        )
    backend, probe_error = _native_isolation_backend()
    if backend is None:
        return (
            f"Strict {framework} black-box mode requires a working Docker daemon; "
            f"refusing to expose the evaluator or repository. {probe_error}"
        )
    image = str(config.get("docker_image") or "agentic-bbo-analysis-sandbox:v1").strip()
    if image.lower() in {"disabled", "none", "off", "false"}:
        return (
            f"Strict {framework} black-box mode requires a real Docker image; "
            f"received docker_image={image!r}."
        )
    docker = shutil.which("docker")
    assert docker is not None
    try:
        inspected = subprocess.run(
            [docker, "image", "inspect", image],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"Docker image preflight failed for {image!r}: {type(exc).__name__}: {exc}"
    if inspected.returncode != 0:
        detail = (inspected.stderr or inspected.stdout).strip()
        return (
            f"Strict {framework} black-box mode requires Docker image {image!r}; "
            "the image is not available locally"
            + (f": {detail[-300:]}" if detail else ".")
        )
    return None


def _native_isolation_backend() -> tuple[str | None, str | None]:
    # Docker is the only supported native-agent isolation path. A host-specific
    # sandbox fallback would make local and remote shards observably different.
    docker = shutil.which("docker")
    if docker:
        try:
            checked = subprocess.run(
                [docker, "info", "--format", "{{.ServerVersion}}"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if checked.returncode == 0:
                return "docker", None
            docker_error = (checked.stderr or checked.stdout).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            docker_error = f"{type(exc).__name__}: {exc}"
    else:
        docker_error = "docker is unavailable"

    return None, f"Docker is required for native agent isolation: {docker_error}"


def _probe_bubblewrap(executable: str) -> str | None:
    """Verify that bubblewrap can create the namespace used by native agents."""

    probe_command = [
        executable,
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--share-net",
    ]
    for root in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
        if Path(root).exists():
            probe_command.extend(["--ro-bind", root, root])
    probe_command.extend(["--proc", "/proc", "--dev", "/dev", "--", "/usr/bin/true"])
    try:
        completed = subprocess.run(
            probe_command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env={"LANG": "C", "PATH": "/usr/bin:/bin"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"Isolation probe failed: {type(exc).__name__}: {exc}"
    if completed.returncode == 0:
        return None
    detail = (
        completed.stderr or completed.stdout or "unknown bubblewrap failure"
    ).strip()
    return f"Isolation probe exited {completed.returncode}: {detail[-300:]}"


def _strict_native_env(
    configured: dict[str, str],
    extra_env: dict[str, str] | None,
) -> dict[str, str]:
    """Build an allowlisted environment without leaking host credentials."""

    env = {
        "HOME": "/workspace",
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "NO_COLOR": "1",
        "PATH": "/runtime/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "SSL_CERT_DIR": "/etc/ssl/certs",
        "TMPDIR": "/tmp",
        # The local SGLang-compatible provider intentionally uses a sentinel
        # key.  Keep it explicit inside sealed containers; host credentials
        # are still never forwarded.
        "LOCAL_LLM_API_KEY": "EMPTY",
    }
    env.update({str(key): str(value) for key, value in configured.items()})
    env.update({str(key): str(value) for key, value in (extra_env or {}).items()})
    return env


def _sealed_bwrap_command(
    command: list[str],
    *,
    workspace: Path,
    state_dir: Path,
    read_only_mounts: list[tuple[Path, str]] | None = None,
    writable_mounts: list[tuple[Path, str]] | None = None,
    host_tool_socket: Path | None = None,
) -> list[str]:
    """Wrap a command in a minimal mount namespace that omits the repository."""

    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("bubblewrap is unavailable")
    wrapped = [
        bwrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--share-net",
    ]
    for root in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
        path = Path(root)
        if path.exists():
            wrapped.extend(["--ro-bind", root, root])
    wrapped.extend(
        [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/workspace",
            "--bind",
            str(workspace.resolve()),
            "/workspace",
            "--dir",
            "/state",
            "--bind",
            str(state_dir.resolve()),
            "/state",
            "--dir",
            "/opt",
        ]
    )
    for source, destination in read_only_mounts or []:
        wrapped.extend(["--ro-bind", str(source.resolve()), destination])
    for source, destination in writable_mounts or []:
        wrapped.extend(
            ["--dir", destination, "--bind", str(source.resolve()), destination]
        )
    if host_tool_socket is not None:
        wrapped.extend(
            ["--dir", "/run", "--bind", str(host_tool_socket.resolve()), "/run/bbo-tool.sock"]
        )
    wrapped.extend(["--chdir", "/workspace", "--"])
    return [*wrapped, *command]


def _sealed_docker_command(
    command: list[str],
    *,
    workspace: Path,
    state_dir: Path,
    read_only_mounts: list[tuple[Path, str]],
    writable_mounts: list[tuple[Path, str]] | None = None,
    container_env: dict[str, str],
    image: str,
    host_tool_socket: Path | None = None,
) -> list[str]:
    """Run a native agent in a read-only, capability-free Docker container."""

    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("docker is unavailable")
    if command[:2] == ["/usr/bin/env", "node"]:
        command = ["/opt/node", *command[2:]]
    wrapped = [
        docker,
        "run",
        "--rm",
        "--init",
        "--network",
        "host",
        "--read-only",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "512",
        "--memory",
        "8g",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=1g",
        "--workdir",
        "/workspace",
        "--mount",
        f"type=bind,src={workspace.resolve()},dst=/workspace",
        "--mount",
        f"type=bind,src={state_dir.resolve()},dst=/state",
    ]
    if host_tool_socket is not None:
        wrapped.extend(
            [
                "--mount",
                f"type=bind,src={host_tool_socket.resolve()},dst=/run/bbo-tool.sock",
            ]
        )
    node = Path(shutil.which("node") or "/usr/local/bin/node").resolve()
    wrapped.extend(["--mount", f"type=bind,src={node},dst=/opt/node,readonly"])
    for source, destination in read_only_mounts:
        wrapped.extend(
            ["--mount", f"type=bind,src={source.resolve()},dst={destination},readonly"]
        )
    for source, destination in writable_mounts or []:
        wrapped.extend(
            ["--mount", f"type=bind,src={source.resolve()},dst={destination}"]
        )
    for key, value in sorted(container_env.items()):
        wrapped.extend(["--env", f"{key}={value}"])
    return [*wrapped, image, *command]


def _sealed_executable(
    executable: str,
) -> tuple[list[str], list[tuple[Path, str]]]:
    """Map one native launcher into the sealed filesystem."""

    resolved = Path(executable).resolve()
    if resolved.suffix == ".js":
        package_root = resolved.parent.parent
        relative = resolved.relative_to(package_root)
        return (
            ["/usr/bin/env", "node", f"/opt/native-agent/{relative.as_posix()}"],
            [(package_root, "/opt/native-agent")],
        )
    return ["/opt/native-agent"], [(resolved, "/opt/native-agent")]


def _sealed_python() -> tuple[str, list[tuple[Path, str]]]:
    """Map the active Python environment without mounting the source checkout."""

    prefix = Path(sys.prefix).resolve()
    executable = Path(sys.executable).resolve()
    if prefix == Path("/usr") or prefix == Path("/usr/local"):
        return str(executable), []
    candidate = Path("/runtime/bin") / Path(sys.executable).name
    return str(candidate), [(prefix, "/runtime")]


class GeneralAgentEngine(ABC):
    """Minimal async agent execution interface borrowed from ClawArena."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Framework name surfaced in logs."""

    @abstractmethod
    async def run_agent(
        self,
        session_id: str,
        message: str,
        work_copy: AgentWorkCopy,
        *,
        agent_id: str | None = None,
        timeout: float | None = None,
        extra_env: dict[str, str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: BBOToolExecutor | None = None,
        max_tool_calls: int = 0,
        final_instruction: str | None = None,
    ) -> AgentResult:
        """Execute a single agent call."""




class CodexEngine(GeneralAgentEngine):
    """Codex CLI engine backed by a per-run isolated ``CODEX_HOME``."""

    @property
    def name(self) -> str:
        return "codex"

    async def run_agent(
        self,
        session_id: str,
        message: str,
        work_copy: AgentWorkCopy,
        *,
        agent_id: str | None = None,
        timeout: float | None = None,
        extra_env: dict[str, str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: BBOToolExecutor | None = None,
        max_tool_calls: int = 0,
        final_instruction: str | None = None,
    ) -> AgentResult:
        if tools:
            return await self._run_with_host_tools(
                session_id,
                message,
                work_copy,
                agent_id=agent_id,
                timeout=timeout,
                extra_env=extra_env,
                tools=tools,
                tool_executor=tool_executor,
                max_tool_calls=max_tool_calls,
                final_instruction=final_instruction,
            )

        if final_instruction:
            message = f"{message.rstrip()}\n\n{final_instruction}"

        cfg = work_copy.extra.get("codex_config", {})
        boundary_error = _native_black_box_error(cfg, framework="Codex")
        if boundary_error:
            return AgentResult(
                status="failed", answer="", error=boundary_error, returncode=-3
            )
        strict_boundary = bool(cfg.get("black_box_required"))

        isolated = cfg.get("execution_backend") == "isolated_docker"
        if isolated and cfg.get("responses_api_compat") not in {"sglang", "deepseek", "chat_completions"}:
            return AgentResult(status="failed", answer="", returncode=-3,
                error="isolated_docker requires a supported Responses-to-Chat-Completions bridge.")
        executable = str(
            cfg.get("executable") or os.environ.get("BBO_CODEX_BIN") or "codex"
        )
        resolved_executable = (
            executable if Path(executable).is_file() else shutil.which(executable)
        )
        if not resolved_executable:
            return AgentResult(
                status="failed",
                answer="",
                error=(
                    f"Codex backend could not find `{executable}`. Install the Codex CLI or set "
                    "`--agent-executable`/`BBO_CODEX_BIN`."
                ),
                returncode=127,
            )
        # Some installations expose Codex through a shell wrapper that locates
        # the real CLI below ``$HOME``.  A clean benchmark HOME must not break
        # that launcher, so resolve the installation-owned entry point before
        # replacing HOME in the child environment.
        resolved_path = Path(str(resolved_executable))
        try:
            launcher_text = resolved_path.read_text(encoding="utf-8")[:4096]
        except (OSError, UnicodeDecodeError):
            launcher_text = ""
        if "$HOME/.npm-global/bin/codex" in launcher_text:
            host_home = os.environ.get("HOME")
            installed_entrypoint = (
                Path(host_home) / ".npm-global" / "bin" / "codex"
                if host_home
                else None
            )
            if installed_entrypoint is not None and installed_entrypoint.is_file():
                resolved_executable = str(installed_entrypoint)

        workspace_path = (
            _resolve_workspace(work_copy, agent_id or "") or work_copy.project_root
        )
        isolation_backend, isolation_error = _native_isolation_backend()
        if strict_boundary and isolation_backend is None:
            return AgentResult(
                status="failed", answer="", error=isolation_error, returncode=-3
            )
        if isolated:
            from .isolated_docker import native_codex_binary
            isolated_executable = native_codex_binary(str(resolved_executable))
            launcher, read_only_mounts = ["/opt/native-agent"], []
        elif strict_boundary:
            launcher, read_only_mounts = _sealed_executable(str(resolved_executable))
        else:
            launcher, read_only_mounts = [str(resolved_executable)], []
        cmd = [
            *launcher,
            "--strict-config",
            "-C",
            "/workspace" if strict_boundary else str(workspace_path),
            "-s",
            str(cfg.get("sandbox") or "workspace-write"),
            "-a",
            str(cfg.get("approval_policy") or "never"),
        ]
        # A resumed validation retry already has the complete prompt, workspace
        # reads, reasoning, and command outputs in its conversation.  Keeping
        # the shell available here can trap reasoning-heavy models in an
        # unbounded "one more analysis step" loop: the CLI turn succeeds, but
        # the final answer is prose rather than the required candidate JSON.
        # Disable only the native shell on these corrective turns so the model
        # must submit from evidence it has already gathered.  First attempts
        # and non-resumed retries retain the normal native Codex tool surface.
        if session_id and final_instruction and not cfg.get("reliable_runtime"):
            cmd.extend(["--disable", "shell_tool"])
        responses_proxy = None
        isolated_runtime = None
        if cfg.get("responses_api_compat") in {"sglang", "deepseek", "chat_completions"}:
            from .codex_responses_compat import SGLangResponsesCompatibilityProxy

            upstream_base_url = cfg.get("api_base")
            if not upstream_base_url:
                return AgentResult(
                    status="failed",
                    answer="",
                    error="Codex SGLang compatibility mode requires an API base URL.",
                )
            responses_proxy = SGLangResponsesCompatibilityProxy(
                upstream_base_url,
                dialect=str(cfg.get("responses_api_compat")),
                # Keep the upstream connection alive longer than the outer
                # agent deadline so the runner, not the proxy, owns timeout
                # classification and retry behavior.
                # This is socket inactivity, not a total generation deadline.
                upstream_timeout_seconds=(float(timeout) + 60.0) if timeout is not None else 600.0,
                **({"round_guard": cfg["round_guard"]} if cfg.get("round_guard") is not None else {}),
                **({
                    "reliable_runtime": True,
                    "audit_path": workspace_path / ".agent_runtime/transport_events.jsonl",
                    "deadline": (time.monotonic() + float(timeout)) if timeout is not None else None,
                } if cfg.get("reliable_runtime") else {}),
            )
            responses_proxy.start()
            model_base_url = responses_proxy.base_url
            if isolated:
                from .isolated_docker import IsolatedDockerRuntime
                try:
                    isolated_runtime = IsolatedDockerRuntime(
                        state_dir=work_copy.state_dir,
                        config_path=work_copy.config_path or work_copy.state_dir / "config.toml",
                        target=model_base_url,
                        api_key=str((cfg.get("env") or {}).get(cfg.get("api_key_env"), "EMPTY")),
                        timeout=(float(timeout) + 60.0) if timeout is not None else 600.0,
                    )
                except Exception:
                    responses_proxy.close()
                    raise
                model_base_url = isolated_runtime.model_base_url
            cmd.extend(
                [
                    "-c",
                    (
                        "model_providers.bbo_sglang.base_url="
                        f"{json.dumps(model_base_url)}"
                    ),
                ]
            )
        if session_id:
            cmd.extend(
                [
                    "exec",
                    "--ignore-rules",
                    "resume",
                    "--json",
                    "--skip-git-repo-check",
                    session_id,
                    message,
                ]
            )
        else:
            cmd.extend(
                [
                    "exec",
                    "--ignore-rules",
                    "--json",
                    "--skip-git-repo-check",
                    "--color",
                    "never",
                    message,
                ]
            )
        if isolated:
            # Only round identifiers and tool timing options enter the container.
            # The actual provider key is injected by the host model gateway.
            container_env = _strict_native_env({}, None)
            if cfg.get("native_round_guard"):
                # Small matrices should not spawn one BLAS thread per host core.
                # Agents may explicitly override these for their own commands.
                for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    container_env[key] = "1"
            container_env.update({"CODEX_HOME": "/state", "HOME": "/state/home"})
            for key in ("BBO_AGENT_CALL_ID", "BBO_HOST_TOOL_DEADLINE", "BBO_HOST_TOOL_UNLIMITED", "BBO_TOOL_RECEIPT_PATH"):
                if key in (extra_env or {}):
                    container_env[key] = str(extra_env[key])
            if cfg.get("api_key_env"):
                container_env[str(cfg["api_key_env"])] = "HOST_GATEWAY_ONLY"
            if cfg.get("host_tool_socket"):
                container_env["BBO_HOST_TOOL_SOCKET"] = "/run/bbo-tool.sock"
            sandbox_index = cmd.index("-s")
            cmd[sandbox_index + 1] = "danger-full-access"
            assert isolated_runtime is not None
            try:
                cmd = isolated_runtime.command(cmd, workspace=workspace_path,
                    executable=isolated_executable, image=str(cfg["docker_image"]),
                    env=container_env, tool_socket=cfg.get("host_tool_socket"),
                    cpus=cfg.get("docker_cpus", 16.0))
            except Exception:
                isolated_runtime.close()
                responses_proxy.close()
                raise
            env = {**os.environ, "NO_COLOR": "1"}
        elif strict_boundary:
            container_env = _strict_native_env(dict(cfg.get("env") or {}), extra_env)
            container_env["CODEX_HOME"] = "/state"
            if isolation_backend != "docker":
                return AgentResult(
                    status="failed",
                    answer="",
                    error=isolation_error or "Docker is required for native agent isolation.",
                    returncode=-3,
                )
            else:
                sandbox_index = cmd.index("-s")
                cmd[sandbox_index + 1] = "danger-full-access"
                cmd = _sealed_docker_command(
                    cmd,
                    workspace=workspace_path,
                    state_dir=work_copy.state_dir,
                    read_only_mounts=read_only_mounts,
                    container_env=container_env,
                    image=str(
                        cfg.get("docker_image") or "agentic-bbo-analysis-sandbox:v1"
                    ),
                )
                env = {**os.environ, "NO_COLOR": "1"}
        else:
            isolated_home = work_copy.state_dir / "home"
            isolated_config = isolated_home / ".config"
            isolated_cache = isolated_home / ".cache"
            isolated_data = isolated_home / ".local" / "share"
            for directory in (
                isolated_home,
                isolated_config,
                isolated_cache,
                isolated_data,
            ):
                directory.mkdir(parents=True, exist_ok=True)
            inherited_env = {
                key: value
                for key, value in os.environ.items()
                if key
                in {
                    "PATH",
                    "LANG",
                    "LC_ALL",
                    "SSL_CERT_FILE",
                    "SSL_CERT_DIR",
                    "HTTP_PROXY",
                    "HTTPS_PROXY",
                    "NO_PROXY",
                    "http_proxy",
                    "https_proxy",
                    "no_proxy",
                }
            }
            env = {
                **inherited_env,
                **(cfg.get("env") or {}),
                **(extra_env or {}),
                "HOME": str(isolated_home),
                "CODEX_HOME": str(work_copy.state_dir),
                "XDG_CONFIG_HOME": str(isolated_config),
                "XDG_CACHE_HOME": str(isolated_cache),
                "XDG_DATA_HOME": str(isolated_data),
                "NO_COLOR": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                cwd=str(workspace_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            # Keep the same reader alive across timeout handling. Cancelling
            # communicate() discards stdout already consumed by that reader.
            communication = asyncio.create_task(proc.communicate())
            try:
                if timeout is None:
                    stdout, stderr = await asyncio.shield(communication)
                else:
                    stdout, stderr = await asyncio.wait_for(
                        asyncio.shield(communication),
                        timeout=timeout,
                    )
            except asyncio.TimeoutError:
                _kill_process(proc, process_group=True)
                stdout, stderr = await communication
                stdout_text = stdout.decode(errors="replace").strip()
                stderr_text = stderr.decode(errors="replace").strip()
                events, invalid_lines = _parse_codex_jsonl(stdout_text)
                return AgentResult(
                    status="timeout",
                    answer=_codex_final_answer(events),
                    error=_agent_timeout_error(timeout),
                    returncode=-1,
                    raw=events,
                    llm_log=_build_codex_llm_log(
                        events=events,
                        invalid_lines=invalid_lines,
                        stderr=stderr_text,
                        agent_id=agent_id,
                    ),
                )
            except asyncio.CancelledError:
                _kill_process(proc, process_group=True)
                await asyncio.shield(communication)
                raise
        finally:
            if isolated_runtime is not None:
                await asyncio.to_thread(isolated_runtime.close)
            if responses_proxy is not None:
                await asyncio.to_thread(responses_proxy.close)

        stdout_text = stdout.decode(errors="replace").strip()
        stderr_text = stderr.decode(errors="replace").strip()
        events, invalid_lines = _parse_codex_jsonl(stdout_text)
        answer = _codex_final_answer(events)
        llm_log = _build_codex_llm_log(
            events=events,
            invalid_lines=invalid_lines,
            stderr=stderr_text,
            agent_id=agent_id,
        )
        if proc.returncode == 0 and answer:
            return AgentResult(
                status="success",
                answer=answer,
                returncode=0,
                raw=events,
                llm_log=llm_log,
            )
        error = (
            _codex_error(events)
            or stderr_text
            or (
                "Codex exited successfully but did not emit a final agent message."
                if proc.returncode == 0
                else stdout_text
            )
        )
        return AgentResult(
            status="failed",
            answer=answer,
            error=error,
            returncode=proc.returncode,
            raw=events,
            llm_log=llm_log,
        )

    async def _run_with_host_tools(
        self,
        session_id: str,
        message: str,
        work_copy: AgentWorkCopy,
        *,
        agent_id: str | None,
        timeout: float | None,
        extra_env: dict[str, str] | None,
        tools: list[dict[str, Any]],
        tool_executor: BBOToolExecutor | None,
        max_tool_calls: int,
        final_instruction: str | None,
    ) -> AgentResult:
        """Expose the original CLI tool surface through a private host Unix socket."""
        if tool_executor is None or max_tool_calls < 0:
            return AgentResult(
                status="failed",
                answer="",
                error="Host tools require an executor and non-negative call budget (0 means unlimited).",
                returncode=-2,
            )
        transport_spec_provider = work_copy.extra.get(
            "tool_transport_spec_provider"
        )
        advertised_tools = (
            transport_spec_provider()
            if callable(transport_spec_provider)
            else tools
        )
        allowed = {
            str(item.get("function", {}).get("name"))
            for item in advertised_tools
            if isinstance(item, dict) and item.get("type") == "function"
        }
        allowed.discard("")
        workspace = work_copy.workspace_root or work_copy.project_root
        client_path = workspace / "bbo_tool.py"
        client_path.write_text(_HOST_TOOL_CLIENT, encoding="utf-8")
        client_path.chmod(0o755)
        reliable = bool(work_copy.extra.get("codex_config", {}).get("reliable_runtime"))
        deadline = time.monotonic() + float(timeout) if reliable and timeout is not None else None
        cfg = work_copy.extra.setdefault("codex_config", {})
        guard = None
        if cfg.get("native_round_guard") and "submit_candidate" in allowed:
            from .native_round_guard import NativeRoundGuard
            guard = NativeRoundGuard(native_tool_limit=max_tool_calls)
            guard.required_tool_choice = cfg.get("required_tool_choice_supported", True)
            cfg["round_guard"] = guard
        original_executor = tool_executor

        async def guarded_executor(name, arguments, call_id):
            output = await original_executor(name, arguments, call_id)
            if guard is not None:
                guard.observe_tool_result(name, output)
            return output

        socket_directory = tempfile.TemporaryDirectory(prefix="bbo-tool-") if cfg.get("execution_backend") == "isolated_docker" else None
        unix_path = Path(socket_directory.name) / "tool.sock" if socket_directory else None
        server, thread = _start_host_tool_server(
            allowed=allowed,
            executor=guarded_executor,
            loop=asyncio.get_running_loop(),
            max_calls=max_tool_calls,
            deadline=deadline,
            delivery_log=(workspace / ".agent_runtime/tool_delivery.jsonl") if reliable else None,
            unlimited_wait=reliable and timeout is None,
            **({"unix_path": unix_path} if unix_path is not None else {}),
        )
        previous_socket = cfg.get("host_tool_socket")
        if unix_path is not None:
            cfg["host_tool_socket"] = unix_path
        cfg["host_tool_address"] = "/run/bbo-tool.sock" if unix_path else f"tcp://127.0.0.1:{server.server_address[1]}"
        call_env = {**(extra_env or {}), "BBO_HOST_TOOL_SOCKET": cfg["host_tool_address"]}
        if reliable:
            receipt_path = workspace / ".agent_runtime/tool_client_receipts.jsonl"
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            call_env.update({"BBO_HOST_TOOL_DEADLINE": str(deadline) if deadline is not None else "0",
                             "BBO_HOST_TOOL_UNLIMITED": "1" if timeout is None else "0",
                             "BBO_TOOL_RECEIPT_PATH": ".agent_runtime/tool_client_receipts.jsonl"})
        if "validate_candidate" in allowed:
            tool_example = (
                "For example, validate one config with "
                "python3 bbo_tool.py validate_candidate "
                "'{\"candidate\":{\"x\":0.5}}'; the required outer argument key is "
                "`candidate`, whose value may be a raw config or an object with a "
                "`config` field. Do not pass `config` as the outer key. "
            )
        elif "optimizer_suggest" in allowed:
            backend = _first_optimizer_backend(advertised_tools) or "BACKEND"
            tool_example = (
                "For example, request one optimizer proposal with "
                "python3 bbo_tool.py optimizer_suggest "
                f"'{{\"backend\":\"{backend}\"}}'. "
            )
        else:
            tool_example = ""
        protocol = (
            "\n\nBBO TOOL CLI\nUse the existing workspace CLI exactly as follows: "
            "python3 bbo_tool.py TOOL_NAME '<JSON arguments object>'. "
            + tool_example
            + "Use the current workspace CLI and do not inspect, override, or reuse "
            "BBO_HOST_TOOL_SOCKET from an earlier attempt. "
            "Tool output is JSON on stdout. Do not import benchmark modules and do not emulate tool results. "
            "Available tool schemas: "
            + json.dumps(advertised_tools, ensure_ascii=False, sort_keys=True)
        )
        if "commit_candidate" in allowed:
            protocol += (
                "\n\nSTATE-GATED ROUND PROTOCOL\n"
                "For this round, use the CLI actions in this exact lifecycle: "
                "assess_backend_suitability, then optimizer_suggest with one explicit "
                "backend, then validate_candidate with the proposal_id returned by "
                "optimizer_suggest, then commit_candidate with the validation_token "
                "returned by validate_candidate. get_trial_history is read-only and may "
                "be called whenever available. The host rejects out-of-order and stale-ID "
                "calls and reports the current stage in its JSON error. After a successful "
                "commit_candidate, stop calling tools and return the candidate_payload from "
                "that tool result as raw JSON. For this state-gated round, commit_candidate "
                "is authoritative and supersedes the generic final_candidate.json handoff."
            )
        if "submit_candidate" in allowed:
            protocol += (
                "\nCall submit_candidate directly with config or a relative workspace JSON path; "
                "base_trial_id plus changes or a saved candidate_id are optional alternatives. "
                "write_candidate is optional, not a prerequisite. "
                "An accepted submission is authoritative and ends this round. Stop calling "
                "tools and acknowledge briefly; do not print or resubmit the configuration."
            )
        # On-demand schemas are static across rounds. Remember which native
        # session received them, including after process-level resume. Dynamic
        # optimizer protocols retain their existing per-round advertisement.
        schema_path = work_copy.state_dir / "context_tool_schema_sessions.json"
        schema_sessions = {}
        schema_hash = hashlib.sha256(protocol.encode()).hexdigest()
        if "submit_candidate" in allowed:
            try:
                schema_sessions = json.loads(schema_path.read_text())
                if not isinstance(schema_sessions, dict):
                    schema_sessions = {}
            except (OSError, ValueError):
                pass
            if session_id and schema_sessions.get(session_id) == schema_hash:
                protocol = (
                    "\n\nUse the same BBO tool CLI and schemas already provided. "
                    "context_tools.json contains the current schemas if needed. "
                    "Use the current CLI connection; submit_candidate accepts config or a workspace JSON path directly."
                )
        full_message = message + protocol
        if final_instruction:
            full_message += "\n\n" + final_instruction
        try:
            result = await self.run_agent(
                session_id,
                full_message,
                work_copy,
                agent_id=agent_id,
                timeout=timeout,
                extra_env=call_env,
                tools=None,
                tool_executor=None,
                max_tool_calls=0,
                final_instruction=None,
            )
            if "submit_candidate" in allowed and result.status == "success":
                resulting_session = str((result.llm_log or {}).get("sessionId") or session_id or "")
                if resulting_session:
                    schema_sessions[resulting_session] = schema_hash
                    schema_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = schema_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(schema_sessions, sort_keys=True) + "\n")
                    temporary.replace(schema_path)
            if result.llm_log is not None:
                result.llm_log = {
                    "hostToolTransport": "unix_socket_cli" if unix_path else "loopback_tcp_cli",
                    "effectivePrompt": full_message,
                    "advertisedTools": sorted(allowed),
                    "codexTurn": result.llm_log,
                }
            if guard is not None:
                from .native_round_guard import VERSION
                result.llm_log = {**(result.llm_log or {}), "nativeRoundGuard": {
                    "version": VERSION, "native_tool_limit": guard.limit,
                    "native_tool_calls": guard.used, "failure": guard.failure,
                    "required_tool_choice_supported": guard.required_tool_choice,
                    "submission_accepted": guard.receipt is not None,
                }}
            return result
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            cfg.pop("host_tool_address", None)
            cfg.pop("round_guard", None)
            if guard is not None:
                cfg["required_tool_choice_supported"] = guard.required_tool_choice
            if previous_socket is None:
                cfg.pop("host_tool_socket", None)
            else:
                cfg["host_tool_socket"] = previous_socket
            if socket_directory is not None:
                socket_directory.cleanup()






class MockAgentEngine(GeneralAgentEngine):
    """Deterministic local agent used by tests and offline examples."""

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = int(seed)
        self.calls = 0

    @property
    def name(self) -> str:
        return "mock"

    async def run_agent(
        self,
        session_id: str,
        message: str,
        work_copy: AgentWorkCopy,
        *,
        agent_id: str | None = None,
        timeout: float | None = None,
        extra_env: dict[str, str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: BBOToolExecutor | None = None,
        max_tool_calls: int = 0,
        final_instruction: str | None = None,
    ) -> AgentResult:
        del session_id, agent_id, timeout, extra_env, max_tool_calls
        if final_instruction:
            message = f"{message.rstrip()}\n\n{final_instruction}"
        del message
        import json

        if tools and tool_executor is not None:
            sample_raw = await tool_executor(
                "sample_candidates",
                {"n": 4, "seed": self.seed + self.calls, "strategy": "random"},
                "mock_sample_candidates",
            )
            self.calls += 1
            try:
                sample_payload = json.loads(sample_raw)
                candidates = [
                    {"config": item["config"], "rationale": "mock tool sample"}
                    for item in sample_payload["result"]["candidates"]
                ]
            except Exception:
                candidates = []
            if candidates:
                return AgentResult(
                    status="success",
                    answer=json.dumps({"candidates": candidates}, sort_keys=True),
                )

        space_path = (work_copy.workspace_root or work_copy.project_root) / "space.json"
        payload = json.loads(space_path.read_text(encoding="utf-8"))
        rng = random.Random(self.seed + self.calls)
        self.calls += 1
        candidates = []
        for _ in range(4):
            config: dict[str, Any] = {}
            for param in payload["parameters"]:
                if param["type"] == "float":
                    config[param["name"]] = rng.uniform(
                        float(param["low"]), float(param["high"])
                    )
                elif param["type"] == "int":
                    config[param["name"]] = rng.randint(
                        int(param["low"]), int(param["high"])
                    )
                elif param["type"] == "categorical":
                    config[param["name"]] = rng.choice(list(param["choices"]))
                else:
                    raise ValueError(
                        f"Unsupported mock parameter type: {param['type']}"
                    )
            candidates.append(
                {"config": config, "rationale": "mock deterministic sample"}
            )
        return AgentResult(
            status="success",
            answer=json.dumps({"candidates": candidates}, sort_keys=True),
        )


def create_general_agent_engine(framework: str) -> GeneralAgentEngine:
    normalized = normalize_agent_framework(framework)
    if normalized == "codex":
        return CodexEngine()
    if normalized == "mock":
        return MockAgentEngine()
    raise ValueError(f"Unknown general-agent framework `{framework}`.")


def normalize_agent_framework(framework: str) -> str:
    normalized = framework.strip().lower().replace("-", "_")
    if normalized in {"codex", "codex_cli", "openai_codex"}:
        return "codex"
    if normalized == "mock":
        return "mock"
    raise ValueError("Only the Codex Docker harness (and the test-only mock) is included")


def _resolve_workspace(work_copy: AgentWorkCopy, agent_id: str) -> Path | None:
    if work_copy.workspace_root and agent_id:
        candidate = work_copy.workspace_root / agent_id
        if candidate.exists():
            return candidate
    return work_copy.workspace_root




class WorkspaceToolCallLimitExceeded(RuntimeError):
    """Raised when a workspace-backed agent call exceeds the configured tool limit."""


async def _await_process_with_tool_limit(
    proc: Any,
    communicate_task: asyncio.Task[tuple[bytes, bytes]],
    *,
    timeout: float | None,
    tool_calls_path: Path | None,
    tool_calls_baseline: int,
    max_tool_calls: int,
) -> tuple[bytes, bytes]:
    started = asyncio.get_running_loop().time()
    while True:
        remaining = (
            None
            if timeout is None
            else timeout - (asyncio.get_running_loop().time() - started)
        )
        if remaining is not None and remaining <= 0:
            raise asyncio.TimeoutError
        wait_seconds = 0.25 if remaining is None else min(0.25, remaining)
        done, _ = await asyncio.wait({communicate_task}, timeout=wait_seconds)
        if done:
            return communicate_task.result()
        if max_tool_calls <= 0 or tool_calls_path is None:
            continue
        new_tool_calls = _count_nonempty_lines(tool_calls_path) - tool_calls_baseline
        if new_tool_calls >= max_tool_calls:
            _kill_process(proc)
            await communicate_task
            raise WorkspaceToolCallLimitExceeded(
                f"Exceeded max BBO workspace tool calls ({max_tool_calls}) in one agent invocation."
            )


def _workspace_tool_calls_path(workspace_path: Path | None) -> Path | None:
    if workspace_path is None:
        return None
    config_path = workspace_path / "bbo_tool_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw_path = config.get("tool_calls_path")
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = workspace_path / path
    return path


def _count_nonempty_lines(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _agent_timeout_error(timeout: float | None) -> str:
    return (
        f"Agent invocation timed out after {timeout}s because thinking/tool use took too long. "
        "Retry with concise reasoning and return exactly the required raw JSON object."
    )


def _kill_process(proc: Any, *, process_group: bool = False) -> None:
    try:
        if getattr(proc, "returncode", None) is None:
            if process_group and getattr(proc, "pid", None):
                try:
                    os.killpg(int(proc.pid), signal.SIGKILL)
                    return
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            proc.kill()
    except ProcessLookupError:
        pass










def _parse_codex_jsonl(stdout_text: str) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    invalid_lines: list[str] = []
    for line in stdout_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            invalid_lines.append(stripped)
            continue
        if isinstance(payload, dict):
            events.append(payload)
        else:
            invalid_lines.append(stripped)
    return events, invalid_lines


def _codex_final_answer(events: list[dict[str, Any]]) -> str:
    answers: list[str] = []
    for event in events:
        item = event.get("item")
        if (
            event.get("type") == "item.completed"
            and isinstance(item, dict)
            and item.get("type") == "agent_message"
            and isinstance(item.get("text"), str)
        ):
            answers.append(item["text"].strip())
    return next((answer for answer in reversed(answers) if answer), "")


def _codex_error(events: list[dict[str, Any]]) -> str | None:
    for event in reversed(events):
        if event.get("type") not in {"error", "turn.failed"}:
            continue
        error = event.get("error")
        if isinstance(error, dict):
            for key in ("message", "detail", "code"):
                if error.get(key):
                    return str(error[key])
        if error:
            return str(error)
        if event.get("message"):
            return str(event["message"])
    return None


def _build_codex_llm_log(
    *,
    events: list[dict[str, Any]],
    invalid_lines: list[str],
    stderr: str,
    agent_id: str | None,
) -> dict[str, Any]:
    thread_id = ""
    usage: dict[str, Any] | None = None
    native_tool_calls: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") == "thread.started" and event.get("thread_id"):
            thread_id = str(event["thread_id"])
        if event.get("type") == "turn.completed" and isinstance(
            event.get("usage"), dict
        ):
            usage = dict(event["usage"])
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if item_type in {
            "command_execution",
            "file_change",
            "mcp_tool_call",
            "web_search",
            "browser_use",
            "computer_use",
            "collab_agent_tool_call",
        }:
            native_tool_calls.append(
                {
                    key: item[key]
                    for key in (
                        "id",
                        "type",
                        "command",
                        "status",
                        "name",
                        "server",
                        "query",
                    )
                    if key in item
                }
            )
    return {
        "agentId": agent_id or "",
        "sessionId": thread_id,
        "success": any(event.get("type") == "turn.completed" for event in events),
        "usage": usage or {},
        "nativeToolCalls": native_tool_calls,
        "events": events,
        "invalidStdoutLines": invalid_lines,
        "stderr": stderr,
    }


__all__ = [
    "AgentResult",
    "AgentWorkCopy",
    "CodexEngine",
    "GeneralAgentEngine",
    "MockAgentEngine",
    "create_general_agent_engine",
    "normalize_agent_framework",
]
