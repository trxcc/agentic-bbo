from __future__ import annotations

import asyncio
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import threading

import pytest

from bbo.algorithms.agentic.general_agent import normalize_agent_execution_backend
from bbo.algorithms.agentic.general_agent_engines import _start_host_tool_server
from bbo.algorithms.agentic.isolated_docker import IsolatedDockerRuntime, ModelGateway


class UnixConnection(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("localhost", timeout=5)
        self.path = str(path)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


def test_model_gateway_fixes_route_and_replaces_credentials(tmp_path):
    received = []

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            received.append((self.path, self.headers.get("Authorization"),
                             self.rfile.read(int(self.headers["Content-Length"]))))
            payload = b'data: {"ok":true}\n\n'
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    gateway = ModelGateway(tmp_path / "model.sock", f"http://127.0.0.1:{server.server_port}/v1", "host-secret", 5)
    try:
        for path, status in [("/v1/responses", 200), ("/v1/chat/completions", 403),
                             ("http://127.0.0.1:8096/evaluate", 403), ("/v1/responses?url=evil", 403)]:
            conn = UnixConnection(tmp_path / "model.sock")
            conn.request("POST", path, body=b"{}", headers={"Authorization": "Bearer agent-chosen", "Host": "evil"})
            response = conn.getresponse()
            assert response.status == status
            assert b"host-secret" not in response.read()
            conn.close()
        assert received == [("/v1/responses", "Bearer host-secret", b"{}")]
    finally:
        gateway.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_unix_tool_bridge_enforces_allowlist_and_budget(tmp_path):
    async def run():
        calls = []

        async def executor(name, args, call_id):
            calls.append((name, args, call_id))
            return '{"ok":true}'

        server, thread = _start_host_tool_server(allowed={"get_incumbent"}, executor=executor,
            loop=asyncio.get_running_loop(), max_calls=1, unix_path=tmp_path / "tool.sock")

        def request(name):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(tmp_path / "tool.sock"))
                client.sendall(json.dumps({"name": name, "arguments": {}}).encode() + b"\n")
                return json.loads(client.makefile("rb").readline())

        try:
            assert not (await asyncio.to_thread(request, "evaluate"))["ok"]
            assert (await asyncio.to_thread(request, "get_incumbent"))["ok"]
            assert not (await asyncio.to_thread(request, "get_incumbent"))["ok"]
            assert len(calls) == 1
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
    asyncio.run(run())


def test_container_mounts_exclude_host_state_and_reject_nested_state(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: None)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = tmp_path / "host_state"
    state.mkdir()
    config = state / "config.toml"
    config.write_text('model = "test"\n')
    runtime = IsolatedDockerRuntime(state_dir=state, config_path=config,
        target="http://127.0.0.1:1/v1", api_key="secret", timeout=5)
    try:
        cmd = runtime.command(["/opt/native-agent"], workspace=workspace,
            executable=Path("/bin/true"), image="test-image", env={}, tool_socket=None)
        assert cmd[cmd.index("--network") + 1] == "none"
        assert cmd[cmd.index("--cpus") + 1] == "16"
        assert "--privileged" not in cmd
        assert not any(f"src={state}," in item for item in cmd)
        assert any(f"src={state}/isolated_codex," in item for item in cmd)
        assert "secret" not in " ".join(cmd)
        cmd32 = runtime.command(["/opt/native-agent"], workspace=workspace,
            executable=Path("/bin/true"), image="test-image", env={}, tool_socket=None, cpus=32)
        assert cmd32[cmd32.index("--cpus") + 1] == "32"
        launches = [json.loads(line) for line in runtime.audit_path.read_text().splitlines()]
        assert [row["cpus"] for row in launches] == [16.0, 32.0]
        with pytest.raises(ValueError, match="outside"):
            runtime.command([], workspace=tmp_path, executable=Path("/bin/true"), image="test", env={}, tool_socket=None)
    finally:
        runtime.close()


@pytest.mark.parametrize("framework,image", [("nanobot", "image"), ("claude_code", "image"), ("codex", "disabled")])
def test_isolated_backend_fails_closed_for_unsupported_settings(framework, image):
    with pytest.raises(ValueError):
        normalize_agent_execution_backend("isolated_docker", framework=framework, code_backend="local_disabled", docker_image=image)


def test_anonymous_container_state_stays_outside_workspace_and_uses_opaque_path(tmp_path):
    from bbo.algorithms.agentic import CodexBBOAlgorithm, MockAgentEngine
    from bbo.tasks import create_task

    task = create_task("bbob_f01_d10", max_evaluations=2, seed=1)
    agent = CodexBBOAlgorithm(engine=MockAgentEngine(seed=1),
        run_dir=tmp_path / "bbob_f01_d10", execution_backend="isolated_docker",
        docker_image="test-image", provider="deepseek", tool_mode="no_tool")
    agent.setup(task.spec, seed=1, task_description=task.get_description())
    work = agent._work_copy
    assert not work.state_dir.is_relative_to(work.workspace_root)
    assert "bbob" not in str(work.state_dir)
    assert "f01" not in str(work.workspace_root.parent)
    assert work.extra["codex_config"]["black_box_required"]


@pytest.mark.parametrize("context_access,tool_mode", [("on_demand", "function_calling"), ("files", "no_tool")])
@pytest.mark.parametrize("backend", ["isolated_docker", "direct_workspace"])
def test_process_guidance_is_container_only_and_keeps_task_card_short(tmp_path, context_access, tool_mode, backend):
    from bbo.algorithms import create_algorithm
    from bbo.algorithms.agentic import MockAgentEngine
    from bbo.core import FloatParam, ObjectiveDirection, ObjectiveSpec, SearchSpace, TaskSpec

    task = TaskSpec(name="example", search_space=SearchSpace([FloatParam("x", low=0, high=1)]),
        objectives=(ObjectiveSpec("loss", ObjectiveDirection.MINIMIZE),), max_evaluations=2)
    agent = create_algorithm("raw_agentic_bbo", engine=MockAgentEngine(seed=1),
        run_dir=tmp_path, execution_backend=backend, docker_image="test-image",
        provider="deepseek", context_access=context_access, tool_mode=tool_mode)
    agent.setup(task, seed=1, task_description="# Example\n\nMinimize loss.")
    workspace = Path(agent.artifact_paths["agent_workspace"])
    instructions = (workspace / "instructions.md").read_text()
    card = (workspace / "task.md").read_text()
    assert ("## Local process control" in instructions) == (backend == "isolated_docker")
    assert "Local process control" not in card
    assert "start_new_session" not in instructions  # Examples remain on demand.
    if backend == "isolated_docker":
        assert "tty=true" in instructions
        assert "session IDs are not PIDs" in instructions
        assert "/usr/local/share/agent-process-control.md" in instructions
        assert instructions.isascii()
    if context_access == "on_demand":
        specs = json.loads((workspace / "context_tools.json").read_text())["tools"]
        assert {spec["function"]["name"] for spec in specs} == {
            "get_task_context", "get_search_space", "get_trial_history",
            "get_incumbent", "write_candidate", "submit_candidate",
        }


@pytest.mark.parametrize("cpus", [0, -1, float("nan"), float("inf"), True])
def test_invalid_cpu_quota_is_rejected_before_launch(cpus):
    from bbo.algorithms.agentic.raw_agentic_bbo import create_raw_agentic_bbo
    from bbo.algorithms.agentic.isolated_docker import validate_docker_cpus

    with pytest.raises(ValueError, match="docker_cpus"):
        validate_docker_cpus(cpus)
    with pytest.raises(ValueError, match="docker_cpus"):
        create_raw_agentic_bbo(docker_cpus=cpus)


def test_cpu_quota_flows_from_cli_through_options_to_native_config(tmp_path):
    from bbo.experiments.agent import build_agent
    from bbo.experiments.tasks import PaperTask
    from bbo.algorithms.agentic import MockAgentEngine
    task = PaperTask('bbob_f15_d10')
    agent = build_agent(task, tmp_path, dict(model='test', api_base='http://127.0.0.1:1/v1',
        api_key_env='BBO_TEST_KEY', reasoning_effort='max'), engine=MockAgentEngine())
    assert agent.config.docker_cpus == 32
    agent.setup(task.spec, seed=2, task_description=task.get_description())
    assert agent._work_copy.extra["codex_config"]["docker_cpus"] == 32
    assert json.loads(agent._agent_state_path.read_text())["docker_cpus"] == 32


def test_resume_requires_matching_cpu_quota_including_legacy_two_cpu_state(tmp_path):
    from bbo.algorithms.agentic import CodexBBOAlgorithm, MockAgentEngine
    from bbo.core import FloatParam, ObjectiveDirection, ObjectiveSpec, SearchSpace, TaskSpec

    task = TaskSpec(name="quota_resume", search_space=SearchSpace([FloatParam("x", low=0, high=1)]),
        objectives=(ObjectiveSpec("loss", ObjectiveDirection.MINIMIZE),), max_evaluations=2)

    def build(cpus, resume):
        return CodexBBOAlgorithm(engine=MockAgentEngine(seed=1), run_dir=tmp_path,
            execution_backend="isolated_docker", docker_image="test-image", provider="deepseek",
            tool_mode="no_tool", docker_cpus=cpus, resume=resume)

    agent = build(2, False)
    agent.setup(task, seed=1)
    with pytest.raises(ValueError, match="different CPU quota"):
        build(16, True).setup(task, seed=1)
    data = json.loads(agent._agent_state_path.read_text())
    data.pop("docker_cpus")
    agent._agent_state_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="different CPU quota"):
        build(16, True).setup(task, seed=1)
    build(2, True).setup(task, seed=1)
