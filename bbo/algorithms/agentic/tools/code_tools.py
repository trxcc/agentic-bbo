"""BBO code execution tools backed by SandboxFusion-compatible services."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
import time
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from .base import BaseBBOTool
from .context import BBOToolContext


@dataclass
class SandboxFusionBBOCodeBackend:
    """HTTP backend for Bytedance SandboxFusion's `/run_code` API."""

    base_url: str
    timeout_seconds: float = 120.0

    async def execute(self, *, code: str, language: str = "python") -> dict[str, Any]:
        return await asyncio.to_thread(self._execute_blocking, code=code, language=language)

    def _execute_blocking(self, *, code: str, language: str) -> dict[str, Any]:
        url = urljoin(self.base_url.rstrip("/") + "/", "run_code")
        body = json.dumps({"code": code, "language": language}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            return {"status": "Error", "message": f"SandboxFusion request failed: {exc}", "run_result": None}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {"status": "Error", "message": "SandboxFusion returned non-JSON response.", "raw": raw}
        return payload if isinstance(payload, dict) else {"status": "Error", "message": "Unexpected response shape."}


@dataclass
class DockerBBOCodeBackend:
    """Run offline Python in a resource-limited, network-disabled container."""

    workspace_dir: Path
    image: str = "agentic-bbo-analysis-sandbox:v1"
    docker_executable: str = "docker"
    timeout_seconds: float = 30.0
    memory: str = "512m"
    cpus: float = 1.0
    pids_limit: int = 64
    output_chars: int = 50000

    async def execute(self, *, code: str, language: str = "python") -> dict[str, Any]:
        return await asyncio.to_thread(
            self.execute_blocking, code=code, language=language
        )

    def execute_blocking(self, *, code: str, language: str = "python") -> dict[str, Any]:
        if language.strip().lower() not in {"python", "python3", "py"}:
            return {
                "status": "Error",
                "message": "Restricted local code backend supports Python only.",
                "run_result": None,
            }
        if not shutil.which(self.docker_executable):
            return {
                "status": "Error",
                "message": f"Docker executable {self.docker_executable!r} was not found.",
                "run_result": None,
            }
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="bbo_analysis_inputs_") as raw_dir:
            input_dir = Path(raw_dir)
            for name in ("history.jsonl", "space.json", "objective.json", "incumbent.json"):
                source = self.workspace_dir / name
                if source.exists():
                    shutil.copy2(source, input_dir / name)
                    (input_dir / name).chmod(0o444)
            input_dir.chmod(0o555)
            command = [
                self.docker_executable,
                "run",
                "--rm",
                "-i",
                "--network",
                "none",
                "--read-only",
                "--cpus",
                str(self.cpus),
                "--memory",
                self.memory,
                "--pids-limit",
                str(self.pids_limit),
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=64m",
                "--workdir",
                "/tmp",
                "--env",
                "OPENBLAS_NUM_THREADS=1",
                "--env",
                "OMP_NUM_THREADS=1",
                "--env",
                "MKL_NUM_THREADS=1",
                "--env",
                "NUMEXPR_NUM_THREADS=1",
                "--env",
                "VECLIB_MAXIMUM_THREADS=1",
                "--env",
                "MPLCONFIGDIR=/tmp/matplotlib",
                "--mount",
                f"type=bind,src={input_dir},dst=/inputs,readonly",
                self.image,
                "python",
                "-I",
                "-",
            ]
            try:
                completed = subprocess.run(
                    command,
                    input=code,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
                status = "Success" if completed.returncode == 0 else "Error"
                stdout = completed.stdout[: self.output_chars]
                stderr = completed.stderr[: self.output_chars]
                return_code = int(completed.returncode)
                message = "" if return_code == 0 else "Restricted Docker Python exited non-zero."
            except subprocess.TimeoutExpired as exc:
                status = "Error"
                stdout = str(exc.stdout or "")[: self.output_chars]
                stderr = str(exc.stderr or "")[: self.output_chars]
                return_code = 124
                message = f"Restricted Docker Python timed out after {self.timeout_seconds}s."
            except OSError as exc:
                return {
                    "status": "Error",
                    "message": f"Restricted Docker Python failed to start: {exc}",
                    "run_result": None,
                }
        return {
            "status": status,
            "message": message,
            "compile_result": None,
            "run_result": {
                "status": "Finished",
                "execution_time": round(time.monotonic() - started, 6),
                "return_code": return_code,
                "stdout": stdout,
                "stderr": stderr,
            },
            "files": {},
            "policy": {
                "network": "none",
                "root_filesystem": "read_only",
                "inputs": "/inputs:read_only",
                "writable": "/tmp:tmpfs",
                "evaluator_access": False,
            },
        }


class DisabledBBOCodeBackend:
    """Code backend that reports a clear disabled result."""

    async def execute(self, *, code: str, language: str = "python") -> dict[str, Any]:
        del code, language
        return {
            "status": "Disabled",
            "message": "BBO code execution is disabled. Configure SandboxFusion to enable this tool.",
            "run_result": None,
        }


@dataclass
class MockBBOCodeBackend:
    """Deterministic backend for tests."""

    stdout: str = ""
    stderr: str = ""
    return_code: int = 0

    async def execute(self, *, code: str, language: str = "python") -> dict[str, Any]:
        return {
            "status": "Success",
            "message": "",
            "compile_result": None,
            "run_result": {
                "status": "Finished",
                "execution_time": 0.0,
                "return_code": self.return_code,
                "stdout": self.stdout or f"mock {language}: {len(code)} chars\n",
                "stderr": self.stderr,
            },
            "files": {},
        }


class CodeInterpreterTool(BaseBBOTool):
    name = "code_interpreter"
    description = (
        "Run analysis code in the configured BBO sandbox. This tool must not call task evaluators or consume trial budget."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Code to execute for offline analysis."},
            "language": {"type": "string", "default": "python", "description": "SandboxFusion language id."},
        },
        "required": ["code"],
    }

    async def execute(
        self,
        context: BBOToolContext,
        code: str,
        language: str = "python",
        **_: Any,
    ) -> dict[str, Any]:
        if not code.strip():
            raise ValueError("code must be non-empty.")
        policy = context.manifest.tool_policy.get("code_interpreter", {})
        if isinstance(policy, dict) and policy.get("enabled") is False and context.code_backend is None:
            return {
                "enabled": False,
                "message": "The BBO manifest disables code_interpreter for this benchmark.",
            }
        backend = context.code_backend or DisabledBBOCodeBackend()
        if not hasattr(backend, "execute"):
            raise TypeError("context.code_backend must provide an async execute(code=..., language=...) method.")
        result = await backend.execute(code=code, language=language)  # type: ignore[attr-defined]
        return {
            "backend": type(backend).__name__,
            "language": language,
            "sandbox_result": result,
            "budget_consumed": False,
        }


__all__ = [
    "CodeInterpreterTool",
    "DisabledBBOCodeBackend",
    "DockerBBOCodeBackend",
    "MockBBOCodeBackend",
    "SandboxFusionBBOCodeBackend",
]
