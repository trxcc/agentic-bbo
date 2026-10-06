"""Workspace-local BBO tool bridge for shell/file based agents."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable


ToolHandler = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
VISIBLE_FLOAT_DECIMALS = 4


def _visible_payload(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return value
        rounded = round(value, VISIBLE_FLOAT_DECIMALS)
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, dict):
        return {key: _visible_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_visible_payload(item) for item in value]
    return value


def _visible_config(config: dict[str, Any]) -> dict[str, Any]:
    return _visible_payload(config)


def main(argv: list[str] | None = None, *, default_config_path: str | None = None) -> int:
    """Run one workspace BBO tool and print a JSON result."""

    parser = argparse.ArgumentParser(description="Call a BBO workspace tool.")
    parser.add_argument("tool_name", help="Tool name from tool_specs.json.")
    parser.add_argument("arguments", nargs="?", default="{}", help="JSON object with tool arguments.")
    parser.add_argument("--config", default=default_config_path or "bbo_tool_config.json")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config = _read_json(config_path)
    if args.tool_name.startswith("optimizer_"):
        hosted = _execute_via_optimizer_host(
            tool_name=args.tool_name,
            raw_arguments=args.arguments,
            config=config,
            config_path=config_path,
        )
        if hosted is not None:
            print(json.dumps(hosted, ensure_ascii=False, sort_keys=True))
            return 0 if hosted.get("ok") is True else 2
    started = time.monotonic()
    timestamp = time.time()
    try:
        arguments = json.loads(args.arguments)
        if not isinstance(arguments, dict):
            raise TypeError("arguments must decode to a JSON object.")
        result = _execute(args.tool_name, arguments, config)
        payload = {"ok": True, "result": result}
        success = True
    except Exception as exc:
        payload = {"ok": False, "error": "exception", "message": str(exc)}
        success = False
    _log_call(config, args.tool_name, args.arguments, payload, started, timestamp, success, interface="workspace_cli")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if success else 2


def _execute_via_optimizer_host(
    *,
    tool_name: str,
    raw_arguments: str,
    config: dict[str, Any],
    config_path: Path,
) -> dict[str, Any] | None:
    """Run optimizer actions with the benchmark's dependency-complete Python.

    Workspace analysis tools remain stdlib-only. Optimizer actions need the
    repository package and optional BO dependencies, so they are delegated to
    the exact interpreter that created the run. The delegated process still
    reads only the agent-visible workspace files and never calls the evaluator.
    """

    if os.environ.get("BBO_OPTIMIZER_HOST_PROCESS") == "1":
        return None
    executable = str(config.get("optimizer_python_executable") or "").strip()
    repository_root = str(config.get("optimizer_repository_root") or "").strip()
    if not executable or not repository_root:
        return None
    workspace_dir = Path(str(config.get("workspace_dir") or config_path.parent))
    bridge_path = workspace_dir / "bbo_tool.py"
    env = dict(os.environ)
    env["BBO_OPTIMIZER_HOST_PROCESS"] = "1"
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        repository_root
        if not existing_pythonpath
        else repository_root + os.pathsep + existing_pythonpath
    )
    try:
        completed = subprocess.run(
            [
                executable,
                str(bridge_path),
                tool_name,
                raw_arguments,
                "--config",
                str(config_path),
            ],
            cwd=str(workspace_dir),
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": "optimizer_host_failure",
            "message": f"Optimizer host process failed: {exc}",
        }
    for line in reversed(completed.stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "ok" in payload:
            return payload
    detail = completed.stderr.strip() or completed.stdout.strip()
    return {
        "ok": False,
        "error": "optimizer_host_failure",
        "message": (
            f"Optimizer host process exited with code {completed.returncode}"
            + (f": {detail[-1000:]}" if detail else ".")
        ),
    }


def _execute(tool_name: str, arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "get_history": "get_trial_history",
        "get_space": "get_search_space",
        "get_objective": "get_objective",
        "get_tool_specs": "get_tool_specs",
        "get_manifest": "get_manifest",
    }
    tool_name = aliases.get(tool_name, tool_name)
    enabled = config.get("enabled_tool_names")
    if isinstance(enabled, list) and tool_name not in {str(item) for item in enabled}:
        raise PermissionError(
            f"Workspace tool {tool_name!r} is not enabled for this experiment condition."
        )
    repository_root = str(config.get("optimizer_repository_root") or "")
    if repository_root and repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    try:
        service = importlib.import_module("bbo.algorithms.agentic.tools.tool_service")
    except ImportError as exc:
        raise RuntimeError("The canonical BBO tool service is unavailable.") from exc
    if tool_name in service.CANONICAL_WORKSPACE_TOOLS:
        return service.execute_workspace_tool(config, tool_name, arguments)

    optimizer_actions = {
        "optimizer_suggest",
        "optimizer_recommend_backends",
        "optimizer_portfolio_suggest",
        "optimizer_predict",
        "optimizer_score",
        "optimizer_diagnostics",
        "optimizer_status",
        "optimizer_set_backend",
        "optimizer_set_bounds",
        "optimizer_set_acquisition",
        "optimizer_reset_policy",
    }
    if tool_name in optimizer_actions:
        return _optimizer_action(tool_name, arguments, config)

    handlers: dict[str, ToolHandler] = {
        "get_task_context": _get_task_context,
        "get_manifest": _get_manifest,
        "get_objective": _get_objective,
        "get_tool_specs": _get_tool_specs,
        "optimizer_suggest": _optimizer_suggest,
        "memory_read": _memory_read,
        "memory_write": _memory_write,
        "code_interpreter": _code_interpreter,
        "web_search": _web_search,
        "fetch_url": _fetch_url,
    }
    if tool_name not in handlers:
        raise ValueError(f"Unknown BBO workspace tool `{tool_name}`.")
    return handlers[tool_name](arguments, config)


def _optimizer_action(
    tool_name: str,
    arguments: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Enforce per-round decision quotas and call the shared optimizer controller."""

    decision_tools = {
        "optimizer_suggest",
        "optimizer_score",
        "optimizer_portfolio_suggest",
    }
    if tool_name in decision_tools:
        max_calls = max(
            0, int(config.get("optimizer_max_calls_per_round", 4))
        )
        if max_calls == 0:
            raise RuntimeError(
                "Optimizer candidate-decision actions are disabled for this condition."
            )
        call_id = os.environ.get("BBO_AGENT_CALL_ID")
        prior_count = 0
        calls_path = Path(str(config.get("tool_calls_path", "")))
        if call_id and calls_path.exists():
            for line in calls_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record_call_id = (
                    record.get("agent_call_id") or record.get("call_id")
                )
                if (
                    record_call_id == call_id
                    and record.get("tool_name") in decision_tools
                    and record.get("success") is True
                ):
                    prior_count += 1
        if prior_count >= max_calls:
            raise RuntimeError(
                "Optimizer candidate-decision quota exceeded for agent round "
                f"{call_id!r}: {prior_count}/{max_calls}."
            )
    try:
        module = importlib.import_module(
            "bbo.algorithms.agentic.tools.optimizer_tools"
        )
        execute_from_workspace = module.execute_from_workspace
    except ImportError as exc:
        raise RuntimeError(
            "Optimizer tools require the repository bbo package in the "
            "workspace Python environment."
        ) from exc
    return execute_from_workspace(
        workspace_dir=_workspace(config),
        config=config,
        action=tool_name,
        arguments=arguments,
    )

def _optimizer_suggest(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    backend = str(arguments.get("backend", "")).strip()
    if not backend:
        raise ValueError("backend is required.")
    call_id = os.environ.get("BBO_AGENT_CALL_ID")
    max_calls = max(0, int(config.get("optimizer_max_calls_per_round", 3)))
    if max_calls == 0:
        raise RuntimeError("Optimizer suggestions are disabled for this condition.")
    prior_backends: list[str] = []
    calls_path = Path(str(config.get("tool_calls_path", "")))
    if call_id and calls_path.exists():
        for line in calls_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_call_id = record.get("agent_call_id") or record.get("call_id")
            if record_call_id != call_id or record.get("tool_name") != "optimizer_suggest":
                continue
            raw_arguments = record.get("arguments")
            if isinstance(raw_arguments, str):
                try:
                    raw_arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    raw_arguments = {}
            if isinstance(raw_arguments, dict):
                prior_backends.append(str(raw_arguments.get("backend", "")).strip())
    if backend in prior_backends:
        raise RuntimeError(
            f"Backend {backend!r} may be called at most once in agent round {call_id!r}."
        )
    if len(prior_backends) >= max_calls:
        raise RuntimeError(
            f"Optimizer suggestion quota exceeded for agent round {call_id!r}: "
            f"{len(prior_backends)}/{max_calls}."
        )
    try:
        module = importlib.import_module(
            "bbo.algorithms.agentic.tools.optimizer_tools"
        )
        suggest_from_workspace = module.suggest_from_workspace
    except ImportError as exc:
        raise RuntimeError(
            "optimizer_suggest requires the repository bbo package in the workspace Python environment."
        ) from exc
    seed = arguments.get("seed")
    return suggest_from_workspace(
        workspace_dir=_workspace(config),
        config=config,
        backend=backend,
        seed=None if seed is None else int(seed),
    )


def _get_task_context(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    workspace = _workspace(config)
    max_chars = int(arguments.get("max_chars_per_section", 4000))
    include_manifest = bool(arguments.get("include_manifest", True))
    task_md = (workspace / "task.md").read_text(encoding="utf-8")
    requested = arguments.get("sections")
    section_map = _markdown_sections(task_md)
    if requested:
        wanted = {str(item) for item in requested}
        sections = {name: text for name, text in section_map.items() if name in wanted}
    else:
        sections = section_map
    return {
        "task_id": _manifest(config).get("task_id"),
        "objective": _read_json(workspace / "objective.json"),
        "sections": {key: _truncate(value, max_chars) for key, value in sections.items()},
        "manifest": _manifest(config) if include_manifest else None,
    }




def _get_manifest(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    del arguments
    return {"manifest": _manifest(config)}


def _get_objective(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    del arguments
    return {"objective": _objective(config)}


def _get_tool_specs(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    del arguments
    return _read_json(_workspace(config) / "tool_specs.json")


def _memory_read(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    path = _memory_path(config)
    kind = arguments.get("kind")
    tags = set(str(tag) for tag in arguments.get("tags", []) or [])
    limit = max(1, int(arguments.get("limit", 20)))
    if not path.exists():
        return {"enabled": True, "records": [], "count": 0}
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if kind is not None and record.get("kind") != kind:
            continue
        if tags and not tags.issubset(set(str(tag) for tag in record.get("tags", []))):
            continue
        records.append(record)
    records = records[-limit:]
    return {"enabled": True, "records": records, "count": len(records)}


def _memory_write(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    kind = str(arguments.get("kind", "")).strip()
    content = str(arguments.get("content", "")).strip()
    if not kind or not content:
        raise ValueError("memory_write requires non-empty kind and content.")
    path = _memory_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "id": f"mem_{int(time.time() * 1000)}",
        "timestamp": time.time(),
        "kind": kind,
        "content": content,
        "tags": list(arguments.get("tags", []) or []),
        "source_call_id": arguments.get("source_call_id"),
        "trial_range": arguments.get("trial_range"),
        "metadata": arguments.get("metadata") or {},
    }
    _append_jsonl(path, record)
    summary_path = Path(config["memory_summary_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"updated_at": time.time(), "record_count": _line_count(path)}, indent=2), encoding="utf-8")
    return {"enabled": True, "written": True, "record": record}


def _code_interpreter(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    code = str(arguments.get("code", ""))
    language = str(arguments.get("language", "python"))
    if not code.strip():
        raise ValueError("code must be non-empty.")
    backend = str(config.get("code_backend", "disabled")).strip().lower().replace("-", "_")
    manifest_policy = (_manifest(config).get("tool_policy") or {}).get("code_interpreter") or {}
    if manifest_policy.get("enabled") is False and backend not in {"mock", "docker", "restricted_docker", "local_docker"}:
        sandbox_result = {
            "status": "Disabled",
            "message": "The BBO manifest disables code_interpreter for this benchmark.",
            "run_result": None,
        }
    elif backend == "mock":
        sandbox_result = {
            "status": "Success",
            "message": "",
            "compile_result": None,
            "run_result": {
                "status": "Finished",
                "execution_time": 0.0,
                "return_code": 0,
                "stdout": f"mock {language}: {len(code)} chars\n",
                "stderr": "",
            },
            "files": {},
        }
    elif backend in {"docker", "restricted_docker", "local_docker"}:
        module = importlib.import_module("bbo.algorithms.agentic.tools.code_tools")
        docker_backend = module.DockerBBOCodeBackend(
            workspace_dir=_workspace(config),
            image=str(config.get("docker_image") or "agentic-bbo-analysis-sandbox:v1"),
        )
        sandbox_result = docker_backend.execute_blocking(
            code=code,
            language=language,
        )
    elif backend == "sandboxfusion" and config.get("sandbox_fusion_base_url"):
        sandbox_result = _sandboxfusion_run(str(config["sandbox_fusion_base_url"]), code, language)
    else:
        sandbox_result = {
            "status": "Disabled",
            "message": "BBO code execution is disabled. Configure SandboxFusion to enable this tool.",
            "run_result": None,
        }
    return {
        "backend": backend,
        "language": language,
        "sandbox_result": sandbox_result,
        "budget_consumed": False,
    }


def _web_search(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    query = str(arguments.get("query", "")).strip()
    if not query:
        raise ValueError("query must be non-empty.")
    limit = min(max(1, int(arguments.get("limit", 5))), 10)
    provider = str(config.get("web_search_provider", "disabled")).strip().lower().replace("-", "_")
    policy = (_manifest(config).get("tool_policy") or {}).get("web_search") or {}
    if policy.get("enabled") is False and provider != "mock":
        return {"enabled": False, "query": query, "results": [], "count": 0}
    if provider in {"", "disabled", "none"}:
        return {"enabled": False, "query": query, "results": [], "count": 0}
    if provider == "mock":
        raw_results = [
            {
                "title": "Mock BBO prior",
                "url": "https://example.test/bbo-prior",
                "snippet": f"Mock search result for {query}",
            }
        ][:limit]
    elif provider == "serpapi":
        raw_results = _serpapi_search(
            query,
            limit,
            str(config.get("web_search_api_key_env") or "SERPAPI_API_KEY"),
            str(config.get("web_search_api_key") or ""),
            str(config.get("serpapi_endpoint") or "https://serpapi.com/search.json"),
        )
    else:
        raise ValueError(f"Unknown BBO web search provider `{provider}`.")
    logged = [_log_source(config, {"kind": "search_result", "query": query, **item}) for item in raw_results]
    return {"enabled": True, "query": query, "results": logged, "count": len(logged)}


def _fetch_url(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    url = str(arguments.get("url", "")).strip()
    max_chars = min(max(200, int(arguments.get("max_chars", 4000))), 20000)
    if not url:
        raise ValueError("url must be non-empty.")
    allowed = _fetch_allowed(url, _manifest(config).get("research_policy") or {})
    if not allowed["ok"]:
        return {"enabled": False, "url": url, "reason": allowed["reason"], "content": ""}
    fetched = _fetch_text(url, max_chars)
    return _log_source(config, {"kind": "fetched_url", "url": url, **fetched})


def _workspace(config: dict[str, Any]) -> Path:
    return Path(config["workspace_dir"])


def _parameters(config: dict[str, Any]) -> list[dict[str, Any]]:
    return list(_read_json(_workspace(config) / "space.json").get("parameters", []))


def _objective(config: dict[str, Any]) -> dict[str, Any]:
    return _read_json(_workspace(config) / "objective.json")


def _manifest(config: dict[str, Any]) -> dict[str, Any]:
    return _read_json(_workspace(config) / "manifest.json")


def _history(config: dict[str, Any]) -> list[dict[str, Any]]:
    path = _workspace(config) / "history.jsonl"
    if not path.exists():
        return []
    trials = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            trials.append(json.loads(line))
    return trials


def _memory_path(config: dict[str, Any]) -> Path:
    return Path(config["memory_path"])


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _log_call(
    config: dict[str, Any],
    tool_name: str,
    raw_arguments: str,
    payload: dict[str, Any],
    started: float,
    timestamp: float,
    success: bool,
    *,
    interface: str,
) -> None:
    path = Path(config["tool_calls_path"])
    try:
        arguments: Any = json.loads(raw_arguments)
    except Exception:
        arguments = raw_arguments
    record = {
            "timestamp": timestamp,
            "call_id": "workspace_cli",
            "agent_call_id": os.environ.get("BBO_AGENT_CALL_ID") or None,
            "tool_call_id": f"workspace_cli_{int(timestamp * 1000)}",
            "tool_name": tool_name,
            "arguments": arguments,
            "success": success,
            "duration_ms": round((time.monotonic() - started) * 1000.0, 3),
            "result_preview": _truncate(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str), 1200),
            "interface": interface,
        }
    if tool_name == "optimizer_suggest" and isinstance(payload, dict):
        nested = payload.get("result", payload)
        if isinstance(nested, dict):
            candidate = nested.get("candidate")
            if not isinstance(candidate, dict):
                candidates = nested.get("candidates")
                if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
                    candidate = candidates[0].get("candidate")
            if isinstance(candidate, dict):
                record["result_data"] = {
                    "candidate": candidate,
                    "identity": nested.get("identity"),
                    "backend": nested.get("backend"),
                }
    _append_jsonl(path, record)


def _log_source(config: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "source_id": f"src_{int(time.time() * 1000)}_{abs(hash(json.dumps(record, sort_keys=True, default=str))) % 1000000:06d}",
        "timestamp": time.time(),
        **record,
    }
    _append_jsonl(Path(config["sources_path"]), payload)
    return payload


def _markdown_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {"document": []}
    current = "document"
    for line in text.splitlines():
        if line.startswith("#"):
            name = line.lstrip("#").strip().lower().replace(" ", "_").replace("-", "_")
            if name:
                current = name
                sections.setdefault(current, [])
                continue
        sections.setdefault(current, []).append(line)
    return {key: "\n".join(value).strip() for key, value in sections.items() if "\n".join(value).strip()}


def _sandboxfusion_run(base_url: str, code: str, language: str) -> dict[str, Any]:
    url = urllib.parse.urljoin(base_url.rstrip("/") + "/", "run_code")
    body = json.dumps({"code": code, "language": language}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120.0) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        return {"status": "Error", "message": f"SandboxFusion request failed: {exc}", "run_result": None}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"status": "Error", "message": "SandboxFusion returned non-JSON response.", "raw": raw}
    return payload if isinstance(payload, dict) else {"status": "Error", "message": "Unexpected response shape."}


def _serpapi_search(query: str, limit: int, api_key_env: str, api_key_value: str, endpoint: str) -> list[dict[str, Any]]:
    api_key = api_key_value or _required_env(api_key_env)
    params = urllib.parse.urlencode({"engine": "google", "q": query, "api_key": api_key, "num": limit})
    data = _get_json(f"{endpoint}?{params}", 30.0)
    return [
        {
            "title": str(item.get("title", "")),
            "url": str(item.get("link", "")),
            "snippet": str(item.get("snippet", "")),
        }
        for item in data.get("organic_results", [])
        if isinstance(item, dict)
    ][:limit]


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable `{name}`.")
    return value


def _fetch_allowed(url: str, policy: dict[str, Any]) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return {"ok": False, "reason": "fetch_url requires an http(s) URL"}
    allowed = set(policy.get("allowed_fetch_domains", []) or [])
    if allowed and parsed.netloc not in allowed:
        return {"ok": False, "reason": f"Domain `{parsed.netloc}` is not allowed by the BBO manifest"}
    if policy.get("allow_external_research") is False and not allowed:
        return {"ok": False, "reason": "External URL fetching is disabled by the BBO manifest"}
    return {"ok": True}


def _fetch_text(url: str, max_chars: int) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "agentic-bbo/0.1"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=20.0) as resp:
            raw = resp.read(max_chars + 1)
            content_type = resp.headers.get("Content-Type", "")
    except urllib.error.URLError as exc:
        return {"ok": False, "error": str(exc), "content": "", "content_type": ""}
    text = raw.decode("utf-8", errors="replace")
    return {
        "ok": True,
        "content_type": content_type,
        "content": text[:max_chars],
        "truncated": len(text) > max_chars,
    }


def _get_json(url: str, timeout_seconds: float, headers: dict[str, str] | None = None) -> dict[str, Any]:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
