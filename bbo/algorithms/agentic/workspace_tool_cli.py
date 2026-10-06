"""Thin workspace entrypoint for shell/file based agents."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
def _runtime_from_config(config: dict[str, Any]) -> Any:
    repository_root = str(config.get("optimizer_repository_root") or "")
    if repository_root and repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    return importlib.import_module("bbo.algorithms.agentic.workspace_tool_runtime")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _execute(tool_name: str, arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return _runtime_from_config(config)._execute(tool_name, arguments, config)


def _execute_via_optimizer_host(
    *, tool_name: str, raw_arguments: str, config: dict[str, Any], config_path: Path
) -> dict[str, Any] | None:
    return _runtime_from_config(config)._execute_via_optimizer_host(
        tool_name=tool_name,
        raw_arguments=raw_arguments,
        config=config,
        config_path=config_path,
    )


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
    _runtime_from_config(config)._log_call(
        config, tool_name, raw_arguments, payload, started, timestamp, success,
        interface=interface,
    )
from typing import Any


def _load_runtime(argv: list[str] | None, default_config_path: str | None) -> Any:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("tool_name")
    parser.add_argument("arguments", nargs="?", default="{}")
    parser.add_argument("--config", default=default_config_path or "bbo_tool_config.json")
    args, _ = parser.parse_known_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repository_root = str(config.get("optimizer_repository_root") or "")
    if repository_root and repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    try:
        return importlib.import_module("bbo.algorithms.agentic.workspace_tool_runtime")
    except ImportError as exc:
        raise RuntimeError("The BBO workspace runtime is unavailable.") from exc


def main(argv: list[str] | None = None, *, default_config_path: str | None = None) -> int:
    runtime = _load_runtime(argv, default_config_path)
    return int(runtime.main(argv, default_config_path=default_config_path))


if __name__ == "__main__":
    raise SystemExit(main())
