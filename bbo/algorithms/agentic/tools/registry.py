"""BBO tool registry and call logging."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..serialization import append_jsonl
from .base import BaseBBOTool
from .context import BBOToolContext


@dataclass
class BBOToolCallLogger:
    """Append-only logger for agent tool calls."""

    path: Path
    preview_chars: int = 1200

    def log(self, record: dict[str, Any]) -> None:
        append_jsonl(self.path, record)

    def preview(self, value: Any) -> str:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if len(text) <= self.preview_chars:
            return text
        return text[: self.preview_chars - 3] + "..."

    @staticmethod
    def structured_result(tool_name: str, result: dict[str, Any]) -> dict[str, Any] | None:
        if tool_name != "optimizer_suggest":
            return None
        payload = result.get("result") if isinstance(result, dict) else None
        if not isinstance(payload, dict):
            return None
        candidate = payload.get("candidate")
        if not isinstance(candidate, dict):
            candidates = payload.get("candidates")
            if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
                candidate = candidates[0].get("candidate")
        if not isinstance(candidate, dict):
            return None
        structured = {
            "candidate": candidate,
            "identity": payload.get("identity"),
            "backend": payload.get("backend"),
        }
        return structured


class BBOToolRegistry:
    """Registry for function-callable BBO tools."""

    def __init__(
        self,
        tools: Iterable[BaseBBOTool],
        *,
        logger: BBOToolCallLogger | None = None,
    ) -> None:
        self._tools: dict[str, BaseBBOTool] = {}
        self.logger = logger
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"Duplicate BBO tool name: {tool.name}")
            self._tools[tool.name] = tool

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def get_tool_specs(self) -> list[dict[str, Any]]:
        return [self._tools[name].function_spec() for name in self.names]

    async def execute_payload(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: BBOToolContext,
    ) -> Any:
        """Execute a tool and return its native payload without transport wrapping."""
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ValueError(f"BBO tool `{tool_name}` not found.")
        return await tool.execute(context, **dict(arguments or {}))

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: BBOToolContext,
        *,
        call_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> str:
        started = time.monotonic()
        timestamp = time.time()
        if tool_name not in self._tools:
            result = {"ok": False, "error": "unknown_tool", "message": f"BBO tool `{tool_name}` not found."}
            self._log_call(tool_name, arguments, result, started, timestamp, call_id, tool_call_id, False)
            return json.dumps(result, ensure_ascii=False, sort_keys=True)
        try:
            payload = await self.execute_payload(tool_name, arguments, context)
            result = {"ok": True, "result": payload}
            success = True
        except Exception as exc:
            result = {"ok": False, "error": "exception", "message": str(exc)}
            success = False
        self._log_call(tool_name, arguments, result, started, timestamp, call_id, tool_call_id, success)
        return json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)

    def _log_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        started: float,
        timestamp: float,
        call_id: str | None,
        tool_call_id: str | None,
        success: bool,
    ) -> None:
        if self.logger is None:
            return
        duration_ms = round((time.monotonic() - started) * 1000.0, 3)
        record = {
                "timestamp": timestamp,
                "call_id": call_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "success": success,
                "duration_ms": duration_ms,
                "result_preview": self.logger.preview(result),
            }
        structured = self.logger.structured_result(tool_name, result)
        if structured is not None:
            record["result_data"] = structured
        self.logger.log(record)


__all__ = ["BBOToolCallLogger", "BBOToolRegistry"]
