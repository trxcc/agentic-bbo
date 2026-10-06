"""Normalized append-only events shared by agentic BBO methods."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


EVENT_SCHEMA_VERSION = "agentic-bbo.deliberation.v1"
EVENT_KINDS = frozenset({
    "context", "probe", "propose", "reconfigure", "belief_update", "role_call",
    "candidate_generated", "candidate_rejected", "commit", "observation",
    "fallback", "stop", "error",
})


@dataclass(frozen=True)
class DeliberationEvent:
    method: str
    kind: str
    evaluation_index: int
    payload: Mapping[str, Any] = field(default_factory=dict)
    call_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    schema_version: str = EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"Unknown deliberation event kind: {self.kind!r}")
        if self.evaluation_index < 0:
            raise ValueError("evaluation_index must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["payload"] = dict(self.payload)
        return result


class DeliberationEventWriter:
    """Small JSONL sink; method-specific detailed logs remain untouched."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, event: DeliberationEvent) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), sort_keys=True, default=str) + "\n")


__all__ = [
    "DeliberationEvent",
    "DeliberationEventWriter",
    "EVENT_KINDS",
    "EVENT_SCHEMA_VERSION",
]
