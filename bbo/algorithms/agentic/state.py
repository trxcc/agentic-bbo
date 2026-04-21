"""Dataclasses for Pablo runtime state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateEntry:
    config: dict[str, Any]
    source: str
    role: str
    round_index: int
    task_name: str | None = None
    seed_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PabloResumeState:
    round_index: int = 0
    failure_streak: int = 0
    queue: list[CandidateEntry] = field(default_factory=list)
    task_registry: dict[str, Any] = field(default_factory=dict)
    seen_config_ids: list[str] = field(default_factory=list)
    model_routes: dict[str, str] = field(default_factory=dict)
    best_config: dict[str, Any] | None = None
    best_score: float | None = None
    history_size: int = 0
    provider: str = "mock"


__all__ = ["CandidateEntry", "PabloResumeState"]
