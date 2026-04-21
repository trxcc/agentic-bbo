"""JSON helpers for Pablo artifact persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any


def _json_default(obj: object) -> object:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def to_jsonable(data: Any) -> Any:
    return json.loads(json.dumps(data, default=_json_default, sort_keys=True))


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_default, sort_keys=True) + "\n")


def stable_config_identity(config: dict[str, Any]) -> str:
    return sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def prompt_hash(*parts: str) -> str:
    digest = sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


__all__ = ["append_jsonl", "dump_json", "prompt_hash", "stable_config_identity", "to_jsonable"]
