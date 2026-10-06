"""BBO benchmark manifest loading.

The manifest is intentionally flexible.  It records the agent-facing benchmark
construction without forcing every existing task to carry a new file on day one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .task import TaskSpec


DEFAULT_BBO_TOOL_NAMES = (
    "get_task_context",
    "get_search_space",
    "get_trial_history",
    "get_incumbent",
    "get_history_overview",
    "compare_trials",
    "find_nearest_trials",
    "estimate_local_effects",
    "measure_search_coverage",
    "summarize_objective_metrics",
    "fit_and_check_surrogate",
    "score_virtual_candidates",
    "get_recent_search_actions",
    "validate_candidate",
    "validate_candidates",
    "sample_candidates",
    "analyze_history",
    "memory_read",
    "memory_write",
    "code_interpreter",
    "web_search",
    "fetch_url",
)


@dataclass(frozen=True)
class BBOBenchmarkManifest:
    """Agent-facing benchmark construction metadata."""

    task_id: str
    family: str = "unknown"
    real_world_domain: str | None = None
    workspace_seed_files: tuple[str, ...] = ()
    tool_policy: dict[str, Any] = field(default_factory=dict)
    research_policy: dict[str, Any] = field(default_factory=dict)
    memory_policy: dict[str, Any] = field(default_factory=dict)
    evaluation_endpoint: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    dynamic_updates: tuple[dict[str, Any], ...] = ()
    artifact_policy: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None
    generated: bool = False

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        path: Path | None = None,
        generated: bool = False,
    ) -> "BBOBenchmarkManifest":
        if not isinstance(data, dict):
            raise TypeError("BBO manifest must be a JSON object.")
        task_id = str(data.get("task_id", "")).strip()
        if not task_id:
            raise ValueError("BBO manifest requires a non-empty `task_id`.")
        dynamic_updates = data.get("dynamic_updates", ())
        if not isinstance(dynamic_updates, list | tuple):
            raise TypeError("BBO manifest `dynamic_updates` must be a list when provided.")
        return cls(
            task_id=task_id,
            family=str(data.get("family", "unknown")),
            real_world_domain=_optional_str(data.get("real_world_domain")),
            workspace_seed_files=tuple(str(item) for item in data.get("workspace_seed_files", ())),
            tool_policy=dict(data.get("tool_policy", {})),
            research_policy=dict(data.get("research_policy", {})),
            memory_policy=dict(data.get("memory_policy", {})),
            evaluation_endpoint=dict(data.get("evaluation_endpoint", {})),
            budget=dict(data.get("budget", {})),
            dynamic_updates=tuple(dict(item) for item in dynamic_updates),
            artifact_policy=dict(data.get("artifact_policy", {})),
            provenance=dict(data.get("provenance", {})),
            raw=dict(data),
            path=path,
            generated=generated,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest dictionary."""

        payload = {
            "task_id": self.task_id,
            "family": self.family,
            "real_world_domain": self.real_world_domain,
            "workspace_seed_files": list(self.workspace_seed_files),
            "tool_policy": dict(self.tool_policy),
            "research_policy": dict(self.research_policy),
            "memory_policy": dict(self.memory_policy),
            "evaluation_endpoint": dict(self.evaluation_endpoint),
            "budget": dict(self.budget),
            "dynamic_updates": list(self.dynamic_updates),
            "artifact_policy": dict(self.artifact_policy),
            "provenance": dict(self.provenance),
            "generated": self.generated,
        }
        if self.path is not None:
            payload["path"] = str(self.path)
        return payload


def load_BBO_manifest(task_spec: TaskSpec) -> BBOBenchmarkManifest:
    """Load a task-local BBO manifest or synthesize a compatible default."""

    manifest_path = _manifest_path_for_task(task_spec)
    if manifest_path is None or not manifest_path.exists():
        return build_compatible_BBO_manifest(task_spec)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid BBO manifest JSON at {manifest_path}: {exc}") from exc
    manifest = BBOBenchmarkManifest.from_dict(data, path=manifest_path, generated=False)
    if manifest.task_id != task_spec.name:
        raise ValueError(
            f"BBO manifest task_id `{manifest.task_id}` does not match task spec `{task_spec.name}`."
        )
    return manifest


def build_compatible_BBO_manifest(task_spec: TaskSpec) -> BBOBenchmarkManifest:
    """Build a manifest for tasks that predate explicit agent benchmark metadata."""

    metadata = dict(task_spec.metadata)
    family = str(metadata.get("family") or metadata.get("task_family") or _infer_family(task_spec.name))
    domain = metadata.get("domain") or metadata.get("benchmark") or metadata.get("display_name")
    payload = {
        "task_id": task_spec.name,
        "family": family,
        "real_world_domain": _optional_str(domain),
        "workspace_seed_files": _description_seed_files(task_spec),
        "tool_policy": {
            "enabled_tools": list(DEFAULT_BBO_TOOL_NAMES),
            "code_interpreter": {"enabled": True, "languages": ["python"]},
            "web_search": {"enabled": True},
        },
        "research_policy": {
            "allow_external_research": True,
            "source_logging_required": True,
        },
        "memory_policy": {
            "enabled": True,
            "append_only": True,
        },
        "evaluation_endpoint": {
            "kind": metadata.get("evaluator_kind", "ask_tell"),
            "direct_tool_access_allowed": False,
        },
        "budget": {
            "max_evaluations": task_spec.max_evaluations,
            "default_budget": task_spec.default_budget,
            "supports_budget": task_spec.supports_budget,
            "budget_range": task_spec.budget_range,
        },
        "dynamic_updates": [],
        "artifact_policy": {
            "append_only_jsonl": True,
            "record_tool_calls": True,
            "record_sources": True,
        },
        "provenance": {
            "generated_from_task_spec": True,
            "description_directory": _description_directory(task_spec),
        },
    }
    return BBOBenchmarkManifest.from_dict(payload, generated=True)


def _manifest_path_for_task(task_spec: TaskSpec) -> Path | None:
    ref = task_spec.description_ref
    if ref is None or ref.directory is None:
        return None
    return ref.directory / "manifest.json"


def _description_directory(task_spec: TaskSpec) -> str | None:
    ref = task_spec.description_ref
    if ref is None or ref.directory is None:
        return None
    return str(ref.directory)


def _description_seed_files(task_spec: TaskSpec) -> list[str]:
    ref = task_spec.description_ref
    if ref is None or ref.directory is None:
        return []
    return [
        path.name
        for path in sorted(ref.directory.glob("*.md"))
        if path.suffix == ".md" and not path.name.endswith((".zh.md", ".en.md"))
    ]


def _infer_family(task_name: str) -> str:
    if task_name.startswith("bbob_"):
        return "synthetic"
    if task_name.startswith("knob_"):
        return "dbtune"
    if task_name == "bboplace_bench":
        return "bboplace"
    return "scientific" if task_name.endswith("_demo") else "unknown"


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "BBOBenchmarkManifest",
    "DEFAULT_BBO_TOOL_NAMES",
    "build_compatible_BBO_manifest",
    "load_BBO_manifest",
]
