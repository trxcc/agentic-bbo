"""Task-declared protocols shared by comparable optimization baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..core import TaskSpec, TrialSuggestion


BENCHMARK_PROTOCOL_KEY = "benchmark_protocol"
MODEL_CANDIDATE_POLICY = "min(5000,max(2048,200*d))"


@dataclass(frozen=True)
class FixedInitializationProtocol:
    """A task-owned, seed-specific initialization prefix."""

    name: str
    seed: int
    strategy: str
    source: str
    configurations: tuple[dict[str, Any], ...]

    def suggestion(self, index: int, *, algorithm: str) -> TrialSuggestion:
        if index < 0 or index >= len(self.configurations):
            raise IndexError(f"Initialization index {index} is outside 0..{len(self.configurations) - 1}.")
        return TrialSuggestion(
            config=dict(self.configurations[index]),
            metadata={
                "phase": "initialization",
                "benchmark_initialization": True,
                "initialization_protocol": self.name,
                "initialization_strategy": self.strategy,
                "initialization_source": self.source,
                "initialization_seed": self.seed,
                "initialization_index": index,
                "initialization_count": len(self.configurations),
                "baseline_algorithm": algorithm,
            },
        )


def dimension_scaled_candidate_budget(dimension: int) -> int:
    """Return the shared GP-EI/TuRBO candidate budget for a dimension."""

    if dimension <= 0:
        raise ValueError("Candidate-budget dimension must be positive.")
    return min(5000, max(2048, 200 * int(dimension)))


def resolve_model_candidate_budget(
    task_spec: TaskSpec,
    *,
    target: str,
    configured: int | None,
) -> tuple[int, str]:
    """Resolve a task override, an explicit algorithm value, or the shared rule."""

    protocol = _protocol_mapping(task_spec)
    candidate = protocol.get("candidate_budget")
    if candidate is not None:
        if not isinstance(candidate, Mapping):
            raise TypeError("Task benchmark protocol `candidate_budget` must be a mapping.")
        applies_to = candidate.get("applies_to", ())
        if not isinstance(applies_to, (list, tuple)):
            raise TypeError("Task candidate-budget `applies_to` must be a list or tuple.")
        if not applies_to or target in applies_to:
            value = int(candidate["value"])
            if value <= 0:
                raise ValueError("Task candidate-budget value must be positive.")
            return value, str(candidate.get("policy", "task_override"))

    if configured is not None:
        if int(configured) <= 0:
            raise ValueError("Configured candidate budget must be positive.")
        return int(configured), "algorithm_explicit"

    dimension = int(task_spec.metadata.get("dimension", len(task_spec.search_space)))
    return dimension_scaled_candidate_budget(dimension), MODEL_CANDIDATE_POLICY


def resolve_fixed_initialization(
    task_spec: TaskSpec,
    *,
    seed: int,
) -> FixedInitializationProtocol | None:
    """Load and validate a task's fixed initialization prefix, when present."""

    protocol = _protocol_mapping(task_spec)
    raw = protocol.get("initialization")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError("Task benchmark protocol `initialization` must be a mapping.")
    strategy = str(raw.get("strategy", ""))
    if strategy != "fixed_configurations":
        raise ValueError(f"Unsupported benchmark initialization strategy: {strategy!r}.")
    protocol_seed = int(raw["seed"])
    if protocol_seed != int(seed):
        raise ValueError(
            f"Task initialization seed {protocol_seed} does not match optimizer seed {int(seed)}."
        )
    raw_configs = raw.get("configurations")
    if not isinstance(raw_configs, (list, tuple)) or not raw_configs:
        raise ValueError("Fixed benchmark initialization requires a non-empty configuration list.")
    declared_count = int(raw.get("count", len(raw_configs)))
    if declared_count != len(raw_configs):
        raise ValueError(
            f"Fixed initialization declares {declared_count} points but contains {len(raw_configs)}."
        )

    configurations: list[dict[str, Any]] = []
    identities: set[str] = set()
    for index, raw_config in enumerate(raw_configs):
        if not isinstance(raw_config, Mapping):
            raise TypeError(f"Fixed initialization config {index} must be a mapping.")
        config = task_spec.search_space.coerce_config(dict(raw_config), use_defaults=False)
        identity = repr(sorted(config.items()))
        if identity in identities:
            raise ValueError(f"Fixed initialization contains duplicate config at index {index}.")
        identities.add(identity)
        configurations.append(config)

    return FixedInitializationProtocol(
        name=str(protocol.get("name", "task_fixed_initialization")),
        seed=protocol_seed,
        strategy=str(raw.get("sampling", "fixed_random")),
        source=str(raw.get("source", "task_metadata")),
        configurations=tuple(configurations),
    )


def _protocol_mapping(task_spec: TaskSpec) -> Mapping[str, Any]:
    protocol = task_spec.metadata.get(BENCHMARK_PROTOCOL_KEY, {})
    if protocol is None:
        return {}
    if not isinstance(protocol, Mapping):
        raise TypeError(f"Task metadata `{BENCHMARK_PROTOCOL_KEY}` must be a mapping.")
    return protocol


__all__ = [
    "BENCHMARK_PROTOCOL_KEY",
    "FixedInitializationProtocol",
    "MODEL_CANDIDATE_POLICY",
    "dimension_scaled_candidate_budget",
    "resolve_fixed_initialization",
    "resolve_model_candidate_budget",
]
