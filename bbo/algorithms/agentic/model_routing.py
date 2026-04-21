"""Role-to-model routing for Pablo."""

from __future__ import annotations

from dataclasses import dataclass

ROLE_NAMES = ("planner", "explorer", "worker")


@dataclass(frozen=True)
class PabloModelRoutingConfig:
    """Model fields exposed through the CLI."""

    model: str = "gpt-4.1-mini"
    global_model: str | None = None
    worker_model: str | None = None
    planner_model: str | None = None
    explorer_model: str | None = None


def resolve_role_model(role: str, config: PabloModelRoutingConfig) -> str:
    if role == "planner":
        return config.planner_model or config.global_model or config.model
    if role == "explorer":
        return config.explorer_model or config.global_model or config.model
    if role == "worker":
        return config.worker_model or config.model
    raise ValueError(f"Unknown Pablo role `{role}`.")


def build_routing_table(config: PabloModelRoutingConfig) -> dict[str, str]:
    return {role: resolve_role_model(role, config) for role in ROLE_NAMES}


__all__ = ["PabloModelRoutingConfig", "ROLE_NAMES", "build_routing_table", "resolve_role_model"]
