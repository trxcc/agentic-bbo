"""Small composable declarations used by agentic optimization methods."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


PromptBuilder = Callable[[Any], str]
ResponseParser = Callable[[str], Any]


@dataclass(frozen=True)
class RoleSpec:
    """Runtime-independent declaration of one method role."""

    name: str
    model_route: str = "global"
    tool_profile: str | None = None
    prompt_profile: str | None = None
    prompt_builder: PromptBuilder | None = field(default=None, repr=False, compare=False)
    response_parser: ResponseParser | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class ToolProfile:
    """Named, reusable tool and optimizer-backend allowlist."""

    name: str
    tools: tuple[str, ...]
    optimizer_backends: tuple[str, ...] = ()
    require_optimizer_decision: bool = False
    require_hypothesis_lifecycle: bool = False
    require_evidence_bound_reconfiguration: bool = False

    def apply(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        resolved = dict(kwargs)
        resolved.setdefault("enabled_tool_names", self.tools)
        resolved.setdefault("optimizer_backend_allowlist", self.optimizer_backends)
        if self.require_optimizer_decision:
            resolved.setdefault("require_optimizer_decision_per_round", True)
        if self.require_hypothesis_lifecycle:
            resolved.setdefault("require_hypothesis_lifecycle_per_round", True)
        if self.require_evidence_bound_reconfiguration:
            resolved.setdefault("require_evidence_bound_reconfiguration", True)
        return resolved


@dataclass(frozen=True)
class SingleAgentMethod:
    """Composition recipe for methods driven by one general-agent role."""

    name: str
    role: RoleSpec
    tools: ToolProfile
    default_framework: str = "nanobot"
    prompt_profile: str = "general_bbo"

    def build(self, **kwargs: Any):
        from .general_agent import GeneralAgentBBOAlgorithm

        resolved = self.tools.apply(kwargs)
        resolved.setdefault("framework", self.default_framework)
        resolved.setdefault("algorithm_name", self.name)
        resolved.setdefault("role_name", self.role.name)
        resolved.setdefault("prompt_profile", self.prompt_profile)
        return GeneralAgentBBOAlgorithm(**resolved)
