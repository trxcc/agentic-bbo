"""Raw workspace agent baseline with no benchmark decision tools."""

from __future__ import annotations

from typing import Any

from .general_agent import GeneralAgentBBOAlgorithm


def create_raw_agentic_bbo(**kwargs: Any) -> GeneralAgentBBOAlgorithm:
    """Create a native workspace agent that proposes directly from context."""

    configured = dict(kwargs)
    configured.setdefault("framework", "codex")
    configured.setdefault("algorithm_name", "raw_agentic_bbo")
    configured.setdefault("execution_backend", "isolated_docker")
    configured.setdefault("context_access", "on_demand")
    configured.setdefault("persist_session_across_rounds", True)
    configured.setdefault("reliable_runtime", True)
    configured.setdefault("max_tool_calls", 0)
    configured.setdefault("timeout_seconds", None)
    configured.setdefault("docker_image", "agentic-bbo-frontier-agent:v2")
    configured.setdefault("docker_cpus", 32.0)
    configured.setdefault("experiment_condition", "raw_agentic_bbo")
    configured.setdefault("tool_mode", "function_calling" if configured.get("context_access") == "on_demand" else "no_tool")
    configured.setdefault("prompt_profile", "general_bbo")
    configured.setdefault("context_profile", None)
    configured.setdefault("enable_memory", False)
    configured.setdefault("enable_code_interpreter", False)
    configured.setdefault("code_backend", "local_disabled")
    configured.setdefault("web_search_provider", "disabled")
    configured.setdefault("allow_fallback", False)
    configured.setdefault("enable_bbo_skills", False)
    return GeneralAgentBBOAlgorithm(**configured)


__all__ = ["create_raw_agentic_bbo"]
