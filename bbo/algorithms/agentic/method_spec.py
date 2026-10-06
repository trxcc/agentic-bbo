"""One workspace-agent method; tools are an explicit ablation setting."""
from dataclasses import dataclass
from .raw_agentic_bbo import create_raw_agentic_bbo
from .evented_algorithm import EventedAlgorithm


@dataclass(frozen=True)
class AgenticMethodSpec:
    name: str = "raw_agentic_bbo"
    factory: object = create_raw_agentic_bbo
    requires_agent_runtime: bool = True
    supports_optimizer_tools: bool = True
    supports_resume: bool = True


AGENTIC_METHOD_REGISTRY = {"raw_agentic_bbo": AgenticMethodSpec()}


def get_agentic_method_spec(name):
    return AGENTIC_METHOD_REGISTRY[name]


def create_agentic_method(name, *, run_dir=None, **kwargs):
    spec = get_agentic_method_spec(name)
    return EventedAlgorithm(spec.factory(run_dir=run_dir, **kwargs), method=name, run_dir=run_dir)
