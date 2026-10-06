"""Composable prompt profiles for agentic benchmark methods."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping


@dataclass(frozen=True)
class PromptProfile:
    """Method-owned prompt additions layered over the benchmark protocol."""

    name: str
    protocol_instructions: str = ""
    round_instructions: str = ""

    def compose(self, base: str, *, stage: str) -> str:
        addition = self.protocol_instructions if stage == "protocol" else self.round_instructions
        addition = addition.strip()
        return base.rstrip() if not addition else f"{base.rstrip()}\n\n# Method profile: {self.name}\n{addition}"
    def compose_bundle(self, bundle: Any) -> Any:
        """Add this role's round instructions to a PromptBundle-like value."""
        return replace(bundle, user=self.compose(bundle.user, stage="round"))


@dataclass(frozen=True)
class WorkflowPromptProfile:
    """Complete role-to-prompt contract for a single- or multi-agent method."""

    name: str
    roles: Mapping[str, PromptProfile]

    def __post_init__(self) -> None:
        normalized = {str(role).strip(): profile for role, profile in self.roles.items()}
        if not normalized or any(not role for role in normalized):
            raise ValueError("WorkflowPromptProfile requires at least one named role.")
        object.__setattr__(self, "roles", normalized)

    def for_role(self, role: str) -> PromptProfile:
        try:
            return self.roles[role]
        except KeyError as exc:
            available = ", ".join(sorted(self.roles))
            raise ValueError(
                f"Workflow prompt profile {self.name!r} has no role {role!r}; available: {available}"
            ) from exc

    def validate_roles(self, roles: set[str]) -> None:
        configured = set(self.roles)
        if configured != roles:
            missing = sorted(roles - configured)
            extra = sorted(configured - roles)
            raise ValueError(
                f"Workflow prompt roles do not match method roles; missing={missing}, extra={extra}."
            )

    @classmethod
    def single(cls, name: str, role: str, profile: PromptProfile) -> "WorkflowPromptProfile":
        return cls(name=name, roles={role: profile})


PROMPT_PROFILES: dict[str, PromptProfile] = {
    "general_bbo": PromptProfile(name="general_bbo"),
    "gp_tpe_dynamic_selector": PromptProfile(
        name="gp_tpe_dynamic_selector",
        round_instructions=(
            "This is the Dynamic Selector condition. In this round, obtain exactly two "
            "unevaluated backend proposals from the same currently evaluated history:\n\n"
            "1. call `optimizer_suggest` exactly once with "
            "`{\"backend\":\"gp_ei\"}`;\n"
            "2. call `optimizer_suggest` exactly once with "
            "`{\"backend\":\"tpe\"}`.\n\n"
            "Compare the two proposals using only the task information, evaluated "
            "history, incumbent, and your continuing conversation. Submit exactly one "
            "of those two proposal configurations. The submitted `config` must be "
            "exactly equal, with the same parameter values and native JSON types, to "
            "either the GP-EI proposal or the TPE proposal. Do not edit, round, "
            "interpolate, combine, repair, or replace either proposal. The proposals "
            "are predictions, not observations; do not invent objective values. Do not "
            "call either backend more than once. Finish using the existing "
            "`final_candidate.json` raw-JSON handoff contract."
        ),
    ),
    "gp_tpe_free_form_agent": PromptProfile(
        name="gp_tpe_free_form_agent",
        round_instructions=(
            "This is the Free-form Agent condition. In this round, obtain exactly two "
            "unevaluated backend proposals from the same currently evaluated history:\n\n"
            "1. call `optimizer_suggest` exactly once with "
            "`{\"backend\":\"gp_ei\"}`;\n"
            "2. call `optimizer_suggest` exactly once with "
            "`{\"backend\":\"tpe\"}`.\n\n"
            "Use the two proposals as advice together with the task information, "
            "evaluated history, incumbent, and your continuing conversation. You may "
            "submit the GP-EI proposal unchanged, submit the TPE proposal unchanged, "
            "modify either proposal, combine ideas from both, or propose a different "
            "candidate. Whatever you choose must be one complete legal configuration "
            "in the exact search space and must not duplicate an evaluated "
            "configuration. The proposals are predictions, not observations; do not "
            "invent objective values. Do not call either backend more than once. Finish "
            "using the existing `final_candidate.json` raw-JSON handoff contract."
        ),
    ),
    "gp_only_free_form_agent": PromptProfile(
        name="gp_only_free_form_agent",
        round_instructions=(
            "This is the GP-only Free-form Agent condition. In this round, obtain "
            "exactly one unevaluated backend proposal from the currently evaluated "
            "history by calling `optimizer_suggest` exactly once with "
            "`{\"backend\":\"gp_ei\"}`.\n\n"
            "Use that proposal as advice together with the task information, evaluated "
            "history, incumbent, and your continuing conversation. You may submit the "
            "GP-EI proposal unchanged, modify it, or propose a different candidate. "
            "Whatever you choose must be one complete legal configuration in the exact "
            "search space and must not duplicate an evaluated configuration. The "
            "proposal is a prediction, not an observation; do not invent objective "
            "values. Do not call the backend more than once. Finish using the existing "
            "`final_candidate.json` raw-JSON handoff contract."
        ),
    ),
    "tpe_only_free_form_agent": PromptProfile(
        name="tpe_only_free_form_agent",
        round_instructions=(
            "This is the TPE-only Free-form Agent condition. In this round, obtain "
            "exactly one unevaluated backend proposal from the currently evaluated "
            "history by calling `optimizer_suggest` exactly once with "
            "`{\"backend\":\"tpe\"}`.\n\n"
            "Use that proposal as advice together with the task information, evaluated "
            "history, incumbent, and your continuing conversation. You may submit the "
            "TPE proposal unchanged, modify it, or propose a different candidate. "
            "Whatever you choose must be one complete legal configuration in the exact "
            "search space and must not duplicate an evaluated configuration. The "
            "proposal is a prediction, not an observation; do not invent objective "
            "values. Do not call the backend more than once. Finish using the existing "
            "`final_candidate.json` raw-JSON handoff contract."
        ),
    ),
    "native_harness": PromptProfile(
        name="native_harness",
        protocol_instructions=(
            "Use the harness's native reasoning, file, and shell capabilities. "
            "The benchmark protocol constrains only the final candidate handoff; "
            "it does not prescribe a search workflow."
        ),
        round_instructions=(
            "Choose the next candidate using the native harness and currently visible evidence."
        ),
    ),
    "native_tools": PromptProfile(
        name="native_tools",
        protocol_instructions=(
            "Use only the tools exposed by this run's ToolProfile. Tool availability is a "
            "capability boundary, not a mandatory sequence of calls."
        ),
        round_instructions=(
            "Select tool calls based on the decision needed this round, then commit exactly one candidate."
        ),
    ),
    "agentic_bo": PromptProfile(
        name="agentic_bo",
        protocol_instructions=(
            "Act as the decision maker in a surrogate-assisted optimization loop; the optimizer "
            "is an uncertainty-aware instrument, not an autopilot. Preserve the benchmark's "
            "candidate-submission protocol. Inspect optimizer state and trial evidence, optionally "
            "probe or reconfigure the optimizer, request or score proposals, then commit exactly "
            "one candidate. Never record predictions as observations, fabricate outcomes, or "
            "update optimizer history directly. Adapt the opening strategy to prior strength: test "
            "a concrete hypothesis when the task gives a strong prior, focus bounds when only a "
            "region is credible, and favor space-filling exploration when no useful prior exists. "
            "Use optimizer_suggest(bounds=...) for a one-off regional probe without changing policy. "
            "Use persistent optimizer_set_bounds only after strong prior or observed evidence; inspect "
            "optimizer_diagnostics first when the justification depends on the surrogate, and attach "
            "the evidence basis and provenance to the reconfiguration call. "
            "Use diagnostics to decide how much to trust the surrogate; do not discard informative "
            "task context merely because it conflicts with an uncertain posterior."
        ),
        round_instructions=(
            "Resolve the immediately previous real evaluation as supported, contradicted, inconclusive, "
            "or not applicable; state what changed in your belief. Before choosing the next evaluation, "
            "state your current belief and what this evaluation should learn. Make an "
            "explicit optimizer decision and record whether the committed candidate adopts, "
            "refines, overrides, or directly scores an optimizer proposal."
        ),
    ),
    "multi_backend_agentic_bo": PromptProfile(
        name="multi_backend_agentic_bo",
        protocol_instructions=(
            "Act as the decision maker in a backend-adaptive optimization loop. The only "
            "search backends are GP-EI and TPE, and neither is an autopilot. Treat "
            "assess_backend_suitability as decision-neutral evidence: GP CV R2 and ranking "
            "accuracy describe predictive suitability, while TPE elite separability describes "
            "quantile structure; the tool never recommends a backend. Select the backend "
            "yourself, call optimizer_suggest with that explicit backend, inspect its "
            "unevaluated candidate, validate the exact final candidate with its proposal_id, "
            "then finish by calling commit_candidate with the returned validation_token. "
            "commit_candidate is the authoritative terminal submission; do not generate a "
            "separate final-candidate JSON response. Never fabricate outcomes or update "
            "optimizer history directly."
        ),
        round_instructions=(
            "Resolve the immediately previous real evaluation, update your current landscape "
            "belief, call assess_backend_suitability, and explain which evidence supports this "
            "round's GP-EI or TPE choice. Record the backend actually passed to "
            "optimizer_suggest and whether the final candidate adopts, refines, or overrides "
            "that proposal in commit_candidate. You retain freedom to revise the optimizer "
            "candidate, but validate the exact revision before committing it."
        ),
    ),
}


def resolve_prompt_profile(profile: str | PromptProfile | None) -> PromptProfile:
    if isinstance(profile, PromptProfile):
        return profile
    name = "general_bbo" if profile is None else str(profile).strip().lower().replace("-", "_")
    try:
        return PROMPT_PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(PROMPT_PROFILES))
        raise ValueError(f"Unknown prompt profile {name!r}; available: {available}") from exc


PABLO_WORKFLOW_PROMPT_PROFILE = WorkflowPromptProfile(
    name="pablo",
    roles={
        "planner": PromptProfile(
            name="pablo.planner",
            round_instructions="Return bounded, distinct search tasks for the Worker roles.",
        ),
        "explorer": PromptProfile(
            name="pablo.explorer",
            round_instructions="Make one globally exploratory proposal from c_global evidence.",
        ),
        "worker": PromptProfile(
            name="pablo.worker",
            round_instructions="Refine only the assigned task and current seed for this feedback step.",
        ),
    },
)


def resolve_workflow_prompt_profile(
    profile: WorkflowPromptProfile | PromptProfile | str,
    *,
    roles: set[str],
    single_role: str | None = None,
) -> WorkflowPromptProfile:
    if isinstance(profile, WorkflowPromptProfile):
        resolved = profile
    else:
        atomic = resolve_prompt_profile(profile)
        if len(roles) != 1:
            raise ValueError(
                "A single PromptProfile cannot configure a multi-role workflow; "
                "provide WorkflowPromptProfile with one profile per role."
            )
        role = single_role or next(iter(roles))
        resolved = WorkflowPromptProfile.single(atomic.name, role, atomic)
    resolved.validate_roles(roles)
    return resolved
__all__ = ["PABLO_WORKFLOW_PROMPT_PROFILE", "PROMPT_PROFILES", "PromptProfile", "WorkflowPromptProfile", "resolve_prompt_profile", "resolve_workflow_prompt_profile"]
