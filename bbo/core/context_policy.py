"""Agent-visible task identity and prior-disclosure policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .description import TaskDescriptionBundle
    from .task import TaskSpec


class ContextProfile(str, Enum):
    A0_P0 = "a0_p0"
    A1_P1 = "a1_p1"
    A2_P2 = "a2_p2"
    LEGACY_V6 = "legacy_v6"
    READABLE_DOMAIN_PRIOR = "readable_domain_prior"


class IdentityExposure(str, Enum):
    ANONYMOUS = "anonymous"
    FAMILY = "family"
    PUBLIC_INSTANCE = "public_instance"


class PriorLevel(str, Enum):
    P0 = "p0"
    P1 = "p1"
    P2 = "p2"


class OracleDisclosure(str, Enum):
    NONE = "none"
    SEMANTICS = "semantics"
    COMPONENTS = "components"


@dataclass(frozen=True)
class TaskContextPolicy:
    """One reproducible agent-visible information condition."""

    profile: ContextProfile
    identity_exposure: IdentityExposure
    prior_level: PriorLevel
    oracle_disclosure: OracleDisclosure
    parameter_aliasing: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile.value,
            "identity_exposure": self.identity_exposure.value,
            "prior_level": self.prior_level.value,
            "oracle_disclosure": self.oracle_disclosure.value,
            "parameter_aliasing": self.parameter_aliasing,
        }


CONTEXT_POLICIES: dict[ContextProfile, TaskContextPolicy] = {
    ContextProfile.READABLE_DOMAIN_PRIOR: TaskContextPolicy(
        ContextProfile.READABLE_DOMAIN_PRIOR,
        IdentityExposure.PUBLIC_INSTANCE,
        PriorLevel.P2,
        OracleDisclosure.COMPONENTS,
    ),
    ContextProfile.A0_P0: TaskContextPolicy(
        ContextProfile.A0_P0,
        IdentityExposure.ANONYMOUS,
        PriorLevel.P0,
        OracleDisclosure.SEMANTICS,
    ),
    ContextProfile.A1_P1: TaskContextPolicy(
        ContextProfile.A1_P1,
        IdentityExposure.FAMILY,
        PriorLevel.P1,
        OracleDisclosure.SEMANTICS,
    ),
    ContextProfile.A2_P2: TaskContextPolicy(
        ContextProfile.A2_P2,
        IdentityExposure.PUBLIC_INSTANCE,
        PriorLevel.P2,
        OracleDisclosure.COMPONENTS,
    ),
    ContextProfile.LEGACY_V6: TaskContextPolicy(
        ContextProfile.LEGACY_V6,
        IdentityExposure.PUBLIC_INSTANCE,
        PriorLevel.P2,
        OracleDisclosure.COMPONENTS,
    ),
}


def resolve_context_policy(value: str | ContextProfile | TaskContextPolicy | None) -> TaskContextPolicy:
    if isinstance(value, TaskContextPolicy):
        return value
    if isinstance(value, ContextProfile):
        profile = value
    else:
        normalized = "a0_p0" if value is None else str(value).strip().lower().replace("-", "_")
        try:
            profile = ContextProfile(normalized)
        except ValueError as exc:
            available = ", ".join(profile.value for profile in ContextProfile)
            raise ValueError(f"Unknown agent context profile {value!r}; choose from {available}.") from exc
    return CONTEXT_POLICIES[profile]


def task_context_family(task_spec: "TaskSpec", description: "TaskDescriptionBundle") -> str:
    """Return the stable prompt-family key without exposing it to A0 agents."""

    name = task_spec.name.lower()
    description_path = " ".join(str(doc.path).lower() for doc in description.all_docs)
    if name.startswith("bbob_") or "bbob_10d" in description_path:
        return "bbob"
    if name.startswith("hpo_bayesmark_"):
        return "hpo"
    if name.startswith("guacamol_"):
        return "guacamol"
    if name.startswith("knob_"):
        return "dbtune"
    if "bboplace" in name or "bboplace" in description_path or name.startswith(("adaptec", "bigblue", "superblue")):
        return "bboplace"
    return "generic"


def render_context_for_policy(
    *,
    task_spec: "TaskSpec",
    description: "TaskDescriptionBundle",
    policy: TaskContextPolicy,
    profile_root: Path | None = None,
) -> str:
    """Render only the information admitted by an explicit context policy."""

    if policy.profile == ContextProfile.READABLE_DOMAIN_PRIOR:
        from .readable_context import load_default_prior, render_readable_prior

        visible = load_default_prior(task_spec.name, metadata=task_spec.metadata)
        expected = {p["name"]: p for p in visible["parameters"]}
        actual = {p.name: p for p in task_spec.search_space}
        if set(expected) != set(actual):
            raise ValueError("Reviewed prior does not match the runtime parameter names.")
        for name, parameter in actual.items():
            for key in ("low", "high", "max_length"):
                if key in expected[name] and getattr(parameter, key, None) != expected[name][key]:
                    raise ValueError(f"Reviewed prior disagrees with runtime {name}.{key}.")
        objective = task_spec.primary_objective
        return render_readable_prior(
            visible, objective=f"{objective.direction.value} `{objective.name}`",
            total_evaluations=task_spec.max_evaluations,
        )
    if policy.profile == ContextProfile.LEGACY_V6:
        return description.rendered_context or "# Task context\n\nNo structured task description was supplied."
    root = profile_root or Path(__file__).resolve().parents[1] / "task_context_profiles"
    family = task_context_family(task_spec, description)
    family_root = root / family
    if not (family_root / "interface.md").exists():
        family_root = root / "generic"
    sections = [(family_root / "interface.md").read_text(encoding="utf-8").strip()]
    if policy.prior_level in {PriorLevel.P1, PriorLevel.P2}:
        sections.append((family_root / "domain.md").read_text(encoding="utf-8").strip())
    if policy.profile == ContextProfile.A2_P2:
        disclosed = [
            description.section_map[kind].strip()
            for kind in ("background", "goal", "prior_knowledge")
            if description.section_map.get(kind, "").strip()
        ]
        if disclosed:
            sections.extend(("# Public task identity and expert context", "\n\n".join(disclosed)))
    objective = task_spec.primary_objective
    dynamic = (
        "# Run contract\n\n"
        f"Primary objective: `{objective.name}` ({objective.direction.value}).\n\n"
        f"Total evaluation budget: {task_spec.max_evaluations}. "
        "Use only visible evaluated observations and submit one complete candidate per round."
    )
    sections.append(dynamic)
    return "\n\n".join(section for section in sections if section)


def context_fingerprint(rendered_context: str, policy: TaskContextPolicy) -> str:
    digest = sha256()
    digest.update(policy.profile.value.encode("utf-8"))
    digest.update(b"\0")
    digest.update(rendered_context.encode("utf-8"))
    return digest.hexdigest()[:16]


__all__ = [
    "CONTEXT_POLICIES",
    "ContextProfile",
    "IdentityExposure",
    "OracleDisclosure",
    "PriorLevel",
    "TaskContextPolicy",
    "resolve_context_policy",
    "context_fingerprint",
    "render_context_for_policy",
    "task_context_family",
]
