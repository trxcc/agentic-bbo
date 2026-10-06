"""General coding-agent optimizer for black-box optimization tasks."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import random
import re
import shutil
import sys
import textwrap
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...core import (
    Incumbent,
    ObjectiveDirection,
    SearchSpace,
    TaskContextPolicy,
    context_fingerprint,
    load_BBO_manifest,
    render_context_for_policy,
    resolve_context_policy,
    TaskDescriptionBundle,
    TaskSpec,
    TrialObservation,
    TrialSuggestion,
)
from ...core.algo import Algorithm
from ...core.readable_context import default_context_profile
from ..benchmark_protocol import (
    FixedInitializationProtocol,
    resolve_fixed_initialization,
)
from .isolated_docker import DEFAULT_ISOLATED_DOCKER_CPUS, validate_docker_cpus
from .general_agent_engines import (
    AgentResult,
    AgentWorkCopy,
    GeneralAgentEngine,
    create_general_agent_engine,
    normalize_agent_framework,
)
from .serialization import append_jsonl, dump_json, stable_config_identity, to_jsonable
from .round_protocol import AgentRoundState
from .tools import (
    BBOMemoryStore,
    BBOToolCallLogger,
    BBOToolContext,
    BBOToolRegistry,
    BBOWebSourceLogger,
    CodeInterpreterTool,
    DisabledBBOCodeBackend,
    FetchURLTool,
    DockerBBOCodeBackend,
    MockBBOCodeBackend,
    OPTIMIZER_ACTION_TOOLS,
    OPTIMIZER_DECISION_TOOLS,
    SandboxFusionBBOCodeBackend,
    WebSearchTool,
    create_BBO_web_search_provider,
    create_optimizer_tools,
    create_core_BBO_tools,
)
from .tools.core_tools import (
    agent_visible_config,
    agent_visible_metadata,
    agent_visible_metrics,
    agent_visible_payload,
    sanitize_agent_context_payload,
)
from .agent_candidate import (
    GeneralAgentValidationError,
    ParsedAgentCandidate,
    _paired_xy_parameter_count,
    _retry_feedback_block,
    parse_agent_candidate_payload,
    search_space_schema,
)
from .agent_skill_audit import (
    BBO_NANOBOT_SKILL_NAMES,
    NANOBOT_BUILTIN_SKILL_NAMES,
    BBO_NUMERIC_EVIDENCE_TOOLS,
    BBO_REGION_EVIDENCE_TOOLS,
    BBO_REGION_JOINT_SUPPORT_TOOLS,
    MAX_UNSUPPORTED_MARGINAL_REGION_CHANGES,
    NON_PROPOSAL_BBO_SKILLS,
    SKILL_EVIDENCE_TOOL_GROUPS,
    SKILL_TO_SEARCH_INTENT,
    _NANOBOT_SKILL_NAME_RE,
    _search_action_metadata,
    _declared_agent_skill_names,
    _nanobot_read_skill_names_for_call,
    _bbo_workspace_tool_names_for_call,
    _bbo_tool_names_from_nanobot_session,
    _build_skill_usage_audit,
    _format_tool_group,
)

from .prompt_profiles import (
    PromptProfile,
    WorkflowPromptProfile,
    resolve_workflow_prompt_profile,
)

DEFAULT_AGENT_TIMEOUT_SECONDS = 300.0
DEFAULT_AGENT_HISTORY_LIMIT = 40
DEFAULT_AGENT_CANDIDATES_PER_CALL = 1
FINAL_CANDIDATE_FILENAME = "final_candidate.json"
AGENT_TOOL_MODES = ("function_calling", "workspace_json", "no_tool")
AGENT_EXECUTION_BACKENDS = ("direct_workspace", "sealed_docker_legacy", "isolated_docker")
AGENT_TOOL_MODE_CLI_CHOICES = (
    "function_calling",
    "workspace_json",
    "no_tool",
    "no-tool",
    "no_tools",
    "no-tools",
    "none",
    "disabled",
)
TWO_BACKEND_ROUTING_CONDITIONS = frozenset(
    {"gp_tpe_dynamic_selector", "gp_tpe_free_form_agent"}
)
SINGLE_BACKEND_FREE_FORM_CONDITIONS = {
    "gp_only_free_form_agent": "gp_ei",
    "tpe_only_free_form_agent": "tpe",
}
PROPOSAL_ROUTING_CONDITIONS = frozenset(
    {*TWO_BACKEND_ROUTING_CONDITIONS, *SINGLE_BACKEND_FREE_FORM_CONDITIONS}
)


def _call_id_scope(call_id: str | Sequence[str]) -> frozenset[str]:
    if isinstance(call_id, str):
        return frozenset((call_id,))
    return frozenset(str(value) for value in call_id)


@dataclass
class AgentCandidateEntry:
    """Queued candidate ready to be surfaced through ask()."""

    config: dict[str, Any]
    call_id: str
    candidate_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneralAgentConfig:
    """Configuration for the general-agent optimizer."""

    framework: str
    algorithm_name: str
    timeout_seconds: float | None = DEFAULT_AGENT_TIMEOUT_SECONDS
    max_retries: int = 1
    history_limit: int = DEFAULT_AGENT_HISTORY_LIMIT
    candidates_per_call: int = DEFAULT_AGENT_CANDIDATES_PER_CALL
    model: str | None = None
    provider: str | None = None
    api_base: str | None = None
    api_key_env: str | None = None
    executable: str | None = None
    initial_random: int = 0
    run_dir: Path | None = None
    resume: bool = False
    persist_session_across_rounds: bool = False
    reliable_runtime: bool = False
    tool_mode: str = "function_calling"
    context_access: str = "files"
    prompt_style: str = "workspace"
    role_name: str = "proposer"
    prompt_profile: PromptProfile = field(
        default_factory=lambda: resolve_workflow_prompt_profile(
            "general_bbo", roles={"proposer"}, single_role="proposer"
        ).for_role("proposer")
    )
    max_tool_calls: int = 16
    max_output_tokens: int | None = None
    thinking_mode: str | None = None
    enable_memory: bool = True
    execution_backend: str = "direct_workspace"
    enable_code_interpreter: bool = True
    docker_image: str = "agentic-bbo-analysis-sandbox:v1"
    docker_cpus: float = DEFAULT_ISOLATED_DOCKER_CPUS
    code_backend: str = "sandboxfusion"
    sandbox_fusion_base_url: str | None = None
    web_search_provider: str = "disabled"
    web_search_api_key_env: str | None = None
    allow_fallback: bool = True
    require_visible_cot: bool = False
    enable_bbo_skills: bool = False
    skill_paths: tuple[Path, ...] = field(default_factory=tuple)
    enabled_tool_names: tuple[str, ...] | None = None
    optimizer_backend_allowlist: tuple[str, ...] = field(default_factory=tuple)
    optimizer_max_calls_per_round: int = 3
    experiment_condition: str = "default"
    require_analysis_evidence_per_round: bool = False
    required_tool_names_per_round: tuple[str, ...] = field(default_factory=tuple)
    require_candidate_validation_per_round: bool = False
    require_optimizer_decision_per_round: bool = False
    require_hypothesis_lifecycle_per_round: bool = False
    require_evidence_bound_reconfiguration: bool = False
    context_policy: TaskContextPolicy = field(
        default_factory=lambda: resolve_context_policy("legacy_v6")
    )


class GeneralAgentBBOAlgorithm(Algorithm):
    """Ask/tell wrapper that lets an external general agent propose configs."""

    def __init__(
        self,
        *,
        framework: str,
        algorithm_name: str | None = None,
        engine: GeneralAgentEngine | None = None,
        timeout_seconds: float | None = DEFAULT_AGENT_TIMEOUT_SECONDS,
        max_retries: int = 1,
        history_limit: int = DEFAULT_AGENT_HISTORY_LIMIT,
        candidates_per_call: int = DEFAULT_AGENT_CANDIDATES_PER_CALL,
        model: str | None = None,
        provider: str | None = None,
        api_base: str | None = None,
        api_key_env: str | None = None,
        executable: str | None = None,
        initial_random: int = 0,
        run_dir: Path | str | None = None,
        resume: bool = False,
        persist_session_across_rounds: bool = False,
        reliable_runtime: bool = False,
        tool_mode: str = "function_calling",
        context_access: str = "files",
        prompt_style: str = "workspace",
        prompt_profile: str | PromptProfile | WorkflowPromptProfile | None = None,
        role_name: str = "proposer",
        max_tool_calls: int = 16,
        max_output_tokens: int | None = None,
        thinking_mode: str | None = None,
        enable_memory: bool = True,
        execution_backend: str | None = None,
        docker_image: str = "agentic-bbo-analysis-sandbox:v1",
        docker_cpus: float = DEFAULT_ISOLATED_DOCKER_CPUS,
        enable_code_interpreter: bool = True,
        code_backend: str = "sandboxfusion",
        sandbox_fusion_base_url: str | None = None,
        web_search_provider: str = "disabled",
        web_search_api_key_env: str | None = None,
        allow_fallback: bool = True,
        require_visible_cot: bool = False,
        enable_bbo_skills: bool = False,
        experiment_condition: str = "default",
        require_analysis_evidence_per_round: bool = False,
        required_tool_names_per_round: Sequence[str] = (),
        require_candidate_validation_per_round: bool = False,
        require_optimizer_decision_per_round: bool = False,
        require_hypothesis_lifecycle_per_round: bool = False,
        require_evidence_bound_reconfiguration: bool = False,
        skill_paths: str
        | Path
        | list[str | Path]
        | tuple[str | Path, ...]
        | None = None,
        enabled_tool_names: Sequence[str] | None = None,
        optimizer_backend_allowlist: Sequence[str] = (),
        optimizer_max_calls_per_round: int = 3,
        context_profile: str | TaskContextPolicy | None = None,
    ) -> None:
        self._requested_context_profile = context_profile
        docker_cpus = validate_docker_cpus(docker_cpus)
        if context_access not in {"files", "on_demand"}:
            raise ValueError("context_access must be files or on_demand.")
        if context_access == "on_demand" and (tool_mode != "function_calling" or algorithm_name != "raw_agentic_bbo"):
            raise ValueError("on_demand currently requires raw_agentic_bbo with function_calling (native workspace CLI is supported).")
        self._context_io_session = None
        self._context_io_documents = None
        normalized = normalize_agent_framework(framework)
        normalized_execution_backend = normalize_agent_execution_backend(
            execution_backend,
            framework=normalized,
            code_backend=code_backend,
            docker_image=docker_image,
        )
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive or None (unlimited).")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative.")
        if history_limit < 0:
            raise ValueError("history_limit must be non-negative.")
        if candidates_per_call <= 0:
            raise ValueError("candidates_per_call must be positive.")
        if initial_random < 0:
            raise ValueError("initial_random must be non-negative.")
        normalized_tool_mode = normalize_agent_tool_mode(tool_mode)
        normalized_prompt_style = prompt_style.strip().lower().replace("-", "_")
        normalized_role_name = str(role_name).strip()
        if not normalized_role_name:
            raise ValueError("role_name must be non-empty.")
        workflow_prompts = resolve_workflow_prompt_profile(
            "general_bbo" if prompt_profile is None else prompt_profile,
            roles={normalized_role_name},
            single_role=normalized_role_name,
        )
        if normalized_prompt_style != "workspace":
            raise ValueError("prompt_style must be `workspace`.")
        if max_tool_calls < 0:
            raise ValueError("max_tool_calls must be non-negative.")
        if max_output_tokens is not None and max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive when provided.")
        normalized_thinking_mode = (
            None if thinking_mode is None else str(thinking_mode).strip().lower()
        )
        if normalized_thinking_mode not in {None, "enabled", "disabled"}:
            raise ValueError("thinking_mode must be `enabled`, `disabled`, or None.")
        if optimizer_max_calls_per_round < 0:
            raise ValueError("optimizer_max_calls_per_round must be non-negative.")
        normalized_skill_paths = _normalize_skill_paths(skill_paths)
        normalized_tool_names = (
            None
            if enabled_tool_names is None
            else tuple(
                dict.fromkeys(
                    str(name).strip()
                    for name in enabled_tool_names
                    if str(name).strip()
                )
            )
        )
        normalized_optimizer_backends = tuple(
            dict.fromkeys(
                str(name).strip().lower().replace("-", "_")
                for name in optimizer_backend_allowlist
                if str(name).strip()
            )
        )
        normalized_required_tools = tuple(
            dict.fromkeys(
                str(name).strip()
                for name in required_tool_names_per_round
                if str(name).strip()
            )
        )
        if normalized_tool_names is not None:
            unavailable_required = sorted(
                set(normalized_required_tools) - set(normalized_tool_names)
            )
            if unavailable_required:
                raise ValueError(
                    "Required per-round tools must be enabled: "
                    + ", ".join(unavailable_required)
                )
        if normalized_tool_mode == "no_tool" and (
            enable_bbo_skills or normalized_skill_paths
        ):
            raise ValueError(
                "BBO skills require `tool_mode` to be `workspace_json` or `function_calling`."
            )
        if normalized_tool_mode == "no_tool" and (
            normalized_tool_names or normalized_optimizer_backends
        ):
            raise ValueError(
                "Tool allowlists require workspace_json or function_calling mode."
            )

        self.config = GeneralAgentConfig(
            framework=normalized,
            algorithm_name=algorithm_name or f"agentic_{normalized}",
            timeout_seconds=None if timeout_seconds is None else float(timeout_seconds),
            max_retries=int(max_retries),
            history_limit=int(history_limit),
            candidates_per_call=int(candidates_per_call),
            model=model,
            provider=provider,
            api_base=api_base,
            api_key_env=api_key_env,
            executable=executable,
            initial_random=int(initial_random),
            run_dir=None if run_dir is None else Path(run_dir),
            resume=bool(resume),
            persist_session_across_rounds=bool(persist_session_across_rounds),
            reliable_runtime=bool(reliable_runtime),
            tool_mode=normalized_tool_mode,
            context_access=context_access,
            prompt_style=normalized_prompt_style,
            role_name=normalized_role_name,
            prompt_profile=workflow_prompts.for_role(normalized_role_name),
            max_tool_calls=int(max_tool_calls),
            max_output_tokens=(
                None if max_output_tokens is None else int(max_output_tokens)
            ),
            thinking_mode=normalized_thinking_mode,
            docker_image=str(docker_image),
            docker_cpus=docker_cpus,
            enable_memory=bool(enable_memory),
            execution_backend=normalized_execution_backend,
            enable_code_interpreter=bool(enable_code_interpreter),
            code_backend=code_backend,
            sandbox_fusion_base_url=sandbox_fusion_base_url,
            web_search_provider=web_search_provider,
            experiment_condition=str(experiment_condition).strip().lower(),
            require_analysis_evidence_per_round=bool(
                require_analysis_evidence_per_round
            ),
            required_tool_names_per_round=normalized_required_tools,
            require_candidate_validation_per_round=bool(
                require_candidate_validation_per_round
            ),
            require_optimizer_decision_per_round=bool(
                require_optimizer_decision_per_round
            ),
            require_hypothesis_lifecycle_per_round=bool(
                require_hypothesis_lifecycle_per_round
            ),
            require_evidence_bound_reconfiguration=bool(
                require_evidence_bound_reconfiguration
            ),
            web_search_api_key_env=web_search_api_key_env,
            allow_fallback=bool(allow_fallback),
            require_visible_cot=bool(require_visible_cot),
            enable_bbo_skills=bool(enable_bbo_skills),
            skill_paths=normalized_skill_paths,
            enabled_tool_names=normalized_tool_names,
            optimizer_backend_allowlist=normalized_optimizer_backends,
            optimizer_max_calls_per_round=int(optimizer_max_calls_per_round),
            context_policy=resolve_context_policy(context_profile),
        )
        self._engine = engine or create_general_agent_engine(normalized)
        if (
            normalized in {"nanobot", "codex", "claude_code"}
            and self._engine.name == normalized
            and normalized_tool_mode == "workspace_json"
            and normalized_execution_backend != "direct_workspace"
        ):
            raise ValueError(
                "Strict native black-box runs reject `workspace_json`; workspace "
                "bridges expose optimizer-side runtime paths."
            )
        self._task_spec: TaskSpec | None = None
        self._description = TaskDescriptionBundle.empty(task_id="unknown")
        self._search_space: SearchSpace | None = None
        self._primary_name: str | None = None
        self._primary_direction = ObjectiveDirection.MINIMIZE
        self._seed = 0
        self._rng = random.Random(0)
        self._fixed_initialization: FixedInitializationProtocol | None = None
        self._history: list[TrialObservation] = []
        self._queue: list[AgentCandidateEntry] = []
        self._seen_config_ids: set[str] = set()
        self._best: Incumbent | None = None
        self._call_index = 0
        self._campaign_session_id = ""
        self._run_dir: Path | None = None
        self._workspace_dir: Path | None = None
        self._workspace_snapshot_dir: Path | None = None
        self._state_dir: Path | None = None
        self._memory_dir: Path | None = None
        self._work_copy: AgentWorkCopy | None = None
        self._manifest = None
        self._memory_store: BBOMemoryStore | None = None
        self._tool_registry: BBOToolRegistry | None = None
        self._artifacts: dict[str, str] = {}
        self._loaded_resume_snapshot: dict[str, Any] = {}
        self._agent_task_alias = "anonymous_task"
        self._rendered_agent_context = ""
        self._agent_context_fingerprint = ""

    @property
    def name(self) -> str:
        return self.config.algorithm_name

    @property
    def artifact_paths(self) -> dict[str, str]:
        return dict(self._artifacts)

    def setup(self, task_spec: TaskSpec, seed: int = 0, **kwargs: Any) -> None:
        self.config = replace(
            self.config, context_policy=resolve_context_policy(
                self._requested_context_profile or default_context_profile(task_spec.name, metadata=task_spec.metadata)
            ),
        )
        self._task_spec = task_spec
        self._search_space = task_spec.search_space
        self._primary_name = task_spec.primary_objective.name
        self._primary_direction = task_spec.primary_objective.direction
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        self._fixed_initialization = resolve_fixed_initialization(
            task_spec, seed=self._seed
        )
        description = kwargs.get("task_description")
        self._description = (
            description
            if isinstance(description, TaskDescriptionBundle)
            else TaskDescriptionBundle.empty(task_id=task_spec.name)
        )
        alias_digest = hashlib.sha256(
            f"{task_spec.name}\0{self._seed}\0{self._description.fingerprint}".encode("utf-8")
        ).hexdigest()[:12]
        self._agent_task_alias = f"task_{alias_digest}"
        self._prepare_agent_context(task_spec)

        self._run_dir = Path(
            kwargs.get("run_dir") or self.config.run_dir or Path.cwd()
        ).resolve()
        if self.config.context_policy.identity_exposure.value != "public_instance":
            opaque_id = hashlib.sha256(str(self._run_dir).encode("utf-8")).hexdigest()[:20]
            self._workspace_dir = Path("/tmp/agentic_bbo_workspaces") / opaque_id
            self._workspace_snapshot_dir = self._run_dir / "agent_workspace_snapshot"
        else:
            self._workspace_dir = self._run_dir / "agent_workspace"
            self._workspace_snapshot_dir = None
        identity_hidden = self.config.context_policy.identity_exposure.value != "public_instance"
        self._state_dir = (
            self._workspace_dir / ".agent_runtime" / "state"
            if identity_hidden and self.config.execution_backend != "isolated_docker"
            else self._run_dir / "agent_state"
        )
        if identity_hidden and self.config.execution_backend == "isolated_docker":
            # Keep authority outside /workspace without disclosing task-bearing
            # run-directory names through Linux bind-mount metadata.
            self._state_dir = Path("/tmp/agentic_bbo_states") / opaque_id
        self._memory_dir = (
            self._workspace_dir / ".agent_runtime" / "memory"
            if identity_hidden
            else self._run_dir / "agent_memory"
        )
        reasoning_dir = self._agent_reasoning_traces_dir
        log_dir = (
            self._workspace_dir / ".agent_runtime" / "llm_logs"
            if identity_hidden
            else self._run_dir / "llm_logs"
        )
        self._workspace_dir.mkdir(parents=True, exist_ok=True)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        reasoning_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        context_record_path = self._run_dir / "agent_context.json"
        dump_json(
            context_record_path,
            {
                "schema_version": "agent-context.v1",
                "task_alias": self._agent_task_alias,
                "context_policy": self.config.context_policy.to_dict(),
                "context_fingerprint": self._agent_context_fingerprint,
                "protocol_compliance": "pending",
            },
        )
        self._manifest = load_BBO_manifest(task_spec)
        self._memory_store = (
            BBOMemoryStore(self._agent_memory_path, self._agent_memory_summary_path)
            if self.config.enable_memory
            else None
        )
        self._tool_registry = self._build_tool_registry()

        config_path = self._build_framework_config(log_dir)
        self._work_copy = AgentWorkCopy(
            state_dir=self._state_dir,
            config_path=config_path,
            project_root=self._workspace_dir,
            workspace_root=self._workspace_dir,
            extra={
                "codex_config": self._codex_config(),
                "log_dir": log_dir,
                "reasoning_dir": reasoning_dir,
                "reasoning_metadata_path": self._agent_reasoning_metadata_path,
            },
        )
        artifacts = {
            "agent_context_json": str(context_record_path),
            "agent_workspace": str(self._workspace_dir),
            "agent_final_candidate_json": str(
                self._workspace_dir / FINAL_CANDIDATE_FILENAME
            ),
            "agent_state_dir": str(self._state_dir),
            "agent_calls_jsonl": str(self._agent_calls_path),
            "agent_prompts_jsonl": str(self._agent_prompts_path),
            "llm_logs_dir": str(log_dir),
            "agent_llm_logs_dir": str(log_dir),
            "agent_state_json": str(self._agent_state_path),
            "agent_history_jsonl": str(self._workspace_dir / "history.jsonl"),
            "agent_optimization_trace_jsonl": str(self._agent_optimization_trace_path),
            "agent_round_events_jsonl": str(self._agent_round_events_path),
            "agent_space_json": str(self._workspace_dir / "space.json"),
            "agent_task_md": str(self._workspace_dir / "task.md"),
            "agent_manifest_json": str(self._workspace_dir / "manifest.json"),
            "agent_sources_jsonl": str(self._agent_sources_path),
            "agent_memory_jsonl": str(self._agent_memory_path),
            "agent_memory_summary_json": str(self._agent_memory_summary_path),
            "agent_reasoning_traces_dir": str(reasoning_dir),
            "agent_reasoning_metadata_jsonl": str(self._agent_reasoning_metadata_path),
        }
        if self._workspace_snapshot_dir is not None:
            artifacts["agent_workspace_snapshot"] = str(self._workspace_snapshot_dir)
        if self._agent_tools_enabled():
            artifacts.update(
                {
                    "agent_tool_specs_json": str(self._agent_tool_specs_path),
                    "agent_workspace_tool_py": str(self._workspace_dir / "bbo_tool.py"),
                    "agent_workspace_bbo_tools_py": str(
                        self._workspace_dir / "bbo_tools.py"
                    ),
                    "agent_workspace_tool_config_json": str(
                        self._workspace_dir / "bbo_tool_config.json"
                    ),
                    "agent_workspace_tools_md": str(self._workspace_dir / "TOOLS.md"),
                    "agent_workspace_python_environment_md": str(
                        self._workspace_dir / "python_environment.md"
                    ),
                    "agent_tool_calls_jsonl": str(self._agent_tool_calls_path),
                }
            )
            if self._optimizer_suggestion_enabled():
                artifacts.update(
                    {
                        "agent_workspace_gp_example_py": str(
                            self._workspace_dir
                            / "examples"
                            / "gp_expected_improvement.py"
                        ),
                        "agent_workspace_gp_entrypoint_py": str(
                            self._workspace_dir / "gp_expected_improvement.py"
                        ),
                    }
                )
        if self._agent_skills_enabled():
            artifacts["agent_workspace_skills_dir"] = str(
                self._workspace_dir / "skills"
            )
        self._artifacts = artifacts
        log_paths = [
            self._agent_calls_path,
            self._agent_prompts_path,
            self._agent_sources_path,
            self._agent_optimization_trace_path,
            self._agent_round_events_path,
            self._agent_reasoning_metadata_path,
        ]
        if self._agent_tools_enabled():
            log_paths.append(self._agent_tool_calls_path)
        for path in log_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
        if self._tool_registry is not None:
            dump_json(self._agent_tool_specs_path, {"tools": self._agent_tool_specs()})

        self._history = []
        self._queue = []
        self._seen_config_ids = set()
        self._best = None
        self._call_index = 0
        self._campaign_session_id = ""
        self._loaded_resume_snapshot = self._load_resume_snapshot()
        if (self._loaded_resume_snapshot and
                self._loaded_resume_snapshot.get("native_round_guard_version") != self._native_round_guard_version()):
            raise ValueError("Cannot resume with a different native round guard policy; use the frozen runtime for historical runs.")
        if self._loaded_resume_snapshot and bool(
            self._loaded_resume_snapshot.get("reliable_runtime", False)
        ) != self.config.reliable_runtime:
            raise ValueError("Cannot resume with a different runtime reliability policy.")
        if (self._loaded_resume_snapshot
                and self.config.execution_backend == "isolated_docker"
                and self._loaded_resume_snapshot.get("execution_backend") == "isolated_docker"):
            # Pre-quota snapshots came from the fixed 2-CPU launcher.
            previous_cpus = self._loaded_resume_snapshot.get("docker_cpus", 2.0)
            if float(previous_cpus) != self.config.docker_cpus:
                raise ValueError("Cannot resume an isolated agent run with a different CPU quota.")
        resumed_session_policy = self._loaded_resume_snapshot.get(
            "persist_session_across_rounds"
        )
        if (
            resumed_session_policy is not None
            and bool(resumed_session_policy)
            != self.config.persist_session_across_rounds
        ):
            raise ValueError(
                "Cannot resume an agent run with a different cross-round session policy."
            )
        if self.config.persist_session_across_rounds:
            self._campaign_session_id = str(
                self._loaded_resume_snapshot.get("campaign_session_id") or ""
            ).strip()
        resumed_profile = self._loaded_resume_snapshot.get("context_profile")
        if resumed_profile and resumed_profile != self.config.context_policy.profile.value:
            raise ValueError(
                "Cannot resume an agent run with a different context profile: "
                f"stored={resumed_profile!r}, requested={self.config.context_policy.profile.value!r}."
            )
        resumed_fingerprint = self._loaded_resume_snapshot.get("context_fingerprint")
        if self._loaded_resume_snapshot and self._loaded_resume_snapshot.get("context_access", "files") != self.config.context_access:
            raise ValueError("Cannot resume with a different context_access protocol.")
        if resumed_fingerprint and resumed_fingerprint != self._agent_context_fingerprint:
            previous_budget = self._loaded_resume_snapshot.get("context_max_evaluations")
            budget_extension = False
            if isinstance(previous_budget, int) and task_spec.max_evaluations >= previous_budget:
                previous_context = render_context_for_policy(
                    task_spec=replace(task_spec, max_evaluations=previous_budget),
                    description=self._description, policy=self.config.context_policy,
                )
                budget_extension = context_fingerprint(
                    previous_context, self.config.context_policy
                ) == resumed_fingerprint
            if not budget_extension:
                raise ValueError("Cannot resume because the rendered agent context fingerprint changed.")
        self._restore_call_index()
        self._write_workspace_context()
        self._sync_workspace_snapshot()
        self._persist_state()

    def _prepare_agent_context(self, task_spec: TaskSpec) -> None:
        """Build all visible context before fingerprints, snapshots and resume checks."""
        self._rendered_agent_context = render_context_for_policy(
            task_spec=task_spec,
            description=self._description,
            policy=self.config.context_policy,
        )
        self._agent_context_fingerprint = context_fingerprint(
            self._rendered_agent_context, self.config.context_policy
        )
        if self.config.context_access == "on_demand":
            from .tools.context_io import context_documents
            from ...core.readable_context import load_default_prior

            prior = None
            if self.config.context_policy.profile.value == "readable_domain_prior":
                prior = load_default_prior(task_spec.name, metadata=task_spec.metadata)
            self._context_io_documents = context_documents(task_spec, self._rendered_agent_context, prior)
            self._agent_context_fingerprint = context_fingerprint(
                json.dumps(self._context_io_documents, sort_keys=True), self.config.context_policy
            )


    def ask(self) -> TrialSuggestion:
        self._require_ready()
        if self._fixed_initialization is not None and len(self._history) < len(
            self._fixed_initialization.configurations
        ):
            suggestion = self._fixed_initialization.suggestion(
                len(self._history), algorithm=self.name
            )
            suggestion.metadata.update(
                {
                    "agent_framework": self.config.framework,
                    "agent_source": "benchmark_initialization",
                }
            )
            self._seen_config_ids.add(stable_config_identity(suggestion.config))
            self._persist_state()
            return suggestion
        if len(self._history) < self.config.initial_random:
            return self._initial_random_suggestion()
        if not self._queue:
            self._fill_queue_from_agent()
        if not self._queue:
            raise RuntimeError(
                f"{self.name} could not produce any valid candidate configurations."
            )
        entry = self._queue.pop(0)
        metadata = {
            "agent_framework": self.config.framework,
            "agent_engine": self._engine.name,
            "agent_call_id": entry.call_id,
            "agent_candidate_index": entry.candidate_index,
            "agent_model": self.config.model,
            "agent_provider": self.config.provider,
            **entry.metadata,
        }
        self._persist_state()
        return TrialSuggestion(config=dict(entry.config), metadata=metadata)

    def tell(self, observation: TrialObservation) -> None:
        self._ingest_observation(observation)
        self._write_workspace_context()
        self._persist_state()

    def replay(self, history: list[TrialObservation]) -> None:
        self._require_ready()
        self._history = []
        self._queue = []
        self._seen_config_ids = set()
        self._best = None
        for observation in history:
            self._ingest_observation(observation, replay=True)
        self._restore_queue_from_snapshot()
        self._write_workspace_context()
        self._persist_state()

    def incumbents(self) -> list[Incumbent]:
        return [self._best] if self._best is not None else []

    def _initial_random_suggestion(self) -> TrialSuggestion:
        search_space = self._require_search_space()
        fixed = resolve_fixed_initialization(self._require_task_spec(), seed=self._seed)
        fixed_index = len(self._history)
        if fixed is not None and fixed_index < len(fixed.configurations):
            suggestion = fixed.suggestion(fixed_index, algorithm=self.name)
            config = dict(suggestion.config)
            identity = stable_config_identity(config)
            if identity in self._seen_config_ids:
                raise RuntimeError(
                    f"Fixed initialization point {fixed_index} duplicates observed history."
                )
            self._seen_config_ids.add(identity)
            self._persist_state()
            return TrialSuggestion(
                config=config,
                metadata={
                    **dict(suggestion.metadata),
                    "agent_framework": self.config.framework,
                    "agent_source": "benchmark_initialization",
                    **_search_action_metadata(
                        {
                            "search_intent": "initialization",
                            "change_summary": f"task-owned {fixed.strategy} initialization point",
                        },
                        source="benchmark_initialization",
                    ),
                },
            )
        for _ in range(100):
            config = search_space.sample(self._rng)
            identity = stable_config_identity(config)
            if identity not in self._seen_config_ids:
                self._seen_config_ids.add(identity)
                self._persist_state()
                return TrialSuggestion(
                    config=config,
                    metadata={
                        "agent_framework": self.config.framework,
                        "agent_source": "initial_random",
                        **_search_action_metadata(
                            {
                                "search_intent": "initialization",
                                "change_summary": "framework initial random sample",
                            },
                            source="initial_random",
                        ),
                    },
                )
        config = search_space.sample(self._rng)
        self._seen_config_ids.add(stable_config_identity(config))
        self._persist_state()
        return TrialSuggestion(
            config=config,
            metadata={
                "agent_framework": self.config.framework,
                "agent_source": "initial_random",
                **_search_action_metadata(
                    {
                        "search_intent": "initialization",
                        "change_summary": "framework initial random sample",
                    },
                    source="initial_random",
                ),
            },
        )

    def _fill_queue_from_agent(self) -> None:
        search_space = self._require_search_space()
        if self.config.context_access == "on_demand":
            from .tools.context_io import ContextIOSession

            self._context_io_session = ContextIOSession(
                self._require_task_spec(), self._history, self._context_io_documents,
                self._state_dir, self._workspace_dir, self._agent_context_fingerprint,
            )
            committed = self._context_io_session.committed_payload
            if committed is not None:
                self._enqueue_candidates("submission_recovery", parse_agent_candidate_payload(json.dumps(committed), search_space))
                self._persist_state()
                return
        last_error: str | None = None
        reasoning_requirement_failed = False
        round_call_ids: list[str] = []
        round_attempts: list[dict[str, str]] = []
        routing_proposal_backends: set[str] | None = (
            set()
            if self.config.experiment_condition in PROPOSAL_ROUTING_CONDITIONS
            else None
        )
        original_prompt: str | None = None
        session_id = (
            self._campaign_session_id
            if self.config.persist_session_across_rounds
            else ""
        )
        boundary_failed = False
        round_protocol = (
            AgentRoundState(
                round_id=f"agent_round_{self._call_index:05d}",
                history=self._history,
                base_specs=self._agent_tool_specs(),
                event_path=self._agent_round_events_path,
            )
            if self._controlled_round_protocol_enabled()
            else None
        )
        for attempt_index in range(self.config.max_retries + 1):
            self._write_workspace_context()
            call_id = f"agent_call_{self._call_index:05d}"
            self._call_index += 1
            # Reserve the identifier before invoking an external agent. A failed
            # ask() must not reuse it after --resume while append-only logs still
            # contain tool calls from the failed attempt.
            self._persist_state()
            round_call_ids.append(call_id)
            self._clear_workspace_candidate_file()
            if original_prompt is None:
                original_prompt = self.config.prompt_profile.compose(
                    self._build_agent_prompt(
                        call_id=call_id,
                        attempt_index=attempt_index,
                        last_error=None,
                    ),
                    stage="round",
                )
            prompt = (
                original_prompt
                if attempt_index == 0
                else self._retry_context_prompt(
                    original_prompt=original_prompt,
                    round_attempts=round_attempts,
                    resumed=bool(session_id),
                )
            )
            retry_instruction = self._retry_instruction(
                last_error=last_error,
                round_call_ids=round_call_ids[:-1],
                call_id=call_id,
            )
            logged_prompt = prompt
            if retry_instruction:
                logged_prompt = f"{logged_prompt.rstrip()}\n\n{retry_instruction}"
            append_jsonl(
                self._agent_prompts_path,
                {
                    "call_id": call_id,
                    "attempt_index": attempt_index,
                    "prompt": logged_prompt,
                    "resumed_session_id": session_id or None,
                    "timestamp": time.time(),
                },
            )
            requested_session_id = session_id
            result = self._run_engine(
                prompt,
                call_id=call_id,
                session_id=session_id,
                final_instruction=retry_instruction,
                round_protocol=round_protocol,
                routing_proposal_backends=routing_proposal_backends,
            )
            self._sync_workspace_snapshot()
            self._audit_workspace_boundary(result)
            result_session_id = self._session_id_from_result(result)
            if result_session_id:
                session_id = result_session_id
                if self.config.persist_session_across_rounds:
                    self._campaign_session_id = result_session_id
                    self._persist_state()
            round_attempts.append(
                {
                    "call_id": call_id,
                    "answer": result.answer,
                    "engine_error": result.error or "",
                    "rejection": "",
                }
            )
            call_record = {
                "call_id": call_id,
                "attempt_index": attempt_index,
                "framework": self.config.framework,
                "engine": self._engine.name,
                "status": result.status,
                "returncode": result.returncode,
                "error": result.error,
                "answer": result.answer,
                "llm_log": result.llm_log,
                "timestamp": time.time(),
                "session_scope": (
                    "campaign"
                    if self.config.persist_session_across_rounds
                    else "round"
                ),
                "requested_session_id": requested_session_id or None,
                "result_session_id": result_session_id or None,
            }
            if result.status != "success":
                if self._context_io_session is not None and self._context_io_session.committed_payload is not None:
                    parsed = parse_agent_candidate_payload(json.dumps(self._context_io_session.committed_payload), search_space)
                    accepted = self._enqueue_candidates(call_id, parsed)
                    call_record.update(candidate_source="submit_candidate", accepted_candidates=len(accepted),
                                       submission_recovered_after_transport_error=True)
                    append_jsonl(self._agent_calls_path, call_record)
                    self._persist_state()
                    return
                append_jsonl(self._agent_calls_path, call_record)
                if self.config.reliable_runtime:
                    from .runtime_reliability import AgentTransportError

                    guard_failure = ((result.llm_log or {}).get("nativeRoundGuard") or {}).get("failure") or {}
                    if guard_failure.get("error_type") in {"NativeBudgetExceeded", "InvalidNativeResponse"}:
                        from .native_round_guard import AgentRoundPolicyError
                        raise AgentRoundPolicyError(guard_failure["message"])

                    # The proxy owns bounded HTTP retries. Any remaining engine
                    # failure is operational, not evidence of an invalid candidate.
                    raise AgentTransportError(
                        f"Agent invocation {call_id} failed; stop for recovery: "
                        f"{result.error or result.status}"
                    )
                if result.returncode == -3:
                    boundary_failed = True
                last_error = result.error or result.answer or result.status
                round_attempts[-1]["rejection"] = last_error
                continue
            reasoning_metadata = self._reasoning_metadata_for_call(call_id)
            if reasoning_metadata:
                call_record["reasoning"] = reasoning_metadata
            if (
                self.config.require_visible_cot
                and not self._call_has_visible_reasoning(call_id)
            ):
                call_record[
                    "reasoning_error"
                ] = "Required visible CoT was not captured for this agent call."
                append_jsonl(self._agent_calls_path, call_record)
                last_error = call_record["reasoning_error"]
                round_attempts[-1]["rejection"] = last_error
                reasoning_requirement_failed = True
                continue
            parsed: list[ParsedAgentCandidate] | None = None
            workspace_error: str | None = None
            if self.config.context_access == "on_demand":
                committed = self._context_io_session.committed_payload
                if committed is None:
                    last_error = "No candidate was submitted. Call submit_candidate directly with config or a workspace JSON path; a final text or file alone does not submit."
                    call_record["validation_error"] = last_error
                    append_jsonl(self._agent_calls_path, call_record)
                    round_attempts[-1]["rejection"] = last_error
                    continue
                parsed = parse_agent_candidate_payload(json.dumps(committed), search_space)
                call_record["candidate_source"] = "submit_candidate"
            if round_protocol is not None and round_protocol.committed_payload is not None:
                parsed = parse_agent_candidate_payload(
                    json.dumps(round_protocol.committed_payload), search_space
                )
                call_record["candidate_source"] = "commit_candidate"
                call_record["protocol_version"] = 2
                call_record["round_id"] = round_protocol.round_id
            workspace_candidate = (
                None
                if parsed is not None
                else self._read_workspace_candidate_file(call_id)
            )
            if workspace_candidate is not None:
                workspace_candidate_path, workspace_candidate_text = workspace_candidate
                try:
                    parsed = parse_agent_candidate_payload(
                        workspace_candidate_text, search_space
                    )
                except GeneralAgentValidationError as workspace_exc:
                    workspace_error = str(workspace_exc)
                else:
                    call_record["candidate_source"] = "workspace_candidate_file"
                    call_record["candidate_file"] = workspace_candidate_path
            if parsed is None:
                try:
                    parsed = parse_agent_candidate_payload(result.answer, search_space)
                except GeneralAgentValidationError as response_exc:
                    parsed = self._recover_successfully_validated_candidate(
                        round_call_ids, search_space
                    )
                    if parsed is not None:
                        call_record["candidate_source"] = "validated_tool_recovery"
                        call_record["agent_response_error"] = str(response_exc)
                        if workspace_error is not None:
                            call_record["workspace_candidate_error"] = workspace_error
                    else:
                        if workspace_error is None:
                            validation_error = str(response_exc)
                        else:
                            validation_error = (
                                f"workspace candidate file was invalid: {workspace_error}; "
                                f"agent response was also invalid: {response_exc}"
                            )
                        call_record["validation_error"] = validation_error
                        append_jsonl(self._agent_calls_path, call_record)
                        last_error = validation_error
                        round_attempts[-1]["rejection"] = last_error
                        continue
                else:
                    call_record["candidate_source"] = "agent_response"
                if workspace_error is not None:
                    call_record["workspace_candidate_error"] = workspace_error
            skill_read_error = self._declared_skill_read_error(call_id, parsed)
            condition_tool_error = self._condition_tool_usage_error(
                round_call_ids, parsed
            )
            if condition_tool_error:
                recovered = self._recover_successfully_validated_candidate(
                    round_call_ids, search_space
                )
                if recovered is not None:
                    recovered_condition_error = self._condition_tool_usage_error(
                        round_call_ids, recovered
                    )
                    if recovered_condition_error is None or self._recoverable_optimizer_overage(
                        round_call_ids, recovered, recovered_condition_error
                    ):
                        parsed = recovered
                        call_record["candidate_source"] = (
                            "validated_tool_recovery"
                        )
                        call_record["agent_response_error"] = condition_tool_error
                        if recovered_condition_error is not None:
                            call_record["recovered_contract_warning"] = recovered_condition_error
                        condition_tool_error = None
                    else:
                        condition_tool_error = recovered_condition_error
            if condition_tool_error:
                call_record["validation_error"] = condition_tool_error
                append_jsonl(self._agent_calls_path, call_record)
                last_error = condition_tool_error
                round_attempts[-1]["rejection"] = last_error
                continue
            if skill_read_error:
                call_record["validation_error"] = skill_read_error
                append_jsonl(self._agent_calls_path, call_record)
                last_error = skill_read_error
                round_attempts[-1]["rejection"] = last_error
                continue
            skill_tool_error = self._declared_skill_tool_usage_error(call_id, parsed)
            if skill_tool_error:
                call_record["validation_error"] = skill_tool_error
                append_jsonl(self._agent_calls_path, call_record)
                last_error = skill_tool_error
                round_attempts[-1]["rejection"] = last_error
                continue

            accepted_actions = self._enqueue_candidates(call_id, parsed)
            accepted = len(accepted_actions)
            call_record["accepted_candidates"] = accepted
            if accepted_actions:
                call_record["accepted_search_actions"] = accepted_actions
            append_jsonl(self._agent_calls_path, call_record)
            self._persist_state()
            if accepted > 0:
                return
            last_error = "Agent returned only duplicate candidate configurations."
            round_attempts[-1]["rejection"] = last_error

        if boundary_failed:
            raise RuntimeError(
                f"{self.name} refused to run without its black-box boundary: {last_error}"
            )

        if reasoning_requirement_failed:
            raise RuntimeError(
                f"{self.name} failed the visible CoT requirement: {last_error}"
            )

        if not self.config.allow_fallback:
            raise RuntimeError(
                f"{self.name} failed to produce a valid candidate and fallback is disabled: {last_error}"
            )

        fallback = self._fallback_candidate(last_error or "agent_failed")
        if fallback is not None:
            self._queue.append(fallback)
            self._persist_state()
            append_jsonl(
                self._agent_calls_path,
                {
                    "call_id": fallback.call_id,
                    "framework": self.config.framework,
                    "engine": self._engine.name,
                    "status": "fallback",
                    "reason": last_error,
                    "accepted_candidates": 1,
                    "timestamp": time.time(),
                },
            )
            return
        raise RuntimeError(
            f"{self.name} failed to produce a valid candidate after retries: {last_error}"
        )

    def _recoverable_optimizer_overage(
        self,
        call_ids: Sequence[str],
        candidates: Sequence[ParsedAgentCandidate],
        error: str,
    ) -> bool:
        """Accept a validated final candidate after harmless duplicate optimizer calls.

        A model can repeat an optimizer probe while recovering a response. Once the
        exact submitted candidate has passed host validation, accepting that candidate
        preserves the experiment's no-fallback contract without allowing unvalidated
        or otherwise non-compliant proposals through.
        """
        if "Optimizer candidate-decision call count exceeded the per-round cap:" not in error:
            return False
        if not candidates or not self.config.require_candidate_validation_per_round:
            return False
        validated = {
            stable_config_identity(config)
            for config in self._successfully_validated_configs_for_call(call_ids)
        }
        return all(stable_config_identity(candidate.config) in validated for candidate in candidates)

    def _run_engine(
        self,
        prompt: str,
        *,
        call_id: str,
        session_id: str = "",
        final_instruction: str | None = None,
        round_protocol: AgentRoundState | None = None,
        routing_proposal_backends: set[str] | None = None,
    ) -> AgentResult:
        self._require_ready()
        assert self._work_copy is not None
        tools = None
        tool_executor = None
        if self.config.tool_mode == "function_calling":
            registry = self._require_tool_registry()
            context = self._build_tool_context()
            # Pass the same policy-narrowed schemas that are written to the
            # workspace.  The raw registry schema may advertise arguments
            # (for example bounds/options/seed) that an experiment arm rejects.
            tools = self._agent_tool_specs()

            async def _execute_tool(
                tool_name: str,
                arguments: dict[str, Any],
                tool_call_id: str | None = None,
            ) -> str:
                async def _base_execute(
                    name: str,
                    payload: dict[str, Any],
                    nested_tool_call_id: str | None,
                ) -> str:
                    return await registry.execute_tool(
                        name,
                        payload,
                        context,
                        call_id=call_id,
                        tool_call_id=nested_tool_call_id,
                    )

                if round_protocol is not None:
                    return await round_protocol.execute(
                        _base_execute, tool_name, arguments, tool_call_id
                    )
                if routing_proposal_backends is not None:
                    return await self._execute_routing_guarded_tool(
                        _base_execute,
                        routing_proposal_backends,
                        tool_name,
                        arguments,
                        tool_call_id,
                        call_id=call_id,
                    )
                return await _base_execute(tool_name, arguments, tool_call_id)

            tool_executor = _execute_tool
            if round_protocol is not None:
                self._work_copy.extra["tool_spec_provider"] = round_protocol.tool_specs
                self._work_copy.extra[
                    "tool_transport_spec_provider"
                ] = round_protocol.transport_tool_specs
        coro = self._engine.run_agent(
            session_id,
            prompt,
            self._work_copy,
            agent_id="bbo",
            timeout=self.config.timeout_seconds,
            tools=tools,
            tool_executor=tool_executor,
            max_tool_calls=self.config.max_tool_calls,
            extra_env=self._agent_call_env(call_id),
            final_instruction=final_instruction,
        )
        return _run_coro_sync(coro)

    async def _execute_routing_guarded_tool(
        self,
        executor: Any,
        proposed_backends: set[str],
        tool_name: str,
        arguments: dict[str, Any],
        tool_call_id: str | None,
        *,
        call_id: str,
    ) -> str:
        """Reject a second successful proposal from one backend in a routing round.

        The guard spans retry turns in the same optimization round.  It reserves a
        backend before awaiting execution, so parallel duplicate tool calls cannot
        both consume optimizer work.  A failed first call releases the reservation
        and remains retryable.
        """

        routing_condition = (
            self.config.experiment_condition in PROPOSAL_ROUTING_CONDITIONS
        )
        backend = str(arguments.get("backend", "")).strip()
        if (
            not routing_condition
            or tool_name != "optimizer_suggest"
            or backend not in {"gp_ei", "tpe"}
        ):
            return await executor(tool_name, arguments, tool_call_id)

        if backend in proposed_backends:
            result = {
                "ok": False,
                "error": "duplicate_backend_proposal",
                "message": (
                    f"A successful {backend} proposal already exists in this "
                    "optimization round. Reuse it; do not call this backend again."
                ),
            }
            append_jsonl(
                self._agent_tool_calls_path,
                {
                    "timestamp": time.time(),
                    "call_id": call_id,
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "arguments": dict(arguments),
                    "success": False,
                    "duration_ms": 0.0,
                    "result_preview": json.dumps(result, ensure_ascii=False, sort_keys=True),
                    "protocol_rejection": "duplicate_backend_proposal",
                },
            )
            return json.dumps(result, ensure_ascii=False, sort_keys=True)

        proposed_backends.add(backend)
        try:
            response = await executor(tool_name, arguments, tool_call_id)
        except Exception:
            proposed_backends.discard(backend)
            raise
        try:
            decoded = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            proposed_backends.discard(backend)
        else:
            if not isinstance(decoded, Mapping) or decoded.get("ok") is not True:
                proposed_backends.discard(backend)
        return response

    def _controlled_round_protocol_enabled(self) -> bool:
        return self.config.algorithm_name in {
            "multi_backend_agent",
            "multi_backend_agentic_bo",
            "workflow_multi_backend_agentic_bo",
        }

    @staticmethod
    def _session_id_from_result(result: AgentResult) -> str:
        log = result.llm_log
        if not isinstance(log, Mapping):
            return ""
        nested = log.get("codexTurn")
        if isinstance(nested, Mapping):
            log = nested
        value = log.get("sessionId")
        return str(value).strip() if value else ""

    @staticmethod
    def _retry_context_prompt(
        *,
        original_prompt: str,
        round_attempts: Sequence[Mapping[str, str]],
        resumed: bool,
    ) -> str:
        if resumed:
            return (
                "Continue the same optimization round. The complete task prompt, prior "
                "reasoning, tool calls, tool outputs, and previous answer remain in the "
                "conversation above and are authoritative."
            )
        sections = [original_prompt, "\n\n# Retry conversation history"]
        for attempt in round_attempts:
            sections.append(f"\n\nAssistant ({attempt.get('call_id', 'prior attempt')}):")
            sections.append(attempt.get("answer") or "[No final answer was produced.]")
            engine_error = attempt.get("engine_error")
            if engine_error:
                sections.append(f"\nEngine result: {engine_error}")
            rejection = attempt.get("rejection")
            if rejection:
                sections.append(f"\nHarness feedback: {rejection}")
        return "".join(sections)

    def _retry_instruction(
        self,
        *,
        last_error: str | None,
        round_call_ids: Sequence[str],
        call_id: str,
    ) -> str | None:
        if not last_error:
            return None
        if self.config.context_access == "on_demand":
            return f"Submission correction: {str(last_error)[:800]}\nUse submit_candidate directly with a corrected config or workspace JSON path. write_candidate is optional. Successful submission ends this round."
        if self.config.reliable_runtime:
            return (
                f"CANDIDATE CORRECTION (current call id: {call_id})\n"
                f"The submitted response was rejected: {str(last_error)[:800]}\n"
                "Read the current workspace inputs and available history as needed. "
                "Native shell and the condition's original tools remain available. "
                "Return one complete, legal, non-duplicate candidate in the required "
                "JSON format. Do not assume a valid candidate already exists."
            )
        error_text = " ".join(str(last_error).split())
        if len(error_text) > 800:
            error_text = error_text[:800].rstrip() + "..."
        completed = self._successful_tool_names_for_call(round_call_ids)
        completed_text = ", ".join(sorted(completed)) or "none"
        preamble = (
            "RETRY CORRECTION (this is the final instruction for the continuing "
            f"conversation; current call id: {call_id})\n"
            f"The previous attempt was not accepted: {error_text}\n"
            "Keep the complete task prompt and all prior conversation and tool results "
            "as authoritative context. Continue normal reasoning, but do not repeat an "
            "expensive action that already succeeded in this optimization round.\n"
            f"Successful BBO tools already completed this round: {completed_text}.\n"
        )

        if self.config.experiment_condition in TWO_BACKEND_ROUTING_CONDITIONS:
            proposals = self._successful_optimizer_proposals_for_call(round_call_ids)
            counts = {
                backend: sum(item[0] == backend for item in proposals)
                for backend in ("gp_ei", "tpe")
            }
            missing_backends = [
                backend for backend, count in counts.items() if count == 0
            ]
            if len(missing_backends) == 2:
                return (
                    preamble
                    + "Do not merely announce or describe the calls: execute both CLI "
                    "calls in this turn. Next missing actions: call `optimizer_suggest` "
                    "exactly once with "
                    "`{\"backend\":\"gp_ei\"}`, then exactly once with "
                    "`{\"backend\":\"tpe\"}`. Compare those two proposals and "
                    "immediately return the one final candidate required by this "
                    "condition. Do not call either backend more than once."
                )
            if len(missing_backends) == 1:
                missing_backend = missing_backends[0]
                return (
                    preamble
                    + "Do not merely announce or describe the call: execute the CLI call "
                    "in this turn. Next missing action: call `optimizer_suggest` exactly "
                    "once with "
                    f"`{{\"backend\":\"{missing_backend}\"}}`. Reuse the other backend "
                    "proposal already present above, then immediately return the one "
                    "final candidate required by this condition."
                )
        single_backend = SINGLE_BACKEND_FREE_FORM_CONDITIONS.get(
            self.config.experiment_condition
        )
        if single_backend is not None:
            proposals = self._successful_optimizer_proposals_for_call(round_call_ids)
            backend_count = sum(item[0] == single_backend for item in proposals)
            if backend_count == 0:
                return (
                    preamble
                    + "Do not merely announce or describe the call: execute the CLI call "
                    "in this turn. Next missing action: call `optimizer_suggest` exactly "
                    f"once with `{{\"backend\":\"{single_backend}\"}}`. Use that "
                    "proposal as advice, then immediately return the one final candidate "
                    "required by this condition."
                )

        missing_required = [
            name
            for name in self.config.required_tool_names_per_round
            if name not in completed
        ]
        if missing_required:
            tool_name = missing_required[0]
            arguments = "{}"
            return (
                preamble
                + f"Next missing action: call `{tool_name}` once with arguments "
                f"`{arguments}`. After it succeeds, continue directly to the next "
                "missing optimizer/validation action; do not call it a second time."
            )

        decision_count = self._successful_tool_call_count_for_call(
            round_call_ids, OPTIMIZER_DECISION_TOOLS
        )
        if self.config.require_optimizer_decision_per_round and decision_count == 0:
            allowed = list(self.config.optimizer_backend_allowlist)
            choices = " or ".join(f'{{\"backend\":\"{name}\"}}' for name in allowed)
            return (
                preamble
                + "Next missing action: use the suitability result already present above, "
                "choose one backend yourself, and call `optimizer_suggest` exactly once "
                f"with only the backend argument ({choices}). Do not pass bounds, options, "
                "q, or seed. Then validate that proposal and finish the candidate payload."
            )

        validated = self._successfully_validated_configs_for_call(round_call_ids)
        if self.config.require_candidate_validation_per_round and not validated:
            optimizer_candidate = self._latest_optimizer_candidate(round_call_ids)
            candidate_text = (
                "the exact optimizer candidate already returned above"
                if optimizer_candidate is None
                else "this exact optimizer candidate after applying the declared final precision: "
                + json.dumps(
                    agent_visible_config(optimizer_candidate),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return (
                preamble
                + "Next missing action: call `validate_candidate` once on "
                + candidate_text
                + ". The CLI arguments must use the outer key `candidate`, exactly as "
                + "`{\"candidate\":<config>}`; do not use `config` as the outer key. "
                + "If it is valid, immediately write `final_candidate.json` and return "
                "the matching candidate payload; do not rerun suitability or the optimizer."
            )

        return (
            preamble
            + "All required tool actions are already complete. Reuse the existing validated "
            "candidate and correct only the rejected response or metadata issue above. Write "
            "the exact payload to `final_candidate.json` and return the same payload.\n"
            "Return only raw JSON; do not include explanations or Markdown."
        )

    def _agent_call_env(self, call_id: str) -> dict[str, str]:
        env = {
            "BBO_AGENT_CALL_ID": call_id,
            "BBO_AGENT_MODEL_REQUESTED": self.config.model or "",
            "BBO_AGENT_PROVIDER": self.config.provider or "",
            "BBO_AGENT_REQUIRE_VISIBLE_COT": "1"
            if self.config.require_visible_cot
            else "0",
        }
        if self.config.framework == "nanobot":
            env["BBO_NANOBOT_REASONING_DIR"] = str(self._agent_reasoning_traces_dir)
            env["BBO_NANOBOT_REASONING_METADATA_PATH"] = str(
                self._agent_reasoning_metadata_path
            )
        return env

    def _reasoning_metadata_for_call(self, call_id: str) -> dict[str, Any] | None:
        path = self._agent_reasoning_metadata_path
        if not path.exists():
            return None
        latest: dict[str, Any] | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("call_id") == call_id:
                latest = record
        return latest

    def _call_has_visible_reasoning(self, call_id: str) -> bool:
        record = self._reasoning_metadata_for_call(call_id)
        return bool(record and record.get("reasoning_visible"))

    def _build_tool_registry(self) -> BBOToolRegistry | None:
        if not self._agent_tools_enabled():
            return None
        if self.config.context_access == "on_demand":
            from .tools.context_io import create_context_io_tools, SessionBoundOptimizerTool

            session = lambda: self._context_io_session
            tools = create_context_io_tools(session)
            required = {tool.name for tool in tools}
            if self.config.enabled_tool_names is not None:
                enabled = set(self.config.enabled_tool_names)
                if not required <= enabled:
                    raise ValueError("On-demand menus must retain all six context/submission tools.")
                available = {tool.name: tool for tool in create_optimizer_tools()}
                extra = enabled - required
                unknown = extra - available.keys()
                if unknown:
                    raise ValueError(f"Unknown on-demand optimizer tools: {sorted(unknown)!r}.")
                if extra and not self.config.optimizer_backend_allowlist:
                    raise ValueError("On-demand optimizer tools require an explicit backend allowlist.")
                tools.extend(SessionBoundOptimizerTool(available[name], session) for name in sorted(extra))
            return BBOToolRegistry(tools, logger=BBOToolCallLogger(self._agent_tool_calls_path))
        tools = create_core_BBO_tools(enable_memory=self.config.enable_memory)
        if self.config.optimizer_backend_allowlist or (
            self.config.enabled_tool_names is not None
            and bool(set(self.config.enabled_tool_names) & OPTIMIZER_ACTION_TOOLS)
        ):
            tools.extend(create_optimizer_tools())
        if self.config.enable_code_interpreter:
            tools.append(CodeInterpreterTool())
        if self._web_tools_enabled():
            tools.extend([WebSearchTool(), FetchURLTool()])
        if self.config.enabled_tool_names is not None:
            by_name = {tool.name: tool for tool in tools}
            unknown = sorted(set(self.config.enabled_tool_names) - set(by_name))
            if unknown:
                raise ValueError(f"Unknown enabled BBO tools: {unknown!r}.")
            tools = [by_name[name] for name in self.config.enabled_tool_names]
        return BBOToolRegistry(
            tools, logger=BBOToolCallLogger(self._agent_tool_calls_path)
        )

    def _agent_tool_specs(self) -> list[dict[str, Any]]:
        """Return tool specs narrowed to the optimizer backends exposed in this arm."""

        if self._tool_registry is None:
            return []
        specs = copy.deepcopy(self._tool_registry.get_tool_specs())
        task_spec = self._require_task_spec()
        optimizer_policy = task_spec.metadata.get("optimizer_tool_policy")
        optimizer_policy = (
            dict(optimizer_policy) if isinstance(optimizer_policy, Mapping) else {}
        )
        allowed = list(self.config.optimizer_backend_allowlist)
        if not allowed:
            return specs
        for spec in specs:
            function = spec.get("function") if isinstance(spec, Mapping) else None
            if not isinstance(function, dict):
                continue
            parameters = function.get("parameters")
            properties = (
                parameters.get("properties") if isinstance(parameters, dict) else None
            )
            if not isinstance(properties, dict):
                continue
            if function.get("name") in {"optimizer_suggest", "optimizer_set_backend"}:
                backend = properties.get("backend")
                if isinstance(backend, dict):
                    backend["enum"] = allowed
                    if (
                        function.get("name") == "optimizer_suggest"
                        and optimizer_policy.get("minimal_suggest") is True
                    ):
                        parameters["properties"] = {"backend": backend}
                        parameters["required"] = ["backend"]
                        parameters["additionalProperties"] = False
                        function["description"] = (
                            "Return one candidate using exactly one explicit "
                            "allowlisted backend. Pass only the backend field."
                        )
            elif function.get("name") == "optimizer_portfolio_suggest":
                backends = properties.get("backends")
                items = backends.get("items") if isinstance(backends, dict) else None
                if isinstance(items, dict):
                    items["enum"] = allowed
        return specs

    def _web_tools_enabled(self) -> bool:
        provider = self.config.web_search_provider.strip().lower().replace("-", "_")
        return provider not in {"", "disabled", "none", "off", "false"}

    def _build_tool_context(self) -> BBOToolContext:
        self._require_ready()
        assert self._workspace_dir is not None
        assert self._state_dir is not None
        assert self._manifest is not None
        return BBOToolContext(
            task_spec=self._require_task_spec(),
            description=self._description,
            manifest=self._manifest,
            workspace_dir=self._workspace_dir,
            state_dir=self._state_dir,
            history=self._history,
            incumbent=self._best,
            memory_store=self._memory_store,
            code_backend=self._build_code_backend(),
            web_search_provider=self._build_web_search_provider(),
            source_logger=BBOWebSourceLogger(self._agent_sources_path),
            seed=self._seed,
            optimizer_backend_allowlist=self.config.optimizer_backend_allowlist,
            agent_task_id=self._agent_visible_task_name(),
            agent_description_sections={"context": self._rendered_agent_context},
            agent_manifest_payload=self._agent_workspace_manifest_payload(),
        )

    def _build_code_backend(self) -> object:
        backend = self.config.code_backend.strip().lower().replace("-", "_")
        if backend == "mock":
            return MockBBOCodeBackend()
        if backend in {"disabled", "local_disabled", "none"}:
            return DisabledBBOCodeBackend()
        if backend in {"docker", "restricted_docker", "local_docker"}:
            if self._workspace_dir is None:
                raise RuntimeError(
                    "Agent workspace must exist before building Docker code backend."
                )
            return DockerBBOCodeBackend(
                workspace_dir=self._workspace_dir, image=self.config.docker_image
            )
        if backend == "sandboxfusion":
            base_url = self.config.sandbox_fusion_base_url or os.environ.get(
                "SANDBOX_FUSION_BASE_URL"
            )
            if not base_url:
                return DisabledBBOCodeBackend()
            return SandboxFusionBBOCodeBackend(base_url=base_url)
        raise ValueError(f"Unknown BBO code backend `{self.config.code_backend}`.")

    def _build_web_search_provider(self) -> object:
        return create_BBO_web_search_provider(
            self.config.web_search_provider,
            api_key_env=self.config.web_search_api_key_env,
        )

    def _require_tool_registry(self) -> BBOToolRegistry:
        if self._tool_registry is None:
            raise RuntimeError("BBO tool registry is not initialized.")
        return self._tool_registry

    def _enqueue_candidates(
        self, call_id: str, candidates: list[ParsedAgentCandidate]
    ) -> list[dict[str, Any]]:
        accepted_actions: list[dict[str, Any]] = []
        for candidate in candidates:
            identity = stable_config_identity(candidate.config)
            if identity in self._seen_config_ids:
                continue
            self._seen_config_ids.add(identity)
            metadata = _search_action_metadata(
                candidate.metadata,
                call_id=call_id,
                candidate_index=candidate.candidate_index,
            )
            metadata = self._metadata_with_skill_audit(
                call_id=call_id, candidate=candidate, metadata=metadata
            )
            self._queue.append(
                AgentCandidateEntry(
                    config=dict(candidate.config),
                    call_id=call_id,
                    candidate_index=candidate.candidate_index,
                    metadata=metadata,
                )
            )
            action = metadata.get("search_action")
            if isinstance(action, dict):
                accepted_actions.append(agent_visible_payload(action))
            break
        return accepted_actions

    def _metadata_with_skill_audit(
        self,
        *,
        call_id: str,
        candidate: ParsedAgentCandidate,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if not (self.config.framework == "nanobot" and self._agent_skills_enabled()):
            return metadata
        assert self._run_dir is not None
        read_skills = _nanobot_read_skill_names_for_call(
            self._run_dir / "llm_logs", call_id
        )
        used_tools = _bbo_workspace_tool_names_for_call(
            self._agent_tool_calls_path, call_id
        )
        if not used_tools:
            used_tools = _bbo_tool_names_from_nanobot_session(
                self._run_dir / "llm_logs", call_id
            )
        audit = _build_skill_usage_audit(
            metadata=metadata,
            config=candidate.config,
            read_skills=read_skills,
            used_tools=used_tools,
            history=self._history,
            incumbent=self._best,
        )
        enriched = dict(metadata)
        action = dict(enriched.get("search_action") or {})
        action["skill_audit"] = audit
        enriched["search_action"] = action
        enriched["skill_audit"] = audit
        return enriched

    def _condition_tool_usage_error(
        self,
        call_id: str | Sequence[str],
        candidates: Sequence[ParsedAgentCandidate] | None = None,
    ) -> str | None:
        """Return a retryable error when this condition's round contract was not met."""

        if not (
            self.config.require_hypothesis_lifecycle_per_round
            or self.config.require_evidence_bound_reconfiguration
            or self.config.require_analysis_evidence_per_round
            or self.config.required_tool_names_per_round
            or self.config.require_candidate_validation_per_round
            or self.config.require_optimizer_decision_per_round
        ):
            return None
        lifecycle_error = self._agentic_deliberation_contract_error(call_id, candidates)
        if lifecycle_error is not None:
            return lifecycle_error
        used = self._successful_tool_names_for_call(call_id)
        missing_required = [
            name
            for name in self.config.required_tool_names_per_round
            if name not in used
        ]
        if missing_required:
            return (
                f"Condition {self.config.experiment_condition!r} requires successful "
                "per-round tool calls: " + ", ".join(missing_required) + "."
            )
        if self.config.experiment_condition in PROPOSAL_ROUTING_CONDITIONS:
            routing_error = self._gp_tpe_routing_contract_error(call_id, candidates)
            if routing_error is not None:
                return routing_error
        if self.config.require_analysis_evidence_per_round and not (
            used & BBO_NUMERIC_EVIDENCE_TOOLS
        ):
            return (
                f"Condition {self.config.experiment_condition!r} requires at least one "
                "successful analysis/evidence tool call in every optimization round."
            )
        if self.config.experiment_condition in {
            "t2_guided_analysis",
            "t3_agentic_search",
            "t4_soft_portfolio",
        } and not (used & BBO_REGION_EVIDENCE_TOOLS):
            return (
                f"Condition {self.config.experiment_condition!r} requires at least one "
                "successful search-strategy analysis call in every optimization round: "
                "analyze_search_strategy."
            )
        if (
            candidates
            and self._best is not None
            and (used & BBO_REGION_EVIDENCE_TOOLS)
            and not (used & BBO_REGION_JOINT_SUPPORT_TOOLS)
        ):
            incumbent = agent_visible_config(self._best.config)
            for candidate in candidates:
                visible_candidate = agent_visible_config(candidate.config)
                changed = [
                    name
                    for name in visible_candidate
                    if visible_candidate.get(name) != incumbent.get(name)
                ]
                if len(changed) > MAX_UNSUPPORTED_MARGINAL_REGION_CHANGES:
                    return (
                        "Marginal region evidence may directly change at most "
                        f"{MAX_UNSUPPORTED_MARGINAL_REGION_CHANGES} parameters from the "
                        f"incumbent, but this candidate changes {len(changed)}: "
                        + ", ".join(changed)
                        + ". Keep context-only parameters at incumbent values, or obtain "
                        "successful joint support from analyze_parameter_interactions, "
                        "score_virtual_candidates, optimizer_suggest, "
                        "optimizer_portfolio_suggest, or optimizer_score."
                    )
        validated_configs = self._successfully_validated_configs_for_call(call_id)
        if self.config.require_candidate_validation_per_round and not validated_configs:
            return (
                f"Condition {self.config.experiment_condition!r} requires successful "
                "validation of the final formatted candidate in every optimization round."
            )
        if self.config.require_candidate_validation_per_round and candidates:
            validated_identities = {
                stable_config_identity(config) for config in validated_configs
            }
            unvalidated = [
                candidate
                for candidate in candidates
                if stable_config_identity(candidate.config) not in validated_identities
            ]
            if unvalidated:
                return (
                    f"Condition {self.config.experiment_condition!r} requires successful "
                    "validation of the exact final formatted candidate, but the returned "
                    "candidate does not match any successful validation this round."
                )
        decision_count = self._successful_tool_call_count_for_call(
            call_id, OPTIMIZER_DECISION_TOOLS
        )
        if self.config.require_optimizer_decision_per_round and decision_count == 0:
            return (
                f"Condition {self.config.experiment_condition!r} requires at least one "
                "successful optimizer_suggest, optimizer_portfolio_suggest, or "
                "optimizer_score candidate-decision call "
                "in every optimization round."
            )
        if decision_count > self.config.optimizer_max_calls_per_round:
            return (
                "Optimizer candidate-decision call count exceeded the per-round cap: "
                f"{decision_count}/{self.config.optimizer_max_calls_per_round}."
            )
        return None

    def _gp_tpe_routing_contract_error(
        self,
        call_id: str | Sequence[str],
        candidates: Sequence[ParsedAgentCandidate] | None,
    ) -> str | None:
        """Enforce the condition's proposal contract without adding capabilities."""

        proposals = self._successful_optimizer_proposals_for_call(call_id)
        condition = self.config.experiment_condition
        required_backends = (
            ("gp_ei", "tpe")
            if condition in TWO_BACKEND_ROUTING_CONDITIONS
            else (SINGLE_BACKEND_FREE_FORM_CONDITIONS[condition],)
        )
        by_backend = {
            backend: [config for item_backend, config in proposals if item_backend == backend]
            for backend in required_backends
        }
        bad_counts = {
            backend: len(configs)
            for backend, configs in by_backend.items()
            if len(configs) != 1
        }
        if bad_counts:
            rendered = ", ".join(
                f"{backend}={count}" for backend, count in sorted(bad_counts.items())
            )
            requirement = (
                "exactly one successful GP-EI proposal and exactly one successful "
                "TPE proposal"
                if len(required_backends) == 2
                else f"exactly one successful {required_backends[0]} proposal"
            )
            return (
                f"This routing condition requires {requirement} per round; "
                f"observed {rendered}."
            )
        if condition != "gp_tpe_dynamic_selector":
            return None
        if not candidates or len(candidates) != 1:
            return "Dynamic Selector requires exactly one submitted candidate."
        submitted = stable_config_identity(candidates[0].config)
        allowed = {
            stable_config_identity(
                self._require_search_space().coerce_config(config, use_defaults=False)
            )
            for configs in by_backend.values()
            for config in configs
        }
        if submitted not in allowed:
            return (
                "Dynamic Selector must submit one backend proposal exactly; the returned "
                "configuration matches neither this round's GP-EI proposal nor its TPE "
                "proposal. Do not edit, round, combine, or repair the proposals."
            )
        return None

    def _successful_optimizer_proposals_for_call(
        self, call_id: str | Sequence[str]
    ) -> list[tuple[str, dict[str, Any]]]:
        path = self._agent_tool_calls_path
        if not path.exists():
            return []
        scoped_call_ids = _call_id_scope(call_id)
        proposals: list[tuple[str, dict[str, Any]]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            record_call_id = record.get("agent_call_id") or record.get("call_id")
            if (
                record_call_id not in scoped_call_ids
                or record.get("success") is not True
                or record.get("tool_name") != "optimizer_suggest"
            ):
                continue
            arguments = record.get("arguments")
            backend = (
                str(arguments.get("backend", "")).strip()
                if isinstance(arguments, Mapping)
                else ""
            )
            structured = record.get("result_data")
            candidate = (
                structured.get("candidate")
                if isinstance(structured, Mapping)
                else None
            )
            if backend in {"gp_ei", "tpe"} and isinstance(candidate, Mapping):
                proposals.append((backend, dict(candidate)))
        return proposals

    def _agentic_deliberation_contract_error(
        self,
        call_id: str | Sequence[str],
        candidates: Sequence[ParsedAgentCandidate] | None,
    ) -> str | None:
        """Validate Sara-style belief updates and evidence-backed persistent bounds."""

        if self.config.require_hypothesis_lifecycle_per_round and candidates:
            latest_trial_id = (
                None if not self._history else self._history[-1].suggestion.trial_id
            )
            for candidate in candidates:
                action = dict((candidate.metadata or {}).get("search_action") or {})
                for field_name in ("belief", "expected_information"):
                    value = action.get(field_name)
                    if not isinstance(value, str) or not value.strip():
                        return (
                            f"Agentic BO requires non-empty search_action.{field_name}."
                        )
                update = action.get("hypothesis_update")
                if not isinstance(update, Mapping):
                    return "Agentic BO requires a structured search_action.hypothesis_update."
                if update.get("status") not in {
                    "supported",
                    "contradicted",
                    "inconclusive",
                    "not_applicable",
                }:
                    return "hypothesis_update.status must be supported, contradicted, inconclusive, or not_applicable."
                if (
                    latest_trial_id is not None
                    and update.get("evidence_trial_id") != latest_trial_id
                ):
                    return f"hypothesis_update.evidence_trial_id must resolve latest real trial {latest_trial_id}."
                reason = update.get("reason")
                if not isinstance(reason, str) or not reason.strip():
                    return "hypothesis_update.reason must explain what the latest observation changed."

        if not self.config.require_evidence_bound_reconfiguration:
            return None
        scoped = _call_id_scope(call_id)
        records: list[dict[str, Any]] = []
        if self._agent_tool_calls_path.exists():
            for line in self._agent_tool_calls_path.read_text(
                encoding="utf-8"
            ).splitlines():
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                record_call_id = record.get("agent_call_id") or record.get("call_id")
                if record_call_id in scoped and record.get("success") is True:
                    records.append(record)
        diagnostics_seen: set[str] = set()
        for record in records:
            if record.get("tool_name") == "optimizer_diagnostics":
                tool_call_id = record.get("tool_call_id")
                if isinstance(tool_call_id, str):
                    diagnostics_seen.add(tool_call_id)
                continue
            if record.get("tool_name") != "optimizer_set_bounds":
                continue
            evidence = dict((record.get("arguments") or {}).get("evidence") or {})
            basis = evidence.get("basis")
            if basis not in {"strong_prior", "trial_evidence", "surrogate_diagnostics"}:
                return "optimizer_set_bounds requires evidence.basis: strong_prior, trial_evidence, or surrogate_diagnostics."
            reason = evidence.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                return "optimizer_set_bounds requires a non-empty evidence.reason."
            if basis == "trial_evidence" and not evidence.get("reference_trials"):
                return "trial_evidence bounds require evidence.reference_trials."
            if basis == "surrogate_diagnostics":
                diagnostic_call_id = evidence.get("diagnostic_call_id")
                if diagnostic_call_id not in diagnostics_seen:
                    return "surrogate_diagnostics bounds require the referenced successful optimizer_diagnostics call earlier in this round."
        return None

    def _recover_successfully_validated_candidate(
        self, call_id: str | Sequence[str], search_space: SearchSpace
    ) -> list[ParsedAgentCandidate] | None:
        """Recover only a same-round candidate that the validator accepted.

        Models occasionally stop after a successful ``optimizer_suggest`` call or
        return non-JSON prose instead of the final protocol payload. In strict
        runs we may repair the protocol, but we must never invent a candidate:
        replay the exact optimizer proposal through ``validate_candidate`` and
        accept it only when that validation reports ``valid=true``.
        """

        path = self._agent_tool_calls_path
        if not path.exists():
            return None
        scoped_call_ids = _call_id_scope(call_id)
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_call_id = record.get("agent_call_id") or record.get("call_id")
            if record_call_id in scoped_call_ids and record.get("success") is True:
                records.append(record)
        for record in reversed(records):
            name = str(record.get("tool_name", "")).strip()
            arguments = record.get("arguments")
            if not isinstance(arguments, Mapping):
                continue
            config: Mapping[str, Any] | None = None
            if name == "validate_candidate" and self._validation_record_is_valid(record):
                candidate = arguments.get("candidate")
                if isinstance(candidate, Mapping):
                    nested = candidate.get("config")
                    config = nested if isinstance(nested, Mapping) else candidate
            elif name == "validate_candidates" and self._validation_record_is_valid(record):
                candidates = arguments.get("candidates")
                if isinstance(candidates, list) and len(candidates) == 1:
                    candidate = candidates[0]
                    if isinstance(candidate, Mapping):
                        nested = candidate.get("config")
                        config = nested if isinstance(nested, Mapping) else candidate
            if config is None:
                continue
            metadata = self._recovery_metadata_from_tool_calls(records)
            payload = {
                "candidates": [
                    {"config": dict(config), **metadata}
                ]
            }
            try:
                return parse_agent_candidate_payload(json.dumps(payload), search_space)
            except GeneralAgentValidationError:
                continue

        # A valid optimizer proposal may not have been explicitly validated by
        # the model. Recover only the latest successful proposal and validate
        # that exact config through the normal tool registry. This keeps the
        # strict path deterministic and evaluator-isolated.
        for record in reversed(records):
            if str(record.get("tool_name", "")).strip() != "optimizer_suggest":
                continue
            preview = record.get("result_preview")
            if not isinstance(preview, str):
                continue
            try:
                result = json.loads(preview)
                nested = result.get("result", result)
                candidate = nested.get("candidate")
                if not isinstance(candidate, Mapping):
                    candidates = nested.get("candidates")
                    candidate = (
                        candidates[0].get("candidate")
                        if isinstance(candidates, list)
                        and candidates
                        and isinstance(candidates[0], Mapping)
                        else None
                    )
                if not isinstance(candidate, Mapping):
                    continue
                config = candidate.get("config", candidate)
                if not isinstance(config, Mapping):
                    continue
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            try:
                validation_text = _run_coro_sync(
                    self._require_tool_registry().execute_tool(
                        "validate_candidate",
                        {"candidate": dict(config)},
                        self._build_tool_context(),
                        call_id=(
                            call_id[0]
                            if isinstance(call_id, Sequence)
                            and not isinstance(call_id, str)
                            else call_id
                        ),
                    )
                )
                validation = json.loads(validation_text)
                validation_result = validation.get("result", validation)
                if validation.get("ok") is not True or not isinstance(
                    validation_result, Mapping
                ) or validation_result.get("valid") is not True:
                    continue
                validated_config = validation_result.get("config", config)
                payload = {
                    "candidates": [
                        {"config": dict(validated_config), **self._recovery_metadata_from_tool_calls(records)}
                    ]
                }
                return parse_agent_candidate_payload(
                    json.dumps(payload), search_space
                )
            except (GeneralAgentValidationError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return None

    @staticmethod
    def _validation_record_is_valid(record: Mapping[str, Any]) -> bool:
        preview = record.get("result_preview")
        if not isinstance(preview, str):
            # Older logs did not always retain a parseable validation result.
            return record.get("success") is True
        try:
            payload = json.loads(preview)
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        if payload.get("ok") is not True:
            return False
        result = payload.get("result")
        if not isinstance(result, Mapping):
            return False
        if record.get("tool_name") == "validate_candidate":
            return result.get("valid") is True
        if record.get("tool_name") == "validate_candidates":
            return result.get("invalid_count") == 0 and result.get("valid_count", 0) > 0
        return False

    def _successfully_validated_configs_for_call(
        self, call_id: str | Sequence[str]
    ) -> list[dict[str, Any]]:
        path = self._agent_tool_calls_path
        if not path.exists():
            return []
        scoped_call_ids = _call_id_scope(call_id)
        configs: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            record_call_id = record.get("agent_call_id") or record.get("call_id")
            if (
                record_call_id not in scoped_call_ids
                or record.get("success") is not True
                or not self._validation_record_is_valid(record)
            ):
                continue
            arguments = record.get("arguments")
            if not isinstance(arguments, Mapping):
                continue
            raw_candidates: list[Any]
            if record.get("tool_name") == "validate_candidate":
                raw_candidates = [arguments.get("candidate")]
            elif record.get("tool_name") == "validate_candidates":
                value = arguments.get("candidates")
                raw_candidates = value if isinstance(value, list) else []
            else:
                continue
            for raw_candidate in raw_candidates:
                if not isinstance(raw_candidate, Mapping):
                    continue
                raw_config = raw_candidate.get("config", raw_candidate)
                if not isinstance(raw_config, Mapping):
                    continue
                try:
                    config = self._require_search_space().coerce_config(
                        raw_config, use_defaults=False
                    )
                except Exception:
                    continue
                configs.append(config)
        return configs

    def _latest_optimizer_candidate(
        self, call_id: str | Sequence[str]
    ) -> dict[str, Any] | None:
        path = self._agent_tool_calls_path
        if not path.exists():
            return None
        scoped_call_ids = _call_id_scope(call_id)
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            record_call_id = record.get("agent_call_id") or record.get("call_id")
            if (
                record_call_id not in scoped_call_ids
                or record.get("success") is not True
                or record.get("tool_name") != "optimizer_suggest"
            ):
                continue
            structured = record.get("result_data")
            if isinstance(structured, Mapping):
                candidate = structured.get("candidate")
                if isinstance(candidate, Mapping):
                    return dict(candidate)
            preview = record.get("result_preview")
            if not isinstance(preview, str):
                continue
            try:
                payload = json.loads(preview)
                result = payload.get("result", payload)
                candidate = result.get("candidate")
                if isinstance(candidate, Mapping):
                    return dict(candidate.get("config", candidate))
            except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return None

    def _recovery_metadata_from_tool_calls(
        self, records: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Build protocol metadata around a candidate accepted by the validator."""
        backend = None
        identity = None
        for record in reversed(records):
            if record.get("tool_name") != "optimizer_suggest":
                continue
            arguments = record.get("arguments")
            if isinstance(arguments, Mapping):
                backend = arguments.get("backend")
            structured = record.get("result_data")
            if isinstance(structured, Mapping):
                if backend is None:
                    backend = structured.get("backend")
                identity = structured.get("identity")
            preview = record.get("result_preview")
            if identity is None and isinstance(preview, str):
                try:
                    result = json.loads(preview)
                    nested = result.get("result", result)
                    identity = nested.get("identity")
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
            break
        latest_trial_id = (
            None if not self._history else self._history[-1].suggestion.trial_id
        )
        considered = list(self.config.optimizer_backend_allowlist)
        action = {
            "belief": "The validated optimizer proposal is usable; the model response metadata was repaired by the harness.",
            "expected_information": "Evaluate this validated backend proposal to update the incumbent and backend assessment.",
            "hypothesis_update": {
                "status": "not_applicable",
                "evidence_trial_id": latest_trial_id,
                "reason": "No new observation was available while recovering the validated proposal.",
            },
            "parent_trials": [],
            "reference_trials": [] if latest_trial_id is None else [latest_trial_id],
            "hypothesis": None,
            "change_summary": "Recovered the exact candidate accepted by validate_candidate after repairing agent output metadata.",
            "optimizer": {
                "relationship": "adopt",
                "backend": backend,
                "candidate_identity": identity,
                "considered_backends": considered,
            },
        }
        return {"rationale": "Harness-recovered validated optimizer proposal.", "search_action": action}

    def _successful_tool_names_for_call(self, call_id: str | Sequence[str]) -> set[str]:
        path = self._agent_tool_calls_path
        if not path.exists():
            return set()
        scoped_call_ids = _call_id_scope(call_id)
        names: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_call_id = record.get("agent_call_id") or record.get("call_id")
            if (
                record_call_id not in scoped_call_ids
                or record.get("success") is not True
            ):
                continue
            name = str(record.get("tool_name", "")).strip()
            if name:
                names.add(name)
        return names

    def _successful_tool_call_count_for_call(
        self,
        call_id: str | Sequence[str],
        tool_names: set[str] | frozenset[str],
    ) -> int:
        path = self._agent_tool_calls_path
        if not path.exists():
            return 0
        scoped_call_ids = _call_id_scope(call_id)
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_call_id = record.get("agent_call_id") or record.get("call_id")
            if (
                record_call_id in scoped_call_ids
                and record.get("success") is True
                and record.get("tool_name") in tool_names
            ):
                count += 1
        return count

    def _declared_skill_read_error(
        self, call_id: str, candidates: list[ParsedAgentCandidate]
    ) -> str | None:
        if not (self.config.framework == "nanobot" and self._agent_skills_enabled()):
            return None
        assert self._workspace_dir is not None
        assert self._run_dir is not None
        declared = _declared_agent_skill_names(candidates)
        if not declared:
            return None
        workspace_skills_dir = self._workspace_dir / "skills"
        required = sorted(
            skill
            for skill in declared
            if (workspace_skills_dir / skill / "SKILL.md").exists()
        )
        if not required:
            return None
        read = _nanobot_read_skill_names_for_call(self._run_dir / "llm_logs", call_id)
        missing = [skill for skill in required if skill not in read]
        if not missing:
            return None
        missing_text = ", ".join(f"`{skill}`" for skill in missing)
        if len(missing) == 1:
            skill = missing[0]
            return (
                f"Agent set search_action.skill to `{skill}` but did not read "
                f"`skills/{skill}/SKILL.md` with the read_file tool in this same attempt. "
                "Follow the skill declaration rules in TOOLS.md, or set search_action.skill to JSON null."
            )
        return (
            f"Agent declared skills {missing_text} but did not read each corresponding "
            "`skills/<skill-name>/SKILL.md` file with the read_file tool in this same attempt. "
            "Follow the skill declaration rules in TOOLS.md, or set search_action.skill to JSON null."
        )

    def _declared_skill_tool_usage_error(
        self, call_id: str, candidates: list[ParsedAgentCandidate]
    ) -> str | None:
        if not (self.config.framework == "nanobot" and self._agent_skills_enabled()):
            return None
        assert self._workspace_dir is not None
        declared = _declared_agent_skill_names(candidates)
        if not declared:
            return None
        workspace_skills_dir = self._workspace_dir / "skills"
        checked = sorted(
            skill
            for skill in declared
            if (workspace_skills_dir / skill / "SKILL.md").exists()
        )
        if not checked:
            return None
        used_tools = _bbo_workspace_tool_names_for_call(
            self._agent_tool_calls_path, call_id
        )
        for skill in checked:
            if skill in NON_PROPOSAL_BBO_SKILLS:
                return (
                    f"Agent set search_action.skill to `{skill}`, but `{skill}` is a memory maintenance "
                    "skill and must not be the primary skill for an evaluator-facing candidate. "
                    "Follow TOOLS.md if memory is useful, then either choose a proposal skill with its evidence tools "
                    "or set search_action.skill to JSON null."
                )
            required_groups = SKILL_EVIDENCE_TOOL_GROUPS.get(skill)
            if not required_groups:
                continue
            if skill == "initialize-search" and not self._history:
                required_groups = tuple(
                    group
                    for group in required_groups
                    if group != ("measure_search_coverage",)
                )
            missing = [
                group
                for group in required_groups
                if not any(tool in used_tools for tool in group)
            ]
            if missing:
                missing_text = ", ".join(_format_tool_group(group) for group in missing)
                return (
                    f"Agent declared BBO skill `{skill}` but did not call the required BBO evidence tools "
                    f"in this same attempt. Missing: {missing_text}. Follow the tool protocol in TOOLS.md, "
                    "validate the final candidate, then return the raw JSON; otherwise set search_action.skill "
                    "to JSON null."
                )
        return None

    def _fallback_candidate(self, reason: str) -> AgentCandidateEntry | None:
        search_space = self._require_search_space()
        for index in range(500):
            config = search_space.sample(self._rng)
            identity = stable_config_identity(config)
            if identity in self._seen_config_ids:
                continue
            self._seen_config_ids.add(identity)
            return AgentCandidateEntry(
                config=config,
                call_id=f"fallback_random_{self._call_index:05d}",
                candidate_index=index,
                metadata={
                    "agent_source": "fallback_random",
                    "agent_fallback_reason": reason,
                    **_search_action_metadata(
                        {
                            "search_intent": "exploration",
                            "change_summary": "fallback random sample after agent failure",
                        },
                        source="fallback_random",
                    ),
                },
            )
        return None

    def _clear_workspace_candidate_file(self) -> None:
        if self._workspace_dir is None:
            return
        path = self._workspace_dir / FINAL_CANDIDATE_FILENAME
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def _read_workspace_candidate_file(self, call_id: str) -> tuple[str, str] | None:
        if self._workspace_dir is None:
            return None
        candidates = [
            self._workspace_dir / FINAL_CANDIDATE_FILENAME,
            self._workspace_dir / "scratch" / call_id / FINAL_CANDIDATE_FILENAME,
            self._workspace_dir / "scratch" / call_id / "candidate.json",
            self._workspace_dir / "scratch" / call_id / "candidates.json",
        ]
        for path in candidates:
            if path.is_symlink() or not path.exists() or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8").strip()
            except Exception:
                continue
            if text:
                relative_path = str(path.relative_to(self._workspace_dir))
                return relative_path, text
        return None

    def _ingest_observation(
        self, observation: TrialObservation, *, replay: bool = False
    ) -> None:
        assert self._primary_name is not None
        self._history.append(observation)
        self._seen_config_ids.add(stable_config_identity(observation.suggestion.config))
        if observation.success and self._primary_name in observation.objectives:
            score = float(observation.objectives[self._primary_name])
            incumbent = Incumbent(
                config=dict(observation.suggestion.config),
                score=score,
                objectives=dict(observation.objectives),
                trial_id=observation.suggestion.trial_id,
                metadata={
                    "algorithm": self.name,
                    "agent_framework": self.config.framework,
                },
            )
            if self._best is None:
                self._best = incumbent
            elif (
                self._primary_direction == ObjectiveDirection.MINIMIZE
                and score < float(self._best.score)
            ):
                self._best = incumbent
            elif (
                self._primary_direction == ObjectiveDirection.MAXIMIZE
                and score > float(self._best.score)
            ):
                self._best = incumbent
        if not replay and self._run_dir is not None:
            append_jsonl(
                self._agent_optimization_trace_path,
                {
                    "step": len(self._history),
                    "trial": _observation_summary(observation),
                    "best": None
                    if self._best is None
                    else {
                        "config": agent_visible_config(self._best.config),
                        "score": agent_visible_payload(self._best.score),
                        "objectives": agent_visible_payload(self._best.objectives),
                        "trial_id": self._best.trial_id,
                    },
                    "agent_framework": self.config.framework,
                    "agent_engine": self._engine.name,
                    "timestamp": time.time(),
                },
            )

    def _write_workspace_context(self) -> None:
        self._require_ready()
        assert self._workspace_dir is not None
        task_spec = self._require_task_spec()
        history = (
            self._history[-self.config.history_limit :]
            if self.config.history_limit
            else []
        )
        (self._workspace_dir / "task.md").write_text(
            self._render_task_markdown(), encoding="utf-8"
        )
        dump_json(
            self._workspace_dir / "space.json",
            {"parameters": search_space_schema(task_spec.search_space)},
        )
        if self._manifest is not None:
            dump_json(
                self._workspace_dir / "manifest.json",
                self._agent_workspace_manifest_payload(),
            )
        if self._tool_registry is not None and self._workspace_bridge_enabled():
            dump_json(
                self._workspace_dir / "tool_specs.json",
                {"tools": self._agent_tool_specs()},
            )
        dump_json(
            self._workspace_dir / "objective.json",
            {
                "name": task_spec.primary_objective.name,
                "direction": task_spec.primary_objective.direction.value,
                "all_objectives": [
                    {"name": objective.name, "direction": objective.direction.value}
                    for objective in task_spec.objectives
                ],
            },
        )
        dump_json(
            self._workspace_dir / "incumbent.json",
            {
                "config": None
                if self._best is None
                else agent_visible_config(self._best.config),
                "score": None
                if self._best is None
                else agent_visible_payload(self._best.score),
                "objectives": {}
                if self._best is None
                else agent_visible_payload(self._best.objectives),
                "trial_id": None if self._best is None else self._best.trial_id,
            },
        )
        self._write_history_jsonl(history)
        if self._workspace_bridge_enabled():
            self._write_workspace_tool_bridge()
            self._write_workspace_python_api()
            if self._optimizer_suggestion_enabled():
                self._write_workspace_gp_example()
            else:
                _remove_path(self._workspace_dir / "gp_expected_improvement.py")
                _remove_path(self._workspace_dir / "examples")
            (self._workspace_dir / "TOOLS.md").write_text(
                self._render_tools_markdown(), encoding="utf-8"
            )
            (self._workspace_dir / "python_environment.md").write_text(
                self._render_python_environment(), encoding="utf-8"
            )
        else:
            self._remove_workspace_tool_files()
        self._write_workspace_skills()
        if self._workspace_bridge_enabled():
            self._write_workspace_audit_script()
        if self.config.context_access == "on_demand":
            from .tools.context_io import INSTRUCTIONS

            instructions = INSTRUCTIONS
            dump_json(self._workspace_dir / "task_details.json", self._context_io_documents["sections"])
            dump_json(self._workspace_dir / "parameter_catalog.json", self._context_io_documents["parameters"])
            dump_json(self._workspace_dir / "context_tools.json", {"tools": self._agent_tool_specs()})
        elif self._raw_workspace_prompt_enabled():
            instructions = self._render_raw_workspace_instructions()
        elif self._codex_controlled_workspace_prompt_enabled():
            instructions = self._render_codex_controlled_workspace_instructions()
        else:
            instructions = self.config.prompt_profile.compose(
                self._render_instructions(), stage="protocol"
            )
        if (
            self.config.tool_mode == "function_calling"
            and not self._codex_controlled_workspace_prompt_enabled()
            and not self._raw_workspace_prompt_enabled()
        ):
            instructions = (
                "# Host-mediated BBO tools\n\n"
                "Use the host-mediated `bbo_tool.py` CLI supplied in the workspace. "
                "It accepts only registered BBO tool names and JSON arguments. Do not import "
                "benchmark modules or access host paths. Read task.md, space.json, objective.json, "
                "history.jsonl, and incumbent.json; call the host optimizer "
                "and validation tools; then write the exact final candidate payload to "
                "final_candidate.json and return the same raw JSON.\n"
            )
        if self.config.execution_backend == "isolated_docker":
            instructions += (
                "\n\n## Local process control\n\n"
                "Use bash to manage your programs. The recommended container image provides "
                "`ps`, `pgrep`, `pkill`, `kill`, `setsid`, and `timeout`; examples are in "
                "`/usr/local/share/agent-process-control.md`. Inspect PIDs/PGIDs before sending "
                "TERM, then KILL if needed; verify termination. Native session IDs are not PIDs. "
                "For Ctrl-C via write_stdin, launch exec_command with tty=true; closed stdin "
                "requires cancellation by PID/PGID.\n"
            )
        (self._workspace_dir / "instructions.md").write_text(
            instructions, encoding="utf-8"
        )
        self._sync_workspace_snapshot()

    def _sync_workspace_snapshot(self) -> None:
        if self._workspace_snapshot_dir is None or self._workspace_dir is None:
            return
        if self._workspace_snapshot_dir.exists():
            shutil.rmtree(self._workspace_snapshot_dir)
        shutil.copytree(
            self._workspace_dir,
            self._workspace_snapshot_dir,
            symlinks=True,
        )

    def _agent_workspace_manifest_payload(self) -> dict[str, Any]:
        assert self._manifest is not None
        payload = self._manifest.to_dict()
        tool_policy = dict(payload.get("tool_policy") or {})
        tool_names: list[str] = []
        if self._tool_registry is not None:
            for spec in self._agent_tool_specs():
                function = spec.get("function") if isinstance(spec, Mapping) else None
                name = function.get("name") if isinstance(function, Mapping) else None
                if isinstance(name, str):
                    tool_names.append(name)
        if tool_names or not self._agent_tools_enabled():
            tool_policy["enabled_tools"] = tool_names
        enabled = set(tool_names)
        code_policy = dict(tool_policy.get("code_interpreter") or {})
        code_policy["enabled"] = "code_interpreter" in enabled
        tool_policy["code_interpreter"] = code_policy
        web_policy = dict(tool_policy.get("web_search") or {})
        web_policy["enabled"] = "web_search" in enabled or "fetch_url" in enabled
        tool_policy["web_search"] = web_policy
        payload["tool_policy"] = tool_policy
        payload["harness_policy"] = self._native_harness_policy()
        payload["task_id"] = self._agent_visible_task_name()
        payload["context_policy"] = self.config.context_policy.to_dict()
        payload["context_access"] = self.config.context_access
        payload["context_fingerprint"] = self._agent_context_fingerprint
        if self.config.context_policy.identity_exposure.value == "public_instance":
            return payload
        sanitized = sanitize_agent_context_payload(payload)
        sanitized["task_id"] = self._agent_visible_task_name()
        return sanitized

    def _write_history_jsonl(self, history: list[TrialObservation]) -> None:
        assert self._workspace_dir is not None
        path = self._workspace_dir / "history.jsonl"
        summarize = (
            _observation_summary
            if self._agent_tools_enabled() and not self._raw_workspace_prompt_enabled()
            else _agent_history_summary
        )
        with path.open("w", encoding="utf-8") as handle:
            for observation in history:
                handle.write(
                    json.dumps(
                        to_jsonable(summarize(observation)),
                        sort_keys=True,
                    )
                    + "\n"
                )

    def _write_workspace_tool_bridge(self) -> None:
        assert self._workspace_dir is not None
        cli_source = (
            Path(__file__)
            .with_name("workspace_tool_cli.py")
            .read_text(encoding="utf-8")
        )
        script = "#!/usr/bin/env python3\n" + cli_source
        tool_path = self._workspace_dir / "bbo_tool.py"
        tool_path.write_text(script, encoding="utf-8")
        try:
            tool_path.chmod(0o755)
        except OSError:
            pass
        web_key_env = self.config.web_search_api_key_env
        if (
            not web_key_env
            and self.config.web_search_provider.strip().lower().replace("-", "_")
            == "serpapi"
        ):
            web_key_env = "SERPAPI_API_KEY"
        web_search_api_key = os.environ.get(web_key_env or "")
        config_path = self._workspace_dir / "bbo_tool_config.json"
        dump_json(
            config_path,
            {
                "workspace_dir": str(self._workspace_dir),
                "state_dir": str(self._state_dir),
                "tool_calls_path": str(self._agent_tool_calls_path),
                "sources_path": str(self._agent_sources_path),
                "memory_path": str(self._agent_memory_path),
                "memory_summary_path": str(self._agent_memory_summary_path),
                "max_tool_calls": self.config.max_tool_calls,
                "enabled_tool_names": (
                    None
                    if self.config.enabled_tool_names is None
                    else list(self.config.enabled_tool_names)
                ),
                "optimizer_backend_allowlist": list(
                    self.config.optimizer_backend_allowlist
                ),
                "optimizer_max_calls_per_round": self.config.optimizer_max_calls_per_round,
                "optimizer_state_path": str(
                    self._state_dir / "optimizer_tool_state.json"
                ),
                "optimizer_python_executable": sys.executable,
                "optimizer_repository_root": str(Path(__file__).resolve().parents[3]),
                "experiment_condition": self.config.experiment_condition,
                "context_profile": self.config.context_policy.profile.value,
                "context_policy": self.config.context_policy.to_dict(),
                "context_fingerprint": self._agent_context_fingerprint,
                "agent_task_alias": self._agent_task_alias,
                "require_analysis_evidence_per_round": self.config.require_analysis_evidence_per_round,
                "required_tool_names_per_round": list(
                    self.config.required_tool_names_per_round
                ),
                "require_candidate_validation_per_round": self.config.require_candidate_validation_per_round,
                "require_optimizer_decision_per_round": self.config.require_optimizer_decision_per_round,
                "require_hypothesis_lifecycle_per_round": self.config.require_hypothesis_lifecycle_per_round,
                "require_evidence_bound_reconfiguration": self.config.require_evidence_bound_reconfiguration,
                "optimizer_agent_task_id": self._agent_visible_task_name(),
                "optimizer_max_evaluations": self._require_task_spec().max_evaluations,
                "optimizer_task_metadata": (
                    _optimizer_visible_task_metadata(self._require_task_spec().metadata)
                    if self.config.context_policy.identity_exposure.value == "public_instance"
                    else {
                        "parameter_transforms": dict(
                            self._require_task_spec().metadata.get("parameter_transforms") or {}
                        )
                    }
                ),
                "seed": self._seed,
                "smiles_pool_path": os.environ.get("BBO_SMILES_POOL_PATH"),
                "code_backend": self.config.code_backend,
                "sandbox_fusion_base_url": self.config.sandbox_fusion_base_url
                or os.environ.get("SANDBOX_FUSION_BASE_URL"),
                "docker_image": self.config.docker_image,
                "web_search_provider": self.config.web_search_provider,
                "web_search_api_key_env": self.config.web_search_api_key_env,
                "web_search_api_key": web_search_api_key,
                "serpapi_endpoint": os.environ.get("SERPAPI_ENDPOINT"),
            },
        )
        try:
            config_path.chmod(0o600)
        except OSError:
            pass

    def _write_workspace_python_api(self) -> None:
        assert self._workspace_dir is not None
        api_source = (
            Path(__file__)
            .with_name("workspace_python_api.py")
            .read_text(encoding="utf-8")
        )
        api_path = self._workspace_dir / "bbo_tools.py"
        api_path.write_text(api_source, encoding="utf-8")

    def _write_workspace_gp_example(self) -> None:
        assert self._workspace_dir is not None
        examples_dir = self._workspace_dir / "examples"
        examples_dir.mkdir(parents=True, exist_ok=True)
        source = (
            Path(__file__)
            .with_name("gp_expected_improvement_example.py")
            .read_text(encoding="utf-8")
        )
        path = examples_dir / "gp_expected_improvement.py"
        path.write_text(source, encoding="utf-8")
        try:
            path.chmod(0o755)
        except OSError:
            pass
        entrypoint = self._workspace_dir / "gp_expected_improvement.py"
        entrypoint.write_text(
            textwrap.dedent(
                """
                #!/usr/bin/env python3
                from __future__ import annotations

                import runpy
                from pathlib import Path


                if __name__ == "__main__":
                    workspace = Path(__file__).resolve().parent
                    runpy.run_path(str(workspace / "examples" / "gp_expected_improvement.py"), run_name="__main__")
                """
            ).lstrip(),
            encoding="utf-8",
        )
        try:
            entrypoint.chmod(0o755)
        except OSError:
            pass

    def _write_workspace_skills(self) -> None:
        assert self._workspace_dir is not None
        skill_sources = self._agent_skill_source_dirs()
        if not skill_sources:
            _remove_path(self._workspace_dir / "skills")
            return
        skills_dir = self._workspace_dir / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for source in skill_sources:
            skill_name = _nanobot_skill_name(source)
            target = skills_dir / skill_name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(
                source,
                target,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            entries.append(_skill_index_entry(skill_name))
        dump_json(skills_dir / "index.json", {"skills": entries})

    def _agent_skill_source_dirs(self) -> list[Path]:
        sources: list[Path] = []
        if self.config.enable_bbo_skills:
            packaged_root = _packaged_bbo_nanobot_skills_dir()
            sources.extend(packaged_root / name for name in BBO_NANOBOT_SKILL_NAMES)
        for raw_path in self.config.skill_paths:
            sources.extend(_discover_nanobot_skill_dirs(raw_path))
        seen: set[str] = set()
        unique: list[Path] = []
        for source in sources:
            skill_name = _nanobot_skill_name(source)
            if skill_name in seen:
                continue
            seen.add(skill_name)
            unique.append(source)
        return unique

    def _agent_skills_enabled(self) -> bool:
        return bool(self.config.enable_bbo_skills or self.config.skill_paths)

    def _agent_tools_enabled(self) -> bool:
        return self.config.tool_mode != "no_tool"

    def _workspace_bridge_enabled(self) -> bool:
        return self.config.tool_mode == "workspace_json"

    def _optimizer_suggestion_enabled(self) -> bool:
        if not self._agent_tools_enabled():
            return False
        if self.config.enabled_tool_names is None:
            return True
        return (
            "optimizer_suggest" in self.config.enabled_tool_names
            and "gp_ei" in self.config.optimizer_backend_allowlist
        )

    def _remove_workspace_tool_files(self) -> None:
        assert self._workspace_dir is not None
        for relative_path in (
            "TOOLS.md",
            "tool_specs.json",
            "bbo_tool.py",
            "bbo_tools.py",
            "bbo_tool_config.json",
            "bbo_workspace_audit.py",
            "bbo_workspace_audit_summary.json",
            "gp_expected_improvement.py",
            "python_environment.md",
            "examples",
        ):
            _remove_path(self._workspace_dir / relative_path)

    def _write_workspace_audit_script(self) -> None:
        assert self._workspace_dir is not None
        audit_path = self._workspace_dir / "bbo_workspace_audit.py"
        audit_path.write_text(
            textwrap.dedent(
                """
                #!/usr/bin/env python3
                from __future__ import annotations

                import json
                from pathlib import Path
                from typing import Any, Callable

                from bbo_tools import BBO


                def safe_call(name: str, fn: Callable[[], Any]) -> dict[str, Any]:
                    try:
                        return {"ok": True, "result": fn()}
                    except Exception as exc:  # noqa: BLE001
                        return {"ok": False, "error": type(exc).__name__, "message": str(exc)}


                def main() -> int:
                    bbo = BBO()
                    summary: dict[str, Any] = {}
                    sample_holder: dict[str, Any] = {}

                    summary["task_context"] = safe_call("task_context", lambda: bbo.task_context())
                    summary["manifest"] = safe_call("manifest", bbo.manifest)
                    summary["search_space"] = safe_call("search_space", bbo.search_space)
                    summary["objective"] = safe_call("objective", bbo.objective)
                    summary["tool_specs"] = safe_call("tool_specs", bbo.tool_specs)
                    summary["history"] = safe_call("history", lambda: bbo.history(limit=20))
                    summary["incumbent"] = safe_call("incumbent", bbo.incumbent)
                    summary["history_overview"] = safe_call("history_overview", bbo.history_overview)
                    summary["objective_metrics"] = safe_call("objective_metrics", bbo.summarize_objective_metrics)
                    summary["coverage"] = safe_call("coverage", bbo.measure_search_coverage)
                    summary["recent_actions"] = safe_call("recent_actions", bbo.recent_search_actions)
                    summary["surrogate_check"] = safe_call("surrogate_check", bbo.fit_and_check_surrogate)

                    def sample_once() -> dict[str, Any]:
                        sample = bbo.sample(n=1, seed=0)
                        sample_holder["sample"] = sample
                        return sample

                    summary["sample"] = safe_call("sample", sample_once)
                    summary["analyze_history"] = safe_call("analyze_history", lambda: bbo.analyze_history(limit=100))
                    summary["memory_write"] = safe_call(
                        "memory_write",
                        lambda: bbo.memory_write(
                            kind="note",
                            content="BBO workspace audit completed.",
                            tags=["audit", "workspace"],
                            source_call_id="bbo_workspace_audit",
                        ),
                    )
                    summary["memory_read"] = safe_call("memory_read", lambda: bbo.memory_read(tags=["audit"], limit=5))
                    summary["code_interpreter"] = safe_call(
                        "code_interpreter",
                        lambda: bbo.code_interpreter("print('bbo workspace audit')", language="python"),
                    )
                    summary["web_search"] = safe_call(
                        "web_search",
                        lambda: bbo.web_search("black-box optimization placement benchmark", limit=1),
                    )
                    summary["fetch_url"] = safe_call(
                        "fetch_url",
                        lambda: bbo.fetch_url("https://example.com", max_chars=500),
                    )

                    def validate_sample() -> dict[str, Any]:
                        sample = sample_holder.get("sample")
                        if not isinstance(sample, dict) or not sample.get("candidates"):
                            sample = bbo.sample(n=1, seed=1)
                        return bbo.validate([sample["candidates"][0]])

                    summary["validate"] = safe_call("validate", validate_sample)
                    summary["validate_candidate"] = safe_call(
                        "validate_candidate",
                        lambda: bbo.validate_candidate(sample_holder.get("sample", {}).get("candidates", [{}])[0]),
                    )

                    path = Path("bbo_workspace_audit_summary.json")
                    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
                    ok_count = sum(1 for item in summary.values() if isinstance(item, dict) and item.get("ok"))
                    print(json.dumps({"audit_summary_path": str(path), "ok_count": ok_count, "total": len(summary)}, sort_keys=True))
                    return 0


                if __name__ == "__main__":
                    raise SystemExit(main())
                """
            ).lstrip(),
            encoding="utf-8",
        )
        try:
            audit_path.chmod(0o755)
        except OSError:
            pass

    def _render_python_environment(self) -> str:
        return textwrap.dedent(
            """
            # BBO Python Environment

            The workspace Python API is available with:

            ```python
            from bbo_tools import BBO
            bbo = BBO()
            ```

            Prefer writing small Python scripts that use this API for task inspection,
            history analysis, candidate validation, memory, web research, and code-backed
            analysis. The underlying `bbo_tool.py` CLI remains available as a fallback.

            Recommended libraries for local analysis are the Python standard library,
            `numpy`, `scipy`, `pandas`, and `scikit-learn` when they are available.
            These libraries are optional aids for analysis; they are not required for
            every task or every candidate-generation strategy.

            When `code_interpreter` is configured with SandboxFusion, the SandboxFusion
            image should preinstall `numpy`, `scipy`, `scikit-learn`, `pandas`, and
            `joblib`. Heavy BO stacks such as `torch`, `gpytorch`, and `botorch` are
            optional and not required by the default BBO agent workflow.
            """
        ).strip()

    def _render_task_markdown(self) -> str:
        if self._context_io_documents is not None:
            return self._context_io_documents["short_task"]
        return self._rendered_agent_context

    def _agent_visible_task_name(self) -> str:
        if self.config.context_policy.identity_exposure.value == "anonymous":
            return self._agent_task_alias
        if self.config.context_policy.identity_exposure.value == "family":
            return "anonymous_family_task"
        return self._require_task_spec().name

    def _render_tools_markdown(self) -> str:
        skill_section = ""
        skill_file_hint = ""
        if self._agent_skills_enabled():
            skill_file_hint = ", and relevant `skills/<skill-name>/SKILL.md` files"
            skill_lines = [
                "## BBO Skills",
                "",
                "BBO skills are instruction documents under `skills/<skill-name>/SKILL.md`.",
                "They are not callable tools, Python functions, or `BBO()` methods.",
                "",
                "- Use at most one primary proposal skill for a candidate.",
                "- Read only the relevant `SKILL.md` files when they would improve the candidate.",
                "- If final `search_action.skill` is non-null, read the exact matching `SKILL.md` with `read_file` in the same attempt.",
                "- Do not rely only on a skill summary or description when declaring skill use.",
                "- If you declare a skill, call that skill's required BBO evidence tools from `skills/index.json` in the same attempt.",
                "- `repair-invalid-candidate` is only a secondary corrective skill after validation fails.",
                "- `distill-search-memory` is a memory maintenance skill and must not be the primary skill for an evaluator-facing candidate.",
                "",
                "Common built-in skill evidence requirements:",
            ]
            for skill_name, groups in SKILL_EVIDENCE_TOOL_GROUPS.items():
                formatted_groups = ", ".join(
                    _format_tool_group(group) for group in groups
                )
                skill_lines.append(f"- `{skill_name}`: {formatted_groups}")
            skill_section = "\n\n" + "\n".join(skill_lines)
        enabled = (
            None
            if self.config.enabled_tool_names is None
            else set(self.config.enabled_tool_names)
        )
        optimizer_lines: list[str] = []
        if enabled is None or "optimizer_suggest" in enabled:
            optimizer_lines.append(
                "- `bbo.optimizer_suggest(backend=..., q=1)` returns a candidate menu."
            )
        if enabled is None or "optimizer_score" in enabled:
            optimizer_lines.append(
                "- `bbo.optimizer_score(configs)` scores virtual candidates."
            )
        if enabled is None or "optimizer_recommend_backends" in enabled:
            optimizer_lines.append(
                "- `bbo.optimizer_recommend_backends(k=3)` gives an optional, explainable backend shortlist."
            )
        if enabled is None or "optimizer_portfolio_suggest" in enabled:
            optimizer_lines.append(
                "- `bbo.optimizer_portfolio_suggest(backends=..., q_per_backend=1)` compares registered baselines on the same history and bounds."
            )
        controls = [
            name
            for name in (
                "optimizer_set_backend",
                "optimizer_set_bounds",
                "optimizer_set_acquisition",
                "optimizer_status",
                "optimizer_diagnostics",
                "optimizer_reset_policy",
            )
            if enabled is None or name in enabled
        ]
        if controls:
            optimizer_lines.append(
                "- Available optimizer controls: "
                + ", ".join(f"`bbo.{name}(...)`" for name in controls)
                + "."
            )
        optimizer_api_section = "\n".join(optimizer_lines)
        return textwrap.dedent(
            f"""
            # Tool Usage Notes

            This workspace exposes Nanobot native tools plus a local BBO Python API.

            ## Native Tools

            - Use `read_file` to inspect workspace files such as `task.md`, `space.json`,
              `objective.json`, `history.jsonl`, `incumbent.json`, `instructions.md`,
              and `TOOLS.md`{skill_file_hint}.
            - Use `exec` for short Python snippets that inspect history, call the BBO API,
              or validate candidate JSON.
            - Use relative workspace paths. Do not execute absolute paths when a relative
              path is available.

            ## BBO Python API

            Import the API inside Nanobot's `exec` tool:

            ```python
            from bbo_tools import BBO
            bbo = BBO()
            ```

            BBO is not a native Nanobot function-calling tool. Do not emit
            `<function=BBO>` and do not call invented methods such as
            `BBO().initialize_search()`.

            Useful methods include:

            - `bbo.task_context()`
            - `bbo.search_space()`
            - `bbo.objective()`
            - `bbo.history(mode="recent", limit=20)`
            - `bbo.incumbent()`
            - `bbo.history_overview()`
            - `bbo.summarize_objective_metrics()`
            - `bbo.compare_trials([...])`
            - `bbo.find_nearest_trials(target, k=5)`
            - `bbo.estimate_local_effects(reference, variables=None, local_radius=0.35)`
            - `bbo.measure_search_coverage()`
            - `bbo.sample(n=4, strategy="random")`
            - `bbo.fit_and_check_surrogate()`
            - `bbo.analyze_search_strategy()` for a landscape hypothesis, bias, conservative
              joint subspace, and downstream optimizer policy
            {optimizer_api_section}
            - `bbo.score_virtual_candidates(model_id, candidates)`
            - `bbo.validate_candidate(candidate)`
            - `bbo.validate([candidate])`
            - `bbo.recent_search_actions(limit=8)`
            - `bbo.memory_read()` and `bbo.memory_write(kind=..., content=...)`
            - `bbo.code_interpreter(code)` for restricted offline Python when enabled
            - `bbo.render_search_diagnostics()` for run-local JSON/SVG artifacts

            Use BBO tools selectively when they improve the decision, especially for
            exact trial comparison, objective/metric summaries, nearest trials, local
            effects, search coverage, promising/underexplored region ranking, surrogate
            validation, virtual candidate scoring, and final candidate validation.

            When a decision depends on precise history comparisons, variable
            differences, distances, local effects, coverage, surrogate quality, or
            candidate legality, use a BBO tool for evidence instead of estimating from
            memory.

            Validate the final rounded and formatted candidate with
            `bbo.validate_candidate(...)` or `bbo.validate(...)` when possible. If
            validation fails, repair or replace the candidate and validate again.

            Tool/API calls are append-only logged to `agent_tool_calls.jsonl`. Do not
            call the benchmark evaluator or any operation that consumes real evaluation
            budget.{skill_section}
            """
        ).strip()

    def _condition_tool_guidance(self) -> str:
        """Render explicit, auditable tool-use guidance for this experiment arm."""

        condition = self.config.experiment_condition
        if condition in {"", "default", "t0_bare"}:
            return ""
        guidance = [
            f"Experiment condition: {condition}.",
            "Use only the registered BBO tools exposed for this condition; tool availability is an experimental treatment.",
        ]
        if condition == "t1_analysis_available":
            guidance.append(
                "Search-strategy analysis is available through analyze_search_strategy. It returns a landscape hypothesis, bias, conservative joint subspace, and downstream optimizer policy; use it only when its evidence can improve the decision."
            )
        if self.config.require_analysis_evidence_per_round:
            guidance.append(
                "Before choosing the candidate, call at least one successful analysis/evidence tool and base the decision on its returned evidence."
            )
        if self.config.required_tool_names_per_round:
            required = ", ".join(self.config.required_tool_names_per_round)
            guidance.append(
                "Before choosing the candidate, successfully call every required "
                f"per-round tool: {required}. A failed or merely attempted call does not count."
            )
        if condition in {
            "t2_guided_analysis",
            "t3_agentic_search",
            "t4_soft_portfolio",
        }:
            guidance.extend(
                [
                    "Every optimization round must call analyze_search_strategy before changing search policy or bounds.",
                    "Treat its landscape and bias fields as hypotheses. Apply recommended_subspace.optimizer_bounds only when recommended_subspace.apply is true; otherwise preserve the original domain.",
                    "Use downstream_policy to choose a downstream optimizer/acquisition, but retain agent ownership and override it when task context or surrogate diagnostics provide stronger evidence.",
                    "State the exploit, explore, or balanced intent and the actionable parameters in search_action.change_summary.",
                    "Never concatenate every marginal row into one joint candidate. A candidate changing more than three parameters from the incumbent requires a successful analyze_parameter_interactions, score_virtual_candidates, optimizer_suggest, optimizer_portfolio_suggest, or optimizer_score call in the same round.",
                    "Validate the final joint candidate after applying these safeguards.",
                ]
            )
        if self.config.require_candidate_validation_per_round:
            guidance.append(
                "After final rounding/formatting, successfully call validate_candidate or validate_candidates on the exact final config."
            )
        if self.config.require_optimizer_decision_per_round:
            allowed = list(self.config.optimizer_backend_allowlist)
            if self.config.experiment_condition in {
                "multi_backend_agent",
                "multi_backend_agentic_bo",
            }:
                guidance.extend(
                    [
                        "Every optimization round must successfully call assess_backend_suitability before selecting a backend.",
                        "Choose exactly one backend yourself from gp_ei and tpe; suitability evidence never recommends or selects.",
                        "Call optimizer_suggest with the chosen backend explicitly. This condition fixes GP acquisition to EI and permits no bounds, option, q, or seed overrides.",
                        f"Use at most {self.config.optimizer_max_calls_per_round} optimizer candidate-decision calls, then validate and submit exactly one final candidate.",
                        "Record the backend actually passed to optimizer_suggest and whether the final candidate adopts, refines, or overrides its proposal.",
                        "Only outer-runner observations update optimizer history. Never invent objective values.",
                    ]
                )
                return "\n".join(f"- {item}" for item in guidance)
            if len(allowed) == 1:
                optimizer_policy = (
                    f"This condition exposes one fixed optimizer backend: {allowed[0]}. "
                    "Do not attempt to select or switch to any other backend. "
                    "You may narrow/reset numeric bounds"
                )
                if allowed[0] == "gp_ei":
                    optimizer_policy += (
                        " and select EI, LogEI, or UCB acquisition settings."
                    )
                else:
                    optimizer_policy += "."
            else:
                optimizer_policy = (
                    "You control the single-objective search loop and may choose only "
                    f"among these enabled backends: {', '.join(allowed)}. You may persist "
                    "an enabled backend, narrow/reset numeric bounds, and select EI, "
                    "LogEI, or UCB for gp_ei. optimizer_recommend_backends is "
                    "advisory and never switches automatically. "
                    "optimizer_portfolio_suggest compares registered baselines "
                    "on the same history and bounds; you still decide."
                )
            guidance.extend(
                [
                    "Every optimization round must include at least one successful optimizer_suggest, optimizer_portfolio_suggest, or optimizer_score call; "
                    f"at most {self.config.optimizer_max_calls_per_round} such candidate-decision calls are allowed.",
                    optimizer_policy,
                    "All optimizer candidates come from the same registered implementations and benchmark policy as standalone baselines. Menus never evaluate points. Inspect them and submit exactly one final candidate.",
                    "Record search_action.optimizer with relationship=adopt, refine, override, or direct_scored; include the selected backend and candidate identity when applicable.",
                    "Only outer-runner observations update optimizer history. Never invent or tell virtual objective values.",
                ]
            )
        return "\n".join(f"- {item}" for item in guidance)

    def _render_instructions(self) -> str:
        if not self._agent_tools_enabled():
            return self._render_no_tool_instructions()

        task_spec = self._require_task_spec()
        compact_xy_hint = self._compact_xy_output_hint()
        skills_file_line = ""
        skills_workflow_line = ""
        skill_read_line = ""
        skill_strategy_line = "Choose a candidate-generation strategy appropriate for the current task and evidence."
        tools_md_line = (
            "- TOOLS.md: native tool, BBO Python API, and validation guidance."
        )
        search_action_shape = self._candidate_payload_example()
        null_requirement = (
            '- Use JSON null for absent hypothesis values, not the string "null".'
        )
        if self._agent_skills_enabled():
            tools_md_line = "- TOOLS.md: native tool, BBO Python API, validation, and skill-use guidance."
            skills_file_line = (
                "\n            - skills/: optional Nanobot BBO skill reference library. "
                "Use it according to `TOOLS.md`."
            )
            skill_read_line = (
                "\n            - Read `TOOLS.md` before using workspace tools, the BBO Python API, or BBO\n"
                "              skills."
            )
            skill_strategy_line = (
                "Choose a candidate-generation strategy appropriate for the current task\n"
                "              and evidence. You may propose directly without any skill when no\n"
                "              specialized skill trigger is clearly satisfied."
            )
            skills_workflow_line = (
                "\n            - Skills are optional references, not mandatory steps. "
                "Decide whether any skill applies from the current task and evidence; "
                "do not read or follow every skill by default. Follow `TOOLS.md` for "
                "skill-read, evidence, validation, and declaration rules."
            )
            null_requirement = '- Use JSON null for absent skill or hypothesis values, not the string "null".'
        else:
            skill_read_line = "\n            - Read `TOOLS.md` before using workspace tools or the BBO Python API."
        return textwrap.dedent(
            f"""
            # Agentic BBO Candidate Protocol

            You are proposing configurations for a black-box optimization benchmark.
            Do not call the benchmark evaluator yourself and do not modify benchmark
            result files.

            Files in this workspace:
            - task.md: task background, goal, constraints, and prior knowledge.
            - manifest.json: agent benchmark construction, tool policy, and provenance.
            - space.json: exact parameter schema. Every candidate must include every parameter exactly once.
            - objective.json: primary objective name and optimization direction.
            - history.jsonl: recent evaluated trials.
            - incumbent.json: current best known configuration, if any.
            {tools_md_line}
            - tool_specs.json: available BBO function-calling tools when the backend supports tools.
            - bbo_tools.py: preferred Python API for BBO tools when using shell/file tools.
            - bbo_workspace_audit.py: optional observability script that exercises workspace BBO APIs.
            - examples/: optional candidate-generation or analysis examples. No example is mandatory.
            - python_environment.md: Python and sandbox library guidance.
            - bbo_tool.py: lower-level CLI bridge for BBO tools; use only as a fallback.{skills_file_line}
            - {FINAL_CANDIDATE_FILENAME}: authoritative final candidate handoff. The harness clears it before each attempt.

            Task: {self._agent_visible_task_name()}
            Primary objective: {task_spec.primary_objective.name}
            Direction: {task_spec.primary_objective.direction.value}

            Workspace workflow:
            - Use relative paths from the workspace. Do not execute absolute paths.
            {skill_read_line}
            - Read evaluated history, the incumbent, recent search actions, and the
              remaining budget before choosing this round's one most valuable search action.
            - Choose one search intent for this round: initialization, exploitation,
              directional_extrapolation, hypothesis_test, interaction_test,
              recombination, exploration, stagnation_recovery, surrogate_proposal,
              or repair.
            - {skill_strategy_line}
            - Do not treat any single method or example script as mandatory. Use tools
              to improve judgment, not to follow a fixed recipe.
            - You may create temporary scratch files or short analysis scripts inside
              the workspace when useful, using new relative paths such as
              `candidate.json`, `analysis.py`, or `scratch/candidates.json`.
              Do not overwrite task definitions, history, results, logs, or framework
              state files.{skills_workflow_line}
            - Tool/API calls are append-only logged to agent_tool_calls.jsonl.

            Final candidate handoff:
            - Write the exact top-level payload below to `{FINAL_CANDIDATE_FILENAME}` in the workspace root.
            - Verify JSON syntax with `python -m json.tool {FINAL_CANDIDATE_FILENAME}`.
            - For tool-enabled conditions, validate the exact rounded candidate with
              `validate_candidate` or `validate_candidates`, then immediately write the
              file before any further explanation, analysis, or tool call.
            - Do not announce that you are about to write the file; write it first.
            - Return the same raw JSON in chat for compatibility. The harness prefers
              `{FINAL_CANDIDATE_FILENAME}` and uses chat only as a fallback.

            Exact payload shape:
            {search_action_shape}

            Requirements:
            - Return exactly one candidate configuration. Each real optimization round
              submits one and only one new candidate to the evaluator.
            - Use available tools according to `TOOLS.md` when they improve the
              candidate decision.
            - Validate proposed candidates according to `TOOLS.md` before final output
              when possible.
            - If any script, command, or tool fails, recover with another reasonable
              strategy; never return an error message as the final answer.
            - Temporary candidate JSON files are allowed for checking, but only
              `{FINAL_CANDIDATE_FILENAME}` is the authoritative file handoff.
            - The harness independently rechecks schema, bounds, types, duplicates,
              and condition-specific tool evidence before accepting the candidate.
            - Do not use shell redirection such as `2>/dev/null`; rerun Python scripts only with relative commands.
            - Do not include markdown fences, comments, prose, or partial configurations.
            - Do not force intermediate analysis into JSON; only the final candidate
              payload and persisted action metadata need machine-readable structure.
            - Float and integer values must stay within their declared bounds.
            - Numeric values in final candidate JSON should use at most 4 decimal places.
            - Categorical values must be one of the declared choices.
            {null_requirement}
            {compact_xy_hint}
            """
        ).strip()

    def _build_agent_prompt(
        self, *, call_id: str, attempt_index: int, last_error: str | None = None
    ) -> str:
        if self.config.context_access == "on_demand":
            from .tools.context_io import INSTRUCTIONS

            intro = (
                f"Choose one next candidate for `{self._agent_visible_task_name()}`. "
                f"Call: {call_id}; attempt: {attempt_index}. "
                f"Observed evaluations: {len(self._history)}; remaining: {self._require_task_spec().max_evaluations - len(self._history)}.\n\n"
            )
            if self.config.framework == "openai_compatible":
                return intro + self._render_task_markdown() + "\n" + INSTRUCTIONS
            if self._campaign_session_id and self._history:
                latest = self._history[-1]
                feedback = {"trial_id": latest.suggestion.trial_id, "status": latest.status.value,
                            "objectives": dict(latest.objectives)}
                anchor = self._history[-2].suggestion.trial_id if len(self._history) > 1 else None
                return intro + "Latest host observation: " + json.dumps(feedback, sort_keys=True) + "\n\n" + (
                    f"Use this feedback and prior context; if more detail is needed, query get_trial_history(after_trial_id={json.dumps(anchor)}). "
                    "Earlier history remains available; avoid reprinting it in full. "
                    "Choose any legal candidate and call submit_candidate with config or a workspace JSON path. "
                    "write_candidate is optional. Stop after acceptance and acknowledge briefly."
                )
            return intro + (
                "Read task.md and instructions.md. Retrieve only needed task sections, parameter definitions and observation columns with the available tools. "
                "Full parameter and history files are available but need not be printed. "
                "Submit any complete legal config directly with submit_candidate(config=...) or submit_candidate(path=...) for a workspace JSON file. "
                "write_candidate and modifications to an existing trial are optional. "
                "After acceptance, stop tool use and give a short acknowledgement; do not repeat the configuration."
            )
        if self._raw_workspace_prompt_enabled():
            if self.config.framework == "openai_compatible":
                return self._build_raw_inline_prompt(
                    call_id=call_id,
                    attempt_index=attempt_index,
                    last_error=last_error,
                )
            return self._build_raw_workspace_prompt(
                call_id=call_id,
                attempt_index=attempt_index,
                last_error=last_error,
            )
        if self._codex_controlled_workspace_prompt_enabled():
            return self._build_codex_controlled_workspace_prompt(
                call_id=call_id,
                attempt_index=attempt_index,
                last_error=last_error,
            )
        if not self._agent_tools_enabled():
            return self._build_no_tool_agent_prompt(
                call_id=call_id,
                attempt_index=attempt_index,
                last_error=last_error,
            )

        task_spec = self._require_task_spec()
        best_score = None if self._best is None else self._best.score
        retry_feedback = _retry_feedback_block(last_error)
        retry_feedback_section = (
            ""
            if not retry_feedback
            else "\n\n" + textwrap.indent(retry_feedback, "            ")
        )
        compact_xy_hint = self._compact_xy_output_hint()
        condition_guidance = self._condition_tool_guidance()
        condition_guidance_section = (
            "" if not condition_guidance else "\n\n" + condition_guidance
        )
        tool_prompt_line = (
            "Read `TOOLS.md` before using workspace tools, BBO Python API helpers, or\n"
            "            BBO skills. Skill use is optional: use a skill only when it clearly helps\n"
            "            this proposal, otherwise set `search_action.skill` to JSON `null`."
            if self._agent_skills_enabled()
            else "Read `TOOLS.md` before using workspace tools or BBO Python API helpers."
        )
        if self.config.tool_mode == "function_calling":
            tool_prompt_line = (
                "Use the host-mediated `bbo_tool.py` CLI in the workspace with a registered tool "
                "name and one JSON arguments object. Do not import benchmark modules or access host paths."
            )
        tools_md_scope = (
            "tool protocol, BBO Python API, validation, and skill-use rules"
            if self._agent_skills_enabled()
            else "tool protocol, BBO Python API, and validation rules"
        )
        search_action_fields = self._search_action_prompt_fields()
        candidate_payload_example = self._candidate_payload_example(indent=2)
        return textwrap.dedent(
            f"""
            You are an optimization agent for task `{self._agent_visible_task_name()}`.

            Workspace: `.`
            Call id: `{call_id}`
            Attempt: `{attempt_index}`

            Produce exactly one new candidate configuration.

            Use the workspace as the source of truth:

            * `task.md`: task description, constraints, and prior knowledge
            * `space.json`: parameter names, types, bounds, choices, and precision
            * `objective.json`: objective name and direction
            * `history.jsonl` and `incumbent.json`: evaluated history and current best
            * `instructions.md`: final output and search-action metadata contract
            * BBO tools: {tools_md_scope if self.config.tool_mode == "workspace_json" else "host-mediated bbo_tool.py CLI backed by registered tools"}

            Do not call the benchmark evaluator or any operation that consumes real
            evaluation budget.

            {tool_prompt_line}{condition_guidance_section}

            Requirements:

            * Return exactly one candidate configuration.
            * Include every active required parameter exactly once.
            * Preserve the parameter types declared in `space.json`.
            * Respect all bounds, choices, precision rules, conditional rules, and
              constraints.
            * Do not invent objective values or trial IDs.
            * Do not return an exact duplicate of an evaluated configuration.
            * Near-duplicate candidates are allowed only when justified by local refinement
              or a controlled experiment.
            * Numeric precision should follow `space.json`. If no precision is declared,
              use at most 4 decimal places without changing the intended scale.
            * Validate the final formatted candidate with `validate_candidate` when
              possible. Validation must occur after rounding, formatting, and any
              repair.
            * If validation fails, repair or replace the candidate and validate again.
            * If validation tooling fails, manually check the candidate against
              `space.json` and the evaluated history.

            Describe the search action using:

            {search_action_fields}

            Only include trial IDs that exist in the workspace history.

            Do not modify protected files:

            * `task.md`
            * `space.json`
            * `objective.json`
            * `history.jsonl`
            * `incumbent.json`
            * `TOOLS.md`
            * `trials.jsonl`
            * `agent_*.jsonl`
            * files under `agent_state/`
            * files under `llm_logs/`
            * files under `reasoning_traces/`

            You may create temporary scratch files or short analysis scripts under a new
            call-specific path such as:

            `scratch/{call_id}/`

            Before finishing, write the exact final payload to
            `{FINAL_CANDIDATE_FILENAME}` in the workspace root and verify it with:

            `python -m json.tool {FINAL_CANDIDATE_FILENAME}`

            Validate the exact rounded configuration with `validate_candidate` or
            `validate_candidates` before writing it. Return the same JSON in chat; the
            harness treats the file as authoritative and chat as a compatibility fallback.

            If a tool or command fails, recover using the workspace files and still return
            one valid candidate.

            Current best primary objective: `{best_score}`
            Objective direction: `{task_spec.primary_objective.direction.value}`{retry_feedback_section}

            The final submitted answer must be valid raw JSON with exactly this shape:

            {candidate_payload_example}

            Replace the example config with the exact active parameters from `space.json`.

            Use native JSON types:

            * numbers as numbers
            * integers as integers
            * booleans as booleans
            * categorical values as strings
            * absent hypotheses as JSON `null`, not the string `"null"`

            {compact_xy_hint}

            The final submitted answer must contain no Markdown fences, comments,
            headings, explanations, or additional prose. Complete the reasoning and all
            required tool actions before submitting that final answer; do not stop after
            an intermediate tool result.
            """
        ).strip()

    def _codex_controlled_workspace_prompt_enabled(self) -> bool:
        return (
            self.config.framework == "codex"
            and self.config.tool_mode == "function_calling"
            and self._controlled_round_protocol_enabled()
        )

    def _raw_workspace_prompt_enabled(self) -> bool:
        if self.config.algorithm_name == "raw_agentic_bbo":
            return True
        # Tool-surface ablations can keep the first-class Agentic BO method while
        # explicitly selecting the same neutral prompt contract as the raw/p10
        # comparator.  The default Agentic BO profile remains method-specific.
        return (
            self.config.algorithm_name == "agentic_bo"
            and self.config.prompt_profile.name == "general_bbo"
        )

    @staticmethod
    def _render_raw_workspace_instructions() -> str:
        return textwrap.dedent(
            """
            # Raw black-box optimization workspace

            The workspace contains task background, the exact search space, objective
            direction, real evaluated history, and the current incumbent. Select one
            next candidate using your own reasoning and native workspace capabilities.

            Do not execute the evaluator, inspect its hidden implementation, fabricate
            observations, install packages, modify the execution environment, or modify
            task, history, result, state, or log files. Modeling based only on the visible
            evaluated history is allowed. You may create scratch work only below `scratch/`.

            Submit exactly one candidate as raw JSON with this shape:

            {"candidates": [{"config": {"<each active parameter>": "<value>"}}]}

            Write the same JSON to `final_candidate.json`. Do not include markdown or
            additional prose in the final response.
            """
        ).strip()

    def _build_raw_workspace_prompt(
        self, *, call_id: str, attempt_index: int, last_error: str | None
    ) -> str:
        task_spec = self._require_task_spec()
        retry_feedback = _retry_feedback_block(last_error, mention_tool_calls=False)
        retry_section = "" if not retry_feedback else f"\n\n{retry_feedback}"
        return textwrap.dedent(
            f"""
            Select exactly one next candidate for black-box optimization task
            `{self._agent_visible_task_name()}`.

            Call id: `{call_id}`
            Attempt: `{attempt_index}`

            Read `task.md`, `space.json`, `objective.json`, `history.jsonl`,
            `incumbent.json`, and `instructions.md` in the current workspace. These files
            are the available benchmark background. Use your own approach to choose the
            next candidate.

            Do not execute the evaluator, inspect its hidden implementation, fabricate
            observations, install dependencies, modify the environment, or modify
            protected workspace files. Modeling based only on visible evaluated history
            is allowed. Return one complete, legal candidate using the submission format
            in `instructions.md`.{retry_section}
            """
        ).strip()

    def _build_raw_inline_prompt(
        self, *, call_id: str, attempt_index: int, last_error: str | None
    ) -> str:
        """Render the raw workspace evidence inline for a tool-free chat transport."""

        context = self._no_tool_prompt_context()
        retry_feedback = _retry_feedback_block(last_error, mention_tool_calls=False)
        retry_section = "" if not retry_feedback else f"\n\n{retry_feedback}"
        return textwrap.dedent(
            f"""
            Select exactly one next candidate for black-box optimization task
            `{self._agent_visible_task_name()}`.

            Call id: `{call_id}`
            Attempt: `{attempt_index}`

            The host has inlined the same benchmark evidence that the workspace harness
            reads from task.md, space.json, objective.json, history.jsonl, and
            incumbent.json. There are no tools in this no-tool condition.

            Task description:
            {context["task_markdown"]}

            Search space JSON:
            {context["space_json"]}

            Objective JSON:
            {context["objective_json"]}

            Recent evaluated history JSONL:
            {context["history_jsonl"]}

            Incumbent JSON:
            {context["incumbent_json"]}

            Do not execute the evaluator, inspect its hidden implementation, fabricate
            observations, install dependencies, or modify the environment. Modeling
            based only on the visible evaluated history is allowed.

            Return exactly one complete, legal candidate as raw JSON with this shape:

            {{"candidates": [{{"config": {{"<each active parameter>": "<value>"}}}}]}}

            Include every active parameter exactly once, preserve declared JSON types,
            respect bounds and choices, and do not duplicate an evaluated configuration.
            Numeric values should use at most 4 decimal places unless the search space
            declares a different precision.{retry_section}

            Return no Markdown fences, comments, headings, reasoning, explanations, or
            additional prose.
            """
        ).strip()

    def _render_codex_controlled_workspace_instructions(self) -> str:
        return textwrap.dedent(
            """
            # Multi-backend Agentic BO workspace

            This workspace contains read-only benchmark context for one state-gated
            optimization round. Read `task.md`, `space.json`, `objective.json`,
            `history.jsonl`, and `incumbent.json` before deciding.

            Use only the host-mediated BBO CLI described in the invocation prompt for
            optimizer evidence, proposals, validation, and terminal submission. Do not
            call the evaluator, import benchmark internals, fabricate observations,
            install packages, or modify benchmark, environment, history, state, or log
            files. Scratch calculations may be written only below `scratch/`.

            The required lifecycle is evidence, backend selection, proposal, validation,
            and `commit_candidate`. The commit tool is the sole authoritative handoff;
            do not create `final_candidate.json`.
            """
        ).strip()

    def _build_codex_controlled_workspace_prompt(
        self, *, call_id: str, attempt_index: int, last_error: str | None
    ) -> str:
        task_spec = self._require_task_spec()
        best_score = None if self._best is None else self._best.score
        retry_feedback = _retry_feedback_block(last_error)
        retry_section = "" if not retry_feedback else f"\n\n{retry_feedback}"
        return textwrap.dedent(
            f"""
            You are the decision maker for one backend-adaptive black-box optimization
            round on task `{self._agent_visible_task_name()}`.

            Call id: `{call_id}`
            Attempt: `{attempt_index}`
            Objective direction: `{task_spec.primary_objective.direction.value}`
            Current best primary objective: `{best_score}`

            First read `task.md`, `space.json`, `objective.json`, `history.jsonl`,
            `incumbent.json`, and `instructions.md` from the current workspace. Use the
            evaluated history as the only source of objective outcomes and trial IDs.

            Produce exactly one new, nonduplicate candidate. You must independently:

            1. resolve the immediately previous real evaluation as supported,
               contradicted, inconclusive, or not applicable;
            2. collect decision-neutral backend evidence;
            3. choose exactly one backend from `gp_ei` and `tpe` and justify that choice;
            4. inspect its optimizer proposal and freely adopt, refine, or override it;
            5. round and validate the exact final candidate; and
            6. commit it with an evidence-grounded belief, hypothesis, expected
               information, backend rationale, modification rationale, and hypothesis
               update.

            Use only the host-mediated BBO CLI appended below. Do not call the evaluator,
            invent objective values, update optimizer history, import benchmark modules,
            install dependencies, or modify protected workspace/environment files.
            Scratch analysis is allowed only under `scratch/{call_id}/`.

            `commit_candidate` is the sole authoritative terminal handoff. Do not write
            `final_candidate.json` and do not construct a separate compatibility payload.
            After a successful commit, return its `candidate_payload` as raw JSON and stop.{retry_section}
            """
        ).strip()

    def _render_no_tool_instructions(self) -> str:
        task_spec = self._require_task_spec()
        compact_xy_hint = self._compact_xy_output_hint()
        if self.config.framework == "nanobot":
            native_tool_lines = (
                "- Use `read_file` to inspect the workspace files before proposing.\n"
                "            - Use `exec` only for short local calculations or scratch scripts over\n"
                "              workspace data. Use relative paths from the workspace."
            )
        else:
            native_tool_lines = (
                "- Use the harness's native file-reading tools to inspect the workspace files before proposing.\n"
                "            - Native shell tools may be used for short local calculations or scratch scripts over\n"
                "              workspace data. Use relative paths from the workspace."
            )
        return textwrap.dedent(
            f"""
            # Agentic BBO Candidate Protocol

            You are proposing configurations for a black-box optimization benchmark.
            Do not call the benchmark evaluator yourself and do not modify benchmark
            result files.

            Files in this workspace:
            - task.md: task background, goal, constraints, and prior knowledge.
            - manifest.json: agent benchmark construction and provenance.
            - space.json: exact parameter schema. Every candidate must include every parameter exactly once.
            - objective.json: primary objective name and optimization direction.
            - history.jsonl: recent evaluated trials.
            - incumbent.json: current best known configuration, if any.
            - {FINAL_CANDIDATE_FILENAME}: authoritative final candidate handoff. The harness clears it before each attempt.

            Task: {self._agent_visible_task_name()}
            Primary objective: {task_spec.primary_objective.name}
            Direction: {task_spec.primary_objective.direction.value}

            Workspace workflow:
            {native_tool_lines}
            - Do not call the benchmark evaluator or any operation that consumes real
              evaluation budget.
            - You may create temporary scratch files or short analysis scripts inside
              the workspace when useful, using new relative paths such as
              `candidate.json`, `analysis.py`, or `scratch/candidates.json`.
              Do not overwrite task definitions, history, results, logs, or framework
              state files.

            Final candidate handoff:
            - Write the exact top-level payload below to `{FINAL_CANDIDATE_FILENAME}` in the workspace root.
            - Verify JSON syntax with `python -m json.tool {FINAL_CANDIDATE_FILENAME}`.
            - Manually check the exact rounded candidate against `space.json` and
              `history.jsonl` before writing the file.
            - Return the same raw JSON in chat for compatibility. The harness prefers
              `{FINAL_CANDIDATE_FILENAME}` and uses chat only as a fallback.

            Exact payload shape:
            {self._candidate_payload_example()}

            Requirements:
            - Return exactly one candidate configuration. Each real optimization round
              submits one and only one new candidate to the evaluator.
            - Include every active required parameter exactly once.
            - Respect all bounds, choices, precision rules, conditional rules, and constraints.
            - Do not invent objective values or trial IDs.
            - Do not return an exact duplicate of an evaluated configuration.
            - Numeric values in final candidate JSON should use at most 4 decimal places.
            - Categorical values must be one of the declared choices.
            - Use JSON null for absent hypothesis values, not the string "null".
            - The harness independently rechecks schema, bounds, types, and duplicates.
            {compact_xy_hint}
            """
        ).strip()

    def _build_no_tool_agent_prompt(
        self, *, call_id: str, attempt_index: int, last_error: str | None = None
    ) -> str:
        task_spec = self._require_task_spec()
        best_score = None if self._best is None else self._best.score
        retry_feedback = _retry_feedback_block(last_error, mention_tool_calls=False)
        retry_feedback_section = (
            ""
            if not retry_feedback
            else "\n\n" + textwrap.indent(retry_feedback, "            ")
        )
        compact_xy_hint = self._compact_xy_output_hint()
        native_tool_guidance = (
            "Use `read_file` to inspect these files. You may use `exec` for short\n"
            "                local calculations over workspace data or temporary scratch scripts."
            if self.config.framework == "nanobot"
            else "Use the harness's native file-reading tools to inspect these files. You may use\n"
            "                native shell tools for short local calculations or temporary scratch scripts."
        )
        if self.config.framework in {"nanobot", "codex", "claude_code"}:
            return textwrap.dedent(
                f"""
                You are an optimization agent for task `{self._agent_visible_task_name()}`.

                Workspace: `.`
                Call id: `{call_id}`
                Attempt: `{attempt_index}`

                Produce exactly one new candidate configuration.

                Use the workspace as the source of truth:

                * `task.md`: task description, constraints, and prior knowledge
                * `space.json`: parameter names, types, bounds, choices, and precision
                * `objective.json`: objective name and direction
                * `history.jsonl` and `incumbent.json`: evaluated history and current best
                * `instructions.md`: final output and search-action metadata contract

                {native_tool_guidance}
                Use relative paths from the workspace.

                Do not call the benchmark evaluator or any operation that consumes real
                evaluation budget.

                Requirements:

                * Return exactly one candidate configuration.
                * Include every active required parameter exactly once.
                * Preserve the parameter types declared in `space.json`.
                * Respect all bounds, choices, precision rules, conditional rules, and
                  constraints.
                * Do not invent objective values or trial IDs.
                * Do not return an exact duplicate of an evaluated configuration.
                * Near-duplicate candidates are allowed only when justified by local refinement
                  or a controlled experiment.
                * Numeric precision should follow `space.json`. If no precision is declared,
                  use at most 4 decimal places without changing the intended scale.
                * Manually check the final formatted candidate against `space.json` and
                  the evaluated history.

                Describe the search action using:

                {self._search_action_prompt_fields()}

                Only include trial IDs that exist in the workspace history.

                Do not modify protected files:

                * `task.md`
                * `space.json`
                * `objective.json`
                * `history.jsonl`
                * `incumbent.json`
                * `trials.jsonl`
                * `agent_*.jsonl`
                * files under `agent_state/`
                * files under `llm_logs/`
                * files under `reasoning_traces/`

                You may create temporary scratch files or short analysis scripts under a
                new call-specific path such as:

                `scratch/{call_id}/`

                Before finishing, write the exact final payload to
                `{FINAL_CANDIDATE_FILENAME}` in the workspace root and verify it with:

                `python -m json.tool {FINAL_CANDIDATE_FILENAME}`

                Manually check the exact rounded configuration against `space.json` and
                `history.jsonl` before writing it. Return the same JSON in chat; the
                harness treats the file as authoritative and chat as a compatibility fallback.

                If a command fails, recover using the workspace files and still return
                one valid candidate.

                Current best primary objective: `{best_score}`
                Objective direction: `{task_spec.primary_objective.direction.value}`{retry_feedback_section}

                Return only valid raw JSON with exactly this shape:

                {self._candidate_payload_example(indent=2)}

                Replace the example config with the exact active parameters from
                `space.json`.

                Use native JSON types:

                * numbers as numbers
                * integers as integers
                * booleans as booleans
                * categorical values as strings
                * absent hypotheses as JSON `null`, not the string `"null"`

                {compact_xy_hint}

                Return no Markdown fences, comments, headings, explanations, or
                additional prose.
                """
            ).strip()

        context = self._no_tool_prompt_context()
        return textwrap.dedent(
            f"""
            You are an optimization agent for task `{self._agent_visible_task_name()}`.

            Call id: `{call_id}`
            Attempt: `{attempt_index}`

            Produce exactly one new candidate configuration.

            Task description:
            {context["task_markdown"]}

            Search space JSON:
            {context["space_json"]}

            Objective JSON:
            {context["objective_json"]}

            Recent evaluated history JSONL:
            {context["history_jsonl"]}

            Incumbent JSON:
            {context["incumbent_json"]}

            Do not call the benchmark evaluator or any operation that consumes real
            evaluation budget.

            Requirements:

            * Return exactly one candidate configuration.
            * Include every active required parameter exactly once.
            * Preserve the parameter types declared in the search space JSON.
            * Respect all bounds, choices, precision rules, conditional rules, and
              constraints.
            * Do not invent objective values or trial IDs.
            * Do not return an exact duplicate of an evaluated configuration.
            * Near-duplicate candidates are allowed only when justified by local refinement
              or a controlled experiment.
            * Numeric precision should follow the search space JSON. If no precision is
              declared, use at most 4 decimal places without changing the intended scale.
            * Manually check the final formatted candidate against the search space and
              evaluated history.

            Describe the search action using:

            {self._search_action_prompt_fields()}

            Only include trial IDs that exist in the evaluated history.

            Current best primary objective: `{best_score}`
            Objective direction: `{task_spec.primary_objective.direction.value}`{retry_feedback_section}

            Return only valid raw JSON with exactly this shape:

            {self._candidate_payload_example(indent=2)}

            Replace the example config with the exact active parameters from the search
            space JSON.

            Use native JSON types:

            * numbers as numbers
            * integers as integers
            * booleans as booleans
            * categorical values as strings
            * absent hypotheses as JSON `null`, not the string `"null"`

            {compact_xy_hint}

            Return no Markdown fences, comments, headings, explanations, or additional
            prose.
            """
        ).strip()

    def _no_tool_prompt_context(self) -> dict[str, str]:
        task_spec = self._require_task_spec()
        history = (
            self._history[-self.config.history_limit :]
            if self.config.history_limit
            else []
        )
        objective = {
            "name": task_spec.primary_objective.name,
            "direction": task_spec.primary_objective.direction.value,
            "all_objectives": [
                {"name": objective.name, "direction": objective.direction.value}
                for objective in task_spec.objectives
            ],
        }
        incumbent = {
            "config": None
            if self._best is None
            else agent_visible_config(self._best.config),
            "score": None
            if self._best is None
            else agent_visible_payload(self._best.score),
            "objectives": {}
            if self._best is None
            else agent_visible_payload(self._best.objectives),
            "trial_id": None if self._best is None else self._best.trial_id,
        }
        summarize = (
            _observation_summary
            if self._agent_tools_enabled() and not self._raw_workspace_prompt_enabled()
            else _agent_history_summary
        )
        history_jsonl = "\n".join(
            json.dumps(
                to_jsonable(summarize(item)),
                sort_keys=True,
            )
            for item in history
        )
        return {
            "task_markdown": self._render_task_markdown(),
            "space_json": json.dumps(
                {"parameters": search_space_schema(task_spec.search_space)},
                indent=2,
                sort_keys=True,
            ),
            "objective_json": json.dumps(objective, indent=2, sort_keys=True),
            "history_jsonl": history_jsonl or "(empty)",
            "incumbent_json": json.dumps(incumbent, indent=2, sort_keys=True),
        }

    def _search_action_prompt_fields(self) -> str:
        lines = []
        if self._agent_skills_enabled():
            lines.append(
                "* `skill`: the primary skill used, or JSON `null` if no skill was used or available"
            )
        lines.extend(
            [
                "* `parent_trials`: trials from which the candidate was directly modified",
                "* `reference_trials`: trials used as evidence for the decision",
                "* `hypothesis`: the specific hypothesis tested by this candidate, or JSON `null`",
                "* `change_summary`: a concise description of how the candidate was produced",
            ]
        )
        if self.config.require_hypothesis_lifecycle_per_round:
            lines.extend(
                [
                    "* `belief`: your current evidence-grounded belief about the search landscape",
                    "* `expected_information`: what this real evaluation should learn",
                    "* `hypothesis_update`: resolve the immediately previous real evaluation with `status`, `evidence_trial_id`, and `reason`",
                ]
            )
        if self.config.require_optimizer_decision_per_round:
            lines.append(
                "* `optimizer`: an object with `relationship`, `backend`, "
                "`candidate_identity`, and `considered_backends`; use JSON "
                "`null` for fields that do not apply"
            )
        return "\n".join(lines)

    def _candidate_payload_example(self, *, indent: int | None = None) -> str:
        search_action: dict[str, Any] = {
            "parent_trials": [],
            "reference_trials": [],
            "hypothesis": None,
            "change_summary": "Short description of how the candidate was produced.",
        }
        if self.config.require_hypothesis_lifecycle_per_round:
            search_action = {
                "belief": "Current evidence-grounded belief about the search landscape.",
                "expected_information": "What this real evaluation should learn.",
                "hypothesis_update": {
                    "status": "not_applicable",
                    "evidence_trial_id": (
                        None
                        if not self._history
                        else self._history[-1].suggestion.trial_id
                    ),
                    "reason": "Resolve the immediately previous real evaluation.",
                },
                **search_action,
            }
        if self._agent_skills_enabled():
            search_action = {"skill": None, **search_action}
        if self.config.require_optimizer_decision_per_round:
            allowed = list(self.config.optimizer_backend_allowlist)
            example_backend = allowed[0] if allowed else None
            search_action["optimizer"] = {
                "relationship": "adopt",
                "backend": example_backend,
                "candidate_identity": "backend candidate identity or null",
                "considered_backends": allowed,
            }
        payload = {
            "candidates": [
                {
                    "config": {"param_name": 0.0},
                    "rationale": "Short evidence-based reason for proposing this candidate.",
                    "search_action": search_action,
                }
            ]
        }
        return json.dumps(payload, indent=indent, sort_keys=False)

    def _compact_xy_output_hint(self) -> str:
        search_space = self._search_space
        if search_space is None:
            return ""
        n_pairs = _paired_xy_parameter_count(search_space)
        if n_pairs <= 0:
            return ""
        return (
            "For this paired-coordinate task, you may use compact coordinate arrays in the final config: "
            f'`{{"x": [<exactly {n_pairs} numbers>], "y": [<exactly {n_pairs} numbers>]}}`. '
            "The framework expands them to `x_0...` and `y_0...`. This compact form is preferred for BBOPlace."
        )

    def _build_framework_config(self, log_dir: Path) -> Path | None:
        assert self._state_dir is not None
        if self.config.framework == "codex":
            config_path = self._state_dir / "config.toml"
            provider_name = "bbo_sglang"
            model = self.config.model or "qwen3.5-9b"
            base_url = self.config.api_base or "http://127.0.0.1:18300/v1"
            provider = (self.config.provider or "").strip().lower()
            if provider == "deepseek":
                # DeepSeek V4 exposes a 1M-token context window.  Reusing the
                # old local-SGLang 64k metadata forced Codex to compact every
                # few optimization rounds and eventually left some turns with
                # reasoning only and no candidate JSON.
                model_context_window = 1_000_000
                model_auto_compact_token_limit = 900_000
            else:
                model_context_window = 65_536
                model_auto_compact_token_limit = 52_000
            lines = [
                f"model = {json.dumps(model)}",
                f"model_provider = {json.dumps(provider_name)}",
                f"model_context_window = {model_context_window}",
                (
                    "model_auto_compact_token_limit = "
                    f"{model_auto_compact_token_limit}"
                ),
                "project_root_markers = []",
                "",
                "[features]",
                "apps = false",
                "browser_use = false",
                "computer_use = false",
                "goals = false",
                "image_generation = false",
                "multi_agent = false",
                "plugin_sharing = false",
                "plugins = false",
                "remote_plugin = false",
                "skill_search = false",
                "",
                f"[model_providers.{provider_name}]",
                'name = "BBO SGLang Responses API"',
                f"base_url = {json.dumps(base_url.rstrip('/'))}",
                'wire_api = "responses"',
            ]
            if self.config.api_key_env:
                lines.append(f"env_key = {json.dumps(self.config.api_key_env)}")
            config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return config_path
        del log_dir
        return None

    def _agent_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        api_key = self._api_key()
        provider = (self.config.provider or "").lower()
        if api_key:
            if provider == "openai":
                env["OPENAI_API_KEY"] = api_key
            elif provider == "anthropic":
                env["ANTHROPIC_API_KEY"] = api_key
            elif provider == "google":
                env["GOOGLE_API_KEY"] = api_key
            elif self.config.api_key_env:
                env[self.config.api_key_env] = api_key
        if self.config.api_base:
            if provider == "openai":
                env["OPENAI_BASE_URL"] = self.config.api_base
            elif provider == "anthropic":
                env["ANTHROPIC_BASE_URL"] = self.config.api_base
        web_key_env = self.config.web_search_api_key_env
        if (
            not web_key_env
            and self.config.web_search_provider.strip().lower().replace("-", "_")
            == "serpapi"
        ):
            web_key_env = "SERPAPI_API_KEY"
        if web_key_env and os.environ.get(web_key_env):
            env[web_key_env] = os.environ[web_key_env]
        if os.environ.get("SERPAPI_ENDPOINT"):
            env["SERPAPI_ENDPOINT"] = os.environ["SERPAPI_ENDPOINT"]
        if self.config.framework == "nanobot":
            if not self._agent_tools_enabled():
                env["BBO_NANOBOT_NO_TOOL_MODE"] = "1"
            if not self._agent_skills_enabled():
                env["BBO_NANOBOT_NO_SKILL_MODE"] = "1"
        return env


    def _native_round_guard_version(self) -> str | None:
        if (self.config.framework == "codex" and self.config.context_access == "on_demand"
                and (self.config.provider or "").lower() in {"chat_completions", "sinapisai", "voyageage", "sglang", "deepseek"}):
            from .native_round_guard import VERSION
            return VERSION
        return None

    def _codex_config(self) -> dict[str, Any]:
        provider = (self.config.provider or "").lower()
        direct_workspace = self.config.execution_backend == "direct_workspace"
        return {
            "reliable_runtime": self.config.reliable_runtime,
            "native_round_guard": self._native_round_guard_version() is not None,
            "env": self._agent_env(),
            "executable": self.config.executable,
            # Current Codex releases implement workspace-write with bubblewrap.
            # The legacy external-workspace mode deliberately avoids both bwrap
            # and Docker, so its boundary is the dedicated cwd rather than the
            # CLI sandbox implementation.
            "sandbox": "danger-full-access" if direct_workspace else "workspace-write",
            "approval_policy": "never",
            # Legacy native-harness mode (workflow 57): run Codex directly with
            # both cwd and -C set to the per-run agent workspace.  The workspace
            # and CODEX_HOME remain run-local, while evaluator access continues
            # to be mediated by the registered host tools.
            "black_box_required": not direct_workspace,
            "filesystem_boundary": (
                "external_direct_workspace"
                if direct_workspace
                else "docker_read_only_container"
            ),
            "direct_workspace": direct_workspace,
            "execution_backend": self.config.execution_backend,
            "docker_image": self.config.docker_image,
            "docker_cpus": self.config.docker_cpus,
            "tool_mode": self.config.tool_mode,
            "context_access": self.config.context_access,
            "model": self.config.model,
            "api_base": self.config.api_base,
            "api_key_env": self.config.api_key_env,
            "wire_api": "responses",
            "responses_api_compat": (
                "chat_completions" if provider in {"chat_completions", "sinapisai", "voyageage"}
                else provider if provider in {"sglang", "deepseek"} else None
            ),
        }


    def _native_harness_policy(self) -> dict[str, Any]:
        direct_workspace = self.config.execution_backend == "direct_workspace"
        isolated = self.config.execution_backend == "isolated_docker"
        return {
            "framework": self.config.framework,
            "execution_backend": self.config.execution_backend,
            "black_box_boundary": (
                "external_workspace_operational_boundary"
                if direct_workspace
                else "required"
            ),
            "filesystem_isolation": (
                "none_dedicated_cwd_only"
                if direct_workspace
                else "docker_mount_allowlist" if isolated else "minimal_root_or_framework_sandbox"
            ),
            "network_isolation": "none_unix_gateways_only" if isolated else "host_network",
            "evaluator_access": (
                "policy_only_not_os_enforced" if direct_workspace else "denied"
            ),
            "missing_isolation_behavior": (
                "direct_workspace" if direct_workspace else "fail_closed"
            ),
            "native_tools_preserved": True,
            "native_tool_policy": "framework_default",
            "benchmark_tools_enabled": self._agent_tools_enabled(),
            "benchmark_skills_enabled": self._agent_skills_enabled(),
            "tool_mode": self.config.tool_mode,
            "external_user_configuration": "isolated"
            if self.config.framework in {"codex", "claude_code"}
            else "framework_config",
        }

    def _api_key(self) -> str | None:
        if not self.config.api_key_env:
            return None
        return os.environ.get(self.config.api_key_env)

    def _restore_queue_from_snapshot(self) -> None:
        if not self.config.resume or not self._loaded_resume_snapshot:
            return
        search_space = self._require_search_space()
        restored: list[AgentCandidateEntry] = []
        for item in self._loaded_resume_snapshot.get("queue", []):
            if not isinstance(item, Mapping):
                continue
            try:
                config = search_space.coerce_config(
                    dict(item.get("config", {})), use_defaults=False
                )
            except Exception:
                continue
            identity = stable_config_identity(config)
            if identity in self._seen_config_ids:
                continue
            restored.append(
                AgentCandidateEntry(
                    config=config,
                    call_id=str(item.get("call_id", "restored")),
                    candidate_index=int(item.get("candidate_index", 0)),
                    metadata=dict(item.get("metadata", {})),
                )
            )
            self._seen_config_ids.add(identity)
        self._queue = restored
        self._call_index = max(
            self._call_index, int(self._loaded_resume_snapshot.get("call_index", 0))
        )

    def _restore_call_index(self) -> None:
        """Advance past every call id already present in append-only run logs."""
        if not self.config.resume:
            return
        snapshot_index = self._loaded_resume_snapshot.get("call_index", 0)
        try:
            self._call_index = max(self._call_index, int(snapshot_index))
        except (TypeError, ValueError):
            pass
        pattern = re.compile(r"^agent_call_(\d+)$")
        for path in (
            self._agent_prompts_path,
            self._agent_calls_path,
            self._agent_tool_calls_path,
        ):
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(record, Mapping):
                    continue
                for key in ("agent_call_id", "call_id"):
                    match = pattern.fullmatch(str(record.get(key, "")))
                    if match is not None:
                        self._call_index = max(
                            self._call_index, int(match.group(1)) + 1
                        )

    def _load_resume_snapshot(self) -> dict[str, Any]:
        if not self.config.resume or not self._agent_state_path.exists():
            return {}
        try:
            data = json.loads(self._agent_state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _persist_state(self) -> None:
        if self._run_dir is None:
            return
        dump_json(
            self._agent_state_path,
            {
                "algorithm": self.name,
                "protocol_version": (
                    2 if self._controlled_round_protocol_enabled() else 1
                ),
                "framework": self.config.framework,
                "engine": self._engine.name,
                "call_index": self._call_index,
                "persist_session_across_rounds": self.config.persist_session_across_rounds,
                "reliable_runtime": self.config.reliable_runtime,
                "native_round_guard_version": self._native_round_guard_version(),
                "campaign_session_id": (
                    self._campaign_session_id
                    if self.config.persist_session_across_rounds
                    else None
                ),
                "history_size": len(self._history),
                "queue": [to_jsonable(entry) for entry in self._queue],
                "seen_config_ids": sorted(self._seen_config_ids),
                "best_config": None if self._best is None else self._best.config,
                "best_score": None if self._best is None else self._best.score,
                "model": self.config.model,
                "provider": self.config.provider,
                "execution_backend": self.config.execution_backend,
                "docker_cpus": self.config.docker_cpus,
                "executable": self.config.executable,
                "tool_mode": self.config.tool_mode,
                "max_tool_calls": self.config.max_tool_calls,
                "max_output_tokens": self.config.max_output_tokens,
                "thinking_mode": self.config.thinking_mode,
                "enabled_tool_names": (
                    None
                    if self.config.enabled_tool_names is None
                    else list(self.config.enabled_tool_names)
                ),
                "optimizer_backend_allowlist": list(
                    self.config.optimizer_backend_allowlist
                ),
                "optimizer_max_calls_per_round": self.config.optimizer_max_calls_per_round,
                "experiment_condition": self.config.experiment_condition,
                "context_profile": self.config.context_policy.profile.value,
                "context_policy": self.config.context_policy.to_dict(),
                "context_fingerprint": self._agent_context_fingerprint,
                "context_max_evaluations": self._require_task_spec().max_evaluations,
                "context_access": self.config.context_access,
                "agent_task_alias": self._agent_task_alias,
                "require_analysis_evidence_per_round": self.config.require_analysis_evidence_per_round,
                "required_tool_names_per_round": list(
                    self.config.required_tool_names_per_round
                ),
                "require_candidate_validation_per_round": self.config.require_candidate_validation_per_round,
                "require_optimizer_decision_per_round": self.config.require_optimizer_decision_per_round,
                "require_hypothesis_lifecycle_per_round": self.config.require_hypothesis_lifecycle_per_round,
                "require_evidence_bound_reconfiguration": self.config.require_evidence_bound_reconfiguration,
                "enable_memory": self.config.enable_memory,
                "web_search_provider": self.config.web_search_provider,
                "code_backend": self.config.code_backend,
                "docker_image": self.config.docker_image,
                "allow_fallback": self.config.allow_fallback,
                "require_visible_cot": self.config.require_visible_cot,
                "enable_bbo_skills": self.config.enable_bbo_skills,
                "skill_paths": [str(path) for path in self.config.skill_paths],
                "harness_policy": self._native_harness_policy(),
            },
        )

    def _audit_workspace_boundary(self, result: AgentResult) -> None:
        """Record cooperative-workspace boundary violations without discarding artifacts."""

        if self.config.context_policy.identity_exposure.value == "public_instance":
            return
        evidence: list[str] = []
        texts = [result.answer or "", result.error or ""]
        reasoning_dir = self._agent_reasoning_traces_dir if self._run_dir is not None else None
        if reasoning_dir is not None and reasoning_dir.exists():
            for path in reasoning_dir.glob("*.json*"):
                try:
                    texts.append(path.read_text(encoding="utf-8"))
                except OSError:
                    continue
        patterns = {
            "parent_traversal": re.compile(r"(?:^|[\s\"'])(?:\.\./|cd\s+\.\.)"),
            "host_absolute_path": re.compile(r"(?:/home/|/root/|/workspace/\.\.)"),
            "broad_filesystem_probe": re.compile(r"(?:find|rg|ls)\s+/(?:\s|$)"),
        }
        joined = "\n".join(texts)
        for kind, pattern in patterns.items():
            if pattern.search(joined):
                evidence.append(kind)
        path = self._run_dir / "agent_context.json" if self._run_dir is not None else None
        if path is None:
            return
        existing: dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
        previous = existing.get("protocol_compliance")
        if evidence:
            existing["protocol_compliance"] = "violated"
            existing["violation_type"] = "out_of_workspace_read"
            existing["violation_evidence"] = sorted(
                set(existing.get("violation_evidence", [])) | set(evidence)
            )
        elif previous != "violated":
            existing["protocol_compliance"] = "compliant"
        dump_json(path, existing)

    @property
    def _agent_calls_path(self) -> Path:
        assert self._run_dir is not None
        return self._run_dir / "agent_calls.jsonl"

    @property
    def _agent_prompts_path(self) -> Path:
        assert self._run_dir is not None
        return self._run_dir / "agent_prompts.jsonl"

    @property
    def _agent_state_path(self) -> Path:
        assert self._run_dir is not None
        return self._run_dir / "agent_state.json"

    @property
    def _agent_optimization_trace_path(self) -> Path:
        assert self._run_dir is not None
        return self._run_dir / "agent_optimization_trace.jsonl"

    @property
    def _agent_round_events_path(self) -> Path:
        assert self._run_dir is not None
        return self._run_dir / "agent_round_events.jsonl"

    @property
    def _agent_tool_calls_path(self) -> Path:
        assert self._run_dir is not None
        if self.config.context_policy.identity_exposure.value != "public_instance":
            assert self._workspace_dir is not None
            return self._workspace_dir / ".agent_runtime" / "agent_tool_calls.jsonl"
        return self._run_dir / "agent_tool_calls.jsonl"

    @property
    def _agent_tool_specs_path(self) -> Path:
        assert self._run_dir is not None
        return self._run_dir / "agent_tool_specs.json"

    @property
    def _agent_sources_path(self) -> Path:
        assert self._run_dir is not None
        if self.config.context_policy.identity_exposure.value != "public_instance":
            assert self._workspace_dir is not None
            return self._workspace_dir / ".agent_runtime" / "agent_web_sources.jsonl"
        return self._run_dir / "agent_web_sources.jsonl"

    @property
    def _agent_memory_path(self) -> Path:
        assert self._memory_dir is not None
        return self._memory_dir / "memory.jsonl"

    @property
    def _agent_memory_summary_path(self) -> Path:
        assert self._memory_dir is not None
        return self._memory_dir / "memory_summary.json"

    @property
    def _agent_reasoning_traces_dir(self) -> Path:
        assert self._run_dir is not None
        if self.config.context_policy.identity_exposure.value != "public_instance":
            assert self._workspace_dir is not None
            return self._workspace_dir / ".agent_runtime" / "reasoning_traces"
        return self._run_dir / "reasoning_traces"

    @property
    def _agent_reasoning_metadata_path(self) -> Path:
        assert self._run_dir is not None
        if self.config.context_policy.identity_exposure.value != "public_instance":
            assert self._workspace_dir is not None
            return self._workspace_dir / ".agent_runtime" / "agent_reasoning_metadata.jsonl"
        return self._run_dir / "agent_reasoning_metadata.jsonl"

    def _require_ready(self) -> None:
        if self._task_spec is None or self._search_space is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.setup() must be called before use."
            )

    def _require_task_spec(self) -> TaskSpec:
        self._require_ready()
        assert self._task_spec is not None
        return self._task_spec

    def _require_search_space(self) -> SearchSpace:
        self._require_ready()
        assert self._search_space is not None
        return self._search_space






class CodexBBOAlgorithm(GeneralAgentBBOAlgorithm):
    """General-agent optimizer backed by Codex CLI."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(framework="codex", algorithm_name="agentic_codex", **kwargs)




def _observation_summary(observation: TrialObservation) -> dict[str, Any]:
    return {
        "trial_id": observation.suggestion.trial_id,
        "config": agent_visible_config(observation.suggestion.config),
        "budget": agent_visible_payload(observation.suggestion.budget),
        "status": observation.status.value,
        "objectives": agent_visible_payload(observation.objectives),
        "metrics": agent_visible_metrics(observation.metrics),
        "elapsed_seconds": agent_visible_payload(observation.elapsed_seconds),
        "error_type": observation.error_type,
        "error_message": observation.error_message,
        "timestamp": agent_visible_payload(observation.timestamp),
        "metadata": agent_visible_metadata(observation.metadata),
        "suggestion_metadata": sanitize_agent_context_payload(observation.suggestion.metadata),
        "search_action": agent_visible_payload(
            observation.suggestion.metadata.get("search_action", {})
        ),
    }


def _agent_history_summary(observation: TrialObservation) -> dict[str, Any]:
    """Return compact optimization evidence without benchmark audit metadata."""

    payload: dict[str, Any] = {
        "trial_id": observation.suggestion.trial_id,
        "config": agent_visible_config(observation.suggestion.config),
        "budget": agent_visible_payload(observation.suggestion.budget),
        "status": observation.status.value,
        "objectives": agent_visible_payload(observation.objectives),
        "metrics": agent_visible_metrics(observation.metrics),
    }
    search_action = agent_visible_payload(
        observation.suggestion.metadata.get("search_action", {})
    )
    if search_action:
        payload["search_action"] = search_action
    if observation.error_type:
        payload["error_type"] = observation.error_type
    if observation.error_message:
        payload["error_message"] = observation.error_message
    return payload


def _run_coro_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_box: dict[str, Any] = {}
    error_box: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result_box["result"] = asyncio.run(coro)
        except (
            BaseException
        ) as exc:  # pragma: no cover - defensive cross-thread propagation.
            error_box["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if error_box:
        raise error_box["error"]
    return result_box["result"]


def _normalize_skill_paths(
    skill_paths: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
) -> tuple[Path, ...]:
    if skill_paths is None:
        return ()
    if isinstance(skill_paths, (str, Path)):
        raw_paths = [skill_paths]
    else:
        raw_paths = list(skill_paths)
    return tuple(Path(path).expanduser() for path in raw_paths)


def normalize_agent_tool_mode(raw: str) -> str:
    normalized = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "no_tools": "no_tool",
        "none": "no_tool",
        "disabled": "no_tool",
        "off": "no_tool",
        "false": "no_tool",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in AGENT_TOOL_MODES:
        choices = ", ".join(AGENT_TOOL_MODE_CLI_CHOICES)
        raise ValueError(f"tool_mode must be one of: {choices}.")
    return normalized


def normalize_agent_execution_backend(
    raw: str | None,
    *,
    framework: str,
    code_backend: str,
    docker_image: str,
) -> str:
    """Resolve the model runtime independently from the code-tool backend."""

    if raw is None:
        # v6 formal native runs use a dedicated workspace. Claude's current SDK
        # transport cannot provide that out-of-process boundary, so its legacy
        # sealed behavior remains explicit in recorded settings.
        normalized = (
            "sealed_docker_legacy" if framework == "claude_code" else "direct_workspace"
        )
    else:
        normalized = str(raw).strip().lower().replace("-", "_")
        normalized = {
            "workspace": "direct_workspace",
            "direct": "direct_workspace",
            "docker": "sealed_docker_legacy",
            "sealed_docker": "sealed_docker_legacy",
        }.get(normalized, normalized)
    if normalized not in AGENT_EXECUTION_BACKENDS:
        raise ValueError(
            "execution_backend must be `direct_workspace`, "
            "`sealed_docker_legacy`, or `isolated_docker`."
        )
    if normalized == "isolated_docker" and framework != "codex":
        raise ValueError("isolated_docker currently supports the native Codex harness only.")
    if normalized != "direct_workspace" and str(docker_image).strip().lower() in {
        "disabled",
        "none",
        "off",
        "false",
    }:
        raise ValueError(f"{normalized} requires a real native container image.")
    # code_backend is intentionally unused for selection; accepting it here
    # makes that separation visible at the normalization boundary.
    del code_backend
    return normalized


def _optimizer_visible_task_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    visible: dict[str, Any] = {}
    transforms = metadata.get("parameter_transforms")
    if isinstance(transforms, Mapping):
        visible["parameter_transforms"] = dict(transforms)
    protocol = metadata.get("benchmark_protocol")
    if isinstance(protocol, Mapping):
        candidate_budget = protocol.get("candidate_budget")
        if isinstance(candidate_budget, Mapping):
            visible["benchmark_protocol"] = {"candidate_budget": dict(candidate_budget)}
    return visible


def _remove_path(path: Path) -> None:
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        pass


def _packaged_bbo_nanobot_skills_dir() -> Path:
    return Path(__file__).with_name("skills")


def _discover_nanobot_skill_dirs(path: Path) -> list[Path]:
    source = path if path.is_absolute() else path.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Nanobot skill path not found: {source}")
    if not source.is_dir():
        raise ValueError(f"Nanobot skill path must be a directory: {source}")
    if (source / "SKILL.md").exists():
        return [source]
    skill_dirs = sorted(
        child
        for child in source.iterdir()
        if child.is_dir() and (child / "SKILL.md").exists()
    )
    if not skill_dirs:
        raise ValueError(
            f"Nanobot skill path contains no skill directories with SKILL.md: {source}"
        )
    return skill_dirs


def _nanobot_skill_name(skill_dir: Path) -> str:
    source = skill_dir if skill_dir.is_absolute() else skill_dir.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Nanobot skill directory not found: {source}")
    if not source.is_dir():
        raise ValueError(f"Nanobot skill source must be a directory: {source}")
    skill_file = source / "SKILL.md"
    if not skill_file.exists():
        raise ValueError(f"Nanobot skill directory is missing SKILL.md: {source}")
    declared_name = _read_skill_frontmatter_name(skill_file)
    name = declared_name or source.name
    if name != source.name:
        raise ValueError(
            f"Nanobot skill name `{name}` must match directory name `{source.name}`."
        )
    if not _NANOBOT_SKILL_NAME_RE.fullmatch(name):
        raise ValueError(
            f"Nanobot skill name `{name}` must use lowercase letters, digits, and single hyphens only."
        )
    return name


def _skill_index_entry(skill_name: str) -> dict[str, Any]:
    groups = SKILL_EVIDENCE_TOOL_GROUPS.get(skill_name, ())
    return {
        "name": skill_name,
        "search_intent": SKILL_TO_SEARCH_INTENT.get(skill_name),
        "proposal_allowed": skill_name not in NON_PROPOSAL_BBO_SKILLS,
        "evidence_tools": [list(group) for group in groups],
    }


def _read_skill_frontmatter_name(skill_file: Path) -> str | None:
    try:
        lines = skill_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    if not lines or lines[0].strip() != "---":
        return None
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "---":
            return None
        if stripped.startswith("name:"):
            return stripped.split(":", 1)[1].strip().strip("\"'")
    return None






__all__ = [
    "AGENT_EXECUTION_BACKENDS",
    "CodexBBOAlgorithm",
    "GeneralAgentBBOAlgorithm",
    "GeneralAgentConfig",
    "GeneralAgentValidationError",
    "AGENT_TOOL_MODE_CLI_CHOICES",
    "AGENT_TOOL_MODES",
    "normalize_agent_execution_backend",
    "normalize_agent_tool_mode",
    "parse_agent_candidate_payload",
    "search_space_schema",
]
