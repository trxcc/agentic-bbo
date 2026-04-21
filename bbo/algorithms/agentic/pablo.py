"""Pablo: a benchmark-internal agentic optimizer with stateless role calls."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...core import Incumbent, ObjectiveDirection, SearchSpace, TaskDescriptionBundle, TaskSpec, TrialObservation, TrialSuggestion
from ...core.algo import Algorithm
from .llm_client import PabloLlmClient, PabloProviderConfig, create_llm_client
from .model_routing import PabloModelRoutingConfig, build_routing_table
from .prompts import PromptBundle, build_explorer_prompt, build_planner_prompt, build_worker_prompt
from .serialization import append_jsonl, dump_json, prompt_hash, stable_config_identity, to_jsonable
from .state import CandidateEntry, PabloResumeState
from .task_registry import TaskRegistry
from .validation import PabloValidationError, validate_candidate_payload, validate_planner_tasks

DEFAULT_INIT_POINTS = 4
DEFAULT_MAX_FAILS = 3
DEFAULT_NUM_SEEDS = 2
DEFAULT_MAX_TASKS = 20
DEFAULT_GLOBAL_MODEL = "gpt-4.1-mini"


@dataclass(frozen=True)
class PabloConfig:
    provider: str = "mock"
    base_url: str | None = None
    api_key_env: str = "PABLO_API_KEY"
    model: str = DEFAULT_GLOBAL_MODEL
    global_model: str | None = None
    worker_model: str | None = None
    planner_model: str | None = None
    explorer_model: str | None = None
    init_points: int = DEFAULT_INIT_POINTS
    max_fails: int = DEFAULT_MAX_FAILS
    num_seeds: int = DEFAULT_NUM_SEEDS
    max_tasks: int = DEFAULT_MAX_TASKS
    enable_explorer: bool = True
    enable_planner: bool = True
    enable_worker: bool = True
    run_dir: Path | None = None
    resume: bool = False


class PabloAlgorithm(Algorithm):
    """Ask/tell optimizer that separates Explorer, Planner, and Worker inputs."""

    def __init__(
        self,
        *,
        provider: str = "mock",
        base_url: str | None = None,
        api_key_env: str = "PABLO_API_KEY",
        model: str = DEFAULT_GLOBAL_MODEL,
        global_model: str | None = None,
        worker_model: str | None = None,
        planner_model: str | None = None,
        explorer_model: str | None = None,
        init_points: int = DEFAULT_INIT_POINTS,
        max_fails: int = DEFAULT_MAX_FAILS,
        num_seeds: int = DEFAULT_NUM_SEEDS,
        max_tasks: int = DEFAULT_MAX_TASKS,
        enable_explorer: bool = True,
        enable_planner: bool = True,
        enable_worker: bool = True,
        run_dir: Path | str | None = None,
        resume: bool = False,
    ) -> None:
        if init_points <= 0:
            raise ValueError("init_points must be positive.")
        if max_fails <= 0:
            raise ValueError("max_fails must be positive.")
        if num_seeds <= 0:
            raise ValueError("num_seeds must be positive.")
        if max_tasks < 3:
            raise ValueError("max_tasks must be at least 3.")
        if not (enable_explorer or enable_planner or enable_worker):
            raise ValueError("At least one Pablo role must be enabled.")

        self.config = PabloConfig(
            provider=provider,
            base_url=base_url,
            api_key_env=api_key_env,
            model=model,
            global_model=global_model,
            worker_model=worker_model,
            planner_model=planner_model,
            explorer_model=explorer_model,
            init_points=init_points,
            max_fails=max_fails,
            num_seeds=num_seeds,
            max_tasks=max_tasks,
            enable_explorer=enable_explorer,
            enable_planner=enable_planner,
            enable_worker=enable_worker,
            run_dir=None if run_dir is None else Path(run_dir),
            resume=bool(resume),
        )
        self._routing = build_routing_table(
            PabloModelRoutingConfig(
                model=model,
                global_model=global_model,
                worker_model=worker_model,
                planner_model=planner_model,
                explorer_model=explorer_model,
            )
        )
        self._client: PabloLlmClient | None = None
        self._task_spec: TaskSpec | None = None
        self._description = TaskDescriptionBundle.empty(task_id="unknown")
        self._search_space: SearchSpace | None = None
        self._seed = 0
        self._rng = random.Random(0)
        self._round_index = 0
        self._failure_streak = 0
        self._history: list[TrialObservation] = []
        self._queue: list[CandidateEntry] = []
        self._seen_config_ids: set[str] = set()
        self._best: Incumbent | None = None
        self._primary_name: str | None = None
        self._primary_direction = ObjectiveDirection.MINIMIZE
        self._task_registry: TaskRegistry | None = None
        self._run_dir: Path | None = None
        self._artifacts: dict[str, str] = {}
        self._loaded_resume_snapshot: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "pablo"

    @property
    def routing_table(self) -> dict[str, str]:
        return dict(self._routing)

    @property
    def artifact_paths(self) -> dict[str, str]:
        return dict(self._artifacts)

    def setup(self, task_spec: TaskSpec, seed: int = 0, **kwargs: Any) -> None:
        if len(task_spec.objectives) != 1:
            raise ValueError("PabloAlgorithm currently supports exactly one objective.")
        if task_spec.description_ref is None:
            raise ValueError("PabloAlgorithm requires a structured task description.")

        self._task_spec = task_spec
        self._search_space = task_spec.search_space
        self._primary_name = task_spec.primary_objective.name
        self._primary_direction = task_spec.primary_objective.direction
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        description = kwargs.get("task_description")
        if isinstance(description, TaskDescriptionBundle):
            self._description = description
        else:
            self._description = TaskDescriptionBundle.empty(task_id=task_spec.name)

        self._run_dir = Path(kwargs.get("run_dir") or self.config.run_dir or Path.cwd())
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts = {
            "pablo_rounds_jsonl": str(self._run_dir / "pablo_rounds.jsonl"),
            "task_registry_json": str(self._run_dir / "task_registry.json"),
            "llm_calls_jsonl": str(self._run_dir / "llm_calls.jsonl"),
            "candidate_queue_jsonl": str(self._run_dir / "candidate_queue.jsonl"),
            "resume_state_json": str(self._run_dir / "resume_state.json"),
        }
        for artifact_path in self._artifacts.values():
            Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
            Path(artifact_path).touch(exist_ok=True)
        self._queue = []
        self._seen_config_ids = set()
        self._history = []
        self._best = None
        self._round_index = 0
        self._failure_streak = 0
        self._task_registry = TaskRegistry(default_tasks=self._default_tasks(task_spec), max_tasks=self.config.max_tasks)
        self._client = create_llm_client(
            PabloProviderConfig(
                provider=self.config.provider,
                base_url=self.config.base_url,
                api_key_env=self.config.api_key_env,
            ),
            seed=self._seed,
        )
        self._load_resume_snapshot()
        self._persist_state()

    def ask(self) -> TrialSuggestion:
        self._require_ready()
        if len(self._history) < self.config.init_points:
            return self._next_initial_suggestion()
        if not self._queue:
            self._plan_round()
        if not self._queue:
            raise RuntimeError("Pablo could not produce any candidate configurations.")
        entry = self._queue.pop(0)
        append_jsonl(
            self._candidate_queue_path,
            {
                "event": "dequeue",
                "round_index": entry.round_index,
                "role": entry.role,
                "source": entry.source,
                "task_name": entry.task_name,
                "seed_index": entry.seed_index,
                "config": entry.config,
                "queue_size_after": len(self._queue),
                "timestamp": time.time(),
            },
        )
        metadata = {
            "pablo_source": entry.source,
            "pablo_role": entry.role,
            "pablo_round": entry.round_index,
            "pablo_provider": self.config.provider,
            "pablo_task_name": entry.task_name,
            "pablo_seed_index": entry.seed_index,
            **entry.metadata,
        }
        self._persist_state()
        return TrialSuggestion(config=dict(entry.config), metadata=metadata)

    def tell(self, observation: TrialObservation) -> None:
        self._ingest_observation(observation)
        self._persist_state()

    def replay(self, history: list[TrialObservation]) -> None:
        self._require_ready()
        self._history = []
        self._queue = []
        self._seen_config_ids = set()
        self._best = None
        self._failure_streak = 0
        assert self._task_spec is not None
        self._task_registry = TaskRegistry(default_tasks=self._default_tasks(self._task_spec), max_tasks=self.config.max_tasks)
        for observation in history:
            self._ingest_observation(observation, replay=True)
        if self._loaded_resume_snapshot:
            self._round_index = max(self._round_index, int(self._loaded_resume_snapshot.get("round_index", 0)))
        self._persist_state()

    def incumbents(self) -> list[Incumbent]:
        return [self._best] if self._best is not None else []

    def _ingest_observation(self, observation: TrialObservation, *, replay: bool = False) -> None:
        assert self._primary_name is not None
        self._history.append(observation)
        identity = stable_config_identity(observation.suggestion.config)
        self._seen_config_ids.add(identity)

        task_name = observation.suggestion.metadata.get("pablo_task_name")
        if isinstance(task_name, str) and task_name and task_name != "INIT":
            assert self._task_registry is not None
            self._task_registry.record_attempt(task_name, success=observation.success)
            dump_json(self._task_registry_path, self._task_registry.snapshot())

        if observation.success and self._primary_name in observation.objectives:
            score = float(observation.objectives[self._primary_name])
            incumbent = Incumbent(
                config=dict(observation.suggestion.config),
                score=score,
                objectives=dict(observation.objectives),
                trial_id=observation.suggestion.trial_id,
                metadata={"algorithm": self.name},
            )
            if self._best is None:
                self._best = incumbent
            elif self._primary_direction == ObjectiveDirection.MINIMIZE and score < float(self._best.score):
                self._best = incumbent
            elif self._primary_direction == ObjectiveDirection.MAXIMIZE and score > float(self._best.score):
                self._best = incumbent

        if not replay:
            dump_json(self._task_registry_path, self._task_registry.snapshot() if self._task_registry is not None else {})

    def _next_initial_suggestion(self) -> TrialSuggestion:
        search_space = self._require_search_space()
        index = len(self._history)
        if index == 0:
            config = search_space.defaults()
        else:
            config = self._sample_unique_random_config()
        metadata = {
            "pablo_source": "initial_design",
            "pablo_role": "init",
            "pablo_round": 0,
            "pablo_provider": self.config.provider,
            "pablo_task_name": "INIT",
            "pablo_seed_index": None,
        }
        return TrialSuggestion(config=config, metadata=metadata)

    def _plan_round(self) -> None:
        self._require_ready()
        if self._failure_streak >= self.config.max_fails:
            raise RuntimeError(
                f"Pablo exhausted its planning budget after {self._failure_streak} consecutive empty rounds."
            )

        assert self._task_spec is not None
        assert self._task_registry is not None
        self._round_index += 1
        c_global = self._build_c_global()
        performance_stats = self._build_performance_stats()
        existing_tasks_summary = self._task_registry.summary(limit=12)
        round_record: dict[str, Any] = {
            "round_index": self._round_index,
            "best_before_round": None if self._best is None else self._best.score,
            "history_size": len(self._history),
            "provider": self.config.provider,
            "model_routes": self.routing_table,
            "planner_tasks": [],
            "explorer_candidate_count": 0,
            "worker_batches": [],
            "timestamp": time.time(),
        }

        if self.config.enable_planner:
            planner_prompt = build_planner_prompt(
                task_spec=self._task_spec,
                description=self._description,
                c_global=c_global,
                performance_stats=performance_stats,
                existing_tasks_summary=existing_tasks_summary,
            )
            planner_tasks = self._invoke_planner(planner_prompt)
            self._task_registry.update_from_planner(planner_tasks)
        else:
            planner_tasks = {card.name: card.text for card in self._task_registry.active_tasks(limit=8)}
        round_record["planner_tasks"] = list(planner_tasks)
        dump_json(self._task_registry_path, self._task_registry.snapshot())

        added = 0
        if self.config.enable_explorer:
            explorer_prompt = build_explorer_prompt(
                task_spec=self._task_spec,
                description=self._description,
                c_global=c_global,
                best_objective=None if self._best is None else self._best.score,
            )
            explorer_candidates = self._invoke_candidate_role("explorer", explorer_prompt)
            round_record["explorer_candidate_count"] = len(explorer_candidates)
            for config in explorer_candidates:
                added += self._enqueue_candidate(
                    CandidateEntry(
                        config=config,
                        source="explorer",
                        role="explorer",
                        round_index=self._round_index,
                        task_name=None,
                        metadata={"pablo_model": self._routing["explorer"]},
                    )
                )

        if self.config.enable_worker:
            for task_name, task_text in planner_tasks.items():
                seeds = self._select_worker_seeds()
                batch_counts: list[int] = []
                for seed_index, seed_config in enumerate(seeds):
                    worker_prompt = build_worker_prompt(
                        task_spec=self._task_spec,
                        planner_task_name=task_name,
                        planner_task_text=task_text,
                        current_seed=seed_config,
                    )
                    worker_candidates = self._invoke_candidate_role("worker", worker_prompt)
                    batch_counts.append(len(worker_candidates))
                    for config in worker_candidates:
                        added += self._enqueue_candidate(
                            CandidateEntry(
                                config=config,
                                source="worker",
                                role="worker",
                                round_index=self._round_index,
                                task_name=task_name,
                                seed_index=seed_index,
                                metadata={
                                    "pablo_model": self._routing["worker"],
                                    "pablo_planner_task_text": task_text,
                                },
                            )
                        )
                round_record["worker_batches"].append(
                    {"task_name": task_name, "seed_count": len(seeds), "candidate_counts": batch_counts}
                )

        round_record["queue_size_after_round"] = len(self._queue)
        round_record["queue_added"] = added
        append_jsonl(self._rounds_path, round_record)
        self._persist_state()

        if not self._queue:
            self._failure_streak += 1
            if self._failure_streak >= self.config.max_fails:
                raise RuntimeError(
                    f"Pablo planned round {self._round_index} but produced no usable candidates after deduplication."
                )
            return
        self._failure_streak = 0

    def _invoke_planner(self, prompt: PromptBundle) -> dict[str, str]:
        raw = self._invoke_role_raw("planner", prompt)
        return validate_planner_tasks(raw)

    def _invoke_candidate_role(self, role: str, prompt: PromptBundle) -> list[dict[str, Any]]:
        raw = self._invoke_role_raw(role, prompt)
        return validate_candidate_payload(raw, self._require_search_space())

    def _invoke_role_raw(self, role: str, prompt: PromptBundle) -> str:
        client = self._require_client()
        model = self._routing[role]
        prompt_id = prompt_hash(prompt.system, prompt.user)
        call_record = {
            "role": role,
            "model": model,
            "provider": self.config.provider,
            "prompt_hash": prompt_id,
            "prompt_size": len(prompt.system) + len(prompt.user),
            "timestamp": time.time(),
        }
        try:
            raw = client.complete(role=role, model=model, prompt=prompt)
            if role == "planner":
                parsed = validate_planner_tasks(raw)
                summary = {"task_count": len(parsed), "task_names": list(parsed)[:10]}
            else:
                parsed = validate_candidate_payload(raw, self._require_search_space())
                summary = {"candidate_count": len(parsed)}
            call_record["parse_status"] = "ok"
            call_record["response_summary"] = summary
            append_jsonl(self._llm_calls_path, call_record)
            return raw
        except Exception as exc:
            call_record["parse_status"] = type(exc).__name__
            call_record["response_summary"] = {"error": str(exc)}
            append_jsonl(self._llm_calls_path, call_record)
            if isinstance(exc, PabloValidationError):
                raise
            raise

    def _build_c_global(self) -> list[dict[str, Any]]:
        assert self._primary_name is not None
        scored_successes = [obs for obs in self._history if obs.success and self._primary_name in obs.objectives]
        scored_successes.sort(
            key=lambda obs: float(obs.objectives[self._primary_name]),
            reverse=self._primary_direction == ObjectiveDirection.MAXIMIZE,
        )
        top_successes = scored_successes[:5]
        remainder = [obs for obs in self._history if obs not in top_successes]
        sampled_remainder = self._uniform_history_sample(remainder, limit=5)
        selected = top_successes + sampled_remainder
        context: list[dict[str, Any]] = []
        for observation in selected:
            score = observation.objectives.get(self._primary_name)
            context.append(
                {
                    "trial_id": observation.suggestion.trial_id,
                    "status": observation.status.value,
                    "primary_objective": score,
                    "config": dict(observation.suggestion.config),
                }
            )
        return context

    def _build_performance_stats(self) -> dict[str, Any]:
        assert self._primary_name is not None
        successes = [obs for obs in self._history if obs.success]
        latest_success = successes[-1] if successes else None
        return {
            "n_trials": len(self._history),
            "successful_trials": len(successes),
            "failed_or_invalid_trials": len(self._history) - len(successes),
            "best_primary_objective": None if self._best is None else self._best.score,
            "latest_primary_objective": (
                None
                if latest_success is None
                else latest_success.objectives.get(self._primary_name)
            ),
        }

    def _select_worker_seeds(self) -> list[dict[str, Any]]:
        search_space = self._require_search_space()
        seeds: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(config: dict[str, Any]) -> None:
            identity = stable_config_identity(config)
            if identity in seen:
                return
            seen.add(identity)
            seeds.append(dict(config))

        if self._best is not None:
            add(self._best.config)
        add(search_space.defaults())
        successful_history = [obs for obs in self._history if obs.success]
        for observation in reversed(successful_history):
            add(dict(observation.suggestion.config))
            if len(seeds) >= self.config.num_seeds:
                break
        while len(seeds) < self.config.num_seeds:
            add(self._sample_unique_random_config())
        return seeds[: self.config.num_seeds]

    def _enqueue_candidate(self, entry: CandidateEntry) -> int:
        identity = stable_config_identity(entry.config)
        if identity in self._seen_config_ids:
            return 0
        if any(stable_config_identity(candidate.config) == identity for candidate in self._queue):
            return 0
        self._queue.append(entry)
        append_jsonl(
            self._candidate_queue_path,
            {
                "event": "enqueue",
                "round_index": entry.round_index,
                "role": entry.role,
                "source": entry.source,
                "task_name": entry.task_name,
                "seed_index": entry.seed_index,
                "config": entry.config,
                "queue_size_after": len(self._queue),
                "timestamp": time.time(),
            },
        )
        return 1

    def _sample_unique_random_config(self) -> dict[str, Any]:
        search_space = self._require_search_space()
        for _ in range(512):
            config = search_space.sample(self._rng)
            identity = stable_config_identity(config)
            if identity not in self._seen_config_ids and not any(
                stable_config_identity(candidate.config) == identity for candidate in self._queue
            ):
                return config
        raise RuntimeError("Pablo could not sample a new unique configuration from the search space.")

    def _uniform_history_sample(self, history: list[TrialObservation], *, limit: int) -> list[TrialObservation]:
        if len(history) <= limit:
            return list(history)
        if limit <= 0:
            return []
        stride = max(1, len(history) // limit)
        sampled = history[::stride]
        return sampled[:limit]

    def _default_tasks(self, task_spec: TaskSpec) -> dict[str, str]:
        has_categorical = any(hasattr(param, "choices") for param in task_spec.search_space)
        default_tasks = {
            "EXPLOIT_BEST": "TASK: refine the best-known region with conservative edits.",
            "LOCAL_NEIGHBORS": "TASK: generate close neighbors of strong seeds while preserving validity.",
            "DIVERSE_EXPLORATION": "TASK: search for novel but valid candidates away from recent repeats.",
        }
        if has_categorical:
            default_tasks["DIVERSE_EXPLORATION"] = (
                "TASK: search for novel valid candidates by rotating categorical choices and re-balancing numerics."
            )
        return default_tasks

    def _load_resume_snapshot(self) -> None:
        path = self._resume_state_path
        if not self.config.resume or not path.exists():
            self._loaded_resume_snapshot = {}
            return
        try:
            import json

            self._loaded_resume_snapshot = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self._loaded_resume_snapshot = {}

    def _persist_state(self) -> None:
        resume_state = PabloResumeState(
            round_index=self._round_index,
            failure_streak=self._failure_streak,
            queue=list(self._queue),
            task_registry={} if self._task_registry is None else self._task_registry.snapshot(),
            seen_config_ids=sorted(self._seen_config_ids),
            model_routes=self.routing_table,
            best_config=None if self._best is None else dict(self._best.config),
            best_score=None if self._best is None else self._best.score,
            history_size=len(self._history),
            provider=self.config.provider,
        )
        dump_json(self._resume_state_path, resume_state)
        dump_json(self._task_registry_path, {} if self._task_registry is None else self._task_registry.snapshot())

    def _require_ready(self) -> None:
        if self._task_spec is None or self._search_space is None or self._task_registry is None:
            raise RuntimeError("PabloAlgorithm.setup() must be called before ask/tell.")

    def _require_client(self) -> PabloLlmClient:
        if self._client is None:
            raise RuntimeError("PabloAlgorithm.setup() must be called before role execution.")
        return self._client

    def _require_search_space(self) -> SearchSpace:
        if self._search_space is None:
            raise RuntimeError("PabloAlgorithm.setup() must be called before ask/tell.")
        return self._search_space

    @property
    def _rounds_path(self) -> Path:
        return Path(self._artifacts["pablo_rounds_jsonl"])

    @property
    def _task_registry_path(self) -> Path:
        return Path(self._artifacts["task_registry_json"])

    @property
    def _llm_calls_path(self) -> Path:
        return Path(self._artifacts["llm_calls_jsonl"])

    @property
    def _candidate_queue_path(self) -> Path:
        return Path(self._artifacts["candidate_queue_jsonl"])

    @property
    def _resume_state_path(self) -> Path:
        return Path(self._artifacts["resume_state_json"])


__all__ = [
    "DEFAULT_GLOBAL_MODEL",
    "DEFAULT_INIT_POINTS",
    "DEFAULT_MAX_FAILS",
    "DEFAULT_MAX_TASKS",
    "DEFAULT_NUM_SEEDS",
    "PabloAlgorithm",
    "PabloConfig",
]
