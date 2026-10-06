"""Transparent event logging for agentic algorithms without a second protocol."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from ...core import Algorithm, Incumbent, TaskSpec, TrialObservation, TrialSuggestion
from .events import DeliberationEvent, DeliberationEventWriter

EventMapper = Callable[[Mapping[str, Any]], tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]]


class EventedAlgorithm(Algorithm):
    """Decorate an Algorithm with events while preserving replay semantics."""

    def __init__(self, algorithm: Algorithm, *, method: str, run_dir: Path | str | None,
                 event_mapper: EventMapper | None = None) -> None:
        self.algorithm = algorithm
        self._method = method
        self._run_dir = None if run_dir is None else Path(run_dir)
        self._event_mapper = event_mapper or (lambda m: ({"source": m.get("agent_source")}, []))
        self._writer: DeliberationEventWriter | None = None
        self._evaluation_index = 0

    @property
    def name(self) -> str:
        return self._method

    @property
    def artifact_paths(self) -> dict[str, str]:
        paths = dict(getattr(self.algorithm, "artifact_paths", {}))
        if self._writer is not None:
            paths["deliberation_events_jsonl"] = str(self._writer.path)
        return paths

    @property
    def routing_table(self) -> dict[str, str]:
        return dict(getattr(self.algorithm, "routing_table", {}))

    def setup(self, task_spec: TaskSpec, seed: int = 0, **kwargs: Any) -> None:
        run_dir = kwargs.get("run_dir") or self._run_dir
        if run_dir is not None:
            self._writer = DeliberationEventWriter(Path(run_dir) / "deliberation_events.jsonl")
        self.algorithm.setup(task_spec, seed=seed, **kwargs)
        self._evaluation_index = 0
        self._emit("context", {"history_size": 0})

    def ask(self) -> TrialSuggestion:
        suggestion = self.algorithm.ask()
        self._emit_tool_events(suggestion.metadata)
        proposal, extra = self._event_mapper(suggestion.metadata)
        for kind, payload in extra:
            self._emit(kind, payload)
        self._emit("propose", proposal)
        self._emit("commit", {"config": suggestion.config, **proposal})
        return suggestion

    def tell(self, observation: TrialObservation) -> None:
        self.algorithm.tell(observation)
        self._emit("observation", {"status": observation.status.value,
                                   "objectives": observation.objectives,
                                   "trial_id": observation.suggestion.trial_id})
        self._evaluation_index += 1

    def seed(self, observation: TrialObservation) -> None:
        self.algorithm.seed(observation)
        self._evaluation_index += 1

    def replay(self, history: list[TrialObservation]) -> None:
        self.algorithm.replay(history)
        self._evaluation_index = len(history)

    def incumbents(self) -> list[Incumbent]:
        return self.algorithm.incumbents()

    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        if self._writer is not None:
            self._writer.append(DeliberationEvent(method=self.name, kind=kind,
                                                  evaluation_index=self._evaluation_index,
                                                  payload=payload))

    def _emit_tool_events(self, metadata: Mapping[str, Any]) -> None:
        call_id = metadata.get("agent_call_id")
        path_value = self.artifact_paths.get("agent_tool_calls_jsonl")
        if not call_id or not path_value or not Path(path_value).exists():
            return
        kinds = {"optimizer_predict": "probe", "optimizer_score": "probe",
                 "optimizer_diagnostics": "probe", "optimizer_status": "probe",
                 "optimizer_suggest": "propose", "optimizer_portfolio_suggest": "propose",
                 "optimizer_set_backend": "reconfigure", "optimizer_set_bounds": "reconfigure",
                 "optimizer_set_acquisition": "reconfigure", "optimizer_reset_policy": "reconfigure"}
        for line in Path(path_value).read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            kind = kinds.get(record.get("tool_name"))
            record_call_id = record.get("agent_call_id") or record.get("call_id")
            if kind and record_call_id == call_id and record.get("success", True):
                self._emit(kind, {"tool_name": record["tool_name"],
                                  "arguments": record.get("arguments", {})})
