"""State-gated, single-transcript protocol for backend-adaptive Agentic BO."""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from .serialization import append_jsonl, stable_config_identity


ToolExecutor = Callable[[str, dict[str, Any], str | None], Awaitable[str]]


class RoundStage(str, Enum):
    COLLECT_EVIDENCE = "collect_evidence"
    SELECT_BACKEND = "select_backend"
    VALIDATE = "validate"
    COMMIT = "commit"
    COMPLETED = "completed"


COMMIT_CANDIDATE_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "commit_candidate",
        "description": (
            "Commit the exact candidate bound to the latest validation token. This is "
            "the terminal action for the current optimization round."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "validation_token": {"type": "string"},
                "relationship": {
                    "type": "string",
                    "enum": ["adopt", "refine", "override"],
                },
                "backend_rationale": {"type": "string"},
                "modification_rationale": {"type": "string"},
                "belief": {"type": "string"},
                "expected_information": {"type": "string"},
                "hypothesis": {"type": "string"},
                "hypothesis_update": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": [
                                "supported",
                                "contradicted",
                                "inconclusive",
                                "not_applicable",
                            ],
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["status", "reason"],
                    "additionalProperties": False,
                },
            },
            "required": [
                "validation_token",
                "relationship",
                "backend_rationale",
                "modification_rationale",
                "belief",
                "expected_information",
                "hypothesis",
                "hypothesis_update",
            ],
            "additionalProperties": False,
        },
    },
}


def _nonempty(arguments: Mapping[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _decode(raw: str) -> dict[str, Any]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Tool transport returned a non-object payload.")
    return payload


def _encode(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str)


@dataclass
class AgentRoundState:
    """Enforce protocol order without adding stage prompts to the transcript."""

    round_id: str
    history: Sequence[Any]
    base_specs: list[dict[str, Any]]
    event_path: Path
    stage: RoundStage = RoundStage.COLLECT_EVIDENCE
    evidence_id: str | None = None
    proposal_id: str | None = None
    proposal_backend: str | None = None
    proposal_candidate: dict[str, Any] | None = None
    proposal_identity: str | None = None
    validation_token: str | None = None
    validated_candidate: dict[str, Any] | None = None
    validated_identity: str | None = None
    committed_payload: dict[str, Any] | None = None
    _replay_cache: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._specs_by_name = {
            str(spec.get("function", {}).get("name")): copy.deepcopy(spec)
            for spec in self.base_specs
        }
        validation = self._specs_by_name.get("validate_candidate")
        if validation is not None:
            params = validation["function"]["parameters"]
            params["properties"]["proposal_id"] = {"type": "string"}
            params["required"] = ["candidate", "proposal_id"]
            params["additionalProperties"] = False
        self._record("round_started", {})

    @property
    def completed(self) -> bool:
        return self.stage is RoundStage.COMPLETED

    def tool_specs(self) -> list[dict[str, Any]]:
        allowed: tuple[str, ...]
        if self.stage is RoundStage.COLLECT_EVIDENCE:
            allowed = ("get_trial_history", "assess_backend_suitability")
        elif self.stage is RoundStage.SELECT_BACKEND:
            allowed = ("get_trial_history", "optimizer_suggest")
        elif self.stage is RoundStage.VALIDATE:
            allowed = ("get_trial_history", "validate_candidate")
        elif self.stage is RoundStage.COMMIT:
            allowed = ("get_trial_history", "validate_candidate")
        else:
            allowed = ()
        specs = [copy.deepcopy(self._specs_by_name[name]) for name in allowed]
        if self.stage is RoundStage.COMMIT:
            specs.append(copy.deepcopy(COMMIT_CANDIDATE_SPEC))
        return specs

    def transport_tool_specs(self) -> list[dict[str, Any]]:
        """Return the complete schema set for transports with a static tool menu.

        Chat-completions transports can refresh :meth:`tool_specs` before every
        model request.  Native CLI harnesses such as Codex receive their tool
        documentation once, in the initial prompt, so they must be told about
        every action they may need later in the round.  Execution remains gated
        by :meth:`execute`; advertising a later-stage tool does not make an
        out-of-order call valid.
        """

        ordered = (
            "get_trial_history",
            "assess_backend_suitability",
            "optimizer_suggest",
            "validate_candidate",
        )
        specs = [
            copy.deepcopy(self._specs_by_name[name])
            for name in ordered
            if name in self._specs_by_name
        ]
        specs.append(copy.deepcopy(COMMIT_CANDIDATE_SPEC))
        return specs

    async def execute(
        self,
        base_executor: ToolExecutor,
        tool_name: str,
        arguments: dict[str, Any],
        tool_call_id: str | None,
    ) -> str:
        cache_key = self._cache_key(tool_name, arguments)
        if tool_name in {"assess_backend_suitability", "optimizer_suggest"}:
            cached = self._replay_cache.get(cache_key)
            if cached is not None:
                self._record(
                    "cached_tool_replay",
                    {"tool_name": tool_name, "tool_call_id": tool_call_id},
                )
                return cached
        allowed = {
            str(spec["function"]["name"]) for spec in self.tool_specs()
        }
        if tool_name not in allowed:
            return self._protocol_error(
                tool_name,
                f"Tool is unavailable in stage {self.stage.value}; allowed tools: {sorted(allowed)}.",
            )
        try:
            if tool_name == "commit_candidate":
                return self._commit(arguments, tool_call_id)
            if tool_name == "validate_candidate":
                return await self._validate(
                    base_executor, arguments, tool_call_id
                )
            raw = await base_executor(tool_name, arguments, tool_call_id)
            payload = _decode(raw)
            if payload.get("ok") is not True:
                return raw
            result = payload.get("result")
            if not isinstance(result, dict):
                raise ValueError(f"{tool_name} returned no structured result.")
            if tool_name == "assess_backend_suitability":
                self.evidence_id = self._new_id("evidence", result)
                result["evidence_id"] = self.evidence_id
                self._transition(RoundStage.SELECT_BACKEND, tool_name)
            elif tool_name == "optimizer_suggest":
                candidate = result.get("candidate")
                if not isinstance(candidate, dict):
                    candidates = result.get("candidates")
                    if isinstance(candidates, list) and candidates:
                        candidate = candidates[0].get("candidate")
                if not isinstance(candidate, dict):
                    raise ValueError("optimizer_suggest returned no candidate.")
                backend = result.get("backend") or arguments.get("backend")
                self.proposal_candidate = dict(candidate)
                self.proposal_identity = str(
                    result.get("identity")
                    or stable_config_identity(self.proposal_candidate)
                )
                self.proposal_backend = str(backend)
                self.proposal_id = self._new_id(
                    "proposal",
                    {
                        "evidence_id": self.evidence_id,
                        "backend": backend,
                        "candidate": candidate,
                    },
                )
                result["proposal_id"] = self.proposal_id
                result["evidence_id"] = self.evidence_id
                self._transition(RoundStage.VALIDATE, tool_name)
            encoded = _encode(payload)
            if tool_name in {"assess_backend_suitability", "optimizer_suggest"}:
                self._replay_cache[cache_key] = encoded
            return encoded
        except Exception as exc:
            return self._protocol_error(tool_name, str(exc))

    async def _validate(
        self,
        base_executor: ToolExecutor,
        arguments: dict[str, Any],
        tool_call_id: str | None,
    ) -> str:
        if arguments.get("proposal_id") != self.proposal_id:
            return self._protocol_error(
                "validate_candidate", "proposal_id is missing, stale, or incorrect."
            )
        candidate = arguments.get("candidate")
        forwarded = {"candidate": candidate}
        if "too_similar_threshold" in arguments:
            forwarded["too_similar_threshold"] = arguments["too_similar_threshold"]
        raw = await base_executor("validate_candidate", forwarded, tool_call_id)
        payload = _decode(raw)
        result = payload.get("result")
        if payload.get("ok") is not True or not isinstance(result, dict):
            return raw
        if result.get("valid") is not True or not isinstance(result.get("config"), dict):
            self._record("validation_rejected", {"result": result})
            return raw
        self.validated_candidate = dict(result["config"])
        self.validated_identity = str(
            result.get("identity")
            or stable_config_identity(self.validated_candidate)
        )
        self.validation_token = uuid.uuid4().hex
        result["proposal_id"] = self.proposal_id
        result["validation_token"] = self.validation_token
        self._transition(RoundStage.COMMIT, "validate_candidate")
        return _encode(payload)

    def _commit(
        self, arguments: dict[str, Any], tool_call_id: str | None
    ) -> str:
        if arguments.get("validation_token") != self.validation_token:
            return self._protocol_error(
                "commit_candidate", "validation_token is missing, stale, or incorrect."
            )
        if self.validated_candidate is None or self.validated_identity is None:
            return self._protocol_error(
                "commit_candidate", "No successfully validated candidate is available."
            )
        relationship = arguments.get("relationship")
        if relationship not in {"adopt", "refine", "override"}:
            return self._protocol_error(
                "commit_candidate", "relationship must be adopt, refine, or override."
            )
        changed = self._candidate_diff()
        if relationship == "adopt" and changed and not self._only_rounding_diff(changed):
            return self._protocol_error(
                "commit_candidate",
                "relationship=adopt is invalid because the committed candidate differs from the optimizer proposal.",
            )
        update = arguments.get("hypothesis_update")
        if not isinstance(update, Mapping):
            return self._protocol_error(
                "commit_candidate", "hypothesis_update must be an object."
            )
        status = update.get("status")
        if status not in {
            "supported",
            "contradicted",
            "inconclusive",
            "not_applicable",
        }:
            return self._protocol_error(
                "commit_candidate", "hypothesis_update.status is invalid."
            )
        try:
            backend_rationale = _nonempty(arguments, "backend_rationale")
            modification_rationale = _nonempty(arguments, "modification_rationale")
            belief = _nonempty(arguments, "belief")
            expected_information = _nonempty(arguments, "expected_information")
            hypothesis = _nonempty(arguments, "hypothesis")
            update_reason = _nonempty(update, "reason")
        except ValueError as exc:
            return self._protocol_error("commit_candidate", str(exc))
        latest_trial_id = (
            None if not self.history else self.history[-1].suggestion.trial_id
        )
        search_action = {
            "belief": belief,
            "expected_information": expected_information,
            "hypothesis": hypothesis,
            "hypothesis_update": {
                "status": status,
                "evidence_trial_id": latest_trial_id,
                "reason": update_reason,
            },
            "parent_trials": [],
            "reference_trials": [] if latest_trial_id is None else [latest_trial_id],
            "change_summary": modification_rationale,
            "optimizer": {
                "relationship": relationship,
                "backend": self.proposal_backend,
                "candidate_identity": self.proposal_identity,
                "final_candidate_identity": self.validated_identity,
                "considered_backends": [self.proposal_backend],
                "backend_rationale": backend_rationale,
                "proposal_id": self.proposal_id,
                "evidence_id": self.evidence_id,
                "candidate_diff": changed,
            },
            "protocol_version": 2,
        }
        self.committed_payload = {
            "candidates": [
                {
                    "config": self.validated_candidate,
                    "rationale": modification_rationale,
                    "search_action": search_action,
                    "protocol_version": 2,
                    "proposal_id": self.proposal_id,
                    "validation_identity": self.validated_identity,
                }
            ]
        }
        self._transition(RoundStage.COMPLETED, "commit_candidate")
        result = {
            "ok": True,
            "result": {
                "terminal": True,
                "round_id": self.round_id,
                "candidate_identity": self.validated_identity,
                "candidate_payload": self.committed_payload,
            },
        }
        self._record(
            "candidate_committed",
            {
                "tool_call_id": tool_call_id,
                "proposal_id": self.proposal_id,
                "candidate_identity": self.validated_identity,
                "candidate_diff": changed,
                "relationship": relationship,
            },
        )
        return _encode(result)

    def _candidate_diff(self) -> list[dict[str, Any]]:
        proposal = self.proposal_candidate or {}
        final = self.validated_candidate or {}
        return [
            {"name": name, "proposal": proposal.get(name), "final": final.get(name)}
            for name in sorted(set(proposal) | set(final))
            if proposal.get(name) != final.get(name)
        ]

    @staticmethod
    def _only_rounding_diff(changed: Sequence[Mapping[str, Any]]) -> bool:
        if not changed:
            return True
        for item in changed:
            proposal = item.get("proposal")
            final = item.get("final")
            if isinstance(proposal, bool) or isinstance(final, bool):
                return False
            if not isinstance(proposal, (int, float)) or not isinstance(final, (int, float)):
                return False
            if abs(float(proposal) - float(final)) > 5e-5 + 1e-12:
                return False
        return True

    @staticmethod
    def _cache_key(tool_name: str, arguments: Mapping[str, Any]) -> str:
        return tool_name + ":" + json.dumps(
            dict(arguments), sort_keys=True, separators=(",", ":"), default=str
        )

    def _new_id(self, prefix: str, payload: Mapping[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, default=str).encode()
        digest = hashlib.sha256(self.round_id.encode() + b"\0" + raw).hexdigest()[:20]
        return f"{prefix}_{digest}"

    def _transition(self, stage: RoundStage, tool_name: str) -> None:
        previous = self.stage
        self.stage = stage
        self._record(
            "stage_transition",
            {"from": previous.value, "to": stage.value, "tool_name": tool_name},
        )

    def _protocol_error(self, tool_name: str, message: str) -> str:
        self._record(
            "protocol_error",
            {"tool_name": tool_name, "stage": self.stage.value, "message": message},
        )
        return _encode(
            {
                "ok": False,
                "error": "protocol_error",
                "stage": self.stage.value,
                "message": message,
                "allowed_tools": [
                    spec["function"]["name"] for spec in self.tool_specs()
                ],
            }
        )

    def _record(self, kind: str, payload: Mapping[str, Any]) -> None:
        append_jsonl(
            self.event_path,
            {
                "timestamp": time.time(),
                "round_id": self.round_id,
                "kind": kind,
                "stage": self.stage.value,
                "payload": dict(payload),
            },
        )


__all__ = ["AgentRoundState", "COMMIT_CANDIDATE_SPEC", "RoundStage"]
