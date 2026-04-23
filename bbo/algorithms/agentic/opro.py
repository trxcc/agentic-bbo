"""OPRO-style prompt optimizer adapted to the benchmark ask/tell protocol."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import random
import re
import textwrap
import uuid
from dataclasses import dataclass
from typing import Any, Protocol, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

from ...core import (
    CategoricalParam,
    FloatParam,
    IntParam,
    ObjectiveDirection,
    SearchSpace,
    TaskDescriptionBundle,
    TrialObservation,
    TrialSuggestion,
)
from ...core.adapters import ExternalOptimizerAdapter


def _stable_seed(*parts: Any) -> int:
    payload = "::".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def _strip_markdown_heading(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _truncate_text(text: str, limit: int) -> str:
    compact = " ".join(_strip_markdown_heading(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _score_to_minimization(direction: ObjectiveDirection, score: float) -> float:
    return score if direction == ObjectiveDirection.MINIMIZE else -score


def _format_scalar(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 12)
    return value


def _canonical_config(search_space: SearchSpace, config: dict[str, Any]) -> dict[str, Any]:
    normalized = search_space.coerce_config(config, use_defaults=False)
    return {param.name: _format_scalar(normalized[param.name]) for param in search_space}


def _config_key(search_space: SearchSpace, config: dict[str, Any]) -> str:
    return json.dumps(_canonical_config(search_space, config), sort_keys=True, separators=(",", ":"))


def _serialize_config(
    search_space: SearchSpace,
    config: dict[str, Any],
    *,
    parameter_order: Sequence[str] | None = None,
) -> str:
    normalized = search_space.coerce_config(config, use_defaults=False)
    order = list(parameter_order or search_space.names())
    payload = {name: _format_scalar(normalized[name]) for name in order}
    return json.dumps(payload, separators=(",", ":"))


@dataclass(frozen=True)
class OproObservedPoint:
    """One successful observation surfaced to the OPRO prompt."""

    config: dict[str, Any]
    score: float
    trial_id: int | None


@dataclass(frozen=True)
class OproGenerationRequest:
    """Prompt and structured context for one OPRO candidate-generation call."""

    prompt: str
    n_responses: int
    seed: int
    objective_direction: ObjectiveDirection
    search_space: SearchSpace
    observed_points: tuple[OproObservedPoint, ...]
    parameter_order: tuple[str, ...]


class OproBackend(Protocol):
    """Backend used by `OproAlgorithm` for prompt-based proposal generation."""

    @property
    def name(self) -> str:
        """Human-readable backend identifier."""

    def generate_candidate_texts(self, request: OproGenerationRequest) -> list[str]:
        """Return raw text generations containing candidate configurations."""

    def request_metadata(self) -> dict[str, Any]:
        """Return backend metadata for the latest request, if any."""


class HeuristicOproBackend:
    """Deterministic local backend for offline OPRO smoke tests."""

    @property
    def name(self) -> str:
        return "heuristic"

    def generate_candidate_texts(self, request: OproGenerationRequest) -> list[str]:
        texts: list[str] = []
        for index in range(request.n_responses):
            rng = random.Random(_stable_seed(request.seed, "candidate", index))
            config = self._sample_candidate(request, rng)
            rendered = _serialize_config(
                request.search_space,
                config,
                parameter_order=request.parameter_order,
            )
            texts.append(f"<candidate>{rendered}</candidate>")
        return texts

    def request_metadata(self) -> dict[str, Any]:
        return {}

    def _sample_candidate(self, request: OproGenerationRequest, rng: random.Random) -> dict[str, Any]:
        defaults = request.search_space.defaults()
        anchors = sorted(
            request.observed_points,
            key=lambda point: _score_to_minimization(request.objective_direction, point.score),
        )
        top_k = anchors[: max(1, min(4, len(anchors)))]
        anchor_config = dict(rng.choice(top_k).config) if top_k else defaults
        exploration = max(0.07, 0.30 * math.exp(-len(request.observed_points) / 12.0))

        config: dict[str, Any] = {}
        for param in request.search_space:
            anchor_value = anchor_config.get(param.name, defaults[param.name])
            if isinstance(param, FloatParam):
                config[param.name] = self._sample_numeric(param, float(anchor_value), rng, exploration)
                continue
            if isinstance(param, IntParam):
                config[param.name] = self._sample_numeric(param, int(anchor_value), rng, exploration)
                continue
            assert isinstance(param, CategoricalParam)
            if rng.random() < 0.75:
                config[param.name] = anchor_value
            else:
                config[param.name] = rng.choice(param.choices)
        return request.search_space.coerce_config(config, use_defaults=False)

    @staticmethod
    def _sample_numeric(
        param: FloatParam | IntParam,
        anchor_value: float | int,
        rng: random.Random,
        exploration: float,
    ) -> float | int:
        if param.log:
            low = math.log10(float(param.low))
            high = math.log10(float(param.high))
            anchor = math.log10(float(anchor_value))
        else:
            low = float(param.low)
            high = float(param.high)
            anchor = float(anchor_value)

        if rng.random() < 0.25:
            raw = rng.uniform(low, high)
        else:
            sigma = max((high - low) * exploration, 1e-6)
            raw = min(max(rng.gauss(anchor, sigma), low), high)

        if param.log:
            raw = 10 ** raw

        if isinstance(param, IntParam):
            return param.coerce(int(round(raw)))
        return param.coerce(float(raw))


class OpenAICompatibleOproBackend:
    """OpenAI-compatible chat backend for OPRO using standard Chat Completions API."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise ValueError("An explicit OpenAI API key is required for the online OPRO backend.")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
        self.organization = organization
        self.project = project
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max(0, int(max_retries))

    @property
    def name(self) -> str:
        return "openai"

    def generate_candidate_texts(self, request: OproGenerationRequest) -> list[str]:
        raw_texts = self._chat_text(
            prompt=request.prompt,
            n=request.n_responses,
            temperature=0.8,
            seed=request.seed,
        )
        return raw_texts

    def request_metadata(self) -> dict[str, Any]:
        return {"opro_openai_base_url": self._endpoint()}

    def _chat_text(
        self,
        *,
        prompt: str,
        n: int,
        temperature: float,
        seed: int,
    ) -> list[str]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a careful optimization assistant. Follow the requested output format exactly.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "n": int(max(1, n)),
            "seed": int(seed % (2**31 - 1)),
            "store": False,
        }
        data, _ = self._post_with_retry(payload)
        choices = data.get("choices", [])
        texts: list[str] = []
        for choice in choices:
            message = choice.get("message", {})
            if message.get("refusal"):
                continue
            content = message.get("content", "")
            if isinstance(content, str) and content.strip():
                texts.append(content.strip())
        return texts

    def _post_with_retry(
        self,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], int | None]:
        body = json.dumps(payload).encode("utf-8")
        endpoint = self._endpoint()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        if self.project:
            headers["OpenAI-Project"] = self.project

        last_exception: Exception | None = None
        for attempt in range(self.max_retries + 1):
            req = urllib_request.Request(
                endpoint,
                data=body,
                method="POST",
                headers=headers,
            )
            try:
                with urllib_request.urlopen(req, timeout=self.timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    return data, response.status
            except urllib_error.HTTPError as exc:
                last_exception = exc
                details = exc.read().decode("utf-8", errors="replace")
                if exc.code == 429:
                    pass
                elif 400 <= exc.code < 500:
                    try:
                        parsed = json.loads(details)
                    except json.JSONDecodeError:
                        parsed = {}
                    return parsed, exc.code
                else:
                    pass
            except urllib_error.URLError as exc:
                last_exception = exc
            if attempt < self.max_retries:
                sleep_seconds = min(2**attempt, 30)
                import time
                time.sleep(sleep_seconds)
        if last_exception is not None:
            raise RuntimeError(f"OPRO OpenAI request failed after {self.max_retries + 1} attempts: {last_exception}") from last_exception
        raise RuntimeError("OPRO OpenAI request failed unexpectedly.")

    def _endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"


class OproAlgorithm(ExternalOptimizerAdapter):
    """Generic OPRO-style optimizer using past config/score pairs as a meta-prompt."""

    def __init__(
        self,
        *,
        backend: str = "heuristic",
        backend_impl: OproBackend | None = None,
        model: str = "gpt-4o-mini",
        n_initial_samples: int = 5,
        n_candidates: int = 8,
        max_prompt_pairs: int = 20,
        max_generation_rounds: int = 4,
    ) -> None:
        super().__init__()
        if n_initial_samples <= 0:
            raise ValueError("n_initial_samples must be positive.")
        if n_candidates <= 0:
            raise ValueError("n_candidates must be positive.")
        if max_prompt_pairs <= 0:
            raise ValueError("max_prompt_pairs must be positive.")
        if max_generation_rounds <= 0:
            raise ValueError("max_generation_rounds must be positive.")
        self.backend = backend
        self.backend_impl = backend_impl
        self.model = model
        self.n_initial_samples = int(n_initial_samples)
        self.n_candidates = int(n_candidates)
        self.max_prompt_pairs = int(max_prompt_pairs)
        self.max_generation_rounds = int(max_generation_rounds)
        self._seed = 0
        self._description = TaskDescriptionBundle.empty(task_id="uninitialized")
        self._history: list[TrialObservation] = []
        self._seen_configs: set[str] = set()
        self._backend: OproBackend | None = None

    @property
    def name(self) -> str:
        return "opro"

    def setup(self, task_spec, seed: int = 0, **kwargs: Any) -> None:
        if len(task_spec.objectives) != 1:
            raise ValueError("OproAlgorithm currently supports exactly one objective.")
        self.bind_task_spec(task_spec)
        self._seed = int(seed)
        description = kwargs.get("task_description")
        if isinstance(description, TaskDescriptionBundle):
            self._description = description
        else:
            self._description = TaskDescriptionBundle.empty(task_id=task_spec.name)
        self._history = []
        self._seen_configs = set()
        self._backend = self.backend_impl or self._build_backend()

    def ask(self) -> TrialSuggestion:
        if len(self._history) < min(self.n_initial_samples, self.require_task_spec().max_evaluations):
            config = self._sample_initial_config(index=len(self._history))
            return TrialSuggestion(
                config=config,
                metadata={
                    "opro_phase": "initialization",
                    "opro_backend": self._require_backend().name,
                    "opro_history_size": len(self._history),
                },
            )

        observed_points = self._successful_points()
        if not observed_points:
            config = self._sample_initial_config(index=len(self._history))
            return TrialSuggestion(
                config=config,
                metadata={
                    "opro_phase": "fallback_random",
                    "opro_backend": self._require_backend().name,
                    "opro_history_size": len(self._history),
                },
            )

        candidates = self._propose_candidates(observed_points)
        backend_metadata = self._require_backend().request_metadata()
        if not candidates:
            config = self._sample_initial_config(index=len(self._history))
            return TrialSuggestion(
                config=config,
                metadata={
                    "opro_phase": "fallback_random",
                    "opro_backend": self._require_backend().name,
                    "opro_history_size": len(self._history),
                    "opro_prompt_pairs": min(len(observed_points), self.max_prompt_pairs),
                    **backend_metadata,
                },
            )

        selected = candidates[0]
        self.require_search_space().validate_config(selected)
        return TrialSuggestion(
            config=selected,
            metadata={
                "opro_phase": "prompt_optimization",
                "opro_backend": self._require_backend().name,
                "opro_history_size": len(self._history),
                "opro_candidate_count": len(candidates),
                "opro_prompt_pairs": min(len(observed_points), self.max_prompt_pairs),
                **backend_metadata,
            },
        )

    def tell(self, observation: TrialObservation) -> None:
        config = self.require_search_space().coerce_config(observation.suggestion.config, use_defaults=False)
        normalized = TrialObservation(
            suggestion=TrialSuggestion(
                config=config,
                trial_id=observation.suggestion.trial_id,
                budget=observation.suggestion.budget,
                metadata=dict(observation.suggestion.metadata),
            ),
            status=observation.status,
            objectives=dict(observation.objectives),
            metrics=dict(observation.metrics),
            elapsed_seconds=observation.elapsed_seconds,
            error_type=observation.error_type,
            error_message=observation.error_message,
            timestamp=observation.timestamp,
            metadata=dict(observation.metadata),
        )
        self._history.append(normalized)
        self._seen_configs.add(_config_key(self.require_search_space(), config))
        self.update_best_incumbent(normalized)

    def replay(self, history: list[TrialObservation]) -> None:
        for observation in history:
            self.tell(observation)

    def _build_backend(self) -> OproBackend:
        if self.backend == "heuristic":
            return HeuristicOproBackend()
        if self.backend == "openai":
            raise RuntimeError(
                "The online OPRO backend must be injected from the runner/CLI layer via `backend_impl` "
                "so OpenAI credentials and endpoint settings stay in user-facing configuration."
            )
        raise ValueError(f"Unknown OPRO backend `{self.backend}`.")

    def _require_backend(self) -> OproBackend:
        if self._backend is None:
            raise RuntimeError("OproAlgorithm.setup() must be called before ask/tell.")
        return self._backend

    def _successful_points(self) -> list[OproObservedPoint]:
        assert self._primary_name is not None
        points: list[OproObservedPoint] = []
        for observation in self._history:
            if not observation.success or self._primary_name not in observation.objectives:
                continue
            points.append(
                OproObservedPoint(
                    config=dict(observation.suggestion.config),
                    score=float(observation.objectives[self._primary_name]),
                    trial_id=observation.suggestion.trial_id,
                )
            )
        return points

    def _sample_initial_config(self, *, index: int) -> dict[str, Any]:
        search_space = self.require_search_space()
        for offset in range(128):
            rng = random.Random(_stable_seed(self._seed, "initial", index, offset))
            config = search_space.sample(rng)
            key = _config_key(search_space, config)
            if key not in self._seen_configs:
                return config
        raise RuntimeError("OPRO failed to sample a new initialization point.")

    def _propose_candidates(self, observed_points: Sequence[OproObservedPoint]) -> list[dict[str, Any]]:
        search_space = self.require_search_space()
        candidates: list[dict[str, Any]] = []
        seen = set(self._seen_configs)
        parameter_order = tuple(search_space.names())
        prompt = self._meta_prompt(observed_points, parameter_order=parameter_order)

        for round_index in range(self.max_generation_rounds):
            request = OproGenerationRequest(
                prompt=prompt,
                n_responses=self.n_candidates,
                seed=_stable_seed(self._seed, "opro_prompt", len(self._history), round_index),
                objective_direction=self._primary_direction,
                search_space=search_space,
                observed_points=tuple(observed_points),
                parameter_order=parameter_order,
            )
            for text in self._require_backend().generate_candidate_texts(request):
                for candidate in self._parse_candidate_text(text):
                    key = _config_key(search_space, candidate)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(candidate)
                    if len(candidates) >= self.n_candidates:
                        return candidates
        return candidates

    def _meta_prompt(
        self,
        observed_points: Sequence[OproObservedPoint],
        *,
        parameter_order: Sequence[str],
    ) -> str:
        assert self._primary_name is not None
        direction_text = "lower values are better" if self._primary_direction == ObjectiveDirection.MINIMIZE else "higher values are better"
        ordering_text = "descending" if self._primary_direction == ObjectiveDirection.MINIMIZE else "ascending"
        example_config = _serialize_config(self.require_search_space(), self.require_search_space().defaults(), parameter_order=parameter_order)
        return textwrap.dedent(
            f"""
            You are helping with black-box optimization for task `{self.require_task_spec().name}`.
            Objective: optimize `{self._primary_name}` where {direction_text}.

            Search space:
            {self._search_space_block(parameter_order=parameter_order)}

            Task context:
            {self._task_context_block()}

            Below are some previous configurations and their objective values.
            The configurations are arranged in {ordering_text} order based on their objective values, where {direction_text}.

            {self._observed_pairs_block(observed_points, parameter_order=parameter_order)}

            Give me {self.n_candidates} new configurations that are different from all configurations above and are likely to achieve a better objective value than any configuration above.
            Use the exact parameter names from the search space.
            Return each configuration as compact JSON wrapped in <candidate>...</candidate>.
            Example format: <candidate>{example_config}</candidate>
            Do not write code. Do not add any prose.
            """
        ).strip()

    def _task_context_block(self) -> str:
        sections = []
        for kind in ("background", "goal", "constraints", "prior_knowledge"):
            content = self._description.section_map.get(kind)
            if not content:
                continue
            title = kind.replace("_", " ").title()
            sections.append(f"- {title}: {_truncate_text(content, 320)}")
        if sections:
            return "\n".join(sections)
        return "- No external task-description bundle was supplied."

    def _search_space_block(self, *, parameter_order: Sequence[str]) -> str:
        space = self.require_search_space()
        lines: list[str] = []
        for name in parameter_order:
            param = space[name]
            if isinstance(param, FloatParam):
                line = f"- {name}: float in [{param.low:g}, {param.high:g}]"
                if param.log:
                    line += " on a log scale"
                line += f", default {param.effective_default():g}"
                lines.append(line)
                continue
            if isinstance(param, IntParam):
                line = f"- {name}: int in [{param.low}, {param.high}]"
                if param.log:
                    line += " on a log scale"
                line += f", default {param.effective_default()}"
                lines.append(line)
                continue
            assert isinstance(param, CategoricalParam)
            choices = ", ".join(repr(choice) for choice in param.choices)
            lines.append(f"- {name}: categorical, choices [{choices}], default {param.effective_default()!r}")
        return "\n".join(lines)

    def _observed_pairs_block(
        self,
        observed_points: Sequence[OproObservedPoint],
        *,
        parameter_order: Sequence[str],
    ) -> str:
        ordered = sorted(
            observed_points,
            key=lambda point: _score_to_minimization(self._primary_direction, point.score),
            reverse=True,
        )
        if len(ordered) > self.max_prompt_pairs:
            ordered = ordered[-self.max_prompt_pairs :]

        blocks: list[str] = []
        for point in ordered:
            config_text = _serialize_config(self.require_search_space(), point.config, parameter_order=parameter_order)
            trial_line = ""
            if point.trial_id is not None:
                trial_line = f"trial_id: {point.trial_id}\n"
            blocks.append(f"{trial_line}input:\n{config_text}\nvalue:\n{point.score:.8f}")
        return "\n\n".join(blocks)

    def _parse_candidate_text(self, text: str) -> list[dict[str, Any]]:
        blocks = re.findall(r"<candidate>\s*(.*?)\s*</candidate>", text, flags=re.IGNORECASE | re.DOTALL)
        if not blocks:
            blocks = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
        if not blocks:
            blocks = re.findall(r"\[.*?\]", text, flags=re.DOTALL)

        parsed_candidates: list[dict[str, Any]] = []
        for block in blocks:
            parsed = self._parse_candidate_payload(block)
            if parsed is None:
                continue
            try:
                candidate = self.require_search_space().coerce_config(parsed, use_defaults=False)
            except Exception:
                continue
            parsed_candidates.append(candidate)
        return parsed_candidates

    def _parse_candidate_payload(self, text: str) -> dict[str, Any] | None:
        cleaned = text.strip()
        for loader in (json.loads, ast.literal_eval):
            try:
                value = loader(cleaned)
            except Exception:
                continue
            mapping = self._mapping_from_loaded_value(value)
            if mapping is not None:
                return mapping

        if cleaned.startswith("[") and cleaned.endswith("]"):
            inner = cleaned[1:-1].strip()
            if "=" in inner:
                pairs = [item.strip() for item in inner.split(",") if item.strip()]
                mapping: dict[str, Any] = {}
                for pair in pairs:
                    if "=" not in pair:
                        return None
                    key, raw_value = pair.split("=", 1)
                    value = self._parse_scalar(raw_value.strip())
                    mapping[key.strip()] = value
                if mapping:
                    return mapping
        return None

    def _mapping_from_loaded_value(self, value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, (list, tuple)) and len(value) == len(self.require_search_space()):
            names = self.require_search_space().names()
            return {name: item for name, item in zip(names, value, strict=True)}
        return None

    @staticmethod
    def _parse_scalar(text: str) -> Any:
        for loader in (json.loads, ast.literal_eval):
            try:
                return loader(text)
            except Exception:
                continue
        return text


__all__ = [
    "HeuristicOproBackend",
    "OpenAICompatibleOproBackend",
    "OproAlgorithm",
    "OproBackend",
    "OproGenerationRequest",
    "OproObservedPoint",
]
