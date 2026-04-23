"""LLAMBO-style prompt optimizer adapted to the benchmark ask/tell protocol."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Protocol, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np

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


def _normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


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


def _numeric_transform(param: FloatParam | IntParam, value: float) -> float:
    if param.log:
        return math.log10(float(value))
    return float(value)


def _parameter_distance(param: FloatParam | IntParam | CategoricalParam, left: Any, right: Any) -> float:
    if isinstance(param, (FloatParam, IntParam)):
        low = _numeric_transform(param, param.low)
        high = _numeric_transform(param, param.high)
        width = max(high - low, 1e-12)
        left_value = _numeric_transform(param, float(left))
        right_value = _numeric_transform(param, float(right))
        return abs(left_value - right_value) / width
    return 0.0 if left == right else 1.0


def _serialize_config(
    search_space: SearchSpace,
    config: dict[str, Any],
    *,
    parameter_order: Sequence[str] | None = None,
) -> str:
    normalized = search_space.coerce_config(config, use_defaults=False)
    order = list(parameter_order or search_space.names())
    payload = {name: normalized[name] for name in order}
    return json.dumps(payload, separators=(",", ":"))


@dataclass(frozen=True)
class ObservedPoint:
    """One successful observation made available to the LLAMBO prompts."""

    config: dict[str, Any]
    score: float
    trial_id: int | None


@dataclass(frozen=True)
class CandidateGenerationRequest:
    """Prompt and structured context for LLAMBO candidate generation."""

    prompt: str
    n_responses: int
    seed: int
    objective_direction: ObjectiveDirection
    search_space: SearchSpace
    observed_points: tuple[ObservedPoint, ...]
    desired_score: float
    parameter_order: tuple[str, ...]


@dataclass(frozen=True)
class ScorePredictionRequest:
    """Prompt and structured context for LLAMBO surrogate predictions."""

    prompt: str
    n_responses: int
    seed: int
    objective_direction: ObjectiveDirection
    search_space: SearchSpace
    observed_points: tuple[ObservedPoint, ...]
    candidate_config: dict[str, Any]
    parameter_order: tuple[str, ...]


@dataclass(frozen=True)
class CandidatePrediction:
    """Predicted score statistics for one candidate configuration."""

    mean: float
    std: float
    expected_improvement: float


class LlamboBackend(Protocol):
    """Backend used by `LlamboAlgorithm` for acquisition and surrogate prompts."""

    @property
    def name(self) -> str:
        """Human-readable backend identifier."""

    def generate_candidate_texts(self, request: CandidateGenerationRequest) -> list[str]:
        """Return raw text generations containing candidate configurations."""

    def generate_score_texts(self, request: ScorePredictionRequest) -> list[str]:
        """Return raw text generations containing predicted scores."""


class HeuristicLlamboBackend:
    """Deterministic local backend for offline LLAMBO smoke tests."""

    @property
    def name(self) -> str:
        return "heuristic"

    def generate_candidate_texts(self, request: CandidateGenerationRequest) -> list[str]:
        texts: list[str] = []
        for index in range(request.n_responses):
            rng = random.Random(_stable_seed(request.seed, "candidate", index))
            config = self._sample_candidate(request, rng)
            texts.append(f"<candidate>{_serialize_config(request.search_space, config, parameter_order=request.parameter_order)}</candidate>")
        return texts

    def generate_score_texts(self, request: ScorePredictionRequest) -> list[str]:
        mean, std = self._estimate_score(request)
        texts: list[str] = []
        for index in range(request.n_responses):
            rng = random.Random(_stable_seed(request.seed, "score", index))
            sampled = rng.gauss(mean, max(std * 0.35, 1e-4))
            texts.append(f"<score>{sampled:.8f}</score>")
        return texts

    def _sample_candidate(self, request: CandidateGenerationRequest, rng: random.Random) -> dict[str, Any]:
        defaults = request.search_space.defaults()
        anchors = sorted(
            request.observed_points,
            key=lambda point: _score_to_minimization(request.objective_direction, point.score),
        )
        top_k = anchors[: max(1, min(3, len(anchors)))]
        anchor_config = dict(rng.choice(top_k).config) if top_k else defaults
        exploration = max(0.06, 0.32 * math.exp(-len(request.observed_points) / 10.0))

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

        if rng.random() < 0.2:
            raw = rng.uniform(low, high)
        else:
            sigma = max((high - low) * exploration, 1e-6)
            raw = min(max(rng.gauss(anchor, sigma), low), high)

        if param.log:
            raw = 10 ** raw

        if isinstance(param, IntParam):
            return param.coerce(int(round(raw)))
        return param.coerce(float(raw))

    def _estimate_score(self, request: ScorePredictionRequest) -> tuple[float, float]:
        scores = np.asarray([point.score for point in request.observed_points], dtype=float)
        if scores.size == 0:
            return 0.0, 1.0
        if scores.size == 1:
            base = float(scores[0])
            return base, max(abs(base) * 0.05, 1e-3)

        distances = np.asarray(
            [
                self._config_distance(request.search_space, request.candidate_config, point.config)
                for point in request.observed_points
            ],
            dtype=float,
        )
        bandwidth = max(float(np.median(distances)), 0.15)
        weights = np.exp(-(distances**2) / max(2.0 * bandwidth * bandwidth, 1e-6))
        if not np.isfinite(weights).all() or float(np.sum(weights)) <= 1e-9:
            weights = np.ones_like(scores)

        mean = float(np.average(scores, weights=weights))
        variance = float(np.average((scores - mean) ** 2, weights=weights))
        nearest = float(np.min(distances))
        spread = max(float(np.std(scores)), 1e-3)
        std = math.sqrt(max(variance, 1e-8) + (0.1 + nearest) * spread * spread / (1.0 + len(request.observed_points)))
        return mean, max(std, 1e-3)

    @staticmethod
    def _config_distance(search_space: SearchSpace, left: dict[str, Any], right: dict[str, Any]) -> float:
        components = [
            _parameter_distance(param, left[param.name], right[param.name])
            for param in search_space
        ]
        if not components:
            return 0.0
        return math.sqrt(sum(component * component for component in components) / len(components))


class OpenAICompatibleLlamboBackend:
    """OpenAI-compatible chat backend with only stdlib HTTP dependencies.

    Supports structured JSON-schema outputs when the endpoint advertises it,
    and falls back to plain text completion + parsing for older or compatible
    endpoints that do not implement ``response_format: {type: json_schema}``.
    """

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
        use_structured_outputs: bool = True,
    ) -> None:
        if not api_key:
            raise ValueError("An explicit OpenAI API key is required for the online LLAMBO backend.")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
        self.organization = organization
        self.project = project
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max(0, int(max_retries))
        self.use_structured_outputs = bool(use_structured_outputs)
        self._structured_outputs_unavailable = False

    @property
    def name(self) -> str:
        return "openai"

    def generate_candidate_texts(self, request: CandidateGenerationRequest) -> list[str]:
        if self.use_structured_outputs and not self._structured_outputs_unavailable:
            payloads = self._chat_json(
                prompt=request.prompt,
                n=request.n_responses,
                temperature=0.8,
                seed=request.seed,
                schema_name="llambo_candidate",
                schema=self._candidate_schema(request.search_space),
            )
            texts: list[str] = []
            for payload in payloads:
                try:
                    config = request.search_space.coerce_config(payload, use_defaults=False)
                except Exception:
                    continue
                texts.append(
                    f"<candidate>{_serialize_config(request.search_space, config, parameter_order=request.parameter_order)}</candidate>"
                )
            if texts:
                return texts
        # Fallback: plain text completion and manual parsing.
        raw_texts = self._chat_text(
            prompt=request.prompt,
            n=request.n_responses,
            temperature=0.8,
            seed=request.seed,
        )
        return raw_texts

    def generate_score_texts(self, request: ScorePredictionRequest) -> list[str]:
        if self.use_structured_outputs and not self._structured_outputs_unavailable:
            payloads = self._chat_json(
                prompt=request.prompt,
                n=request.n_responses,
                temperature=0.35,
                seed=request.seed,
                schema_name="llambo_score",
                schema={
                    "type": "object",
                    "properties": {
                        "predicted_objective": {"type": "number"},
                    },
                    "required": ["predicted_objective"],
                    "additionalProperties": False,
                },
            )
            texts: list[str] = []
            for payload in payloads:
                try:
                    value = float(payload["predicted_objective"])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    texts.append(f"<score>{value:.8f}</score>")
            if texts:
                return texts
        # Fallback: plain text completion and manual parsing.
        raw_texts = self._chat_text(
            prompt=request.prompt,
            n=request.n_responses,
            temperature=0.35,
            seed=request.seed,
        )
        return raw_texts

    def _chat_json(
        self,
        *,
        prompt: str,
        n: int,
        temperature: float,
        seed: int,
        schema_name: str,
        schema: dict[str, Any],
    ) -> list[dict[str, Any]]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a careful optimization assistant. Return only data that satisfies the provided JSON schema.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "n": int(max(1, n)),
            "seed": int(seed % (2**31 - 1)),
            "store": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        data, status = self._post_with_retry(payload)
        if status is not None and status // 100 != 2:
            # If the endpoint rejects json_schema (common with compatible APIs),
            # mark it unavailable and return empty so callers fallback to text.
            body_str = json.dumps(data)
            if "json_schema" in body_str or "response_format" in body_str or "unsupported" in body_str.lower():
                self._structured_outputs_unavailable = True
            return []

        choices = data.get("choices", [])
        items: list[dict[str, Any]] = []
        for choice in choices:
            message = choice.get("message", {})
            if message.get("refusal"):
                continue
            content = message.get("content", "")
            if isinstance(content, list):
                content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
            try:
                parsed = json.loads(str(content))
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                items.append(parsed)
        return items

    def _chat_text(
        self,
        *,
        prompt: str,
        n: int,
        temperature: float,
        seed: int,
    ) -> list[str]:
        """Plain chat completion without structured outputs; returns raw assistant texts."""
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
        """POST to chat/completions with exponential backoff on transient errors."""
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
                # Do not retry on 4xx client errors except 429 rate-limit.
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
            raise RuntimeError(f"LLAMBO OpenAI request failed after {self.max_retries + 1} attempts: {last_exception}") from last_exception
        raise RuntimeError("LLAMBO OpenAI request failed unexpectedly.")

    @staticmethod
    def _candidate_schema(search_space: SearchSpace) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param in search_space:
            required.append(param.name)
            if isinstance(param, FloatParam):
                properties[param.name] = {
                    "type": "number",
                    "minimum": float(param.low),
                    "maximum": float(param.high),
                }
                continue
            if isinstance(param, IntParam):
                properties[param.name] = {
                    "type": "integer",
                    "minimum": int(param.low),
                    "maximum": int(param.high),
                }
                continue
            assert isinstance(param, CategoricalParam)
            properties[param.name] = {
                "enum": list(param.choices),
            }
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def _endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"


class LlamboAlgorithm(ExternalOptimizerAdapter):
    """LLAMBO-style optimizer with repository-native task/context adapters."""

    def __init__(
        self,
        *,
        backend: str = "heuristic",
        backend_impl: LlamboBackend | None = None,
        model: str = "gpt-4o-mini",
        n_candidates: int = 8,
        n_templates: int = 2,
        n_predictions: int = 6,
        alpha: float = -0.2,
        n_initial_samples: int = 5,
        max_prompt_history: int = 12,
    ) -> None:
        super().__init__()
        if n_candidates <= 0:
            raise ValueError("n_candidates must be positive.")
        if n_templates <= 0:
            raise ValueError("n_templates must be positive.")
        if n_predictions <= 0:
            raise ValueError("n_predictions must be positive.")
        if n_initial_samples <= 0:
            raise ValueError("n_initial_samples must be positive.")
        if max_prompt_history <= 0:
            raise ValueError("max_prompt_history must be positive.")
        self.backend = backend
        self.backend_impl = backend_impl
        self.model = model
        self.n_candidates = int(n_candidates)
        self.n_templates = int(n_templates)
        self.n_predictions = int(n_predictions)
        self.alpha = float(alpha)
        self.n_initial_samples = int(n_initial_samples)
        self.max_prompt_history = int(max_prompt_history)
        self._seed = 0
        self._description = TaskDescriptionBundle.empty(task_id="uninitialized")
        self._history: list[TrialObservation] = []
        self._seen_configs: set[str] = set()
        self._backend: LlamboBackend | None = None

    @property
    def name(self) -> str:
        return "llambo"

    def setup(self, task_spec, seed: int = 0, **kwargs: Any) -> None:
        if len(task_spec.objectives) != 1:
            raise ValueError("LlamboAlgorithm currently supports exactly one objective.")
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
        search_space = self.require_search_space()
        if len(self._history) < min(self.n_initial_samples, self.require_task_spec().max_evaluations):
            config = self._sample_initial_config(index=len(self._history))
            return TrialSuggestion(
                config=config,
                metadata={
                    "llambo_phase": "initialization",
                    "llambo_backend": self._require_backend().name,
                    "llambo_history_size": len(self._history),
                },
            )

        observed_points = self._successful_points()
        if not observed_points:
            config = self._sample_initial_config(index=len(self._history))
            return TrialSuggestion(
                config=config,
                metadata={
                    "llambo_phase": "fallback_random",
                    "llambo_backend": self._require_backend().name,
                    "llambo_history_size": len(self._history),
                },
            )

        desired_score = self._desired_score(observed_points)
        candidate_configs = self._propose_candidates(observed_points, desired_score)
        if not candidate_configs:
            config = self._sample_initial_config(index=len(self._history))
            return TrialSuggestion(
                config=config,
                metadata={
                    "llambo_phase": "fallback_random",
                    "llambo_backend": self._require_backend().name,
                    "llambo_history_size": len(self._history),
                    "llambo_target_score": desired_score,
                },
            )

        ranked_candidates = [
            (config, self._predict_candidate(config, observed_points))
            for config in candidate_configs
        ]
        selected_config, prediction = max(
            ranked_candidates,
            key=lambda item: (
                item[1].expected_improvement,
                -item[1].mean if self._primary_direction == ObjectiveDirection.MINIMIZE else item[1].mean,
            ),
        )
        search_space.validate_config(selected_config)
        return TrialSuggestion(
            config=selected_config,
            metadata={
                "llambo_phase": "prompt_optimization",
                "llambo_backend": self._require_backend().name,
                "llambo_history_size": len(self._history),
                "llambo_candidate_count": len(candidate_configs),
                "llambo_target_score": desired_score,
                "llambo_predicted_mean": prediction.mean,
                "llambo_predicted_std": prediction.std,
                "llambo_expected_improvement": prediction.expected_improvement,
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

    def _build_backend(self) -> LlamboBackend:
        if self.backend == "heuristic":
            return HeuristicLlamboBackend()
        if self.backend == "openai":
            raise RuntimeError(
                "The online LLAMBO backend must be injected from the runner/CLI layer via `backend_impl` "
                "so OpenAI credentials and endpoint settings stay in user-facing configuration."
            )
        raise ValueError(f"Unknown LLAMBO backend `{self.backend}`.")

    def _require_backend(self) -> LlamboBackend:
        if self._backend is None:
            raise RuntimeError("LlamboAlgorithm.setup() must be called before ask/tell.")
        return self._backend

    def _successful_points(self) -> list[ObservedPoint]:
        assert self._primary_name is not None
        points: list[ObservedPoint] = []
        for observation in self._history:
            if not observation.success or self._primary_name not in observation.objectives:
                continue
            points.append(
                ObservedPoint(
                    config=dict(observation.suggestion.config),
                    score=float(observation.objectives[self._primary_name]),
                    trial_id=observation.suggestion.trial_id,
                )
            )
        return points

    def _desired_score(self, observed_points: Sequence[ObservedPoint]) -> float:
        scores = np.asarray([point.score for point in observed_points], dtype=float)
        observed_best = float(np.min(scores) if self._primary_direction == ObjectiveDirection.MINIMIZE else np.max(scores))
        observed_worst = float(np.max(scores) if self._primary_direction == ObjectiveDirection.MINIMIZE else np.min(scores))
        score_range = abs(observed_best - observed_worst)
        if score_range <= 1e-12:
            score_range = max(abs(observed_best), 1.0) * 0.1
        if self._primary_direction == ObjectiveDirection.MINIMIZE:
            return observed_best - self.alpha * score_range
        return observed_best + self.alpha * score_range

    def _sample_initial_config(self, *, index: int) -> dict[str, Any]:
        search_space = self.require_search_space()
        for offset in range(64):
            rng = random.Random(_stable_seed(self._seed, "initial", index, offset))
            config = search_space.sample(rng)
            key = _config_key(search_space, config)
            if key not in self._seen_configs:
                return config
        raise RuntimeError("LLAMBO failed to sample a new initialization point.")

    def _propose_candidates(
        self,
        observed_points: Sequence[ObservedPoint],
        desired_score: float,
    ) -> list[dict[str, Any]]:
        search_space = self.require_search_space()
        candidates: list[dict[str, Any]] = []
        seen = set(self._seen_configs)
        responses_per_template = max(1, math.ceil(self.n_candidates / self.n_templates))
        for template_index in range(self.n_templates):
            parameter_order = tuple(self._parameter_order(template_index))
            prompt = self._candidate_prompt(
                observed_points,
                desired_score=desired_score,
                parameter_order=parameter_order,
                template_index=template_index,
                requested_candidates=responses_per_template,
            )
            request = CandidateGenerationRequest(
                prompt=prompt,
                n_responses=responses_per_template,
                seed=_stable_seed(self._seed, "candidate_prompt", len(self._history), template_index),
                objective_direction=self._primary_direction,
                search_space=search_space,
                observed_points=tuple(observed_points),
                desired_score=desired_score,
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

        while len(candidates) < self.n_candidates:
            candidate = self._sample_initial_config(index=len(self._history) + len(candidates))
            key = _config_key(search_space, candidate)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
        return candidates

    def _predict_candidate(
        self,
        candidate_config: dict[str, Any],
        observed_points: Sequence[ObservedPoint],
    ) -> CandidatePrediction:
        values: list[float] = []
        responses_per_template = max(1, math.ceil(self.n_predictions / self.n_templates))
        for template_index in range(self.n_templates):
            parameter_order = tuple(self._parameter_order(template_index))
            prompt = self._score_prompt(
                observed_points,
                candidate_config=candidate_config,
                parameter_order=parameter_order,
                template_index=template_index,
            )
            request = ScorePredictionRequest(
                prompt=prompt,
                n_responses=responses_per_template,
                seed=_stable_seed(
                    self._seed,
                    "score_prompt",
                    len(self._history),
                    template_index,
                    _config_key(self.require_search_space(), candidate_config),
                ),
                objective_direction=self._primary_direction,
                search_space=self.require_search_space(),
                observed_points=tuple(observed_points),
                candidate_config=dict(candidate_config),
                parameter_order=parameter_order,
            )
            values.extend(self._parse_score_texts(self._require_backend().generate_score_texts(request)))

        if not values:
            fallback = self._fallback_score_prediction(observed_points)
            values = [fallback]

        mean = float(np.mean(values))
        std = float(np.std(values)) if len(values) > 1 else max(abs(mean) * 0.05, 1e-3)
        expected_improvement = self._expected_improvement(mean=mean, std=max(std, 1e-3), observed_points=observed_points)
        return CandidatePrediction(mean=mean, std=max(std, 1e-3), expected_improvement=expected_improvement)

    def _expected_improvement(
        self,
        *,
        mean: float,
        std: float,
        observed_points: Sequence[ObservedPoint],
    ) -> float:
        if not observed_points:
            return 0.0
        scores = [point.score for point in observed_points]
        incumbent = min(scores) if self._primary_direction == ObjectiveDirection.MINIMIZE else max(scores)
        if self._primary_direction == ObjectiveDirection.MINIMIZE:
            delta = incumbent - mean
        else:
            delta = mean - incumbent
        if std <= 0:
            return max(delta, 0.0)
        z_value = delta / std
        return float(delta * _normal_cdf(z_value) + std * _normal_pdf(z_value))

    def _fallback_score_prediction(self, observed_points: Sequence[ObservedPoint]) -> float:
        scores = np.asarray([point.score for point in observed_points], dtype=float)
        if scores.size == 0:
            return 0.0
        return float(np.mean(scores))

    def _parse_candidate_text(self, text: str) -> list[dict[str, Any]]:
        search_space = self.require_search_space()
        blocks = re.findall(r"<candidate>\s*(.*?)\s*</candidate>", text, flags=re.IGNORECASE | re.DOTALL)
        if not blocks:
            blocks = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
        parsed_candidates: list[dict[str, Any]] = []
        for block in blocks:
            parsed = self._parse_mapping(block)
            if parsed is None:
                continue
            try:
                candidate = search_space.coerce_config(parsed, use_defaults=False)
            except Exception:
                continue
            parsed_candidates.append(candidate)
        return parsed_candidates

    @staticmethod
    def _parse_mapping(text: str) -> dict[str, Any] | None:
        cleaned = text.strip()
        for loader in (json.loads, ast.literal_eval):
            try:
                value = loader(cleaned)
            except Exception:
                continue
            if isinstance(value, dict):
                return dict(value)
        return None

    @staticmethod
    def _parse_score_texts(texts: Sequence[str]) -> list[float]:
        scores: list[float] = []
        for text in texts:
            matches = re.findall(r"<score>\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*</score>", text)
            if not matches:
                matches = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text)
            for match in matches:
                try:
                    scores.append(float(match))
                except ValueError:
                    continue
        return scores

    def _parameter_order(self, template_index: int) -> list[str]:
        names = self.require_search_space().names()
        if self.n_templates == 1 or template_index % 2 == 0:
            return names
        rng = random.Random(_stable_seed(self._seed, "parameter_order", template_index))
        shuffled = list(names)
        rng.shuffle(shuffled)
        return shuffled

    def _history_examples(
        self,
        observed_points: Sequence[ObservedPoint],
        *,
        template_index: int,
    ) -> list[ObservedPoint]:
        ordered = sorted(
            observed_points,
            key=lambda point: _score_to_minimization(self._primary_direction, point.score),
        )
        best = ordered[: max(1, min(len(ordered), self.max_prompt_history // 2 or 1))]
        recent = list(observed_points)[-max(1, self.max_prompt_history - len(best)) :]
        combined: list[ObservedPoint] = []
        seen: set[tuple[int | None, float]] = set()
        for point in [*best, *recent]:
            key = (point.trial_id, point.score)
            if key in seen:
                continue
            seen.add(key)
            combined.append(point)
        if template_index % 3 == 1:
            combined = list(reversed(combined))
        elif template_index % 3 == 2:
            rng = random.Random(_stable_seed(self._seed, "history_examples", template_index))
            rng.shuffle(combined)
        return combined[: self.max_prompt_history]

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

    def _observed_trials_block(
        self,
        observed_points: Sequence[ObservedPoint],
        *,
        parameter_order: Sequence[str],
        template_index: int,
    ) -> str:
        examples = self._history_examples(observed_points, template_index=template_index)
        lines = []
        for point in examples:
            config_text = _serialize_config(self.require_search_space(), point.config, parameter_order=parameter_order)
            trial_label = f"trial {point.trial_id}" if point.trial_id is not None else "trial"
            lines.append(f"- {trial_label}: score={point.score:.8f}, config={config_text}")
        return "\n".join(lines)

    def _candidate_prompt(
        self,
        observed_points: Sequence[ObservedPoint],
        *,
        desired_score: float,
        parameter_order: Sequence[str],
        template_index: int,
        requested_candidates: int,
    ) -> str:
        direction_text = "minimize" if self._primary_direction == ObjectiveDirection.MINIMIZE else "maximize"
        assert self._primary_name is not None
        return textwrap.dedent(
            f"""
            You are helping with black-box optimization for task `{self.require_task_spec().name}`.
            Objective: {direction_text} `{self._primary_name}`.

            Search space:
            {self._search_space_block(parameter_order=parameter_order)}

            Task context:
            {self._task_context_block()}

            Observed trials:
            {self._observed_trials_block(observed_points, parameter_order=parameter_order, template_index=template_index)}

            Recommend {requested_candidates} promising new configurations whose objective value could approach {desired_score:.8f}.
            Return each configuration as compact JSON wrapped in <candidate>...</candidate>.
            Do not repeat observed configurations. Do not add any prose.
            """
        ).strip()

    def _score_prompt(
        self,
        observed_points: Sequence[ObservedPoint],
        *,
        candidate_config: dict[str, Any],
        parameter_order: Sequence[str],
        template_index: int,
    ) -> str:
        direction_text = "lower is better" if self._primary_direction == ObjectiveDirection.MINIMIZE else "higher is better"
        assert self._primary_name is not None
        candidate_text = _serialize_config(self.require_search_space(), candidate_config, parameter_order=parameter_order)
        return textwrap.dedent(
            f"""
            You are acting as a surrogate model for task `{self.require_task_spec().name}`.
            Predict objective `{self._primary_name}` where {direction_text}.

            Search space:
            {self._search_space_block(parameter_order=parameter_order)}

            Task context:
            {self._task_context_block()}

            Observed trials:
            {self._observed_trials_block(observed_points, parameter_order=parameter_order, template_index=template_index)}

            Candidate configuration:
            {candidate_text}

            Return only the predicted objective value wrapped in <score>...</score>.
            """
        ).strip()


__all__ = [
    "CandidateGenerationRequest",
    "CandidatePrediction",
    "HeuristicLlamboBackend",
    "LlamboAlgorithm",
    "LlamboBackend",
    "ObservedPoint",
    "OpenAICompatibleLlamboBackend",
    "ScorePredictionRequest",
]
