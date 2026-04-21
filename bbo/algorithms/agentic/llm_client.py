"""Provider clients for Pablo role calls."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from ...core import CategoricalParam, FloatParam, IntParam, SearchSpace
from .prompts import PromptBundle


@dataclass(frozen=True)
class PabloProviderConfig:
    provider: str = "mock"
    base_url: str | None = None
    api_key_env: str = "PABLO_API_KEY"


class PabloLlmClient:
    """Stateless JSON completion interface used by Pablo roles."""

    provider_name = "unknown"

    def complete(self, *, role: str, model: str, prompt: PromptBundle) -> str:
        raise NotImplementedError


class MockPabloLlmClient(PabloLlmClient):
    provider_name = "mock"

    def __init__(self, *, seed: int = 0):
        self.seed = int(seed)

    def complete(self, *, role: str, model: str, prompt: PromptBundle) -> str:
        rng = random.Random(self._seed_for(role=role, model=model, prompt=prompt))
        search_space: SearchSpace = prompt.context["search_space"]
        if role == "planner":
            payload = self._planner_payload(prompt, search_space)
        elif role == "explorer":
            payload = self._candidate_payload(
                search_space,
                rng,
                anchors=self._candidate_anchors(prompt.context),
                count=12,
                hint_text="explore global context",
            )
        elif role == "worker":
            payload = self._candidate_payload(
                search_space,
                rng,
                anchors=[dict(prompt.context["current_seed"])],
                count=6,
                hint_text=str(prompt.context["planner_task_text"]),
            )
        else:
            raise ValueError(f"Unsupported Pablo mock role `{role}`.")
        return json.dumps(payload, indent=2, sort_keys=True)

    def _planner_payload(self, prompt: PromptBundle, search_space: SearchSpace) -> dict[str, str]:
        has_categorical = any(isinstance(param, CategoricalParam) for param in search_space)
        names = search_space.names()
        prefix = names[:3]
        payload = {
            "EXPLOIT_TOP": f"TASK: refine the strongest known region with small edits around {prefix}.",
            "LOCAL_STABILITY": "TASK: keep the current promising pattern and perturb only one or two parameters at a time.",
            "BOUNDARY_SCAN": "TASK: probe low/high bounds to test whether the oracle improves near extremes.",
            "INTERACTION_SCAN": "TASK: explore pairwise interactions that the current history may have under-sampled.",
            "DIVERSE_JUMP": "TASK: generate candidates that are intentionally different from recent successes.",
            "DEFAULT_RECHECK": "TASK: include conservative candidates near the default configuration for calibration.",
            "FAILURE_RECOVERY": "TASK: avoid exact repeats of failed candidates and move to adjacent valid regions.",
            "SEED_MIXING": "TASK: combine useful traits from different successful seeds into one valid candidate.",
        }
        if has_categorical:
            payload["CATEGORY_ROTATION"] = "TASK: rotate categorical choices while keeping the strongest numeric pattern stable."
        else:
            payload["SMOOTH_NUMERIC_PUSH"] = "TASK: move smoothly along the best-performing numeric direction without leaving the valid bounds."
        return payload

    def _candidate_anchors(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        c_global = context.get("c_global", [])
        anchors = [entry.get("config", {}) for entry in c_global if isinstance(entry, dict) and isinstance(entry.get("config"), dict)]
        if not anchors:
            anchors = [context["search_space"].defaults()]
        return [dict(anchor) for anchor in anchors]

    def _candidate_payload(
        self,
        search_space: SearchSpace,
        rng: random.Random,
        *,
        anchors: list[dict[str, Any]],
        count: int,
        hint_text: str,
    ) -> dict[str, Any]:
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        local_anchors = anchors or [search_space.defaults()]
        mode = self._hint_mode(hint_text)
        while len(candidates) < count:
            anchor = dict(local_anchors[len(candidates) % len(local_anchors)])
            candidate = self._mutate_candidate(search_space, rng, anchor=anchor, mode=mode)
            identity = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append(candidate)
        return {"candidates": candidates}

    def _mutate_candidate(
        self,
        search_space: SearchSpace,
        rng: random.Random,
        *,
        anchor: dict[str, Any],
        mode: str,
    ) -> dict[str, Any]:
        mutated: dict[str, Any] = {}
        for param in search_space:
            base_value = anchor.get(param.name, param.effective_default())
            if isinstance(param, FloatParam):
                span = float(param.high) - float(param.low)
                if mode == "diverse":
                    value = rng.uniform(float(param.low), float(param.high))
                else:
                    scale = 0.08 if mode == "exploit" else 0.22
                    value = float(base_value) + rng.uniform(-scale, scale) * span
                value = min(max(value, float(param.low)), float(param.high))
                mutated[param.name] = param.coerce(value)
            elif isinstance(param, IntParam):
                span = int(param.high) - int(param.low)
                if mode == "diverse":
                    value = rng.randint(int(param.low), int(param.high))
                else:
                    radius = max(1, int(round(max(1, span) * (0.10 if mode == "exploit" else 0.25))))
                    value = int(base_value) + rng.randint(-radius, radius)
                value = min(max(value, int(param.low)), int(param.high))
                mutated[param.name] = param.coerce(value)
            elif isinstance(param, CategoricalParam):
                choices = list(param.choices)
                if len(choices) == 1:
                    mutated[param.name] = choices[0]
                    continue
                if mode == "exploit" and base_value in choices and rng.random() < 0.7:
                    mutated[param.name] = base_value
                elif mode == "boundary" and base_value in choices and rng.random() < 0.4:
                    mutated[param.name] = choices[0 if base_value != choices[0] else -1]
                else:
                    if base_value in choices and rng.random() < 0.4:
                        pool = [choice for choice in choices if choice != base_value]
                    else:
                        pool = choices
                    mutated[param.name] = rng.choice(pool)
            else:
                raise TypeError(f"Unsupported parameter type: {type(param).__name__}")
        return search_space.coerce_config(mutated, use_defaults=False)

    @staticmethod
    def _hint_mode(text: str) -> str:
        lowered = text.lower()
        if any(token in lowered for token in ("boundary", "extreme", "edge")):
            return "boundary"
        if any(token in lowered for token in ("diverse", "rotate", "different", "explore")):
            return "diverse"
        return "exploit"

    def _seed_for(self, *, role: str, model: str, prompt: PromptBundle) -> int:
        digest = sha256()
        digest.update(str(self.seed).encode("utf-8"))
        digest.update(role.encode("utf-8"))
        digest.update(model.encode("utf-8"))
        digest.update(prompt.system.encode("utf-8"))
        digest.update(prompt.user.encode("utf-8"))
        return int(digest.hexdigest()[:16], 16)


class OpenAICompatiblePabloLlmClient(PabloLlmClient):
    provider_name = "openai-compatible"

    def __init__(self, *, api_key_env: str, base_url: str | None = None):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable `{api_key_env}` is required for the openai-compatible provider.")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional extra.
            raise ImportError(
                "The openai-compatible Pablo provider requires the optional `pablo` extra. "
                "Install it with `uv sync --extra dev --extra pablo` or the equivalent environment sync command."
            ) from exc
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, *, role: str, model: str, prompt: PromptBundle) -> str:
        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
        }
        try:
            response = self._client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
        except TypeError:
            response = self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if not content:
            raise ValueError(f"Provider returned an empty response for Pablo role `{role}`.")
        return str(content)


def create_llm_client(config: PabloProviderConfig, *, seed: int = 0) -> PabloLlmClient:
    if config.provider == "mock":
        return MockPabloLlmClient(seed=seed)
    if config.provider == "openai-compatible":
        return OpenAICompatiblePabloLlmClient(api_key_env=config.api_key_env, base_url=config.base_url)
    raise ValueError(f"Unknown Pablo provider `{config.provider}`. Expected `mock` or `openai-compatible`.")


__all__ = [
    "MockPabloLlmClient",
    "OpenAICompatiblePabloLlmClient",
    "PabloLlmClient",
    "PabloProviderConfig",
    "create_llm_client",
]
