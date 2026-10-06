"""Workspace-local Python API for BBO shell/file agents.

This file is copied into each agent workspace as ``bbo_tools.py``.  It must stay
free of imports from the installed ``bbo`` package because shell/file agents may
run it from an isolated workspace or a framework-created environment.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import bbo_tool as _bridge


class BBOToolError(RuntimeError):
    """Raised when a workspace BBO tool call fails."""


class BBO:
    """Python API for the BBO workspace tool bridge."""

    def __init__(self, config_path: str | Path = "bbo_tool_config.json") -> None:
        path = Path(config_path)
        self.config_path = path if path.is_absolute() else Path.cwd() / path

    def call(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call one BBO workspace tool and return its result object."""

        args = dict(arguments or {})
        raw_arguments = json.dumps(args, ensure_ascii=False, sort_keys=True)
        config = _bridge._read_json(self.config_path)
        if tool_name.startswith("optimizer_"):
            hosted = _bridge._execute_via_optimizer_host(
                tool_name=tool_name,
                raw_arguments=raw_arguments,
                config=config,
                config_path=self.config_path,
            )
            if hosted is not None:
                if hosted.get("ok") is not True:
                    raise BBOToolError(
                        f"{tool_name} failed: {hosted.get('message', 'unknown optimizer host error')}"
                    )
                return hosted["result"]
        started = time.monotonic()
        timestamp = time.time()
        try:
            result = _bridge._execute(tool_name, args, config)
            payload = {"ok": True, "result": result}
            success = True
        except Exception as exc:
            payload = {"ok": False, "error": "exception", "message": str(exc)}
            success = False
        _bridge._log_call(
            config,
            tool_name,
            raw_arguments,
            payload,
            started,
            timestamp,
            success,
            interface="workspace_python_api",
        )
        if not success:
            raise BBOToolError(f"{tool_name} failed: {payload['message']}")
        return payload["result"]

    def task_context(self, **kwargs: Any) -> dict[str, Any]:
        return self.call("get_task_context", kwargs)

    def manifest(self) -> dict[str, Any]:
        return self.call("get_manifest", {})

    def search_space(self) -> dict[str, Any]:
        return self.call("get_search_space", {})

    def objective(self) -> dict[str, Any]:
        return self.call("get_objective", {})

    def tool_specs(self) -> dict[str, Any]:
        return self.call("get_tool_specs", {})

    def history(self, mode: str = "recent", limit: int = 20, offset: int = 0) -> dict[str, Any]:
        return self.call("get_trial_history", {"mode": mode, "limit": limit, "offset": offset})

    def incumbent(self) -> dict[str, Any]:
        return self.call("get_incumbent", {})

    def history_overview(self, recent_limit: int = 8, progression_limit: int = 40) -> dict[str, Any]:
        return self.call(
            "get_history_overview",
            {"recent_limit": recent_limit, "progression_limit": progression_limit},
        )

    def compare_trials(self, trial_ids: list[int | str]) -> dict[str, Any]:
        return self.call("compare_trials", {"trial_ids": trial_ids})

    def find_nearest_trials(self, target: Any, k: int = 5) -> dict[str, Any]:
        return self.call("find_nearest_trials", {"target": target, "k": k})

    def estimate_local_effects(
        self,
        reference: Any,
        *,
        variables: list[str] | None = None,
        local_radius: float = 0.35,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"reference": reference, "local_radius": local_radius}
        if variables is not None:
            args["variables"] = variables
        return self.call("estimate_local_effects", args)

    def measure_search_coverage(self, recent_limit: int = 8) -> dict[str, Any]:
        return self.call("measure_search_coverage", {"recent_limit": recent_limit})

    def summarize_objective_metrics(self, recent_limit: int = 8, progression_limit: int = 40) -> dict[str, Any]:
        return self.call(
            "summarize_objective_metrics",
            {"recent_limit": recent_limit, "progression_limit": progression_limit},
        )

    def fit_and_check_surrogate(self, min_observations: int = 6) -> dict[str, Any]:
        return self.call("fit_and_check_surrogate", {"min_observations": min_observations})

    def score_virtual_candidates(self, model_id: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        return self.call("score_virtual_candidates", {"model_id": model_id, "candidates": candidates})

    def validate_candidate(self, candidate: dict[str, Any], too_similar_threshold: float | None = None) -> dict[str, Any]:
        args: dict[str, Any] = {"candidate": candidate}
        if too_similar_threshold is not None:
            args["too_similar_threshold"] = too_similar_threshold
        return self.call("validate_candidate", args)

    def validate(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        return self.call("validate_candidates", {"candidates": candidates})

    def sample(
        self,
        n: int = 4,
        *,
        strategy: str = "random",
        seed: int | None = None,
        jitter_fraction: float = 0.1,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"n": n, "strategy": strategy, "jitter_fraction": jitter_fraction}
        if seed is not None:
            args["seed"] = seed
        return self.call("sample_candidates", args)

    def analyze_history(self, limit: int = 100) -> dict[str, Any]:
        return self.call("analyze_history", {"limit": limit})

    def profile_history_quality(self, limit: int = 200) -> dict[str, Any]:
        return self.call("profile_history_quality", {"limit": limit})

    def analyze_convergence(self, limit: int = 200) -> dict[str, Any]:
        return self.call("analyze_convergence", {"limit": limit})

    def rank_parameter_importance(self, limit: int = 200, top_k: int = 8) -> dict[str, Any]:
        return self.call("rank_parameter_importance", {"limit": limit, "top_k": top_k})

    def analyze_parameter_interactions(self, limit: int = 200, top_k: int = 8) -> dict[str, Any]:
        return self.call("analyze_parameter_interactions", {"limit": limit, "top_k": top_k})

    def locate_promising_regions(self, limit: int = 200, top_fraction: float = 0.25) -> dict[str, Any]:
        return self.call(
            "locate_promising_regions",
            {"limit": limit, "top_fraction": top_fraction},
        )

    def locate_underexplored_regions(self, limit: int = 200, bins: int = 5) -> dict[str, Any]:
        return self.call("locate_underexplored_regions", {"limit": limit, "bins": bins})

    def recommend_search_regions(
        self,
        limit: int = 200,
        bins: int = 5,
        mode: str = "auto",
    ) -> dict[str, Any]:
        return self.call(
            "recommend_search_regions",
            {"limit": limit, "bins": bins, "mode": mode},
        )

    def analyze_search_strategy(
        self,
        limit: int = 200,
        elite_fraction: float = 0.3,
        min_width_fraction: float = 0.35,
        max_cv_folds: int = 5,
    ) -> dict[str, Any]:
        return self.call(
            "analyze_search_strategy",
            {
                "limit": limit,
                "elite_fraction": elite_fraction,
                "min_width_fraction": min_width_fraction,
                "max_cv_folds": max_cv_folds,
            },
        )

    def render_search_diagnostics(self, title: str = "Search diagnostics") -> dict[str, Any]:
        return self.call("render_search_diagnostics", {"title": title})

    def optimizer_suggest(
        self,
        backend: str | None = None,
        *,
        q: int = 1,
        bounds: dict[str, list[float | int]] | None = None,
        around: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"q": q}
        if backend is not None:
            args["backend"] = backend
        if bounds is not None:
            args["bounds"] = bounds
        if around is not None:
            args["around"] = around
        if options is not None:
            args["options"] = options
        if seed is not None:
            args["seed"] = int(seed)
        return self.call("optimizer_suggest", args)

    def optimizer_predict(self, configs: list[dict[str, Any]]) -> dict[str, Any]:
        return self.call("optimizer_predict", {"configs": configs})

    def optimizer_recommend_backends(self, k: int = 3) -> dict[str, Any]:
        return self.call("optimizer_recommend_backends", {"k": k})

    def optimizer_portfolio_suggest(
        self,
        backends: list[str] | None = None,
        *,
        q_per_backend: int = 1,
        bounds: dict[str, list[float | int]] | None = None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"q_per_backend": q_per_backend}
        if backends is not None:
            args["backends"] = backends
        if bounds is not None:
            args["bounds"] = bounds
        return self.call("optimizer_portfolio_suggest", args)

    def optimizer_score(self, configs: list[dict[str, Any]]) -> dict[str, Any]:
        return self.call("optimizer_score", {"configs": configs})

    def optimizer_diagnostics(self) -> dict[str, Any]:
        return self.call("optimizer_diagnostics", {})

    def optimizer_status(self) -> dict[str, Any]:
        return self.call("optimizer_status", {})

    def optimizer_set_backend(self, backend: str) -> dict[str, Any]:
        return self.call("optimizer_set_backend", {"backend": backend})

    def optimizer_set_bounds(self, bounds: dict[str, list[float | int]]) -> dict[str, Any]:
        return self.call("optimizer_set_bounds", {"bounds": bounds})

    def optimizer_set_acquisition(
        self,
        name: str,
        parameters: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"name": name}
        if parameters is not None:
            args["parameters"] = parameters
        return self.call("optimizer_set_acquisition", args)

    def optimizer_reset_policy(self) -> dict[str, Any]:
        return self.call("optimizer_reset_policy", {})
    def recent_search_actions(self, limit: int = 10) -> dict[str, Any]:
        return self.call("get_recent_search_actions", {"limit": limit})

    def memory_read(
        self,
        *,
        kind: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"limit": limit}
        if kind is not None:
            args["kind"] = kind
        if tags is not None:
            args["tags"] = tags
        return self.call("memory_read", args)

    def memory_write(
        self,
        *,
        kind: str,
        content: str,
        tags: list[str] | None = None,
        source_call_id: str | None = None,
        trial_range: list[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"kind": kind, "content": content}
        if tags is not None:
            args["tags"] = tags
        if source_call_id is not None:
            args["source_call_id"] = source_call_id
        if trial_range is not None:
            args["trial_range"] = trial_range
        if metadata is not None:
            args["metadata"] = metadata
        return self.call("memory_write", args)

    def code_interpreter(self, code: str, language: str = "python") -> dict[str, Any]:
        return self.call("code_interpreter", {"code": code, "language": language})

    def web_search(self, query: str, limit: int = 5) -> dict[str, Any]:
        return self.call("web_search", {"query": query, "limit": limit})

    def fetch_url(self, url: str, max_chars: int = 4000) -> dict[str, Any]:
        return self.call("fetch_url", {"url": url, "max_chars": max_chars})


__all__ = ["BBO", "BBOToolError"]
