"""BBOPlace-Bench macro placement benchmark (HTTP black-box HPWL)."""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ...core import (
    EvaluationResult,
    FloatParam,
    ObjectiveDirection,
    ObjectiveSpec,
    SearchSpace,
    Task,
    TaskDescriptionRef,
    TaskSpec,
    TrialStatus,
    TrialSuggestion,
)

_TASK_FILE = Path(__file__).resolve()
PACKAGE_ROOT = _TASK_FILE.parents[2]
TASK_DESCRIPTION_ROOT = PACKAGE_ROOT / "task_descriptions"

BBOPLACE_TASK_KEY = "bboplace_bench"
DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_EVALUATE_PATH = "/evaluate"
DEFAULT_N_GRID = 224
DEFAULT_N_MACRO = 32
DEFAULT_BENCHMARK = "adaptec1"
DEFAULT_PLACER = "mgo"
DEFAULT_HTTP_TIMEOUT_S = 300.0


def _build_macro_placement_space(*, n_macro: int, n_grid_x: int, n_grid_y: int) -> SearchSpace:
    """Build ordered search space: x_0..x_{n-1}, then y_0..y_{n-1}."""
    params: list[FloatParam] = []
    for i in range(n_macro):
        params.append(
            FloatParam(
                f"x_{i}",
                low=0.0,
                high=float(n_grid_x),
                default=float(n_grid_x) / 2.0,
            )
        )
    for i in range(n_macro):
        params.append(
            FloatParam(
                f"y_{i}",
                low=0.0,
                high=float(n_grid_y),
                default=float(n_grid_y) / 2.0,
            )
        )
    return SearchSpace(params)


def _default_post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """POST JSON and parse response (stdlib only)."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


PostJsonFn = Callable[[str, dict[str, Any], float], dict[str, Any]]


@dataclass(frozen=True)
class BBOPlaceDefinition:
    """Static packaging for one BBOPlace-Bench instance."""

    key: str
    display_name: str
    description: str
    search_space: SearchSpace
    description_dir: Path
    default_max_evaluations: int
    benchmark: str
    placer: str
    base_url: str
    evaluate_path: str
    n_macro: int
    n_grid_x: int
    n_grid_y: int
    bench_seed: int

    @property
    def dimension(self) -> int:
        return len(self.search_space)


def default_bboplace_definition(
    *,
    key: str = BBOPLACE_TASK_KEY,
    n_macro: int = DEFAULT_N_MACRO,
    n_grid_x: int = DEFAULT_N_GRID,
    n_grid_y: int = DEFAULT_N_GRID,
    benchmark: str = DEFAULT_BENCHMARK,
    bench_seed: int = 1,
    placer: str = DEFAULT_PLACER,
    base_url: str | None = None,
    evaluate_path: str = DEFAULT_EVALUATE_PATH,
    default_max_evaluations: int = 40,
    description_dir: Path | None = None,
) -> BBOPlaceDefinition:
    """Default BBOPlace task matching the reference HTTP API."""
    resolved_base = base_url or os.environ.get("BBOPLACE_BASE_URL", DEFAULT_BASE_URL)
    resolved_description_dir = description_dir or (TASK_DESCRIPTION_ROOT / key)
    space = _build_macro_placement_space(n_macro=n_macro, n_grid_x=n_grid_x, n_grid_y=n_grid_y)
    return BBOPlaceDefinition(
        key=key,
        display_name=f"BBOPlace-Bench ({benchmark}, {n_macro} macros)",
        description=(
            "Macro placement on chip benchmarks via HTTP: minimize HPWL under grid bounds. "
            "True optimum is unknown."
        ),
        search_space=space,
        description_dir=resolved_description_dir,
        default_max_evaluations=default_max_evaluations,
        benchmark=benchmark,
        placer=placer,
        base_url=resolved_base.rstrip("/"),
        evaluate_path=evaluate_path if evaluate_path.startswith("/") else f"/{evaluate_path}",
        n_macro=n_macro,
        n_grid_x=n_grid_x,
        n_grid_y=n_grid_y,
        bench_seed=bench_seed,
    )


@dataclass(frozen=True)
class BBOPlaceTaskConfig:
    """Runtime options for `BBOPlaceTask`."""

    problem: str = BBOPLACE_TASK_KEY
    max_evaluations: int | None = None
    seed: int = 0
    definition: BBOPlaceDefinition | None = None
    http_timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_S
    post_json: PostJsonFn | None = None
    metadata: dict[str, str] = field(default_factory=dict)


class BBOPlaceTask(Task):
    """Black-box task that queries the BBOPlace-Bench HTTP evaluator for HPWL."""

    def __init__(
        self,
        config: BBOPlaceTaskConfig,
        definition: BBOPlaceDefinition | None = None,
    ) -> None:
        self.config = config
        self.definition = definition or config.definition or default_bboplace_definition()
        self._post_json: PostJsonFn = config.post_json or _default_post_json
        search_space = self.definition.search_space
        cma_initial = search_space.defaults()
        self._spec = TaskSpec(
            name=self.definition.key,
            search_space=search_space,
            objectives=(ObjectiveSpec("hpwl", ObjectiveDirection.MINIMIZE),),
            max_evaluations=config.max_evaluations or self.definition.default_max_evaluations,
            description_ref=TaskDescriptionRef.from_directory(self.definition.key, self.definition.description_dir),
            metadata={
                "problem_key": self.definition.key,
                "display_name": self.definition.display_name,
                "dimension": self.definition.dimension,
                "benchmark": self.definition.benchmark,
                "n_macro": self.definition.n_macro,
                "n_grid_x": self.definition.n_grid_x,
                "n_grid_y": self.definition.n_grid_y,
                "placer": self.definition.placer,
                "bench_seed": int(config.seed),
                "bench_seed_default": int(self.definition.bench_seed),
                "base_url": self.definition.base_url,
                "known_optimum": None,
                "cma_initial_config": cma_initial,
                "task_family": "bboplace",
                **config.metadata,
            },
        )

    @property
    def spec(self) -> TaskSpec:
        return self._spec

    def evaluate(self, suggestion: TrialSuggestion) -> EvaluationResult:
        start = time.perf_counter()
        config = self.spec.search_space.coerce_config(suggestion.config, use_defaults=False)
        vector = self.spec.search_space.to_numeric_vector(config)
        row = [float(value) for value in vector]
        url = f"{self.definition.base_url}{self.definition.evaluate_path}"
        payload: dict[str, Any] = {
            "benchmark": self.definition.benchmark,
            "seed": int(self.config.seed),
            "n_macro": self.definition.n_macro,
            "placer": self.definition.placer,
            "x": [row],
        }
        try:
            response = self._post_json(url, payload, self.config.http_timeout_seconds)
        except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
            elapsed = time.perf_counter() - start
            return EvaluationResult(
                status=TrialStatus.FAILED,
                objectives={},
                metrics={"dimension": float(self.definition.dimension)},
                elapsed_seconds=elapsed,
                error_type=type(exc).__name__,
                error_message=str(exc),
                metadata={"problem_key": self.definition.key},
            )
        elapsed = time.perf_counter() - start
        hpwl_raw = response.get("hpwl")
        if not isinstance(hpwl_raw, list) or not hpwl_raw:
            return EvaluationResult(
                status=TrialStatus.FAILED,
                objectives={},
                metrics={"dimension": float(self.definition.dimension)},
                elapsed_seconds=elapsed,
                error_type="InvalidResponse",
                error_message="Response missing non-empty `hpwl` list.",
                metadata={"problem_key": self.definition.key},
            )
        try:
            hpwl = float(hpwl_raw[0])
        except (TypeError, ValueError) as exc:
            return EvaluationResult(
                status=TrialStatus.FAILED,
                objectives={},
                metrics={"dimension": float(self.definition.dimension)},
                elapsed_seconds=elapsed,
                error_type=type(exc).__name__,
                error_message=f"Response `hpwl[0]` could not be converted to float: {hpwl_raw[0]!r}.",
                metadata={"problem_key": self.definition.key},
            )
        if not math.isfinite(hpwl):
            return EvaluationResult(
                status=TrialStatus.FAILED,
                objectives={},
                metrics={"dimension": float(self.definition.dimension)},
                elapsed_seconds=elapsed,
                error_type="InvalidResponse",
                error_message=f"Response `hpwl[0]` must be finite, got {hpwl!r}.",
                metadata={"problem_key": self.definition.key},
            )
        metrics: dict[str, Any] = {
            "dimension": float(self.definition.dimension),
            "n_macro": float(self.definition.n_macro),
        }
        for name, scalar in zip(self.spec.search_space.names(), vector, strict=True):
            metrics[f"coord::{name}"] = float(scalar)
        return EvaluationResult(
            status=TrialStatus.SUCCESS,
            objectives={"hpwl": hpwl},
            metrics=metrics,
            elapsed_seconds=elapsed,
            metadata={
                "problem_key": self.definition.key,
                "display_name": self.definition.display_name,
            },
        )

    def sanity_check(self):
        report = super().sanity_check()
        if self.definition.n_macro <= 0:
            report.add_error("invalid_n_macro", "n_macro must be positive.")
        expected_dim = int(self.definition.n_macro) * 2
        if self.definition.dimension != expected_dim:
            report.add_error(
                "dimension_mismatch",
                f"Search-space dimension {self.definition.dimension} does not match 2 * n_macro ({expected_dim}).",
            )
        if not self.definition.base_url:
            report.add_error("invalid_base_url", "base_url must be non-empty.")
        if not self.definition.evaluate_path.startswith("/"):
            report.add_error("invalid_evaluate_path", "evaluate_path must start with '/'.")
        return report


def create_bboplace_task(
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    definition: BBOPlaceDefinition | None = None,
    post_json: PostJsonFn | None = None,
    http_timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_S,
    metadata: dict[str, str] | None = None,
) -> BBOPlaceTask:
    """Factory for the default BBOPlace-Bench task."""
    config = BBOPlaceTaskConfig(
        max_evaluations=max_evaluations,
        seed=seed,
        definition=definition,
        post_json=post_json,
        http_timeout_seconds=http_timeout_seconds,
        metadata=dict(metadata or {}),
    )
    return BBOPlaceTask(config=config, definition=definition)


BBOPLACE_DEFAULT_DEFINITION = default_bboplace_definition()

__all__ = [
    "BBOPLACE_DEFAULT_DEFINITION",
    "BBOPLACE_TASK_KEY",
    "BBOPlaceDefinition",
    "BBOPlaceTask",
    "BBOPlaceTaskConfig",
    "DEFAULT_BASE_URL",
    "default_bboplace_definition",
    "create_bboplace_task",
]
