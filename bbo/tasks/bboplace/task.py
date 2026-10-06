"""BBOPlace-Bench macro placement benchmark (service-backed HPWL)."""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

import numpy as np
from scipy.stats import qmc

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

BBOPLACE_TASK_KEY = "bboplace_adaptec1_n32"
# Default host port 8070 avoids the MariaDB evaluator on port 8080.
# For a container listening on 8080, map it with -p 8070:8080.
DEFAULT_BASE_URL = "http://127.0.0.1:8070"
DEFAULT_EVALUATE_PATH = "/evaluate"
DEFAULT_N_GRID = 224
DEFAULT_N_MACRO = 32
DEFAULT_BENCHMARK = "adaptec1"
DEFAULT_PLACER = "geometry_repair"
DEFAULT_HTTP_TIMEOUT_S = 300.0
BBOPLACE_COMPACT_INITIAL_DESIGN_SIZE = 50

# Macro counts in the source placement instances.
BENCHMARK_MAX_N_MACRO: Mapping[str, int] = MappingProxyType(
    {
        "adaptec1": 543,
        "adaptec2": 566,
        "adaptec3": 723,
        "adaptec4": 1329,
        "bigblue1": 560,
        "bigblue3": 1298,
    }
)


def _benchmark_table_key(benchmark: str) -> str:
    """Normalize benchmark string for table lookup (handles paths like ispd2005/adaptec1)."""
    part = benchmark.strip().lower()
    if "/" in part:
        return part.rsplit("/", 1)[-1]
    return part


def max_n_macro_for_benchmark(benchmark: str) -> int | None:
    """Return the packaged macro-count cap for a known benchmark, or None if unknown."""
    return BENCHMARK_MAX_N_MACRO.get(_benchmark_table_key(benchmark))


def n_macro_over_benchmark_cap_message(*, benchmark: str, n_macro: int) -> str | None:
    """Return a human-readable error if ``n_macro`` exceeds the cap, else None."""
    cap = max_n_macro_for_benchmark(benchmark)
    if cap is None or n_macro <= cap:
        return None
    return (
        f"n_macro={n_macro} exceeds the maximum for benchmark {benchmark!r} "
        f"(effective macro cap is {cap}). Reduce n_macro or choose another benchmark."
    )


def _assert_n_macro_within_benchmark_cap(*, benchmark: str, n_macro: int) -> None:
    msg = n_macro_over_benchmark_cap_message(benchmark=benchmark, n_macro=n_macro)
    if msg is not None:
        raise ValueError(msg)


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


def bboplace_compact_initial_configurations(
    *, seed: int, n_macro: int, n_grid_x: int, n_grid_y: int
) -> tuple[dict[str, float], ...]:
    """Return the shared compact scrambled-Sobol initialization.

    Every compared optimizer receives the same 50 configurations. Their
    geometry-repaired outcomes and fallback layout are frozen in the bundle.
    """

    dimension = 2 * int(n_macro)
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=int(seed))
    points = sampler.random_base2(m=math.ceil(math.log2(BBOPLACE_COMPACT_INITIAL_DESIGN_SIZE)))
    lower = np.zeros(dimension, dtype=float)
    upper = np.asarray(
        ([float(n_grid_x)] * int(n_macro)) + ([float(n_grid_y)] * int(n_macro)),
        dtype=float,
    )
    scaled = qmc.scale(points[:BBOPLACE_COMPACT_INITIAL_DESIGN_SIZE], lower, upper)
    names = ([f"x_{i}" for i in range(int(n_macro))] + [f"y_{i}" for i in range(int(n_macro))])
    return tuple(
        {name: float(value) for name, value in zip(names, point, strict=True)}
        for point in scaled
    )


def bboplace_compact_protocol_metadata(
    *, seed: int, n_macro: int, n_grid_x: int, n_grid_y: int
) -> dict[str, Any]:
    configurations = bboplace_compact_initial_configurations(
        seed=seed, n_macro=n_macro, n_grid_x=n_grid_x, n_grid_y=n_grid_y
    )
    return {
        "name": "bboplace_geometry_repair_v1",
        "upstream_protocol": "geometry_repair_worst_initial_v1",
        "variant": "geometry_repair_worst_initial_fallback",
        "initialization": {
            "strategy": "fixed_configurations",
            "sampling": "scrambled_sobol",
            "seed": int(seed),
            "count": len(configurations),
            "configurations": [dict(config) for config in configurations],
            "source": "scipy.stats.qmc.Sobol(scramble=True):first_50_points",
            "scope": "shared_by_task_seed_across_all_algorithms",
        },
    }


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
    bench_seed: int = 2,
    placer: str = DEFAULT_PLACER,
    base_url: str | None = None,
    evaluate_path: str = DEFAULT_EVALUATE_PATH,
    default_max_evaluations: int = 250,
    description_dir: Path | None = None,
) -> BBOPlaceDefinition:
    """Default BBOPlace task matching the published evaluator contract."""
    _assert_n_macro_within_benchmark_cap(benchmark=benchmark, n_macro=n_macro)
    resolved_base = base_url or os.environ.get("BBOPLACE_BASE_URL", DEFAULT_BASE_URL)
    resolved_description_dir = description_dir or (TASK_DESCRIPTION_ROOT / key)
    space = _build_macro_placement_space(n_macro=n_macro, n_grid_x=n_grid_x, n_grid_y=n_grid_y)
    return BBOPlaceDefinition(
        key=key,
        display_name=f"BBOPlace-Bench ({benchmark}, {n_macro} macros)",
        description=(
            "Macro placement on chip benchmarks via an external evaluator service: "
            "minimize HPWL under grid bounds. "
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
    seed: int = 2
    definition: BBOPlaceDefinition | None = None
    http_timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_S
    post_json: PostJsonFn | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    repair_bundle_sha256: str | None = None


class BBOPlaceTask(Task):
    """Black-box task that queries the published BBOPlace-Bench evaluator for HPWL."""

    def __init__(
        self,
        config: BBOPlaceTaskConfig,
        definition: BBOPlaceDefinition | None = None,
    ) -> None:
        self.config = config
        self.definition = definition or config.definition or default_bboplace_definition()
        if self.definition.placer != "geometry_repair" or not config.repair_bundle_sha256:
            raise ValueError("A frozen geometry-repair bundle is required; use create_bboplace_task")
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
                "benchmark_protocol": bboplace_compact_protocol_metadata(
                    seed=int(config.seed),
                    n_macro=int(self.definition.n_macro),
                    n_grid_x=int(self.definition.n_grid_x),
                    n_grid_y=int(self.definition.n_grid_y),
                ),
                "known_optimum": None,
                "cma_initial_config": cma_initial,
                "task_family": "bboplace",
                **config.metadata,
            },
        )
        if config.repair_bundle_sha256:
            self._spec.metadata["placer"] = "geometry_repair"
            self._spec.metadata["evaluation_protocol"] = "geometry_repair_worst_initial_v1"
            self._spec.metadata["repair_bundle_sha256"] = config.repair_bundle_sha256
            protocol = self._spec.metadata["benchmark_protocol"]
            protocol["name"] = "bboplace_geometry_repair_v1"
            protocol["variant"] = "geometry_repair_worst_initial_fallback"
            protocol.pop("upstream_protocol", None)

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
        if self.config.repair_bundle_sha256:
            payload.update(protocol="geometry_repair_worst_initial_v1", placer="geometry_repair",
                           bundle_sha256=self.config.repair_bundle_sha256)
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
        if self.config.repair_bundle_sha256 and (
            response.get("protocol") != "geometry_repair_worst_initial_v1"
            or response.get("bundle_sha256") != self.config.repair_bundle_sha256
        ):
            return EvaluationResult(status=TrialStatus.FAILED, error_type="ProtocolMismatch",
                                    error_message="Service did not use the frozen geometry repair bundle")
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
        if hpwl < 0.0:
            return EvaluationResult(
                status=TrialStatus.FAILED,
                objectives={},
                metrics={"dimension": float(self.definition.dimension)},
                elapsed_seconds=elapsed,
                error_type="DegenerateObjective",
                error_message=(
                    "BBOPlace returned an impossible or sentinel HPWL "
                    f"value: {hpwl!r}. Check the benchmark/macro subset."
                ),
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
                **({"repair": response["repair"][0], "repair_bundle_sha256": self.config.repair_bundle_sha256}
                   if self.config.repair_bundle_sha256 else {}),
            },
        )

    def sanity_check(self):
        report = super().sanity_check()
        if self.definition.n_macro <= 0:
            report.add_error("invalid_n_macro", "n_macro must be positive.")
        cap_msg = n_macro_over_benchmark_cap_message(
            benchmark=self.definition.benchmark,
            n_macro=int(self.definition.n_macro),
        )
        if cap_msg is not None:
            report.add_error("n_macro_exceeds_benchmark_cap", cap_msg)
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
    seed: int = 2,
    definition: BBOPlaceDefinition | None = None,
    post_json: PostJsonFn | None = None,
    http_timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_S,
    metadata: dict[str, str] | None = None,
    bundle_root: Path | None = None,
    **_kwargs: Any,
) -> BBOPlaceTask:
    """Factory for the public geometry-repair BBOPlace task."""
    resolved_definition = definition or default_bboplace_definition(placer=DEFAULT_PLACER)
    bundle_root = bundle_root or Path(__file__).resolve().parent / "assets" / "repair_bundles"
    bundle = bundle_root / f"{resolved_definition.benchmark}__s{seed}.json"
    if not bundle.exists():
        raise FileNotFoundError(f"No geometry-repair bundle for {resolved_definition.benchmark} seed {seed}: {bundle}")
    from .repair_backend import RepairEvaluator
    packet = json.loads(bundle.read_text())
    evaluator = RepairEvaluator(packet)
    if (len(evaluator.data.names) != resolved_definition.n_macro
            or tuple(evaluator.data.grid) != (resolved_definition.n_grid_x, resolved_definition.n_grid_y)
            or packet["seed"] != seed or packet["benchmark"] != resolved_definition.benchmark):
        raise ValueError("Bundle does not match the requested placement task")
    repair_bundle_sha256 = evaluator.sha256
    config = BBOPlaceTaskConfig(
        max_evaluations=max_evaluations,
        seed=seed,
        definition=resolved_definition,
        post_json=post_json,
        http_timeout_seconds=http_timeout_seconds,
        metadata=dict(metadata or {}),
        repair_bundle_sha256=str(repair_bundle_sha256),
    )
    task = BBOPlaceTask(config=config, definition=resolved_definition)
    task.bundle = packet
    task.spec.metadata["benchmark_protocol"]["initialization"]["configurations"] = [
        o["config"] for o in packet["initializations"]]
    return task


BBOPLACE_DEFAULT_DEFINITION = default_bboplace_definition()

__all__ = [
    "BBOPLACE_DEFAULT_DEFINITION",
    "BBOPLACE_TASK_KEY",
    "BBOPLACE_COMPACT_INITIAL_DESIGN_SIZE",
    "BENCHMARK_MAX_N_MACRO",
    "BBOPlaceDefinition",
    "BBOPlaceTask",
    "BBOPlaceTaskConfig",
    "DEFAULT_BASE_URL",
    "default_bboplace_definition",
    "bboplace_compact_initial_configurations",
    "bboplace_compact_protocol_metadata",
    "create_bboplace_task",
    "max_n_macro_for_benchmark",
    "n_macro_over_benchmark_cap_message",
]
