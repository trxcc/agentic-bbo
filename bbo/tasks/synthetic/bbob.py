"""Official COCO/BBOB tasks for the compact 10-dimensional benchmark suite."""

from __future__ import annotations

import ctypes
import math
import threading
import time
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


BBOB_SUITE_NAME = "bbob"
BBOB_DIMENSION = 10
BBOB_FUNCTION_IDS = tuple(range(1, 25))
BBOB_INSTANCE_IDS = (1, 2, 3)
BBOB_INITIAL_DESIGN_SIZE = 2 * BBOB_DIMENSION
BBOB_OPTIMIZATION_BUDGET = 10 * BBOB_DIMENSION
BBOB_TOTAL_BUDGET = BBOB_INITIAL_DESIGN_SIZE + BBOB_OPTIMIZATION_BUDGET
BBOB_MODEL_CANDIDATES = 2048
BBOB_LOWER_BOUND = -5.0
BBOB_UPPER_BOUND = 5.0
BBOB_TASK_IDS = tuple(
    f"bbob_f{function_id:02d}_d10" for function_id in BBOB_FUNCTION_IDS
)
_SOBOL_INITIALIZATION_LOCK = threading.Lock()

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
BBOB_DESCRIPTION_ROOT = PACKAGE_ROOT / "task_descriptions" / "bbob_10d"


@dataclass(frozen=True)
class BbobProblemDefinition:
    """Internal registry entry for one official BBOB function."""

    key: str
    function_id: int
    dimension: int = BBOB_DIMENSION

    def __post_init__(self) -> None:
        if self.function_id not in BBOB_FUNCTION_IDS:
            raise ValueError(
                f"BBOB function_id must be in 1..24, got {self.function_id}."
            )
        expected = f"bbob_f{self.function_id:02d}_d{self.dimension}"
        if self.key != expected:
            raise ValueError(f"BBOB key {self.key!r} does not match {expected!r}.")


BBOB_PROBLEM_DEFINITIONS = tuple(
    BbobProblemDefinition(key=task_id, function_id=function_id)
    for task_id, function_id in zip(BBOB_TASK_IDS, BBOB_FUNCTION_IDS, strict=True)
)
BBOB_PROBLEM_REGISTRY = {
    definition.key: definition for definition in BBOB_PROBLEM_DEFINITIONS
}


@dataclass
class BbobTaskConfig:
    """Configuration for one fixed-function COCO/BBOB task."""

    task_name: str
    max_evaluations: int | None = None
    seed: int = 0
    instance_id: int | None = None
    description_dir: Path = BBOB_DESCRIPTION_ROOT
    metadata: dict[str, Any] = field(default_factory=dict)


def bbob_instance_for_seed(seed: int) -> int:
    """Map every run seed deterministically onto the agreed instances 1, 2, and 3."""

    return BBOB_INSTANCE_IDS[int(seed) % len(BBOB_INSTANCE_IDS)]


def bbob_search_space() -> SearchSpace:
    """Return the official BBOB domain [-5, 5]^10."""

    return SearchSpace(
        FloatParam(
            name=f"x{index}", low=BBOB_LOWER_BOUND, high=BBOB_UPPER_BOUND, default=0.0
        )
        for index in range(1, BBOB_DIMENSION + 1)
    )


def bbob_initial_configurations(*, seed: int) -> tuple[dict[str, float], ...]:
    """Return the shared 2D scrambled-Sobol initialization for a run seed."""

    exponent = math.ceil(math.log2(BBOB_INITIAL_DESIGN_SIZE))
    # scipy.stats.qmc lazily initializes shared Sobol direction tables.
    # Serialize that short construction step so concurrent benchmark rows remain deterministic.
    with _SOBOL_INITIALIZATION_LOCK:
        unit_points = qmc.Sobol(
            d=BBOB_DIMENSION, scramble=True, seed=int(seed)
        ).random_base2(m=exponent)
    points = qmc.scale(
        unit_points[:BBOB_INITIAL_DESIGN_SIZE],
        np.full(BBOB_DIMENSION, BBOB_LOWER_BOUND),
        np.full(BBOB_DIMENSION, BBOB_UPPER_BOUND),
    )
    return tuple(
        {f"x{index + 1}": float(point[index]) for index in range(BBOB_DIMENSION)}
        for point in points
    )


def bbob_protocol_metadata(*, seed: int) -> dict[str, Any]:
    """Build the task-owned protocol shared by every comparable optimizer."""

    configurations = bbob_initial_configurations(seed=seed)
    return {
        "name": "coco_bbob_10d_compact_v1",
        "suite": BBOB_SUITE_NAME,
        "dimension": BBOB_DIMENSION,
        "instances": list(BBOB_INSTANCE_IDS),
        "optimizer_evaluations": BBOB_OPTIMIZATION_BUDGET,
        "total_evaluations": BBOB_TOTAL_BUDGET,
        "initialization": {
            "strategy": "fixed_configurations",
            "sampling": "scrambled_sobol",
            "seed": int(seed),
            "count": len(configurations),
            "configurations": [dict(config) for config in configurations],
            "source": "scipy.stats.qmc.Sobol(scramble=True):first_2d_points",
            "scope": "shared_by_seed_across_all_bbob_functions_and_algorithms",
        },
        "candidate_budget": {
            "policy": "fixed_2048_for_bbob_10d",
            "value": BBOB_MODEL_CANDIDATES,
            "applies_to": [
                "gp_ei.raw_samples",
                "botorch_turbo.n_candidates",
                "git_bo.n_candidates",
            ],
        },
    }


@lru_cache(maxsize=1)
def _coco_c_api():
    """Load the official C API shipped inside the pinned cocoex extension."""

    import cocoex.interface as coco_interface

    library = ctypes.CDLL(coco_interface.__file__)
    library.coco_suite.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    library.coco_suite.restype = ctypes.c_void_p
    library.coco_suite_get_next_problem.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    library.coco_suite_get_next_problem.restype = ctypes.c_void_p
    library.coco_problem_get_best_value.argtypes = [ctypes.c_void_p]
    library.coco_problem_get_best_value.restype = ctypes.c_double
    library.coco_problem_get_id.argtypes = [ctypes.c_void_p]
    library.coco_problem_get_id.restype = ctypes.c_char_p
    library.coco_suite_free.argtypes = [ctypes.c_void_p]
    library.coco_suite_free.restype = None
    return library


@lru_cache(maxsize=len(BBOB_FUNCTION_IDS) * len(BBOB_INSTANCE_IDS))
def _official_bbob_best_value(function_id: int, instance_id: int) -> float:
    """Read fopt from COCO's own ABI without evaluating or reimplementing a function."""

    library = _coco_c_api()
    options = (
        f"function_indices:{int(function_id)} dimensions:{BBOB_DIMENSION} "
        f"instance_indices:{int(instance_id)}"
    ).encode()
    suite = library.coco_suite(BBOB_SUITE_NAME.encode(), b"", options)
    if not suite:
        raise RuntimeError("COCO C API could not allocate the filtered BBOB suite.")
    try:
        problem = library.coco_suite_get_next_problem(suite, None)
        if not problem:
            raise RuntimeError("COCO C API did not return the requested BBOB problem.")
        observed_id = library.coco_problem_get_id(problem).decode()
        expected_id = (
            f"bbob_f{int(function_id):03d}_i{int(instance_id):02d}_d{BBOB_DIMENSION}"
        )
        if observed_id != expected_id:
            raise RuntimeError(
                f"COCO C API returned {observed_id!r}, expected {expected_id!r}."
            )
        return float(library.coco_problem_get_best_value(problem))
    finally:
        # A problem returned by coco_suite_get_next_problem is suite-owned.
        library.coco_suite_free(suite)


class BbobTask(Task):
    """A lazy, task-owned wrapper around one official cocoex problem."""

    def __init__(
        self, config: BbobTaskConfig, definition: BbobProblemDefinition | None = None
    ):
        self.config = config
        try:
            self.definition = definition or BBOB_PROBLEM_REGISTRY[config.task_name]
        except KeyError as exc:
            available = ", ".join(BBOB_TASK_IDS)
            raise ValueError(
                f"Unknown BBOB task {config.task_name!r}. Available: {available}"
            ) from exc

        self._instance_id = (
            bbob_instance_for_seed(config.seed)
            if config.instance_id is None
            else int(config.instance_id)
        )
        if self._instance_id not in BBOB_INSTANCE_IDS:
            raise ValueError(
                f"Compact BBOB instance_id must be one of {BBOB_INSTANCE_IDS}, got {self._instance_id}."
            )

        self._suite: Any | None = None
        self._problem: Any | None = None
        self._f_opt: float | None = None
        self._lock = threading.Lock()
        search_space = bbob_search_space()
        max_evaluations = (
            BBOB_TOTAL_BUDGET
            if config.max_evaluations is None
            else int(config.max_evaluations)
        )
        self._spec = TaskSpec(
            name=self.definition.key,
            search_space=search_space,
            objectives=(ObjectiveSpec("loss", ObjectiveDirection.MINIMIZE),),
            max_evaluations=max_evaluations,
            description_ref=TaskDescriptionRef.from_directory(
                self.definition.key,
                Path(config.description_dir),
            ),
            metadata={
                "task_family": "synthetic",
                "benchmark_suite": "COCO/BBOB",
                "dimension": BBOB_DIMENSION,
                "domain_bounds": [BBOB_LOWER_BOUND, BBOB_UPPER_BOUND],
                "bbob_function_id": self.definition.function_id,
                "bbob_instance_id": self._instance_id,
                "bbob_problem_id": self.problem_id,
                "initial_evaluations": BBOB_INITIAL_DESIGN_SIZE,
                "optimizer_evaluations": BBOB_OPTIMIZATION_BUDGET,
                "benchmark_protocol": bbob_protocol_metadata(seed=config.seed),
                "cma_initial_config": search_space.defaults(),
                "task_seed": int(config.seed),
                **config.metadata,
            },
        )

    @property
    def spec(self) -> TaskSpec:
        return self._spec

    @property
    def function_id(self) -> int:
        return self.definition.function_id

    @property
    def instance_id(self) -> int:
        return self._instance_id

    @property
    def problem_id(self) -> str:
        return f"bbob_f{self.function_id:03d}_i{self.instance_id:02d}_d{BBOB_DIMENSION}"

    def evaluate(self, suggestion: TrialSuggestion) -> EvaluationResult:
        if suggestion.budget is not None:
            raise ValueError(
                "BBOB uses a fixed-cost evaluation and does not accept per-trial budgets."
            )
        start = time.perf_counter()
        config = self.spec.search_space.coerce_config(
            suggestion.config, use_defaults=False
        )
        vector = self.spec.search_space.to_numeric_vector(config)

        with self._lock:
            problem = self._ensure_problem()
            value = float(problem(vector))
            f_opt = self._f_opt
        if f_opt is None:  # pragma: no cover - guarded by _ensure_problem.
            raise RuntimeError("COCO/BBOB optimum reference was not initialized.")

        regret = max(0.0, value - f_opt)
        return EvaluationResult(
            status=TrialStatus.SUCCESS,
            objectives={"loss": value},
            metrics={
                "regret": regret,
                "log10_regret": math.log10(max(regret, 1e-16)),
                "dimension": float(BBOB_DIMENSION),
            },
            elapsed_seconds=time.perf_counter() - start,
            metadata={
                "benchmark_suite": "COCO/BBOB",
                "bbob_problem_id": self.problem_id,
                "bbob_function_id": self.function_id,
                "bbob_instance_id": self.instance_id,
            },
        )

    def sanity_check(self):
        report = super().sanity_check()
        if len(self.spec.search_space) != BBOB_DIMENSION:
            report.add_error(
                "dimension_mismatch", "BBOB task must have exactly 10 dimensions."
            )
        protocol = self.spec.metadata.get("benchmark_protocol", {})
        configurations = protocol.get("initialization", {}).get("configurations", [])
        if len(configurations) != BBOB_INITIAL_DESIGN_SIZE:
            report.add_error(
                "initialization_count_mismatch",
                f"Expected {BBOB_INITIAL_DESIGN_SIZE} Sobol points, got {len(configurations)}.",
            )
        for index, candidate in enumerate(configurations):
            try:
                self.spec.search_space.validate_config(candidate)
            except Exception as exc:
                report.add_error(
                    "invalid_initialization_config",
                    f"Sobol point {index} is invalid: {exc}",
                )
        report.metadata.update(
            {
                "benchmark_suite": "COCO/BBOB",
                "problem_id": self.problem_id,
                "initial_design_size": BBOB_INITIAL_DESIGN_SIZE,
                "optimizer_budget": BBOB_OPTIMIZATION_BUDGET,
                "candidate_budget": BBOB_MODEL_CANDIDATES,
            }
        )
        return report

    def cleanup(self) -> None:
        with self._lock:
            self._cleanup_unlocked()

    def _cleanup_unlocked(self) -> None:
        problem, suite = self._problem, self._suite
        self._problem = None
        self._suite = None
        self._f_opt = None
        if problem is not None:
            free_problem = getattr(problem, "free", None)
            if callable(free_problem):
                free_problem()
        if suite is not None:
            free_suite = getattr(suite, "free", None)
            if callable(free_suite):
                free_suite()

    def _ensure_problem(self):
        if self._problem is not None:
            return self._problem
        try:
            import cocoex
        except (
            ImportError
        ) as exc:  # pragma: no cover - dependency validation covers this path.
            raise RuntimeError(
                "Official BBOB tasks require coco-experiment==2.8.2; run uv sync."
            ) from exc

        suite_options = (
            f"function_indices:{self.function_id} dimensions:{BBOB_DIMENSION} "
            f"instance_indices:{self.instance_id}"
        )
        self._suite = cocoex.Suite(BBOB_SUITE_NAME, "", suite_options)
        if len(self._suite) != 1:
            raise RuntimeError(
                f"Expected one COCO problem for {self.problem_id}, found {len(self._suite)}."
            )
        self._problem = self._suite.get_problem(0)
        observed_id = str(getattr(self._problem, "id", ""))
        if observed_id != self.problem_id:
            self._cleanup_unlocked()
            raise RuntimeError(
                f"COCO returned problem {observed_id!r}, expected {self.problem_id!r}."
            )
        self._f_opt = _official_bbob_best_value(self.function_id, self.instance_id)
        return self._problem


def create_bbob_task(
    task_name: str,
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    instance_id: int | None = None,
    **kwargs: Any,
) -> BbobTask:
    """Construct one registered compact-suite BBOB task."""

    return BbobTask(
        BbobTaskConfig(
            task_name=task_name,
            max_evaluations=max_evaluations,
            seed=seed,
            instance_id=instance_id,
            **kwargs,
        )
    )


__all__ = [
    "BBOB_DESCRIPTION_ROOT",
    "BBOB_DIMENSION",
    "BBOB_FUNCTION_IDS",
    "BBOB_INITIAL_DESIGN_SIZE",
    "BBOB_INSTANCE_IDS",
    "BBOB_MODEL_CANDIDATES",
    "BBOB_OPTIMIZATION_BUDGET",
    "BBOB_PROBLEM_DEFINITIONS",
    "BBOB_PROBLEM_REGISTRY",
    "BBOB_SUITE_NAME",
    "BBOB_TASK_IDS",
    "BBOB_TOTAL_BUDGET",
    "BbobProblemDefinition",
    "BbobTask",
    "BbobTaskConfig",
    "bbob_initial_configurations",
    "bbob_instance_for_seed",
    "bbob_protocol_metadata",
    "bbob_search_space",
    "create_bbob_task",
]
