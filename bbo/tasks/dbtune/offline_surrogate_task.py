"""Black-box task: sklearn RF surrogate over DB knobs (normalized [0,1]^d design)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

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
from .catalog import (
    SURROGATE_BENCHMARKS,
    SurrogateBenchmarkSpec,
    default_knobs_json_path,
    resolve_bundled_joblib_path,
)
from .joblib_surrogate import JoblibSurrogate
from .knob_space import KnobSpaceFromJson

_PLACEHOLDER_FILENAMES = frozenset({"sysbench_5knob_surrogate.joblib"})


def _sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_unit_hypercube_space(feature_names: tuple[str, ...]) -> SearchSpace:
    params = [
        FloatParam(name, low=0.0, high=1.0, default=0.5, log=False) for name in feature_names
    ]
    return SearchSpace(params)


@dataclass
class SurrogateKnobTaskConfig:
    """Configuration for one surrogate knob task (see ``SURROGATE_BENCHMARKS``)."""

    task_id: str = "knob_surrogate_sysbench_5"
    surrogate_path: Path | None = None
    knobs_json_path: Path | None = None
    max_evaluations: int | None = None
    seed: int = 0
    description_dir: Path | None = None
    display_name: str | None = None
    objective_name: str | None = None
    objective_direction: ObjectiveDirection | None = None
    default_max_evaluations: int = 60
    metadata: dict[str, Any] = field(default_factory=dict)


class SurrogateKnobTask(Task):
    """
    Evaluates ``y = surrogate.predict(decode(x))`` with ``x`` in ``[0,1]^d``.

    Direction (maximize vs minimize) comes from the benchmark catalog (throughput vs latency).
    """

    def __init__(self, config: SurrogateKnobTaskConfig) -> None:
        self.config = config
        bench = SURROGATE_BENCHMARKS.get(config.task_id)
        if bench is None:
            known = ", ".join(sorted(SURROGATE_BENCHMARKS))
            raise ValueError(f"Unknown surrogate task_id `{config.task_id}`. Known: {known}")

        self._bench: SurrogateBenchmarkSpec = bench
        display = config.display_name or bench.display_name
        obj_name = config.objective_name or bench.objective_name
        direction = config.objective_direction or bench.direction

        self._surrogate_path = (
            Path(config.surrogate_path)
            if config.surrogate_path is not None
            else resolve_bundled_joblib_path(bench)
        ).expanduser()
        if self._surrogate_path.name in _PLACEHOLDER_FILENAMES:
            raise ValueError(
                f"Placeholder surrogate checkpoint is not valid for formal task `{config.task_id}`: "
                f"{self._surrogate_path}"
            )
        self._knobs_path = (
            Path(config.knobs_json_path)
            if config.knobs_json_path is not None
            else default_knobs_json_path(bench)
        )

        self._surrogate = JoblibSurrogate.from_path(self._surrogate_path)
        surrogate_sha256 = _sha256_file(self._surrogate_path)
        names = list(self._surrogate.feature_names)
        self._knob_space = KnobSpaceFromJson(self._knobs_path, names)

        if len(names) != self._knob_space.dim:
            raise ValueError("Surrogate X-name length must match knob space dimension.")

        self._search_space = _build_unit_hypercube_space(tuple(names))
        description_dir = config.description_dir
        if description_dir is None:
            package_root = Path(__file__).resolve().parents[2]
            description_dir = package_root / "task_descriptions" / config.task_id

        self._spec = TaskSpec(
            name=config.task_id,
            search_space=self._search_space,
            objectives=(ObjectiveSpec(obj_name, direction),),
            max_evaluations=config.max_evaluations or config.default_max_evaluations,
            description_ref=TaskDescriptionRef.from_directory(config.task_id, description_dir),
            metadata={
                "display_name": display,
                "dimension": float(len(names)),
                "surrogate_path": str(self._surrogate_path.resolve()),
                "surrogate_source": "paper_release",
                "surrogate_sha256": surrogate_sha256,
                "surrogate_filename": self._surrogate_path.name,
                "knobs_json_path": str(self._knobs_path.resolve()),
                "knob_metadata_filename": self._knobs_path.name,
                "feature_order": names,
                "feature_count": len(names),
                "is_placeholder": False,
                "problem_family": "surrogate_knob",
                "workload": bench.task_id,
                **config.metadata,
            },
        )

    @property
    def spec(self) -> TaskSpec:
        return self._spec

    def evaluate(self, suggestion: TrialSuggestion) -> EvaluationResult:
        start = time.perf_counter()
        cfg = self.spec.search_space.coerce_config(suggestion.config, use_defaults=False)
        vector = self.spec.search_space.to_numeric_vector(cfg)
        phys: NDArray[np.float64] = self._knob_space.decode(vector)
        y = float(self._surrogate.predict(phys))
        elapsed = time.perf_counter() - start
        metrics = {
            "dimension": float(len(self._search_space)),
            "surrogate_latency_seconds": elapsed,
        }
        coord_names = self.spec.search_space.names()
        if len(coord_names) != len(vector):
            raise ValueError(
                f"Search-space names ({len(coord_names)}) vs vector length ({len(vector)}) mismatch."
            )
        for name, scalar in zip(coord_names, vector):
            metrics[f"coord::{name}"] = float(scalar)
        obj_key = self.spec.primary_objective.name

        return EvaluationResult(
            status=TrialStatus.SUCCESS,
            objectives={obj_key: y},
            metrics=metrics,
            elapsed_seconds=elapsed,
            metadata={"task_id": self.config.task_id},
        )


def create_surrogate_knob_task(
    task_id: str,
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    surrogate_path: Path | str | None = None,
    knobs_json_path: Path | str | None = None,
) -> SurrogateKnobTask:
    """Factory: ``task_id`` must be a key in ``SURROGATE_BENCHMARKS``."""
    return SurrogateKnobTask(
        SurrogateKnobTaskConfig(
            task_id=task_id,
            max_evaluations=max_evaluations,
            seed=seed,
            surrogate_path=Path(surrogate_path) if surrogate_path is not None else None,
            knobs_json_path=Path(knobs_json_path) if knobs_json_path is not None else None,
        )
    )


def create_sysbench5_surrogate_task(
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    surrogate_path: Path | str | None = None,
    knobs_json_path: Path | str | None = None,
) -> SurrogateKnobTask:
    """Backward-compatible alias for ``knob_surrogate_sysbench_5``."""
    return create_surrogate_knob_task(
        "knob_surrogate_sysbench_5",
        max_evaluations=max_evaluations,
        seed=seed,
        surrogate_path=surrogate_path,
        knobs_json_path=knobs_json_path,
    )


__all__ = [
    "SurrogateKnobTask",
    "SurrogateKnobTaskConfig",
    "create_surrogate_knob_task",
    "create_sysbench5_surrogate_task",
]
