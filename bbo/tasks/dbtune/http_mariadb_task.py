"""BBO task: evaluate dbtune MariaDB knobs via the packaged evaluator service."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

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
from ..http_json import get_json, post_json
from .knob_encode import build_knob_space, feature_order_by_rank, physical_to_mariadb_strings
from .http_mariadb_specs import (
    DBTUNE_MARIADB_TASK_IDS,
    HTTP_DATABASE_TASK_IDS,
    HttpDatabaseTaskSpec,
    by_task_id,
    default_knobs_path_for_spec,
    is_database_task_id,
)

# 可通过环境变量覆盖默认 evaluator 地址与超时
_ENV_BASE_URL = "AGENTBBO_HTTP_EVAL_BASE_URL"
_ENV_TIMEOUT = "AGENTBBO_HTTP_EVAL_TIMEOUT_SEC"
_DEFAULT_BASE_URL = "http://127.0.0.1:8080"
_DEFAULT_TIMEOUT = 300.0
_DEFAULT_IMAGE = "fakerstrawberry/agentbbo-dbtune-mariadb-eval:v1"


def _build_unit_hypercube_space(feature_names: tuple[str, ...]) -> SearchSpace:
    params = [FloatParam(name, low=0.0, high=1.0, default=0.5, log=False) for name in feature_names]
    return SearchSpace(params)


def _resolve_base_url(config_url: str | None) -> str:
    if config_url:
        return config_url
    return os.environ.get(_ENV_BASE_URL, _DEFAULT_BASE_URL).strip()


def _resolve_timeout_sec(config_timeout: float | None) -> float:
    if config_timeout is not None:
        return float(config_timeout)
    raw = os.environ.get(_ENV_TIMEOUT)
    if raw is None or raw == "":
        return _DEFAULT_TIMEOUT
    return float(raw)


@dataclass(frozen=True)
class HttpDatabaseKnobTaskConfig:
    """Configuration for one dbtune MariaDB evaluator task."""

    task_id: str
    base_url: str | None = None
    evaluate_path: str = "/evaluate"
    health_path: str = "/health"
    knobs_json_path: Path | None = None
    request_timeout_sec: float | None = None
    max_evaluations: int | None = None
    seed: int = 0
    description_dir: Path | None = None
    display_name: str | None = None
    skip_health_check: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class HttpDatabaseKnobTask(Task):
    """
    Normalized ``[0,1]^d`` knobs -> decode -> evaluator ``POST /evaluate`` -> TPS (maximize).

    Requires a running compatible evaluator container, normally
    ``fakerstrawberry/agentbbo-dbtune-mariadb-eval:v1``. The evaluator must accept
    ``workload`` in the JSON body; see ``docker_mariadb/server.py``.
    """

    def __init__(self, config: HttpDatabaseKnobTaskConfig) -> None:
        self.config = config
        if not is_database_task_id(config.task_id):
            raise ValueError(
                f"Unknown dbtune MariaDB task_id `{config.task_id}`. Known: {', '.join(DBTUNE_MARIADB_TASK_IDS)}"
            )

        self._task_spec: HttpDatabaseTaskSpec = by_task_id(config.task_id)
        knobs_path = (
            Path(config.knobs_json_path) if config.knobs_json_path is not None else default_knobs_path_for_spec(self._task_spec)
        )
        self._knobs_path = knobs_path
        self._feature_names = feature_order_by_rank(knobs_path)
        self._knob_space = build_knob_space(knobs_path, self._feature_names)
        self._search_space = _build_unit_hypercube_space(self._feature_names)

        self._base_url = _resolve_base_url(config.base_url)
        self._timeout_sec = _resolve_timeout_sec(config.request_timeout_sec)

        description_dir = config.description_dir
        if description_dir is None:
            package_root = Path(__file__).resolve().parents[2]
            description_dir = package_root / "task_descriptions" / config.task_id

        display = config.display_name or self._task_spec.display_name
        max_eval = config.max_evaluations if config.max_evaluations is not None else 60

        self._spec = TaskSpec(
            name=config.task_id,
            search_space=self._search_space,
            objectives=(ObjectiveSpec("throughput", ObjectiveDirection.MAXIMIZE),),
            max_evaluations=max_eval,
            description_ref=TaskDescriptionRef.from_directory(config.task_id, description_dir),
            metadata={
                "display_name": display,
                "dimension": float(len(self._feature_names)),
                "knobs_json_path": str(self._knobs_path.resolve()),
                "http_base_url": self._base_url,
                "evaluate_path": config.evaluate_path,
                "workload": self._task_spec.workload_key,
                "feature_order": list(self._feature_names),
                "problem_family": "dbtune_mariadb",
                **config.metadata,
            },
        )

        if not config.skip_health_check:
            self._probe_health(config.health_path)

    @property
    def spec(self) -> TaskSpec:
        return self._spec

    def _probe_health(self, health_path: str) -> None:
        try:
            get_json(self._base_url, health_path, timeout_sec=min(10.0, self._timeout_sec))
        except RuntimeError as exc:
            raise RuntimeError(
                f"dbtune MariaDB evaluator service not reachable at {self._base_url!r} ({exc}). "
                f"Start Docker image {_DEFAULT_IMAGE!r} or set {_ENV_BASE_URL}."
            ) from exc

    def evaluate(self, suggestion: TrialSuggestion) -> EvaluationResult:
        start = time.perf_counter()
        cfg = self.spec.search_space.coerce_config(suggestion.config, use_defaults=False)
        vector = self.spec.search_space.to_numeric_vector(cfg)
        x_norm = np.asarray(vector, dtype=np.float64).reshape(-1)
        phys: np.ndarray = self._knob_space.decode(x_norm)
        knob_strings = physical_to_mariadb_strings(self._knobs_path, self._feature_names, phys)

        # 与 Docker ``server.py`` 约定：knobs + workload（小写 key）
        payload: dict[str, Any] = {
            "knobs": knob_strings,
            "workload": self._task_spec.workload_key,
        }
        raw = post_json(
            self._base_url,
            self.config.evaluate_path,
            payload,
            timeout_sec=self._timeout_sec,
        )
        status = str(raw.get("status", ""))
        if status != "success":
            msg = str(raw.get("message", raw))
            raise RuntimeError(f"Evaluator returned non-success: {msg!r}")

        y = float(raw.get("y", raw.get("tps", 0.0)))
        elapsed = time.perf_counter() - start

        metrics: dict[str, float | str] = {
            "dimension": float(len(self._search_space)),
            "http_latency_seconds": elapsed,
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
            metadata={"task_id": self.config.task_id, "knobs": knob_strings, "workload": self._task_spec.workload_key},
        )


def create_http_database_task(
    task_id: str,
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    base_url: str | None = None,
    knobs_json_path: str | Path | None = None,
    request_timeout_sec: float | None = None,
    skip_health_check: bool = False,
) -> HttpDatabaseKnobTask:
    """Build ``HttpDatabaseKnobTask`` for any id in ``HTTP_DATABASE_TASK_IDS``."""
    if not is_database_task_id(task_id):
        raise ValueError(
            f"Unknown database task_id `{task_id}`. Knows: {', '.join(HTTP_DATABASE_TASK_IDS)}"
        )
    return HttpDatabaseKnobTask(
        HttpDatabaseKnobTaskConfig(
            task_id=task_id,
            max_evaluations=max_evaluations,
            seed=seed,
            base_url=base_url,
            knobs_json_path=Path(knobs_json_path) if knobs_json_path is not None else None,
            request_timeout_sec=request_timeout_sec,
            skip_health_check=skip_health_check,
        )
    )


create_dbtune_mariadb_task = create_http_database_task


# Backward-compatible name for the previous single-task entry point
def create_http_database_sysbench5_task(
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    base_url: str | None = None,
    knobs_json_path: str | Path | None = None,
    request_timeout_sec: float | None = None,
    skip_health_check: bool = False,
) -> HttpDatabaseKnobTask:
    """``read_write + 5 knobs`` 与旧 ``knob_http_mariadb_sysbench_5`` 行为一致的任务 id 已重命名。"""
    return create_http_database_task(
        "knob_http_mariadb_sysbench_read_write_5",
        max_evaluations=max_evaluations,
        seed=seed,
        base_url=base_url,
        knobs_json_path=knobs_json_path,
        request_timeout_sec=request_timeout_sec,
        skip_health_check=skip_health_check,
    )


__all__ = [
    "DBTUNE_MARIADB_TASK_IDS",
    "HTTP_DATABASE_TASK_IDS",
    "HttpDatabaseKnobTask",
    "HttpDatabaseKnobTaskConfig",
    "create_dbtune_mariadb_task",
    "create_http_database_sysbench5_task",
    "create_http_database_task",
    "is_database_task_id",
]
