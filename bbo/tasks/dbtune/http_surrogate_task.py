"""BBO task: evaluate sklearn joblib surrogate via the packaged dbtune service."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
from .catalog import SURROGATE_BENCHMARKS, default_knobs_json_path, resolve_bundled_joblib_path
from .http_surrogate_specs import (
    DBTUNE_SURROGATE_SERVICE_TASK_IDS,
    _DEFAULT_BASE_URL,
    _DEFAULT_TIMEOUT,
    _ENV_SURROGATE_BASE,
    _ENV_SURROGATE_TIMEOUT,
    canonical_id_from_http_task_id,
    is_dbtune_surrogate_service_task_id,
)
# 与 Docker server 的 JSON 协议一致
_EVALUATE_PATH = "/evaluate"
_HEALTH_PATH = "/health"
_TASK_META_PATH = "/task"  # GET {base}/task/<canonical_task_id>
_DEFAULT_IMAGE = "fakerstrawberry/agentbbo-dbtune-surrogate-http-py37:v1"


def _build_unit_hypercube_space(feature_names: tuple[str, ...]) -> SearchSpace:
    params = [FloatParam(n, low=0.0, high=1.0, default=0.5, log=False) for n in feature_names]
    return SearchSpace(params)


def _resolve_base_url(config_url: str | None) -> str:
    if config_url:
        return config_url
    return os.environ.get(_ENV_SURROGATE_BASE, _DEFAULT_BASE_URL).strip()


def _resolve_timeout_sec(config_timeout: float | None) -> float:
    if config_timeout is not None:
        return float(config_timeout)
    raw = os.environ.get(_ENV_SURROGATE_TIMEOUT)
    if raw is None or raw == "":
        return _DEFAULT_TIMEOUT
    return float(raw)


@dataclass(frozen=True)
class HttpSurrogateKnobTaskConfig:
    """Configuration for :class:`HttpSurrogateKnobTask` (host talks to Py3.7 service)."""

    http_task_id: str
    base_url: str | None = None
    request_timeout_sec: float | None = None
    max_evaluations: int | None = None
    seed: int = 0
    description_dir: Path | None = None
    skip_health_check: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class HttpSurrogateKnobTask(Task):
    """
    与真实库任务一样：宿主机只发请求，评估在 Docker 内完成。

    搜索域是单位超立方体 ``[0,1]^d``；每次评估 ``POST /evaluate`` 发送 **归一化向量** ``x``，
    容器内用 knobs JSON 解码并 ``predict``，返回 **标量** ``y``（throughput / latency 等）。
    """

    def __init__(self, config: HttpSurrogateKnobTaskConfig) -> None:
        self._config = config
        if not is_dbtune_surrogate_service_task_id(config.http_task_id):
            raise ValueError(
                f"Unknown dbtune surrogate-service task_id `{config.http_task_id}`. "
                f"Known ids: {', '.join(sorted(DBTUNE_SURROGATE_SERVICE_TASK_IDS))}"
            )
        self._canonical_id = canonical_id_from_http_task_id(config.http_task_id)
        self._bench = SURROGATE_BENCHMARKS[self._canonical_id]

        self._surrogate_path = resolve_bundled_joblib_path(self._bench)
        self._knobs_path = default_knobs_json_path(self._bench)

        self._base_url = _resolve_base_url(config.base_url)
        self._timeout_sec = _resolve_timeout_sec(config.request_timeout_sec)

        # 从 Py3.7 服务拉取维度与名字；不在本机加载 .joblib，不在本机做 decode
        meta = get_json(
            self._base_url,
            f"{_TASK_META_PATH.rstrip('/')}/{self._canonical_id}",
            timeout_sec=min(30.0, self._timeout_sec),
        )
        if str(meta.get("status", "")) not in ("", "ok", "success"):
            raise RuntimeError(f"Task metadata not ok: {meta!r}")
        names_raw = meta.get("feature_names")
        if not isinstance(names_raw, list) or not names_raw:
            raise RuntimeError(f"Invalid feature_names in metadata: {meta!r}")
        names = tuple(str(x) for x in names_raw)

        self._search_space = _build_unit_hypercube_space(names)

        package_root = Path(__file__).resolve().parents[2]
        description_dir = config.description_dir
        if description_dir is None:
            # 复用同语义 canonical 任务文档，避免维护两套长文
            description_dir = package_root / "task_descriptions" / self._canonical_id

        max_eval = config.max_evaluations if config.max_evaluations is not None else 60
        obj_name = self._bench.objective_name
        direction = self._bench.direction

        self._spec = TaskSpec(
            name=config.http_task_id,
            search_space=self._search_space,
            objectives=(ObjectiveSpec(obj_name, direction),),
            max_evaluations=max_eval,
            description_ref=TaskDescriptionRef.from_directory(config.http_task_id, description_dir),
            metadata={
                "display_name": self._bench.display_name,
                "dimension": float(len(names)),
                "canonical_task_id": self._canonical_id,
                "http_base_url": self._base_url,
                "surrogate_path_ref": str(self._surrogate_path.resolve()),
                "knobs_json_path": str(self._knobs_path.resolve()),
                "feature_order": list(names),
                "problem_family": "dbtune_surrogate_service",
                "http_evaluate_contract": "POST JSON {task_id, x} where x is [0,1]^d; response y",
                **config.metadata,
            },
        )

        if not config.skip_health_check:
            self._probe_health()

    @property
    def spec(self) -> TaskSpec:
        return self._spec

    def _probe_health(self) -> None:
        try:
            get_json(self._base_url, _HEALTH_PATH, timeout_sec=min(10.0, self._timeout_sec))
        except RuntimeError as exc:
            raise RuntimeError(
                f"dbtune surrogate service not reachable at {self._base_url!r} ({exc!s}). "
                f"Start Docker image {_DEFAULT_IMAGE!r} (Python 3.7) or set {_ENV_SURROGATE_BASE}."
            ) from exc

    def evaluate(self, suggestion: TrialSuggestion) -> EvaluationResult:
        start = time.perf_counter()
        cfg = self.spec.search_space.coerce_config(suggestion.config, use_defaults=False)
        vector = self.spec.search_space.to_numeric_vector(cfg)
        x_list = [float(v) for v in vector]

        payload: dict[str, Any] = {
            "task_id": self._canonical_id,
            "x": x_list,
        }
        raw = post_json(
            self._base_url,
            _EVALUATE_PATH,
            payload,
            timeout_sec=self._timeout_sec,
        )
        st = str(raw.get("status", ""))
        if st != "success":
            msg = str(raw.get("message", raw))
            raise RuntimeError(f"Surrogate HTTP returned non-success: {msg!r}")

        obj_key = self.spec.primary_objective.name
        y = float(raw.get("y", raw.get(obj_key, 0.0)))
        elapsed = time.perf_counter() - start

        metrics: dict[str, float] = {
            "dimension": float(len(self._search_space)),
            "http_surrogate_latency_seconds": elapsed,
        }
        coord_names = self.spec.search_space.names()
        for name, scalar in zip(coord_names, vector, strict=True):
            metrics[f"coord::{name}"] = float(scalar)

        return EvaluationResult(
            status=TrialStatus.SUCCESS,
            objectives={obj_key: y},
            metrics=metrics,
            elapsed_seconds=elapsed,
            metadata={
                "http_task_id": self._config.http_task_id,
                "canonical_task_id": self._canonical_id,
            },
        )


def create_http_surrogate_knob_task(
    http_task_id: str,
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    base_url: str | None = None,
    request_timeout_sec: float | None = None,
    skip_health_check: bool = False,
) -> HttpSurrogateKnobTask:
    """Factory: ``http_task_id`` 形如 ``knob_http_surrogate_sysbench_5``。"""
    return HttpSurrogateKnobTask(
        HttpSurrogateKnobTaskConfig(
            http_task_id=http_task_id,
            max_evaluations=max_evaluations,
            seed=seed,
            base_url=base_url,
            request_timeout_sec=request_timeout_sec,
            skip_health_check=skip_health_check,
        )
    )


create_dbtune_surrogate_service_task = create_http_surrogate_knob_task


__all__ = [
    "HttpSurrogateKnobTask",
    "HttpSurrogateKnobTaskConfig",
    "create_dbtune_surrogate_service_task",
    "create_http_surrogate_knob_task",
]
