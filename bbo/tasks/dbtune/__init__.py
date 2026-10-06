"""dbtune tasks: active HTTP surrogate services plus offline surrogate helpers."""

from __future__ import annotations

# --- Offline in-process sklearn surrogates (.joblib) ---
from .catalog import SURROGATE_BENCHMARKS, SurrogateBenchmarkSpec, default_knobs_json_path, resolve_bundled_joblib_path
from .paths import (
    SYSBENCH_5_FEATURE_ORDER,
    bundled_knobs_top5_path,
    bundled_surrogate_sysbench5_path,
)
from .http_surrogate_specs import DBTUNE_SURROGATE_SERVICE_TASK_IDS, HTTP_SURROGATE_TASK_IDS
from .http_surrogate_task import (
    HttpSurrogateKnobTask,
    HttpSurrogateKnobTaskConfig,
    create_dbtune_surrogate_service_task,
    create_http_surrogate_knob_task,
)
from .offline_surrogate_task import (
    SurrogateKnobTask,
    SurrogateKnobTaskConfig,
    create_surrogate_knob_task,
    create_sysbench5_surrogate_task,
)

# Public alias (tests and docs)
create_surrogate_task = create_surrogate_knob_task

__all__ = [
    "DBTUNE_SURROGATE_SERVICE_TASK_IDS",
    "HTTP_SURROGATE_TASK_IDS",
    "HttpSurrogateKnobTask",
    "HttpSurrogateKnobTaskConfig",
    "SURROGATE_BENCHMARKS",
    "SYSBENCH_5_FEATURE_ORDER",
    "SurrogateBenchmarkSpec",
    "SurrogateKnobTask",
    "SurrogateKnobTaskConfig",
    "bundled_knobs_top5_path",
    "bundled_surrogate_sysbench5_path",
    "create_dbtune_surrogate_service_task",
    "create_http_surrogate_knob_task",
    "create_surrogate_knob_task",
    "create_surrogate_task",
    "create_sysbench5_surrogate_task",
    "default_knobs_json_path",
    "resolve_bundled_joblib_path",
]
