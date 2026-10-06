"""Database / knob task registries for the active dbtune task set.

Only the six ``knob_http_surrogate_*`` tasks are active dbtune tasks. Legacy
MariaDB/sysbench implementation files remain in the package, but are not exported
or registered here.
"""

from __future__ import annotations

from .catalog import SURROGATE_BENCHMARKS, SurrogateBenchmarkSpec, default_knobs_json_path, resolve_bundled_joblib_path
from .http_surrogate_specs import (
    DBTUNE_SURROGATE_SERVICE_TASK_IDS,
    HTTP_SURROGATE_TASK_IDS,
    is_dbtune_surrogate_service_task_id,
    is_http_surrogate_task_id,
)

__all__ = [
    "DBTUNE_SURROGATE_SERVICE_TASK_IDS",
    "HTTP_SURROGATE_TASK_IDS",
    "SURROGATE_BENCHMARKS",
    "SurrogateBenchmarkSpec",
    "default_knobs_json_path",
    "is_dbtune_surrogate_service_task_id",
    "is_http_surrogate_task_id",
    "resolve_bundled_joblib_path",
]
