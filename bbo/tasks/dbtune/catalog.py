"""
Registry of surrogate benchmarks backed by ``*.joblib`` files under ``bbo/tasks/dbtune/assets/``.

Large checkpoints are not in git: download from the URL in ``assets/README.md`` (same filenames
as in ``SURROGATE_BENCHMARKS`` / ``default_joblib_filename``). ``knobs_*.json`` live in
``assets/`` and define each benchmark's search space.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ...core import ObjectiveDirection

_ASSETS = Path(__file__).resolve().parent / "assets"


@dataclass(frozen=True)
class SurrogateBenchmarkSpec:
    """Static metadata for one offline surrogate task."""

    task_id: str
    display_name: str
    default_joblib_filename: str
    default_knobs_json_filename: str
    objective_name: str
    direction: ObjectiveDirection
    """MAXIMIZE throughput/TPS-style metrics; MINIMIZE latency workloads."""
    override_env_var: str | None = None
    """If set, ``os.environ[var]`` overrides the default joblib path when present."""


SURROGATE_BENCHMARKS: dict[str, SurrogateBenchmarkSpec] = {
    "knob_surrogate_sysbench_5": SurrogateBenchmarkSpec(
        task_id="knob_surrogate_sysbench_5",
        display_name="Sysbench 5-knob RF surrogate (throughput)",
        default_joblib_filename="RF_SYSBENCH_5knob.joblib",
        default_knobs_json_filename="knobs_SYSBENCH_top5.json",
        objective_name="throughput",
        direction=ObjectiveDirection.MAXIMIZE,
        override_env_var="AGENTIC_BBO_SYSBENCH5_SURROGATE",
    ),
    "knob_surrogate_sysbench_all": SurrogateBenchmarkSpec(
        task_id="knob_surrogate_sysbench_all",
        display_name="Sysbench full-active-knob surrogate (throughput)",
        default_joblib_filename="SYSBENCH_all.joblib",
        default_knobs_json_filename="knobs_mysql_all_197.json",
        objective_name="throughput",
        direction=ObjectiveDirection.MAXIMIZE,
        override_env_var="AGENTIC_BBO_SYSBENCH_ALL_SURROGATE",
    ),
    "knob_surrogate_job_5": SurrogateBenchmarkSpec(
        task_id="knob_surrogate_job_5",
        display_name="JOB 5-knob RF surrogate (latency)",
        default_joblib_filename="RF_JOB_5knob.joblib",
        default_knobs_json_filename="knobs_JOB_top5.json",
        objective_name="latency",
        direction=ObjectiveDirection.MINIMIZE,
        override_env_var="AGENTIC_BBO_JOB5_SURROGATE",
    ),
    "knob_surrogate_job_all": SurrogateBenchmarkSpec(
        task_id="knob_surrogate_job_all",
        display_name="JOB full-active-knob surrogate (95th-percentile latency)",
        default_joblib_filename="JOB_all.joblib",
        default_knobs_json_filename="knobs_mysql_all_197.json",
        objective_name="latency",
        direction=ObjectiveDirection.MINIMIZE,
        override_env_var="AGENTIC_BBO_JOB_ALL_SURROGATE",
    ),
    "knob_surrogate_pg_5": SurrogateBenchmarkSpec(
        task_id="knob_surrogate_pg_5",
        display_name="PostgreSQL JOB 5-knob RF surrogate (95th-percentile latency)",
        default_joblib_filename="pg_5.joblib",
        default_knobs_json_filename="knobs_pg_top5.json",
        objective_name="latency",
        direction=ObjectiveDirection.MINIMIZE,
        override_env_var="AGENTIC_BBO_PG5_SURROGATE",
    ),
    "knob_surrogate_pg_20": SurrogateBenchmarkSpec(
        task_id="knob_surrogate_pg_20",
        display_name="PostgreSQL JOB 20-knob RF surrogate (95th-percentile latency)",
        default_joblib_filename="pg_20.joblib",
        default_knobs_json_filename="knobs_pg_top20.json",
        objective_name="latency",
        direction=ObjectiveDirection.MINIMIZE,
        override_env_var="AGENTIC_BBO_PG20_SURROGATE",
    ),
}


def resolve_bundled_joblib_path(spec: SurrogateBenchmarkSpec) -> Path:
    """Resolve path to ``.joblib``: env override, then ``assets/<filename>``."""
    if spec.override_env_var:
        value = os.environ.get(spec.override_env_var)
        if value:
            return Path(value).expanduser()

    path = _ASSETS / spec.default_joblib_filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Required benchmark checkpoint not found: {path}. "
            "Download the released checkpoint described in "
            "bbo/tasks/dbtune/assets/README.md. "
            "Placeholder surrogates are available only through *_smoke tasks."
        )
    return path


def default_knobs_json_path(spec: SurrogateBenchmarkSpec) -> Path:
    return _ASSETS / spec.default_knobs_json_filename


__all__ = [
    "SURROGATE_BENCHMARKS",
    "SurrogateBenchmarkSpec",
    "default_knobs_json_path",
    "resolve_bundled_joblib_path",
]
