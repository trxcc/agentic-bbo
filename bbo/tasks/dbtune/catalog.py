"""
Registry of surrogate benchmarks backed by legacy sklearn ``*.joblib`` files.

Registered HTTP tasks use the published Python 3.7 sidecar image, which bundles
the full checkpoints. Local ``*.joblib`` files under ``bbo/tasks/dbtune/assets/``
are only needed for image rebuilds or direct in-process surrogate use.
``knobs_*.json`` live in ``assets/`` and define each benchmark's search space.
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
    """MAXIMIZE throughput/TPS-style metrics; MINIMIZE latency for JOB workloads."""
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
        display_name="Sysbench full-knob surrogate (196 dims, throughput)",
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
        display_name="JOB full-knob surrogate (196 dims, latency)",
        default_joblib_filename="JOB_all.joblib",
        default_knobs_json_filename="knobs_mysql_all_197.json",
        objective_name="latency",
        direction=ObjectiveDirection.MINIMIZE,
        override_env_var="AGENTIC_BBO_JOB_ALL_SURROGATE",
    ),
    "knob_surrogate_pg_5": SurrogateBenchmarkSpec(
        task_id="knob_surrogate_pg_5",
        display_name="PostgreSQL 5-knob surrogate (throughput-style score)",
        default_joblib_filename="pg_5.joblib",
        default_knobs_json_filename="knobs_pg_top5.json",
        objective_name="throughput",
        direction=ObjectiveDirection.MAXIMIZE,
        override_env_var="AGENTIC_BBO_PG5_SURROGATE",
    ),
    "knob_surrogate_pg_20": SurrogateBenchmarkSpec(
        task_id="knob_surrogate_pg_20",
        display_name="PostgreSQL 20-knob surrogate (throughput-style score)",
        default_joblib_filename="pg_20.joblib",
        default_knobs_json_filename="knobs_pg_top20.json",
        objective_name="throughput",
        direction=ObjectiveDirection.MAXIMIZE,
        override_env_var="AGENTIC_BBO_PG20_SURROGATE",
    ),
}


def resolve_bundled_joblib_path(spec: SurrogateBenchmarkSpec) -> Path:
    """Resolve path to ``.joblib``: env override, then ``assets/<filename>``."""
    if spec.override_env_var:
        v = os.environ.get(spec.override_env_var)
        if v:
            return Path(v).expanduser()
    primary = _ASSETS / spec.default_joblib_filename
    if primary.is_file():
        return primary
    # Sysbench 5: tiny placeholder from build_placeholder_surrogate
    if spec.task_id == "knob_surrogate_sysbench_5":
        tiny = _ASSETS / "sysbench_5knob_surrogate.joblib"
        if tiny.is_file():
            return tiny
    return primary


def default_knobs_json_path(spec: SurrogateBenchmarkSpec) -> Path:
    return _ASSETS / spec.default_knobs_json_filename


__all__ = [
    "SURROGATE_BENCHMARKS",
    "SurrogateBenchmarkSpec",
    "default_knobs_json_path",
    "resolve_bundled_joblib_path",
]
