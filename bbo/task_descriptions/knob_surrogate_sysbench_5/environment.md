# Environment Setup

This task uses an offline scikit-learn surrogate model. It does not start a live database and does not execute the workload during optimization.

## This Task

| Field | Value |
|---|---|
| Task ID | `knob_surrogate_sysbench_5` |
| Checkpoint | `RF_SYSBENCH_5knob.joblib` |
| Knob metadata | `knobs_SYSBENCH_top5.json` |
| Environment variable | `AGENTIC_BBO_SYSBENCH5_SURROGATE` |

## Installation

For the legacy in-process helper, install its optional dependencies. Formal benchmark runs use the separate Python 3.7 HTTP service described in `bbo/tasks/dbtune/README.md`:

```bash
uv sync --extra dev --extra hpo
```

## Required Assets

The task requires both:

1. the task-specific `.joblib` surrogate checkpoint;
2. the task-specific knob metadata JSON.

The formal benchmark must use the released checkpoint associated with the source database-tuning benchmark. A generated placeholder checkpoint is suitable only for smoke testing and must use a separate task ID ending in `_smoke`.

Place the checkpoint under:

```text
bbo/tasks/dbtune/assets/
```

Alternatively, set the task-specific environment variable documented in `bbo/tasks/dbtune/catalog.py`.

## Artifact Validation

Before creating the task, the implementation must verify that:

- the checkpoint exists;
- the checkpoint is not marked as a placeholder;
- the checkpoint exposes a non-empty ordered feature list;
- every checkpoint feature exists in the knob JSON;
- the number and order of decoded features match the checkpoint input;
- the configured objective name and direction match the benchmark;
- the checkpoint SHA-256 hash is recorded in the run metadata.

The run metadata should contain at least:

```json
{
  "surrogate_source": "paper_release",
  "surrogate_sha256": "<sha256>",
  "surrogate_filename": "<filename>",
  "knob_metadata_filename": "<filename>",
  "feature_count": 0,
  "feature_order": [],
  "is_placeholder": false
}
```

## Smoke Test

Use the task factory to verify that the task loads and passes its sanity check:

```bash
uv run python - <<'PY'
from bbo.tasks.dbtune import create_surrogate_knob_task

task = create_surrogate_knob_task(
    "knob_surrogate_sysbench_5",
    max_evaluations=2,
    seed=0,
)
print(task.spec)
print(task.sanity_check())
PY
```

A successful smoke test confirms only that the artifact can be loaded and evaluated. It does not establish that the artifact matches the source paper. Provenance and feature consistency must be checked separately.

## Active Surrogate Asset Map

| Task ID | Checkpoint | Knob metadata | Environment variable |
|---|---|---|---|
| `knob_surrogate_sysbench_5` | `RF_SYSBENCH_5knob.joblib` | `knobs_SYSBENCH_top5.json` | `AGENTIC_BBO_SYSBENCH5_SURROGATE` |
| `knob_surrogate_sysbench_all` | `SYSBENCH_all.joblib` | `knobs_mysql_all_197.json` | `AGENTIC_BBO_SYSBENCH_ALL_SURROGATE` |
| `knob_surrogate_job_5` | `RF_JOB_5knob.joblib` | `knobs_JOB_top5.json` | `AGENTIC_BBO_JOB5_SURROGATE` |
| `knob_surrogate_job_all` | `JOB_all.joblib` | `knobs_mysql_all_197.json` | `AGENTIC_BBO_JOB_ALL_SURROGATE` |
| `knob_surrogate_pg_5` | `pg_5.joblib` | `knobs_pg_top5.json` | `AGENTIC_BBO_PG5_SURROGATE` |
| `knob_surrogate_pg_20` | `pg_20.joblib` | `knobs_pg_top20.json` | `AGENTIC_BBO_PG20_SURROGATE` |
