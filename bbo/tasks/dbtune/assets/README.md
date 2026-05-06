# Surrogate assets

## Large `*.joblib` checkpoints

Large checkpoint files are **not** committed to this repository. The registered HTTP tasks `knob_http_surrogate_*` use the reusable Docker image `fakerstrawberry/agentbbo-dbtune-surrogate-http-py37:v1`, which already bundles the full checkpoint set under `/app/assets`.

You only need local `.joblib` files in this directory when:

- rebuilding and publishing a new surrogate image tag
- directly using the unregistered in-process helpers such as `create_surrogate_knob_task(...)`

Source folder for maintainers:

[https://drive.google.com/drive/folders/1qalYsF7fuCB6MewOTPvr8DDZzIj7tIRt?usp=sharing](https://drive.google.com/drive/folders/1qalYsF7fuCB6MewOTPvr8DDZzIj7tIRt?usp=sharing)

If a local `joblib.load` fails with **`EOF` / `reading array data`**, the file on disk is **incomplete** (partial download, or Git LFS not pulled for a copy that lives in git). Re-download the same file from the link above, or set an env override to a full path (see table below).

Each `*.joblib` is a **serialized sklearn surrogate** (RF, etc.): it maps physical knob feature vectors to a predicted metric (throughput or latency). Names map to workloads: **Sysbench/MySQL** (`RF_SYSBENCH_*`, `SYSBENCH_all`), **JOB** (`RF_JOB_*`, `JOB_all`), **PostgreSQL** (`pg_5`, `pg_20`). The matching `knobs_*.json` files in this folder define the BBO search space.

`python -m bbo.run` registers **HTTP** tasks `knob_http_surrogate_*` only; the table’s `task_id` column is the **canonical** name (also used by Docker `GET /task/<task_id>`). For in-process loading, call `create_surrogate_knob_task("<task_id>", ...)` from Python.

## Joblib files ↔ benchmark `task_id`

| File (from the link above) | `task_id` | Env override (optional) |
|----------------------------|-----------|-------------------------|
| `RF_SYSBENCH_5knob.joblib` | `knob_surrogate_sysbench_5` | `AGENTIC_BBO_SYSBENCH5_SURROGATE` |
| `SYSBENCH_all.joblib` | `knob_surrogate_sysbench_all` | `AGENTIC_BBO_SYSBENCH_ALL_SURROGATE` |
| `RF_JOB_5knob.joblib` | `knob_surrogate_job_5` | `AGENTIC_BBO_JOB5_SURROGATE` |
| `JOB_all.joblib` | `knob_surrogate_job_all` | `AGENTIC_BBO_JOB_ALL_SURROGATE` |
| `pg_5.joblib` | `knob_surrogate_pg_5` | `AGENTIC_BBO_PG5_SURROGATE` |
| `pg_20.joblib` | `knob_surrogate_pg_20` | `AGENTIC_BBO_PG20_SURROGATE` |

For Sysbench-5, you can also use a **tiny** placeholder from `python -m bbo.tasks.dbtune.build_placeholder_surrogate` (`sysbench_5knob_surrogate.joblib`) for quick smoke tests.

To rebuild/export the reusable HTTP surrogate image for Docker Hub upload, first place the full files in this directory, then run `scripts/package_dbtune_images.sh --tag v1` from the repository root.

## Bundled knobs JSON

`knobs_*.json` files in this directory define knob bounds and types; the mapping from task id to filename is in `bbo/tasks/dbtune/catalog.py` (`default_knobs_json_filename` per benchmark).

## Tests / demo

```bash
uv sync --extra dev --extra surrogate
uv run pytest tests/test_surrogate_task_smoke.py tests/test_surrogate_knob_space.py -v
uv run python examples/run_knob_surrogate_demo.py
```

Use `create_surrogate_task("<task_id>")` from `bbo.tasks`.
