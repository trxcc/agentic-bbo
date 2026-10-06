# Surrogate assets

## Download large `*.joblib` files

Large checkpoint files are **not** committed to this repository. Download them (same filenames as below) from the DBTune checkpoint folder, then place them under `bbo/tasks/dbtune/assets/`:

[https://drive.google.com/drive/folders/1qalYsF7fuCB6MewOTPvr8DDZzIj7tIRt?usp=sharing](https://drive.google.com/drive/folders/1qalYsF7fuCB6MewOTPvr8DDZzIj7tIRt?usp=sharing)

If `joblib.load` fails with **`EOF` / `reading array data`**, the file on disk is **incomplete** (partial download, or Git LFS not pulled for a copy that lives in git). **Re-download** the same file from the link above, or set an env override to a full path (see table below).

Each `*.joblib` is a **serialized sklearn surrogate** (RF, etc.): it maps physical knob feature vectors to a predicted metric (throughput or latency). Names map to workloads: **Sysbench/MySQL** (`RF_SYSBENCH_*`, `SYSBENCH_all`), **JOB** (`RF_JOB_*`, `JOB_all`), **PostgreSQL** (`pg_5`, `pg_20`). The matching `knobs_*.json` files in this folder define the BBO search space.

`python -m bbo.run` registers **HTTP** tasks `knob_http_surrogate_*` only; the table’s `task_id` column is the **canonical** name used by the Docker service at `GET /task/<task_id>`.

## Joblib files ↔ benchmark `task_id`

| File (from the link above) | `task_id` | Env override (optional) |
|----------------------------|-----------|-------------------------|
| `RF_SYSBENCH_5knob.joblib` | `knob_surrogate_sysbench_5` | `AGENTIC_BBO_SYSBENCH5_SURROGATE` |
| `SYSBENCH_all.joblib` | `knob_surrogate_sysbench_all` | `AGENTIC_BBO_SYSBENCH_ALL_SURROGATE` |
| `RF_JOB_5knob.joblib` | `knob_surrogate_job_5` | `AGENTIC_BBO_JOB5_SURROGATE` |
| `JOB_all.joblib` | `knob_surrogate_job_all` | `AGENTIC_BBO_JOB_ALL_SURROGATE` |
| `pg_5.joblib` | `knob_surrogate_pg_5` | `AGENTIC_BBO_PG5_SURROGATE` |
| `pg_20.joblib` | `knob_surrogate_pg_20` | `AGENTIC_BBO_PG20_SURROGATE` |

Formal benchmark task IDs require the released checkpoint files above. Placeholder surrogates are reserved for separately named smoke-test tasks and are not used as fallbacks by the active benchmark ids.

## Bundled knobs JSON

`knobs_*.json` files in this directory define knob bounds and types; the mapping from task id to filename is in `bbo/tasks/dbtune/catalog.py` (`default_knobs_json_filename` per benchmark).

## Service check

```bash
export BBO_DBTUNE_ASSETS=/absolute/path/to/complete/assets
docker compose --profile dbtune up --build -d dbtune
curl --fail http://127.0.0.1:8090/task/knob_surrogate_sysbench_5
```

The task endpoint loads its matching checkpoint and returns an error if an asset is
missing or incompatible. Run `uv run pytest` for the repository's offline checks.
