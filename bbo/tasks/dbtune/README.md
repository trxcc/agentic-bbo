# `bbo.tasks.dbtune` — database knob task family

This package groups **all database-related knob benchmarks** in one place, in the same spirit as
`bbo.tasks.scientific`: a small `registry.py` documents what exists, and co-located assets live under
clear subfolders.

## Layout

| Area | Role |
|------|------|
| `registry.py` | Re-exports catalog metadata for offline surrogates, MariaDB task specs, and surrogate-service id maps. |
| `catalog.py` | Offline `*.joblib` benchmark specs (`SURROGATE_BENCHMARKS`). |
| `http_mariadb_specs.py` | Eight real **MariaDB + sysbench** dbtune tasks (`DBTUNE_MARIADB_TASK_IDS`). |
| `http_mariadb_task.py` | Task implementation: `HttpDatabaseKnobTask`. |
| `offline_surrogate_task.py` | In-process sklearn surrogate: `SurrogateKnobTask`. |
| `http_surrogate_task.py` | Remote evaluator service (Python 3.7 Docker) for the same surrogates. |
| `cli_*.py` | Hooks for `bbo.tasks.registry` / `python -m bbo.run` (no changes to `bbo.run` needed for new task ids). |
| `assets/` | Shared `knobs_*.json`; local `*.joblib` checkpoints are only needed for image rebuilds or in-process surrogate use. |
| `docker_mariadb/` | Dockerfile and docs for the **live** MariaDB + sysbench evaluator (Flask API). |
| `docker_surrogate/` | Dockerfile and docs for **offline** sklearn inference via JSON (isolated old numpy/sklearn). |
| `gen_task_markdown.py` | One-off generator for `bbo/task_descriptions/knob_http_mariadb_sysbench_*/` packs. |

## Reusable evaluator images

The default sidecar images are:

```text
fakerstrawberry/agentbbo-dbtune-mariadb-eval:v1
fakerstrawberry/agentbbo-dbtune-surrogate-http-py37:v1
```

Start the shared stack from the repository root with:

```bash
docker compose -f docker-compose.task-services.yml up -d
```

Override image refs with `AGENTBBO_DBTUNE_MARIADB_IMAGE` and `AGENTBBO_DBTUNE_SURROGATE_IMAGE`.
To build both images locally and export the two tar packages for Docker Hub upload, run:

```bash
scripts/package_dbtune_images.sh --tag v1
```

## Import surface

User code typically uses the stable exports from `bbo.tasks` / `bbo.tasks.registry` (e.g.
`create_task("knob_http_surrogate_sysbench_5")` or the MariaDB `knob_http_mariadb_sysbench_*` ids). In-process
`create_surrogate_knob_task("knob_surrogate_sysbench_5", ...)` remains available but is not registered on
`python -m bbo.run`. For a
**direct** import, prefer:

```python
from bbo.tasks.dbtune import create_dbtune_mariadb_task, create_surrogate_knob_task
```

## See also

- `bbo/tasks/scientific/` — same “family + registry + data/” pattern for non-database scientific benchmarks.
