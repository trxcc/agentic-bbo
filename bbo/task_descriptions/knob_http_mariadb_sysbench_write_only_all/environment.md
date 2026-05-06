# Environment

## Shared reusable Docker image (all eight dbtune MariaDB/sysbench tasks)

```bash
docker pull fakerstrawberry/agentbbo-dbtune-mariadb-eval:v1
docker rm -f agentbbo_http_mariadb_eval 2>/dev/null
docker run -d --name agentbbo_http_mariadb_eval -p 8080:8080 fakerstrawberry/agentbbo-dbtune-mariadb-eval:v1
```

If the evaluator implementation changes, rebuild and export the two dbtune images with `scripts/package_dbtune_images.sh`, then publish the new tag before using it for comparisons.

`docker-compose.task-services.yml` uses the same image by default. Override it with `AGENTBBO_DBTUNE_MARIADB_IMAGE` if you publish under another Docker Hub namespace or tag.

## Client-side environment (Python)

| Variable | Role |
|----------|------|
| `AGENTBBO_HTTP_EVAL_BASE_URL` | Base URL, default `http://127.0.0.1:8080` |
| `AGENTBBO_HTTP_EVAL_TIMEOUT_SEC` | **Per-POST** timeout (seconds), default `300` |

## This task

| Field | Value |
|------|--------|
| `task_id` | `knob_http_mariadb_sysbench_write_only_all` |
| `workload` (JSON) | `write_only` -> `oltp_write_only` |
| Knob JSON (default) | `bbo/tasks/dbtune/assets/knobs_mysql_all_197.json` |

Health check: `GET /health` on the same base URL.
