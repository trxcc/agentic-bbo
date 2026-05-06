# Environment

This legacy directory is kept only as a migration note.
The old single-task id `knob_http_mariadb_sysbench_5` was renamed to `knob_http_mariadb_sysbench_read_write_5`.

Use the shared MariaDB evaluator setup documented in:

- `bbo/task_descriptions/knob_http_mariadb_sysbench_read_write_5/environment.md`

If your Docker installation includes Compose, you can also start all three sidecars from the repository root with:

```bash
docker compose -f docker-compose.task-services.yml up -d
```
