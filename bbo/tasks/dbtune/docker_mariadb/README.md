# MariaDB/sysbench evaluator service

This image runs the live dbtune MariaDB + sysbench evaluator behind a small Flask API.
All eight `knob_http_mariadb_sysbench_*` tasks share the same image; the client selects the workload through the `workload` field in `POST /evaluate`.

## Reusable image

The compose stack and docs default to the reusable Docker Hub tag:

```bash
docker pull fakerstrawberry/agentbbo-dbtune-mariadb-eval:v1
docker rm -f agentbbo_http_mariadb_eval 2>/dev/null
docker run -d --name agentbbo_http_mariadb_eval -p 8080:8080 \
  fakerstrawberry/agentbbo-dbtune-mariadb-eval:v1
```

Override the tag used by `docker-compose.task-services.yml` with:

```bash
export AGENTBBO_DBTUNE_MARIADB_IMAGE=<your-dockerhub-user>/agentbbo-dbtune-mariadb-eval:v1
```

## Build/export for publishing

From the repository root, build both dbtune evaluator images and export them as tar packages:

```bash
scripts/package_dbtune_images.sh --tag v1
```

The script writes two tarballs under `dist/docker-images/` and prints the `docker load` / `docker push` commands for the upload machine.

## API

- `GET /health` -> `{"status":"ok"}`
- `POST /evaluate` -> `{"knobs": {"name": "value"}, "workload": "read_only|write_only|read_write|point_select"}`

The image listens on port `8080` inside the container.
