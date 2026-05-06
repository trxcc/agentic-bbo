# Environment Setup

This task wraps a published BBOPlace-Bench evaluator image as an external service instead of rebuilding the full upstream benchmark inside this repository.
The current published image referenced by this repo is `gaozhixuan/bboplace-bench` on Docker Hub.
The files under `bbo/task_descriptions/bboplace_bench/` are environment descriptions only; this repository does not vendor a task-local `Dockerfile` for BBOPlace itself.

## Host requirements

- Docker Engine with permission to pull and run public images.
- Python 3.11+ and `uv` for this repo.
- Optional: GPU-enabled Docker runtime if your local deployment of the image expects GPU access.

## Pull the evaluator image

```bash
docker pull gaozhixuan/bboplace-bench
```

## Run the evaluator service

CPU-style invocation:

```bash
docker run --rm -p 8070:8080 gaozhixuan/bboplace-bench
```

(The image listens on **8080 inside the container**; this maps it to **8070** on the host, avoiding a clash with the MariaDB evaluator default **8080** in this repo.)

If your local setup requires GPU access, add `--gpus all` before the image name.
The `--rm` flag is convenient for disposable smoke tests; omit it if you want to keep the container around for debugging or to inspect logs after exit.

The packaged task defaults to `http://127.0.0.1:8070` and posts evaluations to `/evaluate`.
Override the base URL when the service is hosted elsewhere:

```bash
export BBOPLACE_BASE_URL=http://127.0.0.1:8070
```

## Install this repo

For the minimal BBOPlace-only host environment:

```bash
uv sync --extra dev
```

For the merged host environment used by mixed-task batches in this repo:

```bash
uv sync --extra dev --extra task-host
```

## Smoke test

With the container running, execute:

```bash
export BBOPLACE_BASE_URL=http://127.0.0.1:8070
uv run python -m bbo.run --algorithm random_search --task bboplace_bench --max-evaluations 1
```

If your Docker installation includes Compose and you also want the MariaDB and surrogate evaluators, start the full sidecar stack from the repository root with:

```bash
docker compose -f docker-compose.task-services.yml up -d
```

## Non-Docker local bridge

If Docker is unavailable but you can prepare an upstream `BBOPlace-Bench` checkout locally, this repo now includes a small HTTP bridge that keeps the same `/evaluate` contract:

```bash
git clone https://github.com/lamda-bbo/BBOPlace-Bench /path/to/BBOPlace-Bench
export BBOPLACE_UPSTREAM_ROOT=/path/to/BBOPlace-Bench
uv run python -m bbo.tasks.bboplace.local_service --host 127.0.0.1 --port 8070
```

This requires the upstream repository's own evaluator dependencies and benchmark assets to already be prepared according to its README. The local bridge is only the HTTP shim.

A healthy setup should complete without connection or JSON-schema errors and write run artifacts under `artifacts/`.
If you need the full upstream workflow, such as rebuilding DREAMPlace, downloading benchmark datasets, or running SP / HPO / GP-HPWL / PPA pipelines, follow the official BBOPlace-Bench repository rather than this lightweight service wrapper.
