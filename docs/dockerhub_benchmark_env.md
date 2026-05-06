# Docker Hub Benchmark Environment

This document describes how to use the published Docker image:

- `johnny114/agentic-bbo:20260504`
- `johnny114/agentic-bbo:latest`

The image is intended to be a practical, low-friction benchmark runtime with as few images as possible.

## What is inside

The image includes:

- the benchmark repository under `/workspace`
- a pre-synced `uv` environment with the `benchmark-main` dependency set
- a self-contained `BBOPlace-miniBench` checkout under `/opt/BBOPlace-miniBench`
- bundled benchmark datasets for the local `bboplace_bench` bridge
- MariaDB + sysbench runtime for the `knob_http_mariadb_*` tasks
- unified runner scripts in `/workspace/scripts`

The image does **not** replace the separate legacy Python 3.7 surrogate service for:

- `knob_http_surrogate_*`

Those surrogate-service tasks still require the dedicated Python 3.7 image `fakerstrawberry/agentbbo-dbtune-surrogate-http-py37:v1` because the old checkpoints are tied to a legacy sklearn/joblib ABI.

## Pull the image

```bash
docker pull johnny114/agentic-bbo:20260504
```

Or use the floating tag:

```bash
docker pull johnny114/agentic-bbo:latest
```

## Start an interactive shell

Minimal:

```bash
docker run --rm -it johnny114/agentic-bbo:20260504 bash
```

Recommended when you want to keep results on the host:

```bash
mkdir -p runs
docker run --rm -it \
  -v "$(pwd)/runs:/workspace/runs" \
  johnny114/agentic-bbo:20260504 bash
```

## Main entrypoint

Inside the container, use:

```bash
bash scripts/run_problem.sh <problem_name> <algorithm_name> [extra args...]
```

Examples:

```bash
bash scripts/run_problem.sh branin_demo random_search --max-evaluations 1 --no-plots
bash scripts/run_problem.sh bboplace_bench random_search --max-evaluations 1 --no-plots
bash scripts/run_problem.sh knob_http_mariadb_sysbench_read_only_5 random_search --max-evaluations 1 --no-plots
```

Behavior:

- `bboplace_bench`
  - the script auto-starts a local HTTP bridge on `127.0.0.1:8070`
  - no extra host mount is required
- `knob_http_mariadb_*`
  - the script auto-starts the local MariaDB/sysbench evaluator on `127.0.0.1:8080`
  - first startup may spend time preparing sysbench tables
- `knob_http_surrogate_*`
  - not handled by this image
  - still requires the separate Python 3.7 surrogate container

## Batch run for the 8 MariaDB tasks

The image includes a helper for the eight `knob_http_mariadb_*` tasks:

```bash
bash scripts/run_mariadb_baselines.sh --max-evaluations 1 --no-plots --results-root /workspace/runs/mariadb_batch
```

This batch script currently runs these methods:

- `random_search`
- `pycma`
- `optuna_tpe`
- `pfns4bo_tabpfn_v2`
- `llambo`
- `opro`
- `skydiscover_interleaved`
- `pablo`

## Methods verified in this environment

### Non-dbtune tasks

The following task set was validated against the following method set:

Tasks:

- `bboplace_bench`
- `bh_demo`
- `branin_demo`
- `budgeted_sphere_demo`
- `guacamol_qed_demo`
- `hea_demo`
- `her_demo`
- `molecule_qed_demo`
- `oer_demo`
- `qed_selfies_demo`
- `sphere_demo`

Methods:

- `random_search`
- `pycma`
- `optuna_tpe`
- `pfns4bo_tabpfn_v2`
- `llambo`
- `opro`
- `skydiscover_interleaved`
- `pablo`

### MariaDB tasks

Validated on:

- `knob_http_mariadb_sysbench_read_only_5`
- `knob_http_mariadb_sysbench_write_only_5`
- `knob_http_mariadb_sysbench_read_write_5`
- `knob_http_mariadb_sysbench_point_select_5`
- `knob_http_mariadb_sysbench_read_only_all`
- `knob_http_mariadb_sysbench_write_only_all`
- `knob_http_mariadb_sysbench_read_write_all`
- `knob_http_mariadb_sysbench_point_select_all`

Methods:

- `random_search`
- `pycma`
- `optuna_tpe`
- `pfns4bo_tabpfn_v2`
- `llambo`
- `opro`
- `skydiscover_interleaved`
- `pablo`

## Results location

By default, runs go under:

- `/workspace/runs`

If you bind-mount a host directory to `/workspace/runs`, all outputs persist outside the container.

## Ports

You usually do **not** need to publish ports to the host when using `scripts/run_problem.sh`, because the benchmark process talks to local evaluators inside the same container.

For debugging, you may expose:

```bash
docker run --rm -it \
  -p 8070:8070 \
  -p 8080:8080 \
  johnny114/agentic-bbo:20260504 bash
```

## Known limitations

- `knob_http_surrogate_*` tasks are intentionally excluded from this image
- general-agent wrappers that depend on external runtimes/credentials are not part of the verified baseline set
- the image is large because it bundles both benchmark code and BBOPlace assets

## Quick smoke checklist

Inside the container:

```bash
bash scripts/run_problem.sh branin_demo random_search --max-evaluations 1 --no-plots
bash scripts/run_problem.sh bboplace_bench random_search --max-evaluations 1 --no-plots
bash scripts/run_problem.sh knob_http_mariadb_sysbench_read_only_5 random_search --max-evaluations 1 --no-plots
```
