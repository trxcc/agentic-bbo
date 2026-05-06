# Task Description Standard

Chinese version: `bbo/task_descriptions/README.zh.md`

This repository treats task descriptions as first-class benchmark artifacts.
Each benchmark task should have its own directory under `bbo/task_descriptions/<task_name>/`.
The core loader validates a standardized schema so agentic methods receive structured context instead of a single ad-hoc prompt.

## Required files

```text
bbo/task_descriptions/<task_name>/
  background.md
  goal.md
  constraints.md
  prior_knowledge.md
```

Recommended optional files:

```text
  evaluation.md
  submission.md
  environment.md
  manifest.json
  notes.md
  history.md
```

## Section intent

- `background.md`: real-world context, workload, and why the problem matters
- `goal.md`: optimization target, evaluation contract, and success criteria
- `constraints.md`: hard limits, forbidden actions, budgets, and safety requirements
- `prior_knowledge.md`: domain priors, heuristics, invariants, and useful starting points
- `evaluation.md`: metrics, aggregation rules, noise model, seeds, and tie-breaking
- `submission.md`: exact knobs, I/O contract, and artifact layout expected by the benchmark
- `environment.md`: manual setup instructions when no task-local Docker workflow is provided

## Environment provisioning requirement

Every task package must provide at least one of the following:

- a task-local Docker workflow such as `Dockerfile`, `docker-compose.yml`, or a `docker/` directory
- an `environment.md` file with explicit setup instructions

The task sanity checks enforce that at least one of these provisioning paths exists.

## Agent benchmark manifest

New agent-facing tasks should include `manifest.json`.
The manifest lets the runtime build a benchmark-like workspace instead of only handing the agent markdown.
It may declare:

- `task_id`, `family`, and `real_world_domain`
- `workspace_seed_files` to copy into the agent workspace
- `tool_policy` for BBO tools such as `code_interpreter`, `web_search`, and `fetch_url`
- `research_policy` for external research and allowed fetch domains
- `memory_policy` for append-only agent memory
- `evaluation_endpoint` and budget metadata
- `dynamic_updates`, `artifact_policy`, and provenance

Existing tasks without `manifest.json` still run.
The loader synthesizes a compatible manifest from `TaskSpec` and the markdown directory.
Use `_template/manifest.json` as the starting point for new tasks.

## Localized companion files

You may add localized documentation companions such as `background.zh.md` or `goal.zh.md`.
These are for collaborators only.
The benchmark loader ignores `*.zh.md` and `*.en.md` files so the runtime task context stays deterministic.

## Included examples

- `bbo/task_descriptions/branin_demo/`: executable synthetic-function demo used by the README and tests
- `bbo/task_descriptions/sphere_demo/`: lightweight sanity-check task
- `bbo/task_descriptions/bboplace_bench/`: service-backed BBOPlace-Bench task with explicit evaluator setup instructions
- `bbo/task_descriptions/collaborator_problem_demo/`: a more complete collaborator-facing packaging example
- `bbo/task_descriptions/_template/`: copyable scaffold for new tasks
- **dbtune MariaDB / sysbench (eight tasks):** directories named `knob_http_mariadb_sysbench_<workload>_<5|all>/` where `<workload>` is `read_only`, `write_only`, `read_write`, or `point_select`, and `<5|all>` is the SHAP top-5 knob JSON or the full ~197-knob list. Each includes English docs plus `*.zh.md` companions. The default evaluator image is `fakerstrawberry/agentbbo-dbtune-mariadb-eval:v1`; after changing the evaluator, rebuild/export it with `scripts/package_dbtune_images.sh` and publish a new tag.
- **dbtune surrogates:** directories `knob_surrogate_sysbench_5/`, `knob_surrogate_job_5/`, etc. They describe the sklearn surrogate setting; on-disk assets and code live under the unified package `bbo/tasks/dbtune/` (`assets/`, `offline_surrogate_task.py`, optional `docker_surrogate/`). Registered `knob_http_surrogate_*` tasks use the reusable sidecar image `fakerstrawberry/agentbbo-dbtune-surrogate-http-py37:v1`. Paths inside these markdown files should reference `bbo/tasks/dbtune/...` (not the removed `database/` or `surrogate/` folders).

Legacy directories such as `bbo/task_descriptions/autoresearch_train/` are retained only for provenance and are not the recommended schema.
