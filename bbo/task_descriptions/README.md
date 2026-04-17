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

## Localized companion files

You may add localized documentation companions such as `background.zh.md` or `goal.zh.md`.
These are for collaborators only.
The benchmark loader ignores `*.zh.md` and `*.en.md` files so the runtime task context stays deterministic.

## Included examples

- `bbo/task_descriptions/branin_demo/`: executable synthetic-function demo used by the README and tests
- `bbo/task_descriptions/sphere_demo/`: lightweight sanity-check task
- `bbo/task_descriptions/collaborator_problem_demo/`: a more complete collaborator-facing packaging example
- `bbo/task_descriptions/_template/`: copyable scaffold for new tasks

Legacy directories such as `bbo/task_descriptions/autoresearch_train/` are retained only for provenance and are not the recommended schema.
