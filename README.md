# Agentic BBO Benchmark Core

Language versions:
- English: `README.md`
- 中文：`README.zh.md`

## Overview

This repository provides a compact benchmark framework for agentic black-box optimization.
It is organized as a standard Python package under `bbo/`, with clear separation between reusable core abstractions, algorithm families, task families, documentation assets, and runnable examples.

The current repository serves three purposes:

- provide a small but well-structured benchmark core for future agent-based optimization methods
- provide executable traditional baselines for validation and comparison
- provide a standardized task-description format that collaborators can extend for new benchmark problems

## Repository layout

```text
.
├── AGENTS.md
├── README.md
├── README.zh.md
├── bbo/
│   ├── __init__.py
│   ├── algorithms/
│   │   ├── model_based/
│   │   ├── registry.py
│   │   └── traditional/
│   ├── core/
│   ├── run.py
│   ├── task_descriptions/
│   └── tasks/
├── docs/
├── examples/
├── tests/
└── pyproject.toml
```

### `bbo/core/`

Reusable benchmark abstractions:

- search-space definitions
- task specification and sanity checks
- trial records
- logging and replay
- experiment orchestration
- task-description loading
- plotting utilities
- external optimizer adapters

### `bbo/algorithms/`

Algorithm implementations are grouped by family.
Current family:

- `bbo/algorithms/model_based/`
  - `optuna_tpe.py`
- `bbo/algorithms/traditional/`
  - `random_search.py`
  - `pycma.py`

### `bbo/tasks/`

Task implementations are grouped by family (see also `bbo/tasks/scientific/` and `bbo/tasks/dbtune/README.md`).

- `bbo/tasks/synthetic/` — toy synthetic objectives (`branin.py`, `sphere.py`, `base.py`)
- `bbo/tasks/scientific/` — tabular / scientific BO tutorial tasks (`registry.py` + per-task modules, `data/` assets)
- `bbo/tasks/dbtune/` — database knob tuning: offline sklearn surrogates, HTTP MariaDB/sysbench evaluators, optional HTTP surrogate servers (`assets/`, `registry.py`, `docker_mariadb/`, `docker_surrogate/`)

### `bbo/task_descriptions/`

Standardized task packaging for benchmark context.
The current repository includes:

- runnable benchmark descriptions for `branin_demo` and `sphere_demo`
- a collaborator-facing packaging example
- a reusable template
- bilingual documentation companions

## Installation

Create the managed environment with `uv`:

```bash
uv sync --extra dev
```

Optional interoperability helpers for ConfigSpace can be installed with:

```bash
uv sync --extra dev --extra interop
```

Optuna TPE is kept optional so the base install stays lightweight:

```bash
uv sync --extra dev --extra optuna
```

Scientific smoke tests for `her_demo`, `oer_demo`, and `molecule_qed_demo` additionally require:

```bash
uv sync --extra dev --extra optuna --extra bo-tutorial
```

## Running the demos

### Full comparison suite

```bash
uv run python -m bbo.run --algorithm suite --task branin_demo
```

Equivalent example script:

```bash
uv run python examples/run_branin_suite.py
```

### Random-search baseline

```bash
uv run python examples/run_random_search_demo.py
```

### CMA-ES baseline

```bash
uv run python examples/run_pycma_demo.py
```

### Optuna TPE baseline

```bash
uv run python examples/run_optuna_tpe_demo.py
```

### Direct CLI example

```bash
uv run python -m bbo.run \
  --algorithm pycma \
  --task branin_demo \
  --max-evaluations 36 \
  --sigma-fraction 0.18 \
  --popsize 6
```

Optuna TPE uses the public algorithm name `optuna_tpe` and supports mixed/categorical search spaces:

```bash
uv run python -m bbo.run \
  --algorithm optuna_tpe \
  --task oer_demo \
  --max-evaluations 6
```

`suite` remains a traditional-only comparison between `random_search` and `pycma`; it does not include `optuna_tpe`.

## Outputs

`python -m bbo.run` writes under `runs/demo/` (or `--results-root`): `trials.jsonl`, `summary.json`, and a `plots/` folder when plotting is enabled (default). Use `--no-plots` to skip PNG generation. `summary.json` includes `plot_paths` listing every generated figure.

Legacy reference bundles may still live under `artifacts/` (e.g. `artifacts/final_demo_v3/`).

Per single-algorithm run, `plots/` typically includes **one file per metric**, for example:

- `trace.png` — objective over evaluations + incumbent curve
- `distribution.png` — histogram of observed objectives
- `per_trial_eval_time.png` / `cumulative_eval_time.png` — evaluation wall time
- `regret.png` — only when the task exports `known_optimum` in metadata (e.g. some synthetic tasks)
- `landscape.png` — only for 2D synthetic tasks with a surface

For `suite` (random_search vs pycma), see also `.../suite/seed_*/plots/`: `comparison.png` (running-best curves), `comparison_cumulative_eval_time.png`, `bar_best_primary_objective.png`, `bar_total_eval_time.png`, plus each algorithm’s own `plots/` under its sub-run directory.

## Task-description standard

Each benchmark task should live under `bbo/task_descriptions/<task_name>/`.
Required files:

```text
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

Localized companion files such as `background.zh.md` are supported for documentation purposes.
They are ignored by the loader during runtime so benchmark context remains deterministic.

Each task must also provide at least one environment provisioning path:

- a task-local Docker workflow
- or explicit setup instructions in `environment.md`

Related documentation:

- `bbo/task_descriptions/README.md`
- `bbo/task_descriptions/README.zh.md`
- `docs/collaborator_demo.md`
- `docs/collaborator_demo.zh.md`
- `bbo/core/DEVELOPER_GUIDE_zh.md`
- `bbo/core/IMPLEMENTATION_PLAN.md`
- `bbo/core/IMPLEMENTATION_PLAN.zh.md`

## Adding a new task

1. Copy `bbo/task_descriptions/_template/` into `bbo/task_descriptions/<task_name>/`.
2. Add or extend a task family under `bbo/tasks/`.
3. Define the search space explicitly with `SearchSpace` and typed parameters.
4. Return normalized `EvaluationResult` objects from the evaluator.
5. Add tests and run the validation commands below.

## Validation commands

```bash
uv run python -m compileall -q bbo examples tests
uv run pytest
uv run python -m bbo.run --algorithm suite --task branin_demo --results-root artifacts/final_demo
```

Optuna smoke examples:

```bash
uv run python -m bbo.run --algorithm optuna_tpe --task branin_demo --max-evaluations 6 --results-root artifacts/optuna_tpe_smoke
uv run python -m bbo.run --algorithm optuna_tpe --task her_demo --max-evaluations 6 --results-root artifacts/optuna_tpe_smoke
uv run python -m bbo.run --algorithm optuna_tpe --task oer_demo --max-evaluations 6 --results-root artifacts/optuna_tpe_smoke
uv run python -m bbo.run --algorithm optuna_tpe --task molecule_qed_demo --max-evaluations 6 --results-root artifacts/optuna_tpe_smoke
```

## Current reference benchmarks

- `branin_demo`: two-dimensional synthetic benchmark for visualization and optimizer comparisons
- `sphere_demo`: convex synthetic benchmark for smoke tests and replay/resume validation
- `collaborator_problem_demo`: documentation-focused example showing how to package a realistic benchmark problem
