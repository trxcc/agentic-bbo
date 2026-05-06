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
Current families:

- `bbo/algorithms/agentic/` — agent algorithms and the general-agent runtime. Pablo is a benchmark-internal role optimizer; the general-agent runtime is for Nanobot / Claude Code style agents.
- `bbo/algorithms/llm_based/`
  - `llambo.py`, `opro.py`
  - `skydiscover_interleaved.py` (SkyDiscover-backed OpenEvolve, GEPA, EvoX, and AdaEvolve integration; ShinkaEvolve can be added as an external backend)
- `bbo/algorithms/model_based/`
  - `optuna_tpe.py`
- `bbo/algorithms/traditional/`
  - `random_search.py`
  - `pycma.py`

### `bbo/tasks/`

Task implementations are grouped by family (see also `bbo/tasks/scientific/` and `bbo/tasks/dbtune/README.md`).

- `bbo/tasks/synthetic/` — toy synthetic objectives (`branin.py`, `sphere.py`, `base.py`)
- `bbo/tasks/bboplace/` — HTTP-backed BBOPlace benchmark task (`task.py`)
- `bbo/tasks/scientific/` — tabular / scientific BO tutorial tasks (`registry.py` + per-task modules, `data/` assets)
- `bbo/tasks/dbtune/` — database knob tuning: offline sklearn surrogates, HTTP MariaDB/sysbench evaluators, optional HTTP surrogate servers (`assets/`, `registry.py`, `docker_mariadb/`, `docker_surrogate/`)

### `bbo/task_descriptions/`

Standardized task packaging for benchmark context.
The current repository includes:

- runnable benchmark descriptions for `branin_demo`, `sphere_demo`, and `bboplace_bench`
- a collaborator-facing packaging example
- a reusable template
- bilingual documentation companions

## Installation

Create the managed environment with `uv`:

```bash
uv sync --extra dev
```

To provision one merged host environment for benchmark tasks, start from the lightweight BBOPlace wrapper and layer in the scientific / HTTP-client dependencies with:

```bash
uv sync --extra dev --extra task-host
```

`task-host` is the recommended host-side stack for mixed-task batches. It keeps the incompatible evaluator runtimes out of the Python 3.11 environment: MariaDB/sysbench still lives in its own Docker image, and the legacy sklearn surrogate runtime still lives in the Python 3.7 sidecar.

Optional interoperability helpers for ConfigSpace can be installed with:

```bash
uv sync --extra dev --extra interop
```

Optuna TPE is kept optional so the base install stays lightweight:

```bash
uv sync --extra dev --extra optuna
```

The **SkyDiscover interleaved** algorithm (`skydiscover_interleaved`) uses the optional PyPI `skydiscover` package:

```bash
uv sync --extra dev --extra skydiscover
```

Scientific smoke tests for `her_demo`, `oer_demo`, and `molecule_qed_demo` additionally require:

```bash
uv sync --extra dev --extra optuna --extra bo-tutorial
```

`uv sync --extra dev --extra task-host` is the equivalent merged install when you want one environment that covers BBOPlace, scientific tasks, and the HTTP task clients together.

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

### LLAMBO baseline

```bash
uv run python examples/run_llambo_demo.py
```

### OPRO baseline

```bash
uv run python examples/run_opro_demo.py
```

### SkyDiscover interleaved baseline (offline)

No LLM / no `skydiscover` extra required: interleave steps refresh the bundled seed strategy only.

```bash
uv run python examples/run_skydiscover_interleaved_demo.py
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

### BBOPlace HTTP task

If your Docker installation includes Compose and you want the full task-side service stack (`bboplace` + reusable dbtune MariaDB + reusable dbtune surrogate HTTP), start it from the repository root with:

```bash
docker compose -f docker-compose.task-services.yml up -d
```

The dbtune images default to `fakerstrawberry/agentbbo-dbtune-mariadb-eval:v1` and `fakerstrawberry/agentbbo-dbtune-surrogate-http-py37:v1`. Override them with `AGENTBBO_DBTUNE_MARIADB_IMAGE` and `AGENTBBO_DBTUNE_SURROGATE_IMAGE`.

To start only the published BBOPlace evaluator service, run:

```bash
docker pull gaozhixuan/bboplace-bench
docker run --rm -p 8070:8080 gaozhixuan/bboplace-bench
```

The task defaults to `http://127.0.0.1:8070` (host port **8070** → container **8080**, to avoid clashing with the MariaDB evaluator on **8080**). Override with `BBOPLACE_BASE_URL` if needed.

Then run a quick smoke test from this repo:

```bash
uv run python -m bbo.run --algorithm random_search --task bboplace_bench --max-evaluations 1
```

Optuna TPE uses the public algorithm name `optuna_tpe` and supports mixed/categorical search spaces:

```bash
uv run python -m bbo.run \
  --algorithm optuna_tpe \
  --task oer_demo \
  --max-evaluations 6
```

`suite` remains a traditional-only comparison between `random_search` and `pycma`; it does not include `optuna_tpe`.

LLAMBO uses the public algorithm name `llambo`. The default `heuristic` backend is an offline smoke-test path that preserves the LLAMBO acquisition/surrogate loop without requiring network access:

```bash
uv run python -m bbo.run \
  --algorithm llambo \
  --task branin_demo \
  --max-evaluations 12 \
  --llambo-backend heuristic
```

For the online OpenAI path, keep credentials and endpoint selection at the user-facing runner layer instead of hard-coding them inside the low-level algorithm. Set the API key in an environment variable and configure model/base-url/timeout through the CLI:

```bash
export OPENAI_API_KEY=your_key_here
uv run python -m bbo.run \
  --algorithm llambo \
  --task branin_demo \
  --max-evaluations 12 \
  --llambo-backend openai \
  --llambo-model gpt-4o-mini \
  --llambo-openai-api-key-env OPENAI_API_KEY \
  --llambo-openai-timeout-seconds 30
```

Optional endpoint overrides such as `--llambo-openai-base-url`, `--llambo-openai-organization`, and `--llambo-openai-project` are also configured at the runner layer.

Additional tuning flags:
- `--llambo-openai-max-retries` (default 3) – retry transient network errors with exponential backoff.
- `--no-llambo-openai-use-structured-outputs` – disable ``json_schema`` structured outputs for endpoints that do not support them; the backend will fall back to plain text completion + parsing.

A ready-made demo script for the online backend is provided at `examples/run_llambo_openai_demo.py`.

### Agentic BBO agents

The general-agent entrypoints expose a BBO function-calling runtime in addition to the workspace JSON files.
Current public names:

- `nanobot` / `agentic_nanobot`
- `claude_code` / `agentic_claude_code`
- `agentic_openai_compatible`

The runtime writes `agent_workspace/`, `agent_state/`, `agent_memory/`, `agent_tool_calls.jsonl`, `agent_web_sources.jsonl`, `llm_logs/`, and `agent_optimization_trace.jsonl` under each run directory.
Tools include task context, search-space inspection, trial history, incumbent lookup, candidate validation, sampling, history analysis, append-only memory, code execution, web search, and URL fetch.
For shell/file agents such as Nanobot and Claude Code, the workspace exposes a Python API in `agent_workspace/bbo_tools.py`; agents should prefer `from bbo_tools import BBO` and write small scripts instead of calling the lower-level `bbo_tool.py` CLI directly. The workspace also includes `examples/gp_expected_improvement.py`, an editable GP/LCB example that reads history, fits a `scikit-learn` GP when possible, validates candidates, and prints the strict candidates JSON schema.
For web research, set `SERPAPI_API_KEY` and use `--agent-web-search-provider serpapi`; setup details are in [`docs/serpapi_web_search.md`](docs/serpapi_web_search.md).

The code tool is designed for SandboxFusion's `/run_code` API; BBO does not provide a custom local security sandbox. Start a SandboxFusion server separately and pass `--sandbox-fusion-base-url` or set `SANDBOX_FUSION_BASE_URL`. Recommended SandboxFusion Python packages for BBO agents are `numpy`, `scipy`, `scikit-learn`, `pandas`, and `joblib`. Check a server before a real run with:

```bash
uv run python -m bbo.algorithms.agentic.sandbox_healthcheck \
  --base-url "$SANDBOX_FUSION_BASE_URL"
```

The full BBO SandboxFusion setup guide is in [`docs/sandboxfusion_bbo.md`](docs/sandboxfusion_bbo.md).

If an OpenAI-compatible endpoint returns model reasoning in `message.reasoning_content` (for example the proxy in `run.sh` with `--agent-model deepseek-reasoner`), Nanobot runs write `reasoning_traces/` and `agent_reasoning_metadata.jsonl`. Use `--agent-require-visible-cot` to fail a call when visible reasoning was not captured. The benchmark parser still consumes only the final strict JSON candidates payload.

```bash
uv run python -m bbo.run \
  --algorithm agentic_openai_compatible \
  --task branin_demo \
  --max-evaluations 8 \
  --agent-api-key-env OPENAI_API_KEY \
  --agent-model gpt-4.1-mini \
  --agent-tool-mode function_calling \
  --agent-web-search-provider disabled
```

For offline smoke tests, use `NanobotBBOAlgorithm(engine=MockAgentEngine(...))` from Python; the mock engine exercises the same BBO tool registry without external credentials.

OPRO uses the public algorithm name `opro`. The default `heuristic` backend is an offline smoke-test path that adapts the original OPRO config/value meta-prompt pattern to repository-native search spaces:

```bash
uv run python -m bbo.run \
  --algorithm opro \
  --task branin_demo \
  --max-evaluations 12 \
  --opro-backend heuristic
```

For the online OpenAI path, credentials and endpoint selection follow the same runner-layer pattern as LLAMBO:

```bash
export OPENAI_API_KEY=your_key_here
uv run python -m bbo.run \
  --algorithm opro \
  --task branin_demo \
  --max-evaluations 12 \
  --opro-backend openai \
  --opro-model gpt-4o-mini \
  --opro-openai-api-key-env OPENAI_API_KEY \
  --opro-openai-timeout-seconds 30
```

Optional endpoint overrides such as `--opro-openai-base-url`, `--opro-openai-organization`, and `--opro-openai-project` are also configured at the runner layer.

Additional tuning flags:
- `--opro-openai-max-retries` (default 3) – retry transient network errors with exponential backoff.

A ready-made demo script for the online backend is provided at `examples/run_opro_openai_demo.py`.

### SkyDiscover interleaved (`skydiscover_interleaved`)

This algorithm alternates **short SkyDiscover runs** (evolving a Python module that implements `suggest_next_config`) with **normal BBO trials** on the task search space. Artifacts live under each run directory: `generated/strategy.py`, `generated/meta_context.json`, and per-round `generated/skydiscover_round_*`.

Public names: `skydiscover_interleaved` (alias: `skydiscover_meta`).

#### Credentials (API keys)

LLAMBO and OPRO expose runner-layer flags such as `--llambo-openai-api-key-env` / `--opro-openai-api-key-env` so `bbo.run` can wire OpenAI-compatible credentials explicitly.

**`skydiscover_interleaved` does not use those flags.** When you enable **`--skydiscover-runner`**, the SkyDiscover `Runner` is invoked in-process and loads credentials the **same way standalone SkyDiscover** does:

- **Environment variables** — e.g. `OPENAI_API_KEY`, plus optional `OPENAI_API_BASE` / `OPENAI_BASE_URL` as resolved by SkyDiscover’s `load_config()` (provider-specific variables such as `AZURE_API_KEY`, `ANTHROPIC_API_KEY`, etc. apply when the model name implies that provider).
- **YAML config** — pass `--skydiscover-config path/to/config.yaml`; under `llm` you can set `api_key`, `api_base`, and `models` entries (per-model overrides are supported by SkyDiscover).
- **Post-load bridging** — SkyDiscover may copy resolved keys into the process environment for backends that read env vars only (`bridge_provider_env`).

So: **put keys in the environment or in the SkyDiscover YAML**, not in LLAMBO/OPRO-specific CLI options.

#### Offline / smoke (no LLM)

With **`--no-skydiscover-runner`** (the default when you do not pass `--skydiscover-runner`), no SkyDiscover `Runner` and no LLM calls run: the bundled seed strategy is copied on the interleave cadence. **No API key is required** — useful for CI and for validating BBO logging, replay, and `generated/strategy.py` refresh without cloud calls.

Example:

```bash
uv run python -m bbo.run \
  --algorithm skydiscover_interleaved \
  --task branin_demo \
  --max-evaluations 20 \
  --skydiscover-interleave-every 5 \
  --no-skydiscover-runner
```

#### End-to-end LLM evolution (online SkyDiscover)

To exercise the full loop — **SkyDiscover evolves `suggest_next_config` via LLM**, then BBO evaluates real task trials using the updated `generated/strategy.py` — do the following:

1. **Install the optional SkyDiscover dependency**:

   ```bash
   uv sync --extra dev --extra skydiscover
   ```

2. **Provide credentials** using either approach (or both, with YAML overriding where applicable):

   - **Environment:** export the key your provider expects (commonly `OPENAI_API_KEY` for OpenAI-style endpoints). Set `OPENAI_API_BASE` / `OPENAI_BASE_URL` if you use a proxy or non-default host.
   - **YAML:** create or reuse a SkyDiscover config and pass it with `--skydiscover-config`. Put `api_key`, `api_base`, and `models` under `llm` per SkyDiscover’s schema so discovery iterations can call the model.

3. **Enable the runner** with **`--skydiscover-runner`** (omit it or use `--no-skydiscover-runner` for offline smoke only).

4. **Tune interleaving:** `--skydiscover-interleave-every` controls how many completed BBO trials occur between SkyDiscover rounds; `--skydiscover-round-iterations` sets how many inner discovery iterations each round runs.

Example (OpenAI key via environment, default SkyDiscover config if you omit `--skydiscover-config`):

```bash
uv sync --extra dev --extra skydiscover
export OPENAI_API_KEY=your_key_here
# Optional: export OPENAI_API_BASE=https://api.openai.com/v1
uv run python -m bbo.run \
  --algorithm skydiscover_interleaved \
  --task branin_demo \
  --max-evaluations 30 \
  --skydiscover-interleave-every 4 \
  --skydiscover-runner \
  --skydiscover-round-iterations 3 \
  --skydiscover-search-type adaevolve
```

If the loaded YAML has **no** `llm.models` entries, this repo injects a single model using **`--skydiscover-model`** (default name `gpt-4o-mini` when unset) so the `Runner` still has a model to call; you should still set the matching API key in env or YAML.

A ready-made script that mirrors `examples/run_opro_openai_demo.py` — including **`--search-type`** for `adaevolve`, `evox`, `gepa`, `openevolve`, etc. — is at **`examples/run_skydiscover_interleaved_openai_demo.py`**.

**Method-specific hyperparameters** (AdaEvolve islands and intensity, GEPA merge/gating, EvoX co-evolution settings, beam width, …) are **not** exposed as separate `bbo.run` flags. Configure them in **SkyDiscover YAML** under `search.database`, exactly as in standalone SkyDiscover. Field names match SkyDiscover's config dataclasses (e.g. `AdaEvolveDatabaseConfig`).

For quick experiments, copy or extend the small repo-local examples in **`examples/skydiscover_configs/`** (`adaevolve_bbo.yaml`, `evox_bbo.yaml`, `gepa_bbo.yaml`, `openevolve_bbo.yaml`) and pass `--config`, or run the demo with **`--preset adaevolve`** / **`--preset evox`** / **`--preset gepa`** / **`--preset openevolve`** (loads the matching example YAML). The YAML `search.type` should match **`--search-type`**; if they disagree, this repo may instantiate a fresh default database for the CLI type and **drop** custom `search.database` values from the file.

**Result directories:** runs with `--algorithm skydiscover_interleaved` (or `skydiscover_meta`) are written to **`<results-root>/<task>/skydiscover_interleaved_<--skydiscover-search-type>/seed_<seed>/`** so each SkyDiscover search type has its own folder (sanitized for the filesystem). To dump **`generated/strategy.py`** plus each **`generated/skydiscover_round_*/best/best_program.py`** from a finished run:

```bash
uv run python examples/print_skydiscover_run_strategies.py runs/demo/branin_demo/skydiscover_interleaved_topk/seed_7
```


Useful CLI flags:

- `--skydiscover-config` — path to a SkyDiscover YAML config (optional; otherwise SkyDiscover defaults apply).
- `--skydiscover-search-type` — e.g. `topk`, `adaevolve`, `evox`, `gepa`, `openevolve` (must be supported by your installed SkyDiscover).
- `--skydiscover-model` — when the loaded config has no `llm.models`, a single `LLMModelConfig(name=...)` is injected (default name `gpt-4o-mini` if unset).
- `--skydiscover-max-meta-history` — cap on successful `(config, score)` pairs passed into the meta context (default 32).

**Note:** How to measure the performance of the intermediate algorithms evolved by SkyDiscover?

## Outputs

`python -m bbo.run` writes under `runs/demo/` (or `--results-root`): `trials.jsonl`, `summary.json`, and a `plots/` folder when plotting is enabled (default). Use `--no-plots` to skip PNG generation. `summary.json` includes `plot_paths` listing every generated figure. For `skydiscover_interleaved` / `skydiscover_meta`, the immediate parent of `seed_*` is **`skydiscover_interleaved_<search-type>`** (from `--skydiscover-search-type`), and `summary.json` adds `skydiscover_search_type` and `runs_subdirectory`.

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
manifest.json
notes.md
history.md
```

Localized companion files such as `background.zh.md` are supported for documentation purposes.
They are ignored by the loader during runtime so benchmark context remains deterministic.

Each task must also provide at least one environment provisioning path:

- a task-local Docker workflow
- or explicit setup instructions in `environment.md`

`manifest.json` is optional for existing tasks but recommended for new agent benchmarks.
It records the benchmark family, real-world domain, workspace seed files, tool policy, web/code research policy, memory policy, evaluation endpoint, budget, dynamic updates, artifact policy, and provenance.
If it is missing, the runner synthesizes a compatible manifest from `TaskSpec` and the markdown description.

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
uv run python -m bbo.run --algorithm opro --task branin_demo --max-evaluations 6 --results-root artifacts/opro_smoke
```

## Current reference benchmarks

- `branin_demo`: two-dimensional synthetic benchmark for visualization and optimizer comparisons
- `sphere_demo`: convex synthetic benchmark for smoke tests and replay/resume validation
- `bboplace_bench`: HTTP-backed macro-placement benchmark adapter for BBOPlace-Bench MGO evaluation
- `collaborator_problem_demo`: documentation-focused example showing how to package a realistic benchmark problem
