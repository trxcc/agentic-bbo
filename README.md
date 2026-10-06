# Agentic BBO Bench

Agentic BBO Bench is a Python benchmark for black-box optimization with language-model agents and numerical optimizers. It provides 77 tasks across five domains, shared initial evaluations, a host-controlled evaluation budget, isolated agent workspaces, append-only run logs, and a common scoring implementation. The benchmark accompanies *A Closer Look at Agentic BBO: Benchmarking LLM Agents for Black-Box Optimization* (Ming Chen et al., 2026).

| Domain | Tasks | Shared initial evaluations | New evaluations per main run |
| --- | ---: | ---: | ---: |
| BBOB | 24 | 20 | 100 |
| Hyperparameter optimization (HPO) | 25 | 5 | 25 |
| Database tuning (DBTune) | 6 | 50 | 200 |
| Placement (BBOPlace) | 12 | 50 | 200 |
| Molecular optimization (GuacaMol) | 10 | 50 | 200 |

A task is an objective and search space; a run fixes a task, seed, method, and budget. The included main-suite contracts cover seeds 2–5 for each task. Frontier and diagnostic suites select smaller, fixed subsets. Task IDs and frozen reference files are retained in `bbo/experiments/assets/`.

## Quick start

Requirements: Linux, Python 3.11, and [uv](https://docs.astral.sh/uv/). The commands below require no API key, Docker service, or downloaded checkpoint.

```bash
uv sync --extra dev
uv run python -m bbo.run --list-tasks
uv run python -m bbo.run --algorithm random --task bbob_f01_d10 --output results/random
uv run python -m bbo.frontier.run --task bbob_f15_d10 --dry-run --output results/inspect
uv run pytest
```

`--dry-run` prepares the run contract, shared initialization, first prompt, and agent workspace audit without contacting a model or objective. Use a new `--output` directory for each run.

## Running benchmarks

The main entry point is `python -m bbo.run`; `python -m bbo.frontier.run` defaults to the five frozen frontier cases. Both accept `--suite`, `--task`, `--seed`, `--algorithm`, `--output`, `--dry-run`, and `--resume`. `--help` lists all options.

### Numerical baselines

```bash
uv run python -m bbo.run --task bbob_f15_d10 --algorithm random --output results/bbob-random
uv sync --extra dev --extra bo
uv run python -m bbo.run --task hpo_bayesmark_breast_svm --algorithm gp_ei --output results/hpo-gp
```

Available methods include Random, Sobol, CMA-ES, GP-EI, TuRBO, TPE, local perturbation, GIT-BO, Graph GA, GPBO, and six included LLM-developed programs. Dependencies are split into `hpo`, `molecular`, `bo`, and `git-bo` extras; choose the extras needed for a method and task. GIT-BO also needs upstream weights and a CUDA GPU. The developed programs are fixed artifacts: `--algorithm developed --program bbob_1_v5` selects one for its corresponding held-out tasks.

### Language-model agent

The agent uses a persistent native Codex session in an isolated Docker workspace. The evaluator, task data, and credentials stay on the host. The agent reads task context and submits candidates through six host-mediated interfaces. Only accepted evaluations consume the task budget; invalid submissions can be corrected within the same round.

Build the agent image and copy a model profile:

```bash
docker build -f docker/agent_process_control/Dockerfile -t agentic-bbo-frontier-agent:v2 .
cp configs/main_deepseek.example.json configs/my-model.json
```

Edit `configs/my-model.json` for your endpoint and model. A profile names `model`, `api_base`, `api_key_env`, and `reasoning_effort`; it contains no credential. Export the named API-key environment variable, then run:

```bash
uv run python -m bbo.run --task bbob_f15_d10 --seed 2 \
  --model-profile configs/my-model.json --output results/bbob-agent
```

The native Codex executable is a separate prerequisite; pass its path with `--codex-executable` or set `BBO_CODEX_BIN`. The experiment environment used `codex-cli 0.153.4`. A provider-specific model catalog, if needed for a historical model alias, can be supplied through `BBO_CODEX_MODEL_CATALOG=/absolute/path/to/model_catalog.json`. It is copied into the isolated runtime when that catalog contains the selected model. The historical catalog is not distributed with this repository.

Resume an interrupted run with the same arguments plus `--resume`. The runner checks the saved settings, shared initialization, and any supplied model catalog hash. An unresolved host evaluation request must be investigated before resuming; it is never silently evaluated a second time. Run directories are locked against concurrent writers.

### Evaluator services

BBOB uses COCO. HPO uses the bundled fixed data splits, and GuacaMol uses RDKit-based SMILES objectives. The other domains use host-side services:

```bash
docker compose up --build -d placement
docker compose --profile dbtune up --build -d dbtune
```

Placement can also run directly with `uv run python -m bbo.tasks.bboplace.local_service --port 8070`. BBOPlace uses the frozen `geometry_repair_worst_initial_v1` repair policy and weighted pin HPWL; the bundled geometry and netlist data never enter the agent container. DBTune requires separately downloaded surrogate checkpoints and a dedicated Python 3.7 / scikit-learn 0.21.3 service. Follow [`bbo/tasks/dbtune/assets/README.md`](bbo/tasks/dbtune/assets/README.md) before starting it. Use `AGENTBBO_HTTP_SURROGATE_BASE_URL` or `--dbtune-url` for a nondefault DBTune endpoint.

## Diagnostic experiments

The diagnostic suite fixes smaller budgets and task selections for agent interface and information studies:

```bash
uv run python -m bbo.run --suite diagnostic --task bbob_f02_d10 \
  --tools T4 --model-profile configs/my-model.json --output results/diagnostic-t4
uv run python -m bbo.run --suite controlled --task coarse_geometry_task_001 \
  --prior geometry_count --model-profile configs/my-model.json --output results/controlled
```

`T0` exposes the six context and submission interfaces. `T1` adds a GP suggestion interface; `T4` exposes all seven GP assistance interfaces. `--information semantic` removes rendered domain priors, while `--information anonymous` also anonymizes real-task parameters. Controlled-prior experiments use three deterministic objectives, seven prior conditions, and the same 16+48 evaluation protocol per condition. The supported handoff study uses `--handoff-history` to continue a verified agent trajectory with GP, TuRBO, or local search. See [`docs/ablations.md`](docs/ablations.md) for exact contracts.

The repository includes frozen deployment of six selected developed optimizers and their development artifacts. It does not expose an executable GP-policy control role or a full optimizer-development feedback loop; those roles require additional implementation to rerun end to end.

## Scoring and outputs

For each new evaluation, the scorer normalizes the best value found so far, including the shared initialization. The run score is

`0.7 × mean(normalized incumbent over new evaluations) + 0.3 × normalized final incumbent`.

HPO, DBTune, and BBOPlace use a task/seed-matched GP reference at quality 0.6 with a smooth tail above it. BBOB uses its theoretical optimum; GuacaMol uses a task-specific upper reference without a GP anchor. See [`docs/scoring.md`](docs/scoring.md) for the normalization and aggregation rules. `bbo-score --help` describes scoring a standalone `trials.jsonl` file.

Each run directory records its effective settings, task context, append-only trials, and host evaluation journal. Agent runs also record tool and model transport audits. The output directory is the unit of resume and comparison; keep it intact when sharing results. Frozen file checksums and protocol contracts can be checked with:

```bash
uv run python -m bbo.experiments.verify_assets
```

## Repository map

| Path | Contents |
| --- | --- |
| `bbo/core/` | Benchmark-independent search spaces, ask/tell flow, adapters, logging, and replay |
| `bbo/algorithms/` | Agent and numerical optimizer implementations |
| `bbo/tasks/` | Objective wrappers, evaluator services, and domain assets |
| `bbo/task_descriptions/`, `bbo/task_context_profiles/` | Agent-visible task documents and context profiles |
| `bbo/experiments/` | Suite contracts, frozen references, scoring, and CLIs |
| `configs/`, `docker/`, `scripts/` | Example model profiles and service/container setup |
| `tests/`, `docs/` | Regression tests and protocol documentation |

For development setup and contribution rules, see [`CONTRIBUTING.md`](CONTRIBUTING.md). Validation commands and their scope are in [`docs/validation.md`](docs/validation.md).

## Citation and license

If you use the benchmark, cite *A Closer Look at Agentic BBO: Benchmarking LLM Agents for Black-Box Optimization* (Ming Chen et al., 2026). The project source is MIT-licensed; see [`LICENSE`](LICENSE). Bundled adaptations and benchmark assets are described in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
