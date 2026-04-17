# Repository Guidelines

## Project Structure & Module Organization
This repository now exposes a real Python package under `bbo/`.
Top-level benchmark entrypoints live in `bbo/run.py`.
Algorithms are grouped under `bbo/algorithms/`, tasks are grouped under `bbo/tasks/`, and reusable benchmark-agnostic abstractions live in `bbo/core/`.
Task descriptions live in `bbo/task_descriptions/<task_name>/`.
Use the standardized task-description schema with required files `background.md`, `goal.md`, `constraints.md`, and `prior_knowledge.md`.

## Build, Test, and Development Commands
- `uv sync --extra dev` creates the managed environment and installs the package in editable mode.
- `uv run pytest` runs the current automated test suite.
- `uv run python -m compileall -q bbo examples tests` is the quick syntax smoke test.
- `uv run python -m bbo.run --algorithm suite --task branin_demo` runs the canonical end-to-end demo.

## Coding Style & Naming Conventions
Use 4-space indentation, type hints, and concise docstrings on public APIs.
Follow existing naming: `snake_case` for functions and modules, `PascalCase` for classes and dataclasses, and `UPPER_SNAKE_CASE` for registries.
Keep `bbo/core/` benchmark-agnostic.
Put task-specific evaluation logic, benchmark packaging helpers, and synthetic/real task wrappers outside `bbo/core/`.
Each task package should provide either a task-local Docker workflow or an `environment.md` with setup instructions.
Preserve append-only JSONL logging and replay-based resume semantics.

## Testing Guidelines
Add or update tests whenever you change ask/tell flow, adapter setup, logging, plotting, or resume behavior.
Prefer lightweight synthetic tasks over expensive evaluators.
Name tests by behavior, for example `test_resume_replays_trials_in_order`.
When changing the task-description logic, also verify that localized files such as `*.zh.md` are ignored by the loader.

## Commit & Pull Request Guidelines
This checkout does not include `.git`, so there is no local history to inspect for conventions.
Use short imperative commit subjects like `Add optimizer comparison plotter`.
In pull requests, state whether the change touches `bbo/core/` or only task/demo layers, describe any JSONL or task-description schema impact, link related issues, and list the exact validation commands you ran.
Include sample CLI output or plot paths only when behavior changes.
