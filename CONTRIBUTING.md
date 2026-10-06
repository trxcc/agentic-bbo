# Contributing

Contributions to evaluators, algorithms, task descriptions, documentation, and
reproducibility tooling are welcome.

## Development setup

Use Python 3.11 and [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
uv run pytest
uv run python -m bbo.experiments.verify_assets
uv run python -m compileall -q bbo tests
```

Optional evaluator and optimizer dependencies are in the `hpo`, `molecular`,
`bo`, and `git-bo` extras. The default test suite runs without provider
credentials, Docker, paid model calls, or DBTune checkpoints.

## Change guidelines

- Keep reusable ask/tell, space, logging, and replay code in `bbo/core/`.
  Evaluator-specific behavior belongs in `bbo/tasks/` or `bbo/experiments/`.
- Keep task descriptions in `bbo/task_descriptions/<task_name>/` with
  `background.md`, `goal.md`, `constraints.md`, and `prior_knowledge.md`.
- Preserve append-only JSONL logging and replay-based resume behavior. Add a
  focused test for changes to these contracts, adapters, or evaluator setup.
- Treat files listed in `bbo/experiments/assets/provenance.json` as frozen.
  If a benchmark protocol changes, add a separately versioned artifact and
  explain its relationship to the existing reference.
- Do not commit API keys, model traces, downloaded checkpoints, local results,
  or the paper manuscript.

In a pull request, describe the behavior changed, any task-description or JSONL
schema impact, and the exact commands used to validate it. For changes to
published reference values, include the provenance and scoring implications.
