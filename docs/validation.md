# Validation

Run these checks from the repository root after `uv sync --extra dev`:

```bash
uv run pytest
uv run python -m bbo.experiments.verify_assets
uv run python -m compileall -q bbo tests
uv build
```

The default test suite covers task inventories and initializations, task
description loading, scoring, numerical and agent ask/tell flow, append-only
evaluation journals, resume guards, Docker workspace boundaries, placement
repair replay, and the frozen frontier context. It uses local fakes or bundled
small evaluators; it does not call a paid model provider.

`verify_assets` checks SHA-256 for every bundled frozen reference and validates
the main, frontier, and controlled-prior task/seed contracts without objective
calls. The excluded historical Codex model catalog retains a recorded hash but
is not required for this check. Docker integration tests and evaluator services
that need external checkpoints can be run separately when those dependencies
are available.

Live provider behavior, model availability, and stochastic model trajectories
depend on the chosen endpoint. Save the run's profile, returned model IDs,
evaluation journal, and package versions when reporting a reproduction.
