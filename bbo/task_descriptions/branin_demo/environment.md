# Environment Setup

This synthetic task runs in the repository-wide Python environment.
A collaborator can provision it with:

```bash
uv sync --extra dev
```

Minimal smoke test:

```bash
uv run python -m bbo.run --algorithm random_search --task branin_demo --max-evaluations 3
```

No task-specific Docker image is required for this benchmark.
