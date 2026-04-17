# Environment Setup

This task shares the same repository environment as the other synthetic benchmarks.
Provision it with:

```bash
uv sync --extra dev
```

Minimal smoke test:

```bash
uv run python -m bbo.run --algorithm random_search --task sphere_demo --max-evaluations 3
```
