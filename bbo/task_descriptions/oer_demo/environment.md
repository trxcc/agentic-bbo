# Environment Setup

This task uses the repository Python environment and does not require a GPU or a task-local Docker image.
Recommended setup:

```bash
uv sync --extra dev --extra bo-tutorial
```

By default, this task reads its bundled source data from `bbo/tasks/scientific/data/examples/` inside the workspace.
If you want to override the bundled files with another uploaded dataset bundle, set `BBO_BO_TUTORIAL_SOURCE_ROOT=/path/to/source_root`; paths such as `examples/OER/OER.csv` are interpreted relative to that root.
You may also redirect staged assets with `BBO_BO_TUTORIAL_CACHE_ROOT=/path/to/cache`.

Minimal smoke test:

```bash
uv run python -m bbo.run --algorithm random_search --task oer_demo --max-evaluations 3
```

Required runtime packages are `pandas` and `scikit-learn`.
Because the interface contains categorical parameters, numeric-only optimizers such as plain CMA-style methods are not valid unless they explicitly support mixed spaces.
