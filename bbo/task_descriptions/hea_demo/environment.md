# Environment Setup

This task uses the repository Python environment and does not require a GPU or a task-local Docker image.
Recommended setup:

```bash
uv sync --extra dev --extra bo-tutorial
```

By default, this task reads its bundled source data from `bbo/tasks/scientific/data/examples/` inside the workspace.
If you want to override the bundled files with another uploaded dataset bundle, set `BBO_BO_TUTORIAL_SOURCE_ROOT=/path/to/source_root`; paths such as `examples/HEA/data/oracle_data.xlsx` are interpreted relative to that root.
You may also redirect staged assets with `BBO_BO_TUTORIAL_CACHE_ROOT=/path/to/cache`.

Minimal smoke test:

```bash
uv run python -m bbo.run --algorithm random_search --task hea_demo --max-evaluations 3
```

Required runtime packages are `pandas`, `openpyxl`, and `scikit-learn`.
