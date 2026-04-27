# Environment Setup

Install the scientific optional dependencies before running this task:

```bash
uv sync --extra dev --extra optuna --extra bo-tutorial
```

Smoke run:

```bash
uv run python -m bbo.run --algorithm optuna_tpe --task qed_selfies_demo --max-evaluations 6
```

