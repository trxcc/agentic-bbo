# Environment Setup

Install the repository with the HPO and development extras:

```bash
uv sync --extra dev --extra hpo
```

The task is CPU-compatible and uses only bundled offline arrays. No external service, network access, or GPU is required for evaluation.

Smoke test:

```bash
uv run python - <<'PY'
from bbo.core import TrialSuggestion
from bbo.tasks import create_task
task = create_task("hpo_bayesmark_breast_decision_tree", max_evaluations=1, seed=0)
print(task.sanity_check().ok)
print(task.evaluate(TrialSuggestion(task.spec.search_space.defaults())))
PY
```
