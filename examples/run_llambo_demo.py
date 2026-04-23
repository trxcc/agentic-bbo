"""Run the LLAMBO Branin demo with the offline heuristic backend."""

from __future__ import annotations

import json

from bbo.run import run_single_experiment


if __name__ == "__main__":
    summary = run_single_experiment(
        task_name="branin_demo",
        algorithm_name="llambo",
        seed=7,
        max_evaluations=12,
        llambo_backend="heuristic",
        llambo_initial_samples=4,
        llambo_candidates=8,
        llambo_templates=2,
        llambo_predictions=6,
        llambo_alpha=-0.1,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
