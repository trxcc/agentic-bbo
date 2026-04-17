"""Run only the pycma Branin demo."""

from __future__ import annotations

import json

from bbo.run import run_single_experiment


if __name__ == "__main__":
    summary = run_single_experiment(
        task_name="branin_demo",
        algorithm_name="pycma",
        seed=7,
        max_evaluations=36,
        sigma_fraction=0.18,
        popsize=6,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
