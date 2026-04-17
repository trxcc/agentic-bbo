"""Run only the random-search Branin demo."""

from __future__ import annotations

import json

from bbo.run import run_single_experiment


if __name__ == "__main__":
    summary = run_single_experiment(task_name="branin_demo", algorithm_name="random_search", seed=7)
    print(json.dumps(summary, indent=2, sort_keys=True))
