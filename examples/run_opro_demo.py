"""Run the OPRO Branin demo with the offline heuristic backend."""

from __future__ import annotations

import json

from bbo.run import run_single_experiment


if __name__ == "__main__":
    summary = run_single_experiment(
        task_name="branin_demo",
        algorithm_name="opro",
        seed=7,
        max_evaluations=12,
        opro_backend="heuristic",
        opro_initial_samples=4,
        opro_candidates=8,
        opro_prompt_pairs=12,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
