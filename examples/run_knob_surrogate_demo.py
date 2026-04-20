"""Run a baseline optimizer on a surrogate knob task (offline sklearn RF)."""

from __future__ import annotations

import argparse
import json

from bbo.run import run_single_experiment
from bbo.tasks import SURROGATE_TASK_IDS


def main() -> None:
    parser = argparse.ArgumentParser(description="Knob surrogate benchmark demo.")
    parser.add_argument(
        "--task",
        default="knob_surrogate_sysbench_all",
        choices=sorted(SURROGATE_TASK_IDS),
        help="Surrogate task id (default: sysbench_5; large models e.g. *_20 need a complete .joblib).",
    )
    parser.add_argument(
        "--algorithm",
        default="random_search",
        choices=("random_search", "pycma"),
        help="Baseline optimizer.",
    )
    parser.add_argument("--max-evaluations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sigma-fraction", type=float, default=0.18)
    parser.add_argument("--popsize", type=int, default=6)
    args = parser.parse_args()

    summary = run_single_experiment(
        task_name=args.task,
        algorithm_name=args.algorithm,
        seed=args.seed,
        max_evaluations=args.max_evaluations,
        sigma_fraction=args.sigma_fraction,
        popsize=args.popsize,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
