"""Run optimizers on dbtune MariaDB/sysbench tasks (Docker evaluator).

Tasks are defined under ``bbo/tasks/dbtune/`` and registered in ``bbo.tasks.registry``.
This script calls ``create_dbtune_mariadb_task`` directly (same shape as ``bbo.run.run_single_experiment``).

Prerequisite: start the reusable dbtune MariaDB evaluator image (see each task's ``environment.md``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow ``python examples/run_database_http_eval_demo.py`` without editable install
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bbo.algorithms import ALGORITHM_REGISTRY, create_algorithm
from bbo.core import ExperimentConfig, Experimenter, JsonlMetricLogger, Task
from bbo.tasks.dbtune import DBTUNE_MARIADB_TASK_IDS, create_dbtune_mariadb_task

_DEFAULT_RESULTS_ROOT = _PROJECT_ROOT / "runs" / "demo"
_DEFAULT_TASK = "knob_http_mariadb_sysbench_read_write_5"


def _allocate_run_dir(base_dir: Path, *, resume: bool) -> Path:
    if resume or not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir
    counter = 1
    while True:
        candidate = base_dir.parent / f"{base_dir.name}_run_{counter:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        counter += 1


def _require_algorithm_support(task: Task, algorithm_name: str) -> None:
    algorithm_spec = ALGORITHM_REGISTRY[algorithm_name]
    if not algorithm_spec.numeric_only:
        return
    try:
        task.spec.search_space.numeric_bounds()
    except TypeError as exc:
        raise ValueError(
            f"Algorithm `{algorithm_name}` only supports fully numeric search spaces; "
            f"task `{task.spec.name}` includes categorical parameters."
        ) from exc


def run_dbtune_mariadb_experiment(
    *,
    task_id: str,
    algorithm_name: str,
    seed: int,
    max_evaluations: int | None,
    results_root: Path,
    resume: bool,
    sigma_fraction: float,
    popsize: int | None,
    http_eval_base_url: str | None,
    http_eval_timeout_sec: float | None,
    http_skip_health_check: bool,
    knobs_json_path: str | None,
) -> dict[str, Any]:
    task = create_dbtune_mariadb_task(
        task_id,
        max_evaluations=max_evaluations,
        seed=seed,
        base_url=http_eval_base_url,
        knobs_json_path=knobs_json_path,
        request_timeout_sec=http_eval_timeout_sec,
        skip_health_check=http_skip_health_check,
    )
    _require_algorithm_support(task, algorithm_name)

    run_dir = _allocate_run_dir(
        results_root / task.spec.name / algorithm_name / f"seed_{seed}",
        resume=resume,
    )
    results_jsonl = run_dir / "trials.jsonl"

    algorithm_kwargs: dict[str, Any] = {}
    if algorithm_name in {"pycma", "cma_es"}:
        algorithm_kwargs = {"sigma_fraction": sigma_fraction, "popsize": popsize}
    algorithm = create_algorithm(algorithm_name, **algorithm_kwargs)

    logger = JsonlMetricLogger(results_jsonl)
    experiment = Experimenter(
        task=task,
        algorithm=algorithm,
        logger_backend=logger,
        config=ExperimentConfig(seed=seed, resume=resume, fail_fast_on_sanity=True),
    )
    summary = experiment.run()
    records = logger.load_records()

    serializable_summary = {
        "task_name": summary.task_name,
        "algorithm_name": summary.algorithm_name,
        "seed": summary.seed,
        "n_completed": summary.n_completed,
        "total_eval_time": summary.total_eval_time,
        "best_primary_objective": summary.best_primary_objective,
        "stop_reason": summary.stop_reason,
        "description_fingerprint": summary.description_fingerprint,
        "incumbents": [
            {
                "config": incumbent.config,
                "score": incumbent.score,
                "objectives": incumbent.objectives,
                "trial_id": incumbent.trial_id,
                "metadata": incumbent.metadata,
            }
            for incumbent in summary.incumbents
        ],
        "logger_summary": summary.logger_summary,
        "results_jsonl": str(results_jsonl),
        "trial_count": len(records),
    }
    (run_dir / "summary.json").write_text(json.dumps(serializable_summary, indent=2, sort_keys=True), encoding="utf-8")
    return serializable_summary


run_database_http_experiment = run_dbtune_mariadb_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="dbtune MariaDB evaluator demo (MariaDB/sysbench TPS via Docker API)."
    )
    parser.add_argument(
        "--task",
        default=_DEFAULT_TASK,
        choices=sorted(DBTUNE_MARIADB_TASK_IDS),
        help="One of the eight dbtune MariaDB task ids (default: read/write, 5 knobs).",
    )
    parser.add_argument(
        "--algorithm",
        default="random_search",
        choices=("random_search", "pycma"),
        help="Baseline optimizer.",
    )
    parser.add_argument("--max-evaluations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sigma-fraction", type=float, default=0.18)
    parser.add_argument("--popsize", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--results-root", type=Path, default=_DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--http-eval-base-url",
        type=str,
        default=None,
        help="Evaluator base URL (overrides AGENTBBO_HTTP_EVAL_BASE_URL).",
    )
    parser.add_argument(
        "--http-eval-timeout-sec",
        type=float,
        default=None,
        help="HTTP timeout per evaluation (overrides AGENTBBO_HTTP_EVAL_TIMEOUT_SEC).",
    )
    parser.add_argument(
        "--http-skip-health-check",
        action="store_true",
        help="Skip GET /health at task construction.",
    )
    parser.add_argument(
        "--knobs-json-path",
        type=str,
        default=None,
        help="Override knob JSON (default: per-task asset from bbo/tasks/dbtune/assets).",
    )
    args = parser.parse_args()

    summary = run_dbtune_mariadb_experiment(
        task_id=args.task,
        algorithm_name=args.algorithm,
        seed=args.seed,
        max_evaluations=args.max_evaluations,
        results_root=args.results_root,
        resume=args.resume,
        sigma_fraction=args.sigma_fraction,
        popsize=args.popsize,
        http_eval_base_url=args.http_eval_base_url,
        http_eval_timeout_sec=args.http_eval_timeout_sec,
        http_skip_health_check=args.http_skip_health_check,
        knobs_json_path=args.knobs_json_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
