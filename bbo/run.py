"""CLI and helpers for running standalone agentic BBO demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .algorithms import ALGORITHM_REGISTRY, create_algorithm
from .core import ExperimentConfig, Experimenter, JsonlMetricLogger
from .tasks import SYNTHETIC_PROBLEM_REGISTRY, SURROGATE_TASK_IDS, create_demo_task, create_surrogate_task


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "runs" / "demo"


def run_single_experiment(
    *,
    task_name: str,
    algorithm_name: str,
    seed: int,
    max_evaluations: int | None = None,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    resume: bool = False,
    sigma_fraction: float = 0.18,
    popsize: int | None = None,
    noise_std: float = 0.0,
    surrogate_path: str | Path | None = None,
    knobs_json_path: str | Path | None = None,
) -> dict[str, Any]:
    # 兼容 synthetic demo 和 surrogate(knob) 任务：
    # - synthetic: 支持 noise_std
    # - surrogate: 支持 surrogate_path / knobs_json_path（以及环境变量覆盖）
    if task_name in SYNTHETIC_PROBLEM_REGISTRY:
        task = create_demo_task(task_name, max_evaluations=max_evaluations, seed=seed, noise_std=noise_std)
    elif task_name in SURROGATE_TASK_IDS:
        task = create_surrogate_task(
            task_name,
            max_evaluations=max_evaluations,
            seed=seed,
            surrogate_path=str(surrogate_path) if surrogate_path is not None else None,
            knobs_json_path=str(knobs_json_path) if knobs_json_path is not None else None,
        )
    else:
        known = ", ".join(sorted((*SYNTHETIC_PROBLEM_REGISTRY.keys(), *SURROGATE_TASK_IDS)))
        raise ValueError(f"Unknown task `{task_name}`. Known tasks: {known}")

    run_dir = _allocate_run_dir(results_root / task_name / algorithm_name / f"seed_{seed}", resume=resume)
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


def run_demo_suite(
    *,
    task_name: str = "branin_demo",
    seed: int = 7,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    random_evaluations: int = 45,
    pycma_evaluations: int = 36,
    sigma_fraction: float = 0.18,
    popsize: int | None = 6,
    resume: bool = False,
) -> dict[str, Any]:
    random_summary = run_single_experiment(
        task_name=task_name,
        algorithm_name="random_search",
        seed=seed,
        max_evaluations=random_evaluations,
        results_root=results_root,
        resume=resume,
    )
    pycma_summary = run_single_experiment(
        task_name=task_name,
        algorithm_name="pycma",
        seed=seed,
        max_evaluations=pycma_evaluations,
        results_root=results_root,
        resume=resume,
        sigma_fraction=sigma_fraction,
        popsize=popsize,
    )

    comparison_dir = _allocate_run_dir(results_root / task_name / "suite" / f"seed_{seed}", resume=resume)
    suite_summary = {
        "task_name": task_name,
        "seed": seed,
        "random_search": random_summary,
        "pycma": pycma_summary,
    }
    (comparison_dir / "suite_summary.json").write_text(json.dumps(suite_summary, indent=2, sort_keys=True), encoding="utf-8")
    return suite_summary


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic and surrogate demos for the agentic BBO benchmark core.")
    all_tasks = tuple(sorted((*SYNTHETIC_PROBLEM_REGISTRY.keys(), *SURROGATE_TASK_IDS)))
    parser.add_argument("--task", default="branin_demo", choices=all_tasks)
    parser.add_argument(
        "--algorithm",
        default="suite",
        choices=["suite", *sorted({name for name in ALGORITHM_REGISTRY if name in {"random_search", "pycma"}})],
        help="Which demo to run. `suite` runs both random_search and pycma (JSONL + summaries only).",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-evaluations", type=int, default=None)
    parser.add_argument("--random-evaluations", type=int, default=45)
    parser.add_argument("--pycma-evaluations", type=int, default=36)
    parser.add_argument("--sigma-fraction", type=float, default=0.18)
    parser.add_argument("--popsize", type=int, default=6)
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Synthetic-only: add Gaussian noise to objective evaluations.",
    )
    parser.add_argument(
        "--surrogate-path",
        type=Path,
        default=None,
        help="Surrogate-only: override path to .joblib (otherwise uses bundled assets or env var override).",
    )
    parser.add_argument(
        "--knobs-json-path",
        type=Path,
        default=None,
        help="Surrogate-only: override path to knobs_*.json (otherwise uses bundled assets).",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.algorithm == "suite":
        summary = run_demo_suite(
            task_name=args.task,
            seed=args.seed,
            results_root=args.results_root,
            random_evaluations=args.random_evaluations,
            pycma_evaluations=args.pycma_evaluations,
            sigma_fraction=args.sigma_fraction,
            popsize=args.popsize,
            resume=args.resume,
        )
    else:
        summary = run_single_experiment(
            task_name=args.task,
            algorithm_name=args.algorithm,
            seed=args.seed,
            max_evaluations=args.max_evaluations,
            results_root=args.results_root,
            resume=args.resume,
            sigma_fraction=args.sigma_fraction,
            popsize=args.popsize,
            noise_std=args.noise_std,
            surrogate_path=args.surrogate_path,
            knobs_json_path=args.knobs_json_path,
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
