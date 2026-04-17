"""CLI and helpers for running standalone agentic BBO demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .algorithms import ALGORITHM_REGISTRY, create_algorithm
from .core import (
    ExperimentConfig,
    Experimenter,
    JsonlMetricLogger,
    Landscape2DPlotter,
    ObjectiveDistributionPlotter,
    OptimizationTracePlotter,
    OptimizerComparisonPlotter,
)
from .tasks import SYNTHETIC_PROBLEM_REGISTRY, SyntheticFunctionTask, create_demo_task


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "artifacts" / "demo_runs"


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
) -> dict[str, Any]:
    task = create_demo_task(task_name, max_evaluations=max_evaluations, seed=seed, noise_std=noise_std)
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
    plot_paths = generate_visualizations(task, logger, run_dir / "plots", algorithm_label=algorithm.name)

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
        "plot_paths": [str(path) for path in plot_paths],
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
    comparison_plot = generate_comparison_plot(
        task=create_demo_task(task_name, max_evaluations=random_evaluations, seed=seed),
        histories={
            "random_search": JsonlMetricLogger(Path(random_summary["results_jsonl"])).load_records(),
            "pycma": JsonlMetricLogger(Path(pycma_summary["results_jsonl"])).load_records(),
        },
        output_dir=comparison_dir / "plots",
    )
    suite_summary = {
        "task_name": task_name,
        "seed": seed,
        "random_search": random_summary,
        "pycma": pycma_summary,
        "comparison_plot": str(comparison_plot),
    }
    (comparison_dir / "suite_summary.json").write_text(json.dumps(suite_summary, indent=2, sort_keys=True), encoding="utf-8")
    return suite_summary


def generate_visualizations(
    task: SyntheticFunctionTask,
    logger: JsonlMetricLogger,
    output_dir: Path,
    *,
    algorithm_label: str,
) -> list[Path]:
    records = logger.load_records()
    if not records:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    objective_name = task.spec.primary_objective.name
    direction = task.spec.primary_objective.direction
    artifacts = [
        OptimizationTracePlotter().plot(
            records,
            objective_name=objective_name,
            direction=direction,
            output_path=output_dir / "trace.png",
            title=f"{task.spec.metadata['display_name']} - {algorithm_label} trace",
        ).path,
        ObjectiveDistributionPlotter().plot(
            records,
            objective_name=objective_name,
            output_path=output_dir / "distribution.png",
            title=f"{task.spec.metadata['display_name']} - {algorithm_label} distribution",
        ).path,
    ]
    if int(task.spec.metadata.get("dimension", 0)) == 2:
        artifacts.append(
            Landscape2DPlotter().plot(
                task,
                records,
                objective_name=objective_name,
                output_path=output_dir / "landscape.png",
                title=f"{task.spec.metadata['display_name']} - sampled landscape",
                resolution=int(task.spec.metadata.get("plot_resolution", 180)),
            ).path
        )
    return artifacts


def generate_comparison_plot(
    *,
    task: SyntheticFunctionTask,
    histories: dict[str, list],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = OptimizerComparisonPlotter().plot(
        histories,
        objective_name=task.spec.primary_objective.name,
        direction=task.spec.primary_objective.direction,
        output_path=output_dir / "comparison.png",
        title=f"{task.spec.metadata['display_name']} - optimizer comparison",
    )
    return artifact.path


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
    parser = argparse.ArgumentParser(description="Run synthetic demos for the agentic BBO benchmark core.")
    parser.add_argument("--task", default="branin_demo", choices=sorted(SYNTHETIC_PROBLEM_REGISTRY))
    parser.add_argument(
        "--algorithm",
        default="suite",
        choices=["suite", *sorted({name for name in ALGORITHM_REGISTRY if name in {"random_search", "pycma"}})],
        help="Which demo to run. `suite` runs both algorithms and a comparison plot.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-evaluations", type=int, default=None)
    parser.add_argument("--random-evaluations", type=int, default=45)
    parser.add_argument("--pycma-evaluations", type=int, default=36)
    parser.add_argument("--sigma-fraction", type=float, default=0.18)
    parser.add_argument("--popsize", type=int, default=6)
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
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
