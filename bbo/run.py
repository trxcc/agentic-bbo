"""CLI and helpers for running standalone agentic BBO demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .algorithms import ALGORITHM_REGISTRY, create_algorithm
from .core import (
    CumulativeEvalTimeComparisonPlotter,
    ExperimentConfig,
    Experimenter,
    JsonlMetricLogger,
    Landscape2DPlotter,
    ObjectiveDistributionPlotter,
    OptimizationTracePlotter,
    CumulativeEvalTimePlotter,
    OptimizerComparisonPlotter,
    PerTrialEvalTimePlotter,
    RegretTracePlotter,
    ScalarBarPlotter,
    Task,
)
from .tasks import ALL_TASK_NAMES, create_task


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
    generate_plots: bool = True,
) -> dict[str, Any]:
    task = create_task(
        task_name,
        max_evaluations=max_evaluations,
        seed=seed,
        noise_std=noise_std,
        surrogate_path=surrogate_path,
        knobs_json_path=knobs_json_path,
    )
    _require_algorithm_support(task, algorithm_name)
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
    plot_paths: list[Path] = []
    if generate_plots:
        plot_paths = generate_visualizations(
            task=task,
            logger=logger,
            output_dir=run_dir / "plots",
            algorithm_label=algorithm_name,
        )
    serializable_summary["plot_paths"] = [str(p) for p in plot_paths]
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
    generate_plots: bool = True,
) -> dict[str, Any]:
    task = create_task(task_name, max_evaluations=random_evaluations, seed=seed)
    _require_algorithm_support(task, "random_search")
    _require_algorithm_support(task, "pycma")
    random_summary = run_single_experiment(
        task_name=task_name,
        algorithm_name="random_search",
        seed=seed,
        max_evaluations=random_evaluations,
        results_root=results_root,
        resume=resume,
        generate_plots=generate_plots,
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
        generate_plots=generate_plots,
    )

    comparison_dir = _allocate_run_dir(results_root / task_name / "suite" / f"seed_{seed}", resume=resume)
    comparison_plot_paths: list[str] = []
    if generate_plots:
        comparison_dir_plots = comparison_dir / "plots"
        generate_comparison_plot(
            task=task,
            histories={
                "random_search": JsonlMetricLogger(Path(random_summary["results_jsonl"])).load_records(),
                "pycma": JsonlMetricLogger(Path(pycma_summary["results_jsonl"])).load_records(),
            },
            output_dir=comparison_dir_plots,
        )
        extra = _generate_two_algorithm_suite_plots(
            task=task,
            random_summary=random_summary,
            pycma_summary=pycma_summary,
            output_dir=comparison_dir_plots,
        )
        comparison_plot_paths = [str(comparison_dir_plots / "comparison.png"), *[str(p) for p in extra]]
    suite_summary: dict[str, Any] = {
        "task_name": task_name,
        "seed": seed,
        "random_search": random_summary,
        "pycma": pycma_summary,
        "comparison_dir": str(comparison_dir),
        "comparison_plot_paths": comparison_plot_paths,
    }
    (comparison_dir / "suite_summary.json").write_text(json.dumps(suite_summary, indent=2, sort_keys=True), encoding="utf-8")
    return suite_summary


def generate_visualizations(
    task: Task,
    logger: JsonlMetricLogger,
    output_dir: Path,
    *,
    algorithm_label: str,
) -> list[Path]:
    """Emit one figure per metric: objective trace, distribution, per-trial time, cumulative time, optional 2D landscape."""
    records = logger.load_records()
    if not records:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    display = str(task.spec.metadata.get("display_name", task.spec.name))
    objective_name = task.spec.primary_objective.name
    direction = task.spec.primary_objective.direction
    artifacts: list[Path] = [
        OptimizationTracePlotter().plot(
            records,
            objective_name=objective_name,
            direction=direction,
            output_path=output_dir / "trace.png",
            title=f"{display} | {algorithm_label} | {objective_name} trace",
        ).path,
        ObjectiveDistributionPlotter().plot(
            records,
            objective_name=objective_name,
            output_path=output_dir / "distribution.png",
            title=f"{display} | {algorithm_label} | {objective_name} distribution",
        ).path,
        PerTrialEvalTimePlotter().plot(
            records,
            output_path=output_dir / "per_trial_eval_time.png",
            title=f"{display} | {algorithm_label} | eval wall time (per trial)",
        ).path,
        CumulativeEvalTimePlotter().plot(
            records,
            output_path=output_dir / "cumulative_eval_time.png",
            title=f"{display} | {algorithm_label} | cumulative eval time",
        ).path,
    ]
    if int(task.spec.metadata.get("dimension", 0)) == 2 and hasattr(task, "surface_grid"):
        artifacts.append(
            Landscape2DPlotter().plot(
                task,
                records,
                objective_name=objective_name,
                output_path=output_dir / "landscape.png",
                title=f"{display} | {algorithm_label} | sampled 2D landscape",
                resolution=int(task.spec.metadata.get("plot_resolution", 180)),
            ).path
        )
    known = task.spec.metadata.get("known_optimum")
    if known is not None and isinstance(known, (int, float)):
        artifacts.append(
            RegretTracePlotter().plot(
                records,
                objective_name=objective_name,
                direction=direction,
                known_optimum=float(known),
                output_path=output_dir / "regret.png",
                title=f"{display} | {algorithm_label} | regret (known optimum in metadata)",
            ).path
        )
    return artifacts


def generate_comparison_plot(
    *,
    task: Task,
    histories: dict[str, list],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    display = str(task.spec.metadata.get("display_name", task.spec.name))
    on = task.spec.primary_objective.name
    artifact = OptimizerComparisonPlotter().plot(
        histories,
        objective_name=on,
        direction=task.spec.primary_objective.direction,
        output_path=output_dir / "comparison.png",
        title=f"{display} | running best {on} (all algorithms)",
    )
    return artifact.path


def _generate_two_algorithm_suite_plots(
    *,
    task: Task,
    random_summary: dict[str, Any],
    pycma_summary: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    """Bar + cumulative-time comparison for the fixed suite (random_search vs pycma). One metric per figure."""
    display = str(task.spec.metadata.get("display_name", task.spec.name))
    on = task.spec.primary_objective.name
    r_rs = JsonlMetricLogger(Path(random_summary["results_jsonl"])).load_records()
    r_pc = JsonlMetricLogger(Path(pycma_summary["results_jsonl"])).load_records()
    paths: list[Path] = [
        CumulativeEvalTimeComparisonPlotter().plot(
            {"random_search": r_rs, "pycma": r_pc},
            output_path=output_dir / "comparison_cumulative_eval_time.png",
            title=f"{display} | cumulative eval time (all algorithms)",
        ).path,
        ScalarBarPlotter().plot(
            {
                "random_search": float(random_summary["best_primary_objective"]),
                "pycma": float(pycma_summary["best_primary_objective"]),
            },
            ylabel=f"best {on}",
            output_path=output_dir / "bar_best_primary_objective.png",
            title=f"{display} | best {on} (final, one bar per algorithm)",
        ).path,
        ScalarBarPlotter().plot(
            {
                "random_search": float(random_summary["total_eval_time"]),
                "pycma": float(pycma_summary["total_eval_time"]),
            },
            ylabel="Total eval time (s)",
            output_path=output_dir / "bar_total_eval_time.png",
            title=f"{display} | total eval time (sum of trial times)",
        ).path,
    ]
    return paths


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
    parser = argparse.ArgumentParser(description="Run demos for the agentic BBO benchmark core.")
    parser.add_argument("--task", default="branin_demo", choices=sorted(ALL_TASK_NAMES))
    parser.add_argument(
        "--algorithm",
        default="suite",
        choices=["suite", *sorted(ALGORITHM_REGISTRY)],
        help="Which demo to run. `suite` runs both algorithms and a comparison plot.",
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
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG plots under run_dir/plots (faster, for CI or headless runs).",
    )
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
            generate_plots=not args.no_plots,
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
            generate_plots=not args.no_plots,
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
