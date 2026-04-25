"""CLI and helpers for running standalone agentic BBO demos."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .algorithms import ALGORITHM_REGISTRY, OpenAICompatibleLlamboBackend, OpenAICompatibleOproBackend, create_algorithm
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


def _resolve_optional_env(*names: str) -> str | None:
    for name in names:
        if not name:
            continue
        value = os.environ.get(name)
        if value:
            return value
    return None


def _build_llambo_algorithm_kwargs(
    *,
    llambo_backend: str,
    llambo_model: str,
    llambo_initial_samples: int,
    llambo_candidates: int,
    llambo_templates: int,
    llambo_predictions: int,
    llambo_alpha: float,
    llambo_openai_api_key_env: str,
    llambo_openai_base_url: str | None,
    llambo_openai_organization: str | None,
    llambo_openai_project: str | None,
    llambo_openai_timeout_seconds: float,
    llambo_openai_max_retries: int = 3,
    llambo_openai_use_structured_outputs: bool = True,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "backend": llambo_backend,
        "model": llambo_model,
        "n_initial_samples": llambo_initial_samples,
        "n_candidates": llambo_candidates,
        "n_templates": llambo_templates,
        "n_predictions": llambo_predictions,
        "alpha": llambo_alpha,
    }
    if llambo_backend != "openai":
        return kwargs

    api_key = _resolve_optional_env(llambo_openai_api_key_env)
    if not api_key:
        raise ValueError(
            "LLAMBO OpenAI backend requires an API key in the user-facing environment. "
            f"Set `{llambo_openai_api_key_env}` or choose `--llambo-backend heuristic`."
        )
    kwargs["backend_impl"] = OpenAICompatibleLlamboBackend(
        model=llambo_model,
        api_key=api_key,
        base_url=llambo_openai_base_url or _resolve_optional_env("OPENAI_BASE_URL", "OPENAI_API_BASE"),
        organization=llambo_openai_organization or _resolve_optional_env("OPENAI_ORGANIZATION"),
        project=llambo_openai_project or _resolve_optional_env("OPENAI_PROJECT"),
        timeout_seconds=llambo_openai_timeout_seconds,
        max_retries=llambo_openai_max_retries,
        use_structured_outputs=llambo_openai_use_structured_outputs,
    )
    return kwargs


def _build_opro_algorithm_kwargs(
    *,
    opro_backend: str,
    opro_model: str,
    opro_initial_samples: int,
    opro_candidates: int,
    opro_prompt_pairs: int,
    opro_openai_api_key_env: str,
    opro_openai_base_url: str | None,
    opro_openai_organization: str | None,
    opro_openai_project: str | None,
    opro_openai_timeout_seconds: float,
    opro_openai_max_retries: int = 3,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "backend": opro_backend,
        "model": opro_model,
        "n_initial_samples": opro_initial_samples,
        "n_candidates": opro_candidates,
        "max_prompt_pairs": opro_prompt_pairs,
    }
    if opro_backend != "openai":
        return kwargs

    api_key = _resolve_optional_env(opro_openai_api_key_env)
    if not api_key:
        raise ValueError(
            "OPRO OpenAI backend requires an API key in the user-facing environment. "
            f"Set `{opro_openai_api_key_env}` or choose `--opro-backend heuristic`."
        )
    kwargs["backend_impl"] = OpenAICompatibleOproBackend(
        model=opro_model,
        api_key=api_key,
        base_url=opro_openai_base_url or _resolve_optional_env("OPENAI_BASE_URL", "OPENAI_API_BASE"),
        organization=opro_openai_organization or _resolve_optional_env("OPENAI_ORGANIZATION"),
        project=opro_openai_project or _resolve_optional_env("OPENAI_PROJECT"),
        timeout_seconds=opro_openai_timeout_seconds,
        max_retries=opro_openai_max_retries,
    )
    return kwargs


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
    pfns_device: str | None = None,
    pfns_pool_size: int = 256,
    pfns_model: str = "hebo_plus",
    pablo_provider: str = "mock",
    pablo_base_url: str | None = None,
    pablo_api_key_env: str = "PABLO_API_KEY",
    pablo_global_model: str | None = None,
    pablo_worker_model: str | None = None,
    pablo_planner_model: str | None = None,
    pablo_explorer_model: str | None = None,
    pablo_model: str = "gpt-4.1-mini",
    pablo_init_points: int = 4,
    pablo_max_fails: int = 3,
    pablo_num_seeds: int = 2,
    pablo_max_tasks: int = 20,
    pablo_enable_explorer: bool = True,
    pablo_enable_planner: bool = True,
    pablo_enable_worker: bool = True,
    llambo_backend: str = "heuristic",
    llambo_model: str = "gpt-4o-mini",
    llambo_initial_samples: int = 5,
    llambo_candidates: int = 8,
    llambo_templates: int = 2,
    llambo_predictions: int = 6,
    llambo_alpha: float = -0.2,
    llambo_openai_api_key_env: str = "OPENAI_API_KEY",
    llambo_openai_base_url: str | None = None,
    llambo_openai_organization: str | None = None,
    llambo_openai_project: str | None = None,
    llambo_openai_timeout_seconds: float = 30.0,
    llambo_openai_max_retries: int = 3,
    llambo_openai_use_structured_outputs: bool = True,
    opro_backend: str = "heuristic",
    opro_model: str = "gpt-4o-mini",
    opro_initial_samples: int = 5,
    opro_candidates: int = 8,
    opro_prompt_pairs: int = 20,
    opro_openai_api_key_env: str = "OPENAI_API_KEY",
    opro_openai_base_url: str | None = None,
    opro_openai_organization: str | None = None,
    opro_openai_project: str | None = None,
    opro_openai_timeout_seconds: float = 30.0,
    opro_openai_max_retries: int = 3,
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
    elif algorithm_name == "pfns4bo":
        algorithm_kwargs = {
            "device": pfns_device,
            "pool_size": pfns_pool_size,
            "model_name": pfns_model,
        }
    elif algorithm_name in {"pablo", "palbo"}:
        algorithm_kwargs = {
            "provider": pablo_provider,
            "base_url": pablo_base_url,
            "api_key_env": pablo_api_key_env,
            "global_model": pablo_global_model,
            "worker_model": pablo_worker_model,
            "planner_model": pablo_planner_model,
            "explorer_model": pablo_explorer_model,
            "model": pablo_model,
            "init_points": pablo_init_points,
            "max_fails": pablo_max_fails,
            "num_seeds": pablo_num_seeds,
            "max_tasks": pablo_max_tasks,
            "enable_explorer": pablo_enable_explorer,
            "enable_planner": pablo_enable_planner,
            "enable_worker": pablo_enable_worker,
            "run_dir": run_dir,
            "resume": resume,
        }
    elif algorithm_name == "llambo":
        algorithm_kwargs = _build_llambo_algorithm_kwargs(
            llambo_backend=llambo_backend,
            llambo_model=llambo_model,
            llambo_initial_samples=llambo_initial_samples,
            llambo_candidates=llambo_candidates,
            llambo_templates=llambo_templates,
            llambo_predictions=llambo_predictions,
            llambo_alpha=llambo_alpha,
            llambo_openai_api_key_env=llambo_openai_api_key_env,
            llambo_openai_base_url=llambo_openai_base_url,
            llambo_openai_organization=llambo_openai_organization,
            llambo_openai_project=llambo_openai_project,
            llambo_openai_timeout_seconds=llambo_openai_timeout_seconds,
            llambo_openai_max_retries=llambo_openai_max_retries,
            llambo_openai_use_structured_outputs=llambo_openai_use_structured_outputs,
        )
    elif algorithm_name == "opro":
        algorithm_kwargs = _build_opro_algorithm_kwargs(
            opro_backend=opro_backend,
            opro_model=opro_model,
            opro_initial_samples=opro_initial_samples,
            opro_candidates=opro_candidates,
            opro_prompt_pairs=opro_prompt_pairs,
            opro_openai_api_key_env=opro_openai_api_key_env,
            opro_openai_base_url=opro_openai_base_url,
            opro_openai_organization=opro_openai_organization,
            opro_openai_project=opro_openai_project,
            opro_openai_timeout_seconds=opro_openai_timeout_seconds,
            opro_openai_max_retries=opro_openai_max_retries,
        )
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
        "run_dir": str(run_dir),
        "results_jsonl": str(results_jsonl),
        "trial_count": len(records),
        "internal_artifacts": getattr(algorithm, "artifact_paths", {}),
        "role_model_routes": getattr(algorithm, "routing_table", {}),
    }
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
    }
    if generate_plots:
        suite_summary["comparison_plot_paths"] = comparison_plot_paths
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
    try:
        task.spec.search_space.numeric_bounds()
        return
    except TypeError as exc:
        if algorithm_spec.categorical_to_continuous is not None:
            return
        if not algorithm_spec.numeric_only:
            return
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
    parser = argparse.ArgumentParser(description="Run tasks for the agentic BBO benchmark core.")
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
    parser.add_argument("--pfns-device", default=None)
    parser.add_argument("--pfns-pool-size", type=int, default=256)
    parser.add_argument("--pfns-model", default="hebo_plus")
    parser.add_argument("--pablo-provider", default="mock", choices=["mock", "openai-compatible"])
    parser.add_argument("--pablo-base-url", default=None)
    parser.add_argument("--pablo-api-key-env", default="PABLO_API_KEY")
    parser.add_argument("--pablo-global-model", default=None)
    parser.add_argument("--pablo-worker-model", default=None)
    parser.add_argument("--pablo-planner-model", default=None)
    parser.add_argument("--pablo-explorer-model", default=None)
    parser.add_argument("--pablo-model", default="gpt-4.1-mini")
    parser.add_argument("--pablo-init-points", type=int, default=4)
    parser.add_argument("--pablo-max-fails", type=int, default=3)
    parser.add_argument("--pablo-num-seeds", type=int, default=2)
    parser.add_argument("--pablo-max-tasks", type=int, default=20)
    parser.add_argument("--pablo-enable-explorer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pablo-enable-planner", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pablo-enable-worker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--llambo-backend", choices=["heuristic", "openai"], default="heuristic")
    parser.add_argument("--llambo-model", default="gpt-4o-mini")
    parser.add_argument("--llambo-initial-samples", type=int, default=5)
    parser.add_argument("--llambo-candidates", type=int, default=8)
    parser.add_argument("--llambo-templates", type=int, default=2)
    parser.add_argument("--llambo-predictions", type=int, default=6)
    parser.add_argument("--llambo-alpha", type=float, default=-0.2)
    parser.add_argument("--llambo-openai-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llambo-openai-base-url", default=None)
    parser.add_argument("--llambo-openai-organization", default=None)
    parser.add_argument("--llambo-openai-project", default=None)
    parser.add_argument("--llambo-openai-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--llambo-openai-max-retries", type=int, default=3)
    parser.add_argument(
        "--llambo-openai-use-structured-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use json_schema structured outputs if supported by the endpoint. Disable for older compatible APIs.",
    )
    parser.add_argument("--opro-backend", choices=["heuristic", "openai"], default="heuristic")
    parser.add_argument("--opro-model", default="gpt-4o-mini")
    parser.add_argument("--opro-initial-samples", type=int, default=5)
    parser.add_argument("--opro-candidates", type=int, default=8)
    parser.add_argument("--opro-prompt-pairs", type=int, default=20)
    parser.add_argument("--opro-openai-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--opro-openai-base-url", default=None)
    parser.add_argument("--opro-openai-organization", default=None)
    parser.add_argument("--opro-openai-project", default=None)
    parser.add_argument("--opro-openai-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--opro-openai-max-retries", type=int, default=3)
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
            pfns_device=args.pfns_device,
            pfns_pool_size=args.pfns_pool_size,
            pfns_model=args.pfns_model,
            pablo_provider=args.pablo_provider,
            pablo_base_url=args.pablo_base_url,
            pablo_api_key_env=args.pablo_api_key_env,
            pablo_global_model=args.pablo_global_model,
            pablo_worker_model=args.pablo_worker_model,
            pablo_planner_model=args.pablo_planner_model,
            pablo_explorer_model=args.pablo_explorer_model,
            pablo_model=args.pablo_model,
            pablo_init_points=args.pablo_init_points,
            pablo_max_fails=args.pablo_max_fails,
            pablo_num_seeds=args.pablo_num_seeds,
            pablo_max_tasks=args.pablo_max_tasks,
            pablo_enable_explorer=args.pablo_enable_explorer,
            pablo_enable_planner=args.pablo_enable_planner,
            pablo_enable_worker=args.pablo_enable_worker,
            llambo_backend=args.llambo_backend,
            llambo_model=args.llambo_model,
            llambo_initial_samples=args.llambo_initial_samples,
            llambo_candidates=args.llambo_candidates,
            llambo_templates=args.llambo_templates,
            llambo_predictions=args.llambo_predictions,
            llambo_alpha=args.llambo_alpha,
            llambo_openai_api_key_env=args.llambo_openai_api_key_env,
            llambo_openai_base_url=args.llambo_openai_base_url,
            llambo_openai_organization=args.llambo_openai_organization,
            llambo_openai_project=args.llambo_openai_project,
            llambo_openai_timeout_seconds=args.llambo_openai_timeout_seconds,
            llambo_openai_max_retries=args.llambo_openai_max_retries,
            llambo_openai_use_structured_outputs=args.llambo_openai_use_structured_outputs,
            opro_backend=args.opro_backend,
            opro_model=args.opro_model,
            opro_initial_samples=args.opro_initial_samples,
            opro_candidates=args.opro_candidates,
            opro_prompt_pairs=args.opro_prompt_pairs,
            opro_openai_api_key_env=args.opro_openai_api_key_env,
            opro_openai_base_url=args.opro_openai_base_url,
            opro_openai_organization=args.opro_openai_organization,
            opro_openai_project=args.opro_openai_project,
            opro_openai_timeout_seconds=args.opro_openai_timeout_seconds,
            opro_openai_max_retries=args.opro_openai_max_retries,
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
