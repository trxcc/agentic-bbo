"""Batch-run **HTTP dbtune** tasks (`knob_http_mariadb_*` + `knob_http_surrogate_*`) across algorithms.

编排规则对齐 ``examples/run_all_registered_tasks.py``：同一套端口探测、`--include-http` /
``--skip-http``、`RunOutcome`、`run_single_experiment`、`write_batch_objectives_table`。

默认算法矩阵：**pycma**、**llambo**（默认 heuristic）、**opro**（默认 heuristic）。用于检验各任务在各算法下是否可跑通（含混合搜索空间导致 pycma 抛出预期错误）。

当前 registry 合计 **14** 个 HTTP knob 任务（MariaDB **8** + Surrogate **6**）；可用 ``--list`` 打印列表。

Usage:
    uv run python examples/run_dbtune_http_algorithm_matrix.py --list
    uv run python examples/run_dbtune_http_algorithm_matrix.py --dry-run
    uv run python examples/run_dbtune_http_algorithm_matrix.py --max-evaluations 5 --no-plots
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_run_all_registered_tasks_module():
    """Load sibling ``run_all_registered_tasks.py`` without ``examples`` package."""
    path = Path(__file__).resolve().parent / "run_all_registered_tasks.py"
    spec = importlib.util.spec_from_file_location("_run_all_registered_tasks_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helpers from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_rat = _load_run_all_registered_tasks_module()

from bbo.algorithms.registry import ALGORITHM_REGISTRY
from bbo.tasks.dbtune.cli_http_surrogate import DBTUNE_SURROGATE_SERVICE_TASK_NAMES
from bbo.tasks.dbtune.cli_mariadb_http import DBTUNE_MARIADB_TASK_NAMES


def _dbtune_http_task_names() -> tuple[str, ...]:
    """MariaDB HTTP ∪ surrogate HTTP，按字典序。"""
    return tuple(sorted(DBTUNE_MARIADB_TASK_NAMES | DBTUNE_SURROGATE_SERVICE_TASK_NAMES))


def _filter_tasks_for_probe(
    *,
    tasks: tuple[str, ...],
    include_http: bool,
    skip_http: bool,
    probe: Any,
) -> tuple[str, ...]:
    """与 ``run_all_registered_tasks._non_bboplace_tasks`` 中 knob_http 分支一致。"""
    out: list[str] = []
    for name in tasks:
        k = _rat._knob_http_kind(name)
        if k is None:
            continue
        if skip_http:
            continue
        if include_http:
            out.append(name)
            continue
        if (k == "mariadb" and not probe.mariadb) or (k == "surrogate" and not probe.surrogate):
            continue
        out.append(name)
    return tuple(out)


def _parse_algorithms(raw: str) -> tuple[str, ...]:
    parts = tuple(a.strip() for a in raw.split(",") if a.strip())
    unknown = [a for a in parts if a not in ALGORITHM_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown algorithm(s): {unknown}; known: {sorted(ALGORITHM_REGISTRY)}")
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Matrix: each HTTP dbtune task × each algorithm (default pycma, llambo, opro). "
        "Probe / HTTP flags mirror run_all_registered_tasks.py.",
    )
    http_mode = parser.add_mutually_exclusive_group()
    http_mode.add_argument(
        "--include-http",
        action="store_true",
        help="Force-include all listed HTTP tasks (skip port gating).",
    )
    http_mode.add_argument(
        "--skip-http",
        action="store_true",
        help="Exclude every knob_http_* from this matrix (usually yields empty plan).",
    )
    parser.add_argument("--http-host", default=_rat.DEFAULT_PROBE_HOST)
    parser.add_argument("--http-probe-timeout", type=float, default=_rat.DEFAULT_PROBE_TIMEOUT_S)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-evaluations", type=int, default=10)
    parser.add_argument(
        "--algorithms",
        default="pycma,llambo,opro",
        help="Comma-separated names from ALGORITHM_REGISTRY (default: pycma,llambo,opro).",
    )
    parser.add_argument("--sigma-fraction", type=float, default=0.18)
    parser.add_argument("--popsize", type=int, default=6)
    parser.add_argument("--llambo-backend", choices=["heuristic", "openai"], default="heuristic")
    parser.add_argument("--opro-backend", choices=["heuristic", "openai"], default="heuristic")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-table", action="store_true")
    parser.add_argument(
        "--results-subdir",
        default="batch_dbtune_algo_matrix",
        help="Under runs/demo/<subdir>/",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true", dest="list_only")
    parser.add_argument(
        "--tasks",
        default=None,
        help="Optional comma-separated subset of task ids (default: all registered HTTP dbtune tasks).",
    )
    args = parser.parse_args()

    try:
        algorithms = _parse_algorithms(args.algorithms)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    all_tasks = _dbtune_http_task_names()
    if args.tasks:
        requested = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
        unknown_t = [t for t in requested if t not in all_tasks]
        if unknown_t:
            print(f"Unknown task id(s): {unknown_t}; registered: {list(all_tasks)}", file=sys.stderr)
            return 2
        task_pool = tuple(sorted(requested))
    else:
        task_pool = all_tasks

    probe = _rat.probe_evaluator_ports(args.http_host, timeout_s=args.http_probe_timeout)
    selected = _filter_tasks_for_probe(
        tasks=task_pool,
        include_http=bool(args.include_http),
        skip_http=bool(args.skip_http),
        probe=probe,
    )

    if args.list_only:
        print("TCP probe", probe.host, f"(timeout={args.http_probe_timeout}s):")
        print(f"  {_rat.PORT_MARIADB_HTTP} MariaDB ->", "ok" if probe.mariadb else "closed")
        print(f"  {_rat.PORT_SURROGATE_HTTP} surrogate ->", "ok" if probe.surrogate else "closed")
        print("Registered HTTP dbtune tasks:", len(all_tasks))
        for t in all_tasks:
            print(" ", t)
        print("After probe/filter:", len(selected))
        for t in selected:
            print(" ", t)
        print("Algorithms:", ", ".join(algorithms))
        return 0

    results_base = _PROJECT_ROOT / "runs" / "demo" / args.results_subdir

    # 顺序：按算法外层，便于表格上行=算法（与跨算法汇总习惯一致）
    planned: list[tuple[str, dict[str, Any]]] = []
    for algo in algorithms:
        for task_name in selected:
            label = f"{task_name} seed={args.seed} algorithm={algo}"
            run_kw: dict[str, Any] = {
                "task_name": task_name,
                "algorithm_name": algo,
                "seed": args.seed,
                "max_evaluations": args.max_evaluations,
                "results_root": results_base,
                "generate_plots": not args.no_plots,
            }
            if algo in {"pycma", "cma_es"}:
                run_kw["sigma_fraction"] = args.sigma_fraction
                run_kw["popsize"] = args.popsize
            if algo == "llambo":
                run_kw["llambo_backend"] = args.llambo_backend
            if algo == "opro":
                run_kw["opro_backend"] = args.opro_backend
            planned.append((label, run_kw))

    if not planned:
        print("No experiments planned (empty task list after filters).", file=sys.stderr)
        return 2

    print(
        f"HTTP @ {probe.host}: MariaDB:{_rat.PORT_MARIADB_HTTP}={probe.mariadb} "
        f"surrogate:{_rat.PORT_SURROGATE_HTTP}={probe.surrogate}",
        flush=True,
    )

    if args.dry_run:
        print(f"Dry run — {len(planned)} experiments -> {results_base}")
        for label, _ in planned:
            print(" -", label)
        return 0

    results: list[_rat.RunOutcome] = []
    for label, kwargs in planned:
        print("===", label, flush=True)
        results.append(_rat._run_one(label=label, run=kwargs))

    runs_zip = list(
        zip(
            [lab for lab, _ in planned],
            [kw for _, kw in planned],
            results,
            strict=True,
        )
    )

    if not args.no_table and runs_zip:
        csv_p, json_p = _rat.write_batch_objectives_table(
            runs=runs_zip,
            output_dir=results_base,
            table_basename="dbtune_http_algo_matrix_table",
        )
        print("Wrote summary table:", csv_p)
        print("Wrote JSON:", json_p)

    print(json.dumps([{"label": r.label, "ok": r.ok, "error": r.error} for r in results], indent=2))
    failures = [r for r in results if not r.ok]
    if failures:
        for r in failures:
            print("FAILED:", r.label, r.error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
