from __future__ import annotations

from pathlib import Path

from bbo.core import JsonlMetricLogger
from bbo.run import run_single_experiment


def test_run_single_experiment_generates_plots(tmp_path: Path) -> None:
    summary = run_single_experiment(
        task_name="branin_demo",
        algorithm_name="random_search",
        seed=3,
        max_evaluations=10,
        results_root=tmp_path,
        resume=False,
    )
    logger = JsonlMetricLogger(Path(summary["results_jsonl"]))
    records = logger.load_records()

    assert len(records) == 10
    for plot_path in summary["plot_paths"]:
        path = Path(plot_path)
        assert path.exists()
        assert path.stat().st_size > 0
