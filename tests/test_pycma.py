from __future__ import annotations

from pathlib import Path

from bbo.algorithms import create_algorithm
from bbo.core import ExperimentConfig, Experimenter, JsonlMetricLogger
from bbo.tasks import SyntheticFunctionTask, SyntheticFunctionTaskConfig


def test_pycma_runs_on_numeric_task(tmp_path: Path) -> None:
    task = SyntheticFunctionTask(SyntheticFunctionTaskConfig(problem="sphere_demo", max_evaluations=14, seed=5))
    logger = JsonlMetricLogger(tmp_path / "pycma.jsonl")
    experiment = Experimenter(
        task=task,
        algorithm=create_algorithm("pycma", sigma_fraction=0.15, popsize=4),
        logger_backend=logger,
        config=ExperimentConfig(seed=5, resume=False, fail_fast_on_sanity=True),
    )
    summary = experiment.run()
    records = logger.load_records()

    assert summary.n_completed == 14
    assert len(records) == 14
    assert summary.incumbents
    assert summary.best_primary_objective is not None
    assert summary.best_primary_objective <= records[0].objectives[task.spec.primary_objective.name]
