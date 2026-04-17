from __future__ import annotations

from pathlib import Path

from bbo.algorithms import RandomSearchAlgorithm
from bbo.core import ExperimentConfig, Experimenter, JsonlMetricLogger
from bbo.tasks import SyntheticFunctionTask, SyntheticFunctionTaskConfig


def test_experimenter_resume_keeps_append_only_history(tmp_path: Path) -> None:
    task = SyntheticFunctionTask(SyntheticFunctionTaskConfig(problem="sphere_demo", max_evaluations=6, seed=11))
    logger = JsonlMetricLogger(tmp_path / "trials.jsonl")
    experiment = Experimenter(
        task=task,
        algorithm=RandomSearchAlgorithm(),
        logger_backend=logger,
        config=ExperimentConfig(seed=11, resume=False, fail_fast_on_sanity=True),
    )
    summary = experiment.run()
    assert summary.n_completed == 6
    assert len(logger.load_records()) == 6

    task_resume = SyntheticFunctionTask(SyntheticFunctionTaskConfig(problem="sphere_demo", max_evaluations=6, seed=11))
    resume_experiment = Experimenter(
        task=task_resume,
        algorithm=RandomSearchAlgorithm(),
        logger_backend=logger,
        config=ExperimentConfig(seed=11, resume=True, fail_fast_on_sanity=True),
    )
    resumed = resume_experiment.run()
    assert resumed.n_completed == 6
    assert len(logger.load_records()) == 6
