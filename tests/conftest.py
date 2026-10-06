"""Shared test-only fixtures that are not part of the benchmark registry."""

from __future__ import annotations

from pathlib import Path

from bbo.core import (
    EvaluationResult,
    FloatParam,
    ObjectiveDirection,
    ObjectiveSpec,
    SearchSpace,
    Task,
    TaskDescriptionRef,
    TaskSpec,
    TrialSuggestion,
)


class TwoDimensionalAgentTestTask(Task):
    """Protocol-free numeric task for parser, tool, and agent-loop unit tests."""

    def __init__(self, *, max_evaluations: int = 4, seed: int = 0) -> None:
        del seed
        search_space = SearchSpace(
            [
                FloatParam("x1", low=-5.0, high=5.0, default=0.0),
                FloatParam("x2", low=-5.0, high=5.0, default=0.0),
            ]
        )
        description_dir = (
            Path(__file__).resolve().parents[1]
            / "bbo"
            / "task_descriptions"
            / "bbob_10d"
        )
        self._spec = TaskSpec(
            name="agent_test_2d",
            search_space=search_space,
            objectives=(ObjectiveSpec("loss", ObjectiveDirection.MINIMIZE),),
            max_evaluations=int(max_evaluations),
            description_ref=TaskDescriptionRef.from_directory("agent_test_2d", description_dir),
            metadata={
                "task_family": "test",
                "dimension": 2,
                "cma_initial_config": search_space.defaults(),
            },
        )

    @property
    def spec(self) -> TaskSpec:
        return self._spec

    def evaluate(self, suggestion: TrialSuggestion) -> EvaluationResult:
        config = self.spec.search_space.coerce_config(suggestion.config, use_defaults=False)
        loss = float(config["x1"]) ** 2 + float(config["x2"]) ** 2
        return EvaluationResult(objectives={"loss": loss})


def create_agent_test_task(*, max_evaluations: int = 4, seed: int = 0) -> TwoDimensionalAgentTestTask:
    return TwoDimensionalAgentTestTask(max_evaluations=max_evaluations, seed=seed)
