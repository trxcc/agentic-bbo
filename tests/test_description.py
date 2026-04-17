from __future__ import annotations

from pathlib import Path

from bbo.core import (
    EvaluationResult,
    FloatParam,
    MarkdownDescriptionLoader,
    ObjectiveDirection,
    ObjectiveSpec,
    SearchSpace,
    Task,
    TaskDescriptionRef,
    TaskSpec,
    TrialStatus,
    TrialSuggestion,
    write_task_description_template,
)


def test_write_task_description_template_and_load(tmp_path: Path) -> None:
    written = write_task_description_template(tmp_path)
    assert written

    ref = TaskDescriptionRef.from_directory("demo", tmp_path)
    bundle = MarkdownDescriptionLoader().load(ref)

    assert not ref.missing_sections()
    assert bundle.fingerprint
    assert "Background" in bundle.rendered_context
    assert (tmp_path / "environment.md").exists()
    assert "goal" in bundle.section_map


class _MinimalTask(Task):
    def __init__(self, description_dir: Path) -> None:
        self._spec = TaskSpec(
            name="minimal",
            search_space=SearchSpace([FloatParam("x", low=0.0, high=1.0, default=0.5)]),
            objectives=(ObjectiveSpec("loss", ObjectiveDirection.MINIMIZE),),
            max_evaluations=2,
            description_ref=TaskDescriptionRef.from_directory("minimal", description_dir),
        )

    @property
    def spec(self) -> TaskSpec:
        return self._spec

    def evaluate(self, suggestion: TrialSuggestion) -> EvaluationResult:
        return EvaluationResult(
            status=TrialStatus.SUCCESS,
            objectives={"loss": float(suggestion.config["x"])},
        )


def test_task_requires_environment_provisioning(tmp_path: Path) -> None:
    for name in ("background.md", "goal.md", "constraints.md", "prior_knowledge.md"):
        (tmp_path / name).write_text(f"# {name}\n", encoding="utf-8")

    report = _MinimalTask(tmp_path).sanity_check()
    assert not report.ok
    assert any(issue.code == "missing_environment_setup" for issue in report.errors)
