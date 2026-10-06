from __future__ import annotations

import json
from pathlib import Path

from bbo.core import (
    BBOBenchmarkManifest,
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
    load_BBO_manifest,
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


def test_loads_task_local_BBO_manifest(tmp_path: Path) -> None:
    write_task_description_template(tmp_path)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "task_id": "minimal",
                "family": "scientific",
                "real_world_domain": "demo chemistry",
                "tool_policy": {"enabled_tools": ["get_task_context"], "web_search": {"enabled": True}},
            }
        ),
        encoding="utf-8",
    )

    manifest = load_BBO_manifest(_MinimalTask(tmp_path).spec)

    assert isinstance(manifest, BBOBenchmarkManifest)
    assert manifest.task_id == "minimal"
    assert manifest.family == "scientific"
    assert manifest.real_world_domain == "demo chemistry"
    assert manifest.tool_policy["web_search"]["enabled"] is True
    assert manifest.generated is False


def test_builds_compatible_BBO_manifest_and_ignores_localized_docs(tmp_path: Path) -> None:
    write_task_description_template(tmp_path)
    (tmp_path / "background.zh.md").write_text("# localized\n", encoding="utf-8")

    manifest = load_BBO_manifest(_MinimalTask(tmp_path).spec)

    assert manifest.generated is True
    assert manifest.task_id == "minimal"
    assert "memory_write" in manifest.tool_policy["enabled_tools"]
    assert "code_interpreter" in manifest.tool_policy["enabled_tools"]
    assert "summarize_objective_metrics" in manifest.tool_policy["enabled_tools"]
    assert "estimate_local_effects" in manifest.tool_policy["enabled_tools"]
    assert "validate_candidate" in manifest.tool_policy["enabled_tools"]
    assert manifest.tool_policy["code_interpreter"]["enabled"] is True
    assert manifest.tool_policy["web_search"]["enabled"] is True
    assert manifest.research_policy["allow_external_research"] is True
    assert "background.md" in manifest.workspace_seed_files
    assert "background.zh.md" not in manifest.workspace_seed_files
    assert manifest.budget["max_evaluations"] == 2
