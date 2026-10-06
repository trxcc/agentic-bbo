"""Regression coverage for large incumbent configurations and field selection."""
import asyncio
import json

import pytest

from bbo.algorithms.agentic.tools.context_io import (
    ContextIOSession,
    context_documents,
    create_context_io_tools,
)
from bbo.core import (
    EvaluationResult,
    FloatParam,
    ObjectiveDirection,
    ObjectiveSpec,
    SearchSpace,
    TaskSpec,
    TrialObservation,
    TrialStatus,
    TrialSuggestion,
)


def large_session(tmp_path, count=196, direction=ObjectiveDirection.MINIMIZE):
    names = [f"database_configuration_parameter_{i:03d}" for i in range(count)]
    task = TaskSpec(
        name="large_config",
        search_space=SearchSpace([FloatParam(name, low=0, high=1) for name in names]),
        objectives=(ObjectiveSpec("score", direction),),
        max_evaluations=10,
    )
    rows = [TrialObservation.from_evaluation(
        TrialSuggestion(config=dict.fromkeys(names, value), trial_id=trial_id),
        EvaluationResult(status=status, objectives={"score": score}),
    ) for trial_id, value, score, status in [
        (30, 0.1234, 0.4, TrialStatus.SUCCESS),
        (10, 0.2345, 0.2, TrialStatus.SUCCESS),
        (20, 0.3456, 0.2, TrialStatus.SUCCESS),
        (40, 0.4567, -1.0, TrialStatus.FAILED),
    ]]
    return ContextIOSession(task, rows, context_documents(task, "# Large configuration"),
                            tmp_path / "state", tmp_path / "workspace", "fp")


def tool_call(session, name, **arguments):
    tool = next(t for t in create_context_io_tools(lambda: session) if t.name == name)
    return asyncio.run(tool.execute(None, **arguments))


@pytest.mark.parametrize("direction,trial_id", [
    (ObjectiveDirection.MINIMIZE, 10), (ObjectiveDirection.MAXIMIZE, 30),
])
def test_large_incumbent_returns_complete_config_by_default(tmp_path, direction, trial_id):
    s = large_session(tmp_path, direction=direction)
    expected = next(row for row in s.history if row["trial_id"] == trial_id)
    assert len(json.dumps(expected["config"])) > 6000
    assert "config" not in tool_call(s, "get_incumbent")["incumbent"]
    result = tool_call(s, "get_incumbent", include_config=True)
    assert result["incumbent"] == expected
    assert len(json.dumps(result, separators=(",", ":"))) <= 24000


@pytest.mark.parametrize("tool_name", ["get_incumbent", "get_trial_history"])
def test_explicit_parameter_selection_overrides_include_config(tmp_path, tool_name):
    s = large_session(tmp_path)
    names = s.task.search_space.names()[:2]
    extra = {"mode": "best"} if tool_name == "get_trial_history" else {}
    result = tool_call(s, tool_name, include_config=True, parameter_names=names, **extra)
    row = result["incumbent"] if tool_name == "get_incumbent" else result["items"][0]
    assert row["config"] == {name: 0.2345 for name in names}
    with pytest.raises(ValueError, match="parameter_names"):
        tool_call(s, tool_name, include_config=True, parameter_names=["missing"], **extra)


def test_incumbent_character_budget_can_be_changed_and_errors_offer_export(tmp_path):
    s = large_session(tmp_path)
    with pytest.raises(ValueError, match="output_path"):
        tool_call(s, "get_incumbent", include_config=True, max_chars=6000)
    result = tool_call(s, "get_incumbent", include_config=True, max_chars=24000)
    assert len(result["incumbent"]["config"]) == 196
    for invalid in (True, 999, 24001, "24000"):
        with pytest.raises(ValueError, match="max_chars"):
            tool_call(s, "get_incumbent", max_chars=invalid)


def test_oversized_incumbent_exports_only_the_best_row_without_printing_it(tmp_path):
    s = large_session(tmp_path, count=600)
    with pytest.raises(ValueError, match="output_path"):
        tool_call(s, "get_incumbent", include_config=True)
    result = tool_call(s, "get_incumbent", include_config=True,
                       output_path="scratch/incumbent.json")
    assert result["exported"] == 1 and result["total"] == 1
    assert "incumbent" not in result and "items" not in result
    data = json.loads((s.workspace / "scratch/incumbent.json").read_text())
    assert data["items"] == [s.history[1]]
    assert data["total"] == 1
    assert len(json.dumps(result)) < 1000
    with pytest.raises(ValueError, match="omit max_chars"):
        tool_call(s, "get_incumbent", output_path="scratch/conflict.json", max_chars=24000)
    assert not (s.workspace / "scratch/conflict.json").exists()


def test_incumbent_export_keeps_projection_and_workspace_boundary(tmp_path):
    s = large_session(tmp_path)
    name = s.task.search_space.names()[0]
    tool_call(s, "get_incumbent", include_config=True, parameter_names=[name],
              output_path="scratch/selected.json")
    data = json.loads((s.workspace / "scratch/selected.json").read_text())
    assert data["items"][0]["config"] == {name: 0.2345}
    for path in ("../outside.json", "scratch/../outside.json", "task.md"):
        with pytest.raises(ValueError, match="scratch"):
            tool_call(s, "get_incumbent", output_path=path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (s.workspace / "scratch/link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        tool_call(s, "get_incumbent", output_path="scratch/link/result.json")
    assert not list(outside.iterdir())


def test_no_successful_incumbent_returns_null_or_an_empty_export(tmp_path):
    s = large_session(tmp_path)
    s.history = [row for row in s.history if row["status"] != "success"]
    assert tool_call(s, "get_incumbent")["incumbent"] is None
    result = tool_call(s, "get_incumbent", include_config=True,
                       output_path="scratch/empty.json")
    assert result["exported"] == 0
    assert json.loads((s.workspace / "scratch/empty.json").read_text())["items"] == []
