from __future__ import annotations

import json
import asyncio
from pathlib import Path

import pytest

from bbo.algorithms import create_algorithm
from bbo.algorithms.agentic.general_agent_engines import AgentResult, GeneralAgentEngine
from bbo.algorithms.agentic.tools.context_io import ContextIOSession, context_documents
from bbo.core import FloatParam, IntParam, SearchSpace, TaskSpec, ObjectiveSpec, ObjectiveDirection
from bbo.core import TrialObservation, TrialSuggestion, EvaluationResult, TrialStatus


def observation(config, value, trial_id):
    return TrialObservation.from_evaluation(TrialSuggestion(config=config, trial_id=trial_id),
        EvaluationResult(status=TrialStatus.SUCCESS, objectives={"loss": value}))


def session(tmp_path, n=20, history=None, budget=10):
    task = TaskSpec(name="example", search_space=SearchSpace([FloatParam(f"x{i}", low=0, high=1) for i in range(n)]),
                    objectives=(ObjectiveSpec("loss", ObjectiveDirection.MINIMIZE),), max_evaluations=budget)
    rows = [observation({f"x{i}": 0.123456789 for i in range(n)}, 0.5, 0)] if history is None else history
    return ContextIOSession(task, rows, context_documents(task, "# Example\n\nMinimize loss.\n"),
                            tmp_path / "state", tmp_path / "workspace", "fp")


def test_pages_recover_all_parameters_and_reject_foreign_cursor(tmp_path):
    s = session(tmp_path)
    args = {"view": "details", "limit": 3}
    found = []
    while True:
        r = s.execute("get_search_space", args)
        found += [i["name"] for i in r["items"]]
        if not r["next_cursor"]:
            break
        args["cursor"] = r["next_cursor"]
    assert found == s.task.search_space.names()
    with pytest.raises(ValueError, match="Cursor"):
        s.execute("get_search_space", {**args, "query": "x1"})


def test_history_projection_and_best_direction(tmp_path):
    s = session(tmp_path, n=1, history=[observation({"x0": .2}, .3, 0), observation({"x0": .4}, .1, 1)])
    assert "config" not in s.execute("get_trial_history", {})["items"][0]
    assert s.execute("get_incumbent", {})["incumbent"]["trial_id"] == 1
    assert s.execute("get_incumbent", {"parameter_names": ["x0"]})["incumbent"]["config"] == {"x0": .4}


def test_history_export_contains_all_rows_without_printing_them(tmp_path):
    history = [observation({f"x{j}":i/100 for j in range(20)}, i, i) for i in range(80)]
    s = session(tmp_path, history=history, budget=100)
    page = s.execute("get_trial_history", {"mode":"all","limit":100,"include_config":True,"max_chars":1000})
    assert page['has_more'] and page['returned']<80 and page['next_cursor']
    receipt = s.execute("get_trial_history", {'mode':'all','include_config':True,'output_path':'scratch/history.json'})
    assert receipt['exported']==80 and 'items' not in receipt and not receipt['has_more']
    data=json.loads((s.workspace/'scratch/history.json').read_text())
    assert [r['trial_id'] for r in data['items']]==list(range(80))
    assert data['items'][-1]['config']['x0']==.79
    with pytest.raises(ValueError,match='omit'):
        s.execute('get_trial_history',{'output_path':'scratch/a.json','limit':100})


def test_history_export_respects_projection_and_rejects_symlink_directories(tmp_path):
    s=session(tmp_path,n=2)
    s.execute('get_trial_history',{'parameter_names':['x1'],'output_path':'scratch/sub/a.json'})
    data=json.loads((s.workspace/'scratch/sub/a.json').read_text())
    assert set(data['items'][0]['config'])=={'x1'}
    for path in ['/tmp/a.json','../a.json','task.md','scratch/../a.json']:
        with pytest.raises(ValueError):
            s.execute('get_trial_history',{'output_path':path})
    outside=tmp_path/'outside';outside.mkdir()
    (s.workspace/'scratch/link').symlink_to(outside,target_is_directory=True)
    with pytest.raises(OSError):
        s.execute('get_trial_history',{'output_path':'scratch/link/a.json'})
    assert not list(outside.iterdir())


def test_write_inherits_exact_values_and_submit_is_authoritative_and_idempotent(tmp_path):
    s = session(tmp_path)
    write = s.execute("write_candidate", {"base_trial_id": 0, "changes": {"x0": .6}})
    assert write["inherited_count"] == 19
    assert s.committed_payload is None
    receipt = s.execute("submit_candidate", {"candidate_id": write["candidate_id"]})
    assert receipt == s.execute("submit_candidate", {"candidate_id": write["candidate_id"]})
    assert receipt["objective_value"] is None and receipt["terminal"]
    config = s.committed_payload["candidates"][0]["config"]
    assert config["x0"] == .6 and config["x1"] == .123456789
    with pytest.raises(ValueError, match="submitted"):
        s.execute("write_candidate", {"base_trial_id": 0, "changes": {"x0": .7}})
    restored = session(tmp_path)
    assert restored.committed_payload == s.committed_payload


@pytest.mark.parametrize("changes", [{"unknown": .5}, {"x0": 2}, {"x0": "0.5"}, {"x0": True}, {"x0": float("nan")}])
def test_invalid_changes_never_submit(tmp_path, changes):
    s = session(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        s.execute("write_candidate", {"base_trial_id": 0, "changes": changes})
    assert s.committed_payload is None


def test_missing_fields_duplicates_and_stale_candidates_are_rejected(tmp_path):
    s = session(tmp_path)
    with pytest.raises(ValueError):
        s.execute("write_candidate", {"config": {"x0": .5}})
    with pytest.raises(ValueError, match="duplicates"):
        s.execute("write_candidate", {"base_trial_id": 0, "changes": {}})
    w = s.execute("write_candidate", {"base_trial_id": 0, "changes": {"x0": .5}})
    next_round = session(tmp_path, history=[observation({f"x{i}": .4 for i in range(20)}, .2, 1)])
    with pytest.raises(ValueError, match="stale"):
        next_round.execute("submit_candidate", {"candidate_id": w["candidate_id"]})


def test_budget_and_candidate_tampering_are_rejected(tmp_path):
    s = session(tmp_path, budget=1)
    w = s.execute("write_candidate", {"base_trial_id": 0, "changes": {"x0": .5}})
    with pytest.raises(ValueError, match="budget"):
        s.execute("submit_candidate", {"candidate_id": w["candidate_id"]})
    p = s.directory / (w["candidate_id"] + ".json")
    obj = json.loads(p.read_text()); obj["config"]["x0"] = .7; p.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="integrity"):
        s.execute("submit_candidate", {"candidate_id": w["candidate_id"]})


class SubmitEngine(GeneralAgentEngine):
    name = "submit-test"

    async def run_agent(self, session_id, message, work_copy, **kwargs):
        execute = kwargs["tool_executor"]
        config = {"x": .4}
        result = json.loads(await execute("write_candidate", {"config": config}, "w"))["result"]
        await execute("submit_candidate", {"candidate_id": result["candidate_id"]}, "s")
        # Conflicting final text must never replace the selected candidate.
        return AgentResult(status="success", answer='{"candidates":[{"config":{"x":0.8}}]}')


class MissingSubmitEngine(GeneralAgentEngine):
    name = "missing-submit-test"

    async def run_agent(self, session_id, message, work_copy, **kwargs):
        payload = '{"candidates":[{"config":{"x":0.8}}]}'
        (work_copy.workspace_root / "final_candidate.json").write_text(payload)
        return AgentResult(status="success", answer=payload)


class TransportFailureAfterSubmitEngine(SubmitEngine):
    async def run_agent(self, *args, **kwargs):
        await super().run_agent(*args, **kwargs)
        return AgentResult(status="error", answer="", error="Connection lost after durable submission")


class MustNotRunEngine(GeneralAgentEngine):
    name = "must-not-run-test"

    async def run_agent(self, *args, **kwargs):
        raise AssertionError("A durable pending submission must recover without another model call")


def one_dimensional_task():
    return TaskSpec(name="example", search_space=SearchSpace([FloatParam("x", low=0, high=1)]),
                    objectives=(ObjectiveSpec("loss", ObjectiveDirection.MINIMIZE),), max_evaluations=3)


def test_final_file_and_text_cannot_bypass_submit(tmp_path):
    alg = create_algorithm("raw_agentic_bbo", context_access="on_demand", engine=MissingSubmitEngine(),
                           run_dir=tmp_path, max_retries=0)
    alg.setup(one_dimensional_task())
    with pytest.raises(RuntimeError, match="submit_candidate"):
        alg.ask()


def test_committed_candidate_survives_transport_failure(tmp_path):
    alg = create_algorithm("raw_agentic_bbo", context_access="on_demand", engine=TransportFailureAfterSubmitEngine(),
                           run_dir=tmp_path)
    alg.setup(one_dimensional_task())
    assert alg.ask().config == {"x": .4}


def test_resume_recovers_submission_before_evaluation_without_model_call(tmp_path):
    alg = create_algorithm("raw_agentic_bbo", context_access="on_demand", engine=SubmitEngine(), run_dir=tmp_path)
    task = one_dimensional_task()
    alg.setup(task)
    pending = alg.ask()
    restored = create_algorithm("raw_agentic_bbo", context_access="on_demand", engine=MustNotRunEngine(),
                                run_dir=tmp_path, resume=True)
    restored.setup(task)
    restored.replay([])
    assert restored.ask().config == pending.config


def test_integer_submission_never_silently_truncates(tmp_path):
    task = TaskSpec(name="integer", search_space=SearchSpace([IntParam("x", low=0, high=10)]),
                    objectives=(ObjectiveSpec("loss", ObjectiveDirection.MINIMIZE),), max_evaluations=3)
    s = ContextIOSession(task, [], context_documents(task, "# Integer"), tmp_path / "state", tmp_path / "workspace", "fp")
    for value in (1.1, "2", True):
        with pytest.raises(ValueError, match="integer"):
            s.execute("write_candidate", {"config": {"x": value}})
    assert s.execute("write_candidate", {"config": {"x": 2}})["valid"]


def test_registered_algorithm_uses_submit_not_final_text(tmp_path):
    task = TaskSpec(name="example", search_space=SearchSpace([FloatParam("x", low=0, high=1)]),
                    objectives=(ObjectiveSpec("loss", ObjectiveDirection.MINIMIZE),), max_evaluations=3)
    alg = create_algorithm("raw_agentic_bbo", context_access="on_demand", engine=SubmitEngine(), run_dir=tmp_path)
    alg.setup(task)
    suggestion = alg.ask()
    assert suggestion.config == {"x": .4}
    assert "submit_candidate" in Path(alg.artifact_paths["agent_workspace"], "instructions.md").read_text()


def test_parameter_definitions_match_runtime_and_bbob_stays_anonymous(tmp_path):
    from bbo.tasks import create_task

    task = create_task("bbob_f01_d10", max_evaluations=10, seed=0)
    alg = create_algorithm("raw_agentic_bbo", context_access="on_demand", engine=SubmitEngine(), run_dir=tmp_path)
    alg.setup(task.spec)
    workspace = Path(alg.artifact_paths["agent_workspace"])
    for name in ("task.md", "task_details.json", "parameter_catalog.json"):
        text = (workspace / name).read_text()
        assert "bbob_f01" not in text and "Sphere" not in text


def test_direct_submit_is_idempotent_without_a_write_call(tmp_path):
    s = session(tmp_path, n=1)
    first = s.execute("submit_candidate", {"config": {"x0": .6}})
    assert first["status"] == "accepted"
    assert first == s.execute("submit_candidate", {"config": {"x0": .6}})
    assert s.committed_payload["candidates"][0]["config"] == {"x0": .6}
    with pytest.raises(ValueError, match="different"):
        s.execute("submit_candidate", {"config": {"x0": .7}})


def test_direct_delta_submit_preserves_inherited_precision(tmp_path):
    s = session(tmp_path)
    args = {"base_trial_id": 0, "changes": {"x0": .6}}
    first = s.execute("submit_candidate", args)
    assert first == s.execute("submit_candidate", args)
    assert s.committed_payload["candidates"][0]["config"]["x1"] == .123456789


@pytest.mark.parametrize("wrapped", [False, True])
def test_file_submit_snapshots_configuration_and_rejects_changed_retry(tmp_path, wrapped):
    s = session(tmp_path, n=1)
    path = s.workspace / "scratch" / "candidate.json"
    path.parent.mkdir(parents=True)
    config = {"x0": .6}
    path.write_text(json.dumps({"config": config} if wrapped else config))
    first = s.execute("submit_candidate", {"path": "scratch/candidate.json"})
    assert first == s.execute("submit_candidate", {"path": "scratch/candidate.json"})
    path.write_text('{"x0":0.8}')
    assert s.committed_payload["candidates"][0]["config"] == config
    with pytest.raises(ValueError, match="different"):
        s.execute("submit_candidate", {"path": "scratch/candidate.json"})


def test_candidate_file_cannot_escape_workspace(tmp_path):
    s = session(tmp_path, n=1)
    outside = tmp_path / "outside.json"
    outside.write_text('{"x0":0.6}')
    s.workspace.mkdir(parents=True)
    (s.workspace / "link.json").symlink_to(outside)
    for path in (str(outside), "../outside.json", "link.json"):
        with pytest.raises(ValueError, match="workspace"):
            s.execute("submit_candidate", {"path": path})
    assert s.committed_payload is None


@pytest.mark.parametrize("payload", ['{"x0":0.4,"x0":0.6}', '{"x0":NaN}', '{"x0":2}', '{"x0":true}'])
def test_invalid_candidate_file_cannot_submit(tmp_path, payload):
    s = session(tmp_path, n=1)
    s.workspace.mkdir(parents=True)
    (s.workspace / "bad.json").write_text(payload)
    with pytest.raises((ValueError, TypeError)):
        s.execute("submit_candidate", {"path": "bad.json"})
    assert s.committed_payload is None


def test_submit_requires_one_mode_and_respects_budget(tmp_path):
    s = session(tmp_path, n=1, budget=1)
    with pytest.raises(ValueError, match="exactly one"):
        s.execute("submit_candidate", {"config": {"x0": .6}, "path": "any.json"})
    with pytest.raises(ValueError, match="exactly one"):
        s.execute("submit_candidate", {})
    with pytest.raises(ValueError, match="budget"):
        s.execute("submit_candidate", {"config": {"x0": .6}})
    assert s.committed_payload is None


def test_incremental_history_uses_observation_order_and_query_bound_cursor(tmp_path):
    s = session(tmp_path, n=1, history=[observation({"x0": .2}, .3, 30),
        observation({"x0": .4}, .1, 10), observation({"x0": .6}, .2, 20)])
    args = {"mode": "all", "after_trial_id": 30, "limit": 1}
    first = s.execute("get_trial_history", args)
    assert [x["trial_id"] for x in first["items"]] == [10]
    second = s.execute("get_trial_history", {**args, "cursor": first["next_cursor"]})
    assert [x["trial_id"] for x in second["items"]] == [20]
    assert s.execute("get_trial_history", {"after_trial_id": 20})["items"] == []
    with pytest.raises(ValueError, match="Cursor"):
        s.execute("get_trial_history", {**args, "after_trial_id": 10, "cursor": first["next_cursor"]})
    with pytest.raises(ValueError, match="after_trial_id"):
        s.execute("get_trial_history", {"after_trial_id": "absent"})


class DirectSubmitEngine(GeneralAgentEngine):
    name = "direct-submit-test"

    async def run_agent(self, session_id, message, work_copy, **kwargs):
        path = work_copy.workspace_root / "scratch" / "candidate.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"x":0.35}')
        response = json.loads(await kwargs["tool_executor"]("submit_candidate", {"path": "scratch/candidate.json"}, "s"))
        assert response["ok"]
        return AgentResult(status="success", answer="Submitted.")


def test_registered_algorithm_accepts_direct_file_submit(tmp_path):
    alg = create_algorithm("raw_agentic_bbo", context_access="on_demand", engine=DirectSubmitEngine(), run_dir=tmp_path)
    alg.setup(one_dimensional_task())
    assert alg.ask().config == {"x": .35}


def test_codex_schemas_are_sent_once_per_session_and_resent_on_change(tmp_path):
    from bbo.algorithms.agentic.general_agent_engines import CodexEngine, AgentWorkCopy
    from bbo.algorithms.agentic.tools.context_io import create_context_io_tools

    class CaptureEngine(CodexEngine):
        async def run_agent(self, session_id, message, work_copy, **kwargs):
            self.message = message
            return AgentResult(status="success", answer="Submitted.", llm_log={"sessionId": session_id or "session1"})

    async def execute(*args):
        raise AssertionError("No tool invocation is needed in this protocol fixture")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    work = AgentWorkCopy(state_dir=tmp_path / "state", config_path=None, project_root=workspace, workspace_root=workspace)
    specs = [t.function_spec() for t in create_context_io_tools(lambda: None)]

    def run(session_id, tools):
        engine = CaptureEngine()  # New object tests persisted session bookkeeping.
        asyncio.run(engine._run_with_host_tools(session_id, "Choose a candidate.", work,
            agent_id=None, timeout=5, extra_env=None, tools=tools, tool_executor=execute,
            max_tool_calls=5, final_instruction=None))
        return engine.message

    assert "Available tool schemas:" in run("", specs)
    assert "Available tool schemas:" not in run("session1", specs)
    changed = json.loads(json.dumps(specs))
    changed[0]["function"]["description"] += " Revised."
    assert "Available tool schemas:" in run("session1", changed)
    assert "Available tool schemas:" in run("new_session", specs)
