from __future__ import annotations

import os
from pathlib import Path

import pytest

from bbo.algorithms import ALGORITHM_REGISTRY
from bbo.algorithms.agentic import build_worker_prompt
from bbo.algorithms.agentic.model_routing import PabloModelRoutingConfig, build_routing_table, resolve_role_model
from bbo.algorithms.agentic.validation import validate_candidate_payload
from bbo.run import build_arg_parser, run_single_experiment
from bbo.tasks import create_molecule_qed_task, create_oer_task, create_task
from bbo.tasks.scientific import CACHE_ROOT_ENV, SOURCE_ROOT_ENV, VENDORED_SOURCE_ROOT


REQUIRED_ARTIFACT_KEYS = {
    "pablo_rounds_jsonl",
    "task_registry_json",
    "llm_calls_jsonl",
    "candidate_queue_jsonl",
    "resume_state_json",
}


def _require_bo_tutorial_source() -> Path:
    source_root = Path(os.environ.get(SOURCE_ROOT_ENV, str(VENDORED_SOURCE_ROOT)))
    if not source_root.exists():
        pytest.skip("Bundled scientific task datasets are not available in the workspace.")
    return source_root


@pytest.fixture
def scientific_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    source_root = _require_bo_tutorial_source()
    monkeypatch.setenv(SOURCE_ROOT_ENV, str(source_root))
    monkeypatch.setenv(CACHE_ROOT_ENV, str(tmp_path / "dataset_cache"))
    return source_root


def test_pablo_and_palbo_are_registered_and_cli_visible() -> None:
    parser = build_arg_parser()
    algorithm_action = next(action for action in parser._actions if action.dest == "algorithm")

    assert "pablo" in ALGORITHM_REGISTRY
    assert "palbo" in ALGORITHM_REGISTRY
    assert ALGORITHM_REGISTRY["pablo"].family == "agentic"
    assert ALGORITHM_REGISTRY["palbo"].family == "agentic"
    assert "pablo" in algorithm_action.choices
    assert "palbo" in algorithm_action.choices
    assert parser.parse_args(["--algorithm", "palbo"]).algorithm == "palbo"


def test_pablo_model_routing_and_worker_prompt_boundary() -> None:
    routing = build_routing_table(
        PabloModelRoutingConfig(
            model="base-model",
            global_model="global-model",
            worker_model="worker-model",
            planner_model="planner-model",
        )
    )
    assert routing == {
        "planner": "planner-model",
        "explorer": "global-model",
        "worker": "worker-model",
    }
    assert resolve_role_model("worker", PabloModelRoutingConfig(model="base-model", worker_model=None)) == "base-model"

    task = create_task("branin_demo", max_evaluations=3, seed=7)
    prompt = build_worker_prompt(
        task_spec=task.spec,
        planner_task_name="EXPLOIT_TOP",
        planner_task_text="TASK: refine around the current strong seed.",
        current_seed=task.spec.search_space.defaults(),
    )
    assert "current_seed" in prompt.user
    assert "c_global" not in prompt.user
    assert "c_global" not in prompt.system
    assert prompt.context["planner_task_name"] == "EXPLOIT_TOP"


def test_candidate_validation_accepts_wrapped_config_objects() -> None:
    task = create_task("branin_demo", max_evaluations=3, seed=7)
    payload = """
    {
      "candidates": [
        {"config": {"x1": 0.5, "x2": 0.5}, "rationale": "keep near center"},
        {"x1": 0.1, "x2": 0.9}
      ]
    }
    """

    candidates = validate_candidate_payload(payload, task.spec.search_space)

    assert candidates == [{"x1": 0.5, "x2": 0.5}, {"x1": 0.1, "x2": 0.9}]


@pytest.mark.parametrize("task_name", ["her_demo", "oer_demo"])
def test_pablo_mock_scientific_smoke(task_name: str, scientific_env: Path, tmp_path: Path) -> None:
    summary = run_single_experiment(
        task_name=task_name,
        algorithm_name="pablo",
        seed=5,
        max_evaluations=5,
        results_root=tmp_path,
        resume=False,
        pablo_provider="mock",
    )

    assert summary["trial_count"] == 5
    assert summary["best_primary_objective"] is not None
    assert Path(summary["results_jsonl"]).exists()
    assert REQUIRED_ARTIFACT_KEYS <= set(summary["internal_artifacts"])
    assert summary["role_model_routes"]["planner"] == "gpt-4.1-mini"
    for artifact_path in summary["internal_artifacts"].values():
        assert Path(artifact_path).exists()


def test_pablo_mock_molecule_and_alias_smoke(scientific_env: Path, tmp_path: Path) -> None:
    pytest.importorskip("rdkit")

    summary = run_single_experiment(
        task_name="molecule_qed_demo",
        algorithm_name="pablo",
        seed=5,
        max_evaluations=5,
        results_root=tmp_path,
        resume=False,
        pablo_provider="mock",
    )
    alias_summary = run_single_experiment(
        task_name="her_demo",
        algorithm_name="palbo",
        seed=6,
        max_evaluations=4,
        results_root=tmp_path,
        resume=False,
        pablo_provider="mock",
        pablo_model="base-model",
        pablo_global_model="global-model",
        pablo_worker_model="worker-model",
        pablo_planner_model="planner-model",
        pablo_explorer_model="explorer-model",
    )

    assert summary["trial_count"] == 5
    assert 0.0 <= float(summary["best_primary_objective"]) <= 1.0
    assert alias_summary["trial_count"] == 4
    assert alias_summary["algorithm_name"] == "pablo"
    assert alias_summary["role_model_routes"] == {
        "planner": "planner-model",
        "explorer": "explorer-model",
        "worker": "worker-model",
    }
    for artifact_path in alias_summary["internal_artifacts"].values():
        assert Path(artifact_path).exists()
