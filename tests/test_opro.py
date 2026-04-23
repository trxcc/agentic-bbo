from __future__ import annotations

import json
from pathlib import Path

import pytest

from bbo.algorithms import ALGORITHM_REGISTRY, OproAlgorithm
from bbo.algorithms.agentic.opro import (
    OpenAICompatibleOproBackend,
    OproGenerationRequest,
)
from bbo.core import (
    CategoricalParam,
    EvaluationResult,
    FloatParam,
    IntParam,
    ObjectiveDirection,
    ObjectiveSpec,
    SearchSpace,
    TaskSpec,
    TrialObservation,
    TrialSuggestion,
)
from bbo.run import _build_opro_algorithm_kwargs, build_arg_parser, run_single_experiment


def _mixed_task_spec(*, max_evaluations: int = 7) -> TaskSpec:
    return TaskSpec(
        name="mixed_opro_demo",
        search_space=SearchSpace(
            [
                FloatParam("lr", low=1e-4, high=1e-1, log=True, default=1e-2),
                IntParam("depth", low=2, high=8, default=4),
                CategoricalParam("activation", choices=("relu", "gelu", "tanh"), default="relu"),
            ]
        ),
        objectives=(ObjectiveSpec("loss", ObjectiveDirection.MINIMIZE),),
        max_evaluations=max_evaluations,
        metadata={"display_name": "Mixed OPRO Demo"},
    )


def _make_observation(suggestion: TrialSuggestion, trial_id: int) -> TrialObservation:
    normalized = TrialSuggestion(
        config=dict(suggestion.config),
        trial_id=trial_id,
        budget=suggestion.budget,
        metadata=dict(suggestion.metadata),
    )
    activation_penalty = {"relu": 0.25, "gelu": 0.1, "tanh": 0.4}[str(suggestion.config["activation"])]
    loss = float(suggestion.config["lr"]) * 15.0 + float(suggestion.config["depth"]) + activation_penalty
    return TrialObservation.from_evaluation(
        normalized,
        EvaluationResult(objectives={"loss": loss}),
    )


def test_opro_is_registered_and_cli_visible() -> None:
    parser = build_arg_parser()
    algorithm_action = next(action for action in parser._actions if action.dest == "algorithm")

    assert "opro" in ALGORITHM_REGISTRY
    assert ALGORITHM_REGISTRY["opro"].family == "agentic"
    assert ALGORITHM_REGISTRY["opro"].numeric_only is False
    assert "opro" in algorithm_action.choices
    assert parser.parse_args(["--algorithm", "opro"]).algorithm == "opro"


def test_opro_openai_backend_must_be_injected_from_runner_layer() -> None:
    algorithm = OproAlgorithm(backend="openai")
    with pytest.raises(RuntimeError, match="runner/CLI layer"):
        algorithm.setup(_mixed_task_spec(max_evaluations=3), seed=7)


def test_opro_runner_builds_online_backend_and_backend_uses_chat_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPRO_TEST_KEY", "sk-test")
    captured_requests: list[dict[str, Any]] = []

    class _FakeHttpResponse:
        def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
            self.payload = payload
            self.status = status

        def read(self) -> bytes:
            return json.dumps(self.payload).encode("utf-8")

        def __enter__(self) -> "_FakeHttpResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def _fake_urlopen(request, timeout: float):
        captured_requests.append(
            {
                "url": request.full_url,
                "headers": dict(request.header_items()),
                "body": json.loads(request.data.decode("utf-8")),
            }
        )
        return _FakeHttpResponse(
            {
                "choices": [
                    {"message": {"content": '<candidate>{"lr": 0.02, "depth": 5, "activation": "gelu"}</candidate>'}},
                ]
            }
        )

    monkeypatch.setattr("bbo.algorithms.agentic.opro.urllib_request.urlopen", _fake_urlopen)

    kwargs = _build_opro_algorithm_kwargs(
        opro_backend="openai",
        opro_model="gpt-4o-mini",
        opro_initial_samples=3,
        opro_candidates=6,
        opro_prompt_pairs=8,
        opro_openai_api_key_env="OPRO_TEST_KEY",
        opro_openai_base_url="https://api.openai.com/v1",
        opro_openai_organization="org-test",
        opro_openai_project="proj-test",
        opro_openai_timeout_seconds=12.5,
        opro_openai_max_retries=2,
    )
    assert kwargs["backend"] == "openai"
    backend = kwargs["backend_impl"]
    assert isinstance(backend, OpenAICompatibleOproBackend)

    search_space = _mixed_task_spec().search_space
    request = OproGenerationRequest(
        prompt="Recommend a candidate configuration.",
        n_responses=1,
        seed=11,
        objective_direction=ObjectiveDirection.MINIMIZE,
        search_space=search_space,
        observed_points=(),
        parameter_order=tuple(search_space.names()),
    )
    texts = backend.generate_candidate_texts(request)
    assert len(texts) == 1
    assert '<candidate>{"lr": 0.02, "depth": 5, "activation": "gelu"}</candidate>' in texts[0]

    call = captured_requests[0]
    call_headers = {key.lower(): value for key, value in call["headers"].items()}
    assert call["url"] == "https://api.openai.com/v1/chat/completions"
    assert call_headers["authorization"] == "Bearer sk-test"
    assert call_headers["openai-organization"] == "org-test"
    assert call_headers["openai-project"] == "proj-test"
    assert call["body"]["model"] == "gpt-4o-mini"
    assert call["body"]["messages"][0]["role"] == "system"
    assert call["body"]["messages"][1]["role"] == "user"


def test_opro_replay_reconstructs_next_suggestion_for_mixed_space() -> None:
    task_spec = _mixed_task_spec(max_evaluations=6)
    algorithm = OproAlgorithm(
        backend="heuristic",
        n_initial_samples=2,
        n_candidates=5,
        max_prompt_pairs=6,
    )
    algorithm.setup(task_spec, seed=23)

    history: list[TrialObservation] = []
    for trial_id in range(task_spec.max_evaluations):
        suggestion = algorithm.ask()
        observation = _make_observation(suggestion, trial_id)
        algorithm.tell(observation)
        history.append(observation)

    replayed = OproAlgorithm(
        backend="heuristic",
        n_initial_samples=2,
        n_candidates=5,
        max_prompt_pairs=6,
    )
    replayed.setup(task_spec, seed=23)
    replayed.replay(history[:-1])
    next_suggestion = replayed.ask()

    assert next_suggestion.config == history[-1].suggestion.config
    replayed.tell(history[-1])
    assert replayed.incumbents() == algorithm.incumbents()


def test_opro_branin_summary_and_resume_outputs(tmp_path: Path) -> None:
    summary = run_single_experiment(
        task_name="branin_demo",
        algorithm_name="opro",
        seed=7,
        max_evaluations=6,
        results_root=tmp_path,
        resume=False,
        opro_backend="heuristic",
        opro_initial_samples=3,
        opro_candidates=6,
        opro_prompt_pairs=8,
    )

    results_path = Path(summary["results_jsonl"])
    summary_path = results_path.with_name("summary.json")
    assert summary["trial_count"] == 6
    assert summary["best_primary_objective"] is not None
    assert results_path.exists()
    assert summary_path.exists()
    assert len(summary["incumbents"]) >= 1
    for plot_path in summary["plot_paths"]:
        assert Path(plot_path).exists()

    resumed = run_single_experiment(
        task_name="branin_demo",
        algorithm_name="opro",
        seed=7,
        max_evaluations=6,
        results_root=tmp_path,
        resume=True,
        opro_backend="heuristic",
        opro_initial_samples=3,
        opro_candidates=6,
        opro_prompt_pairs=8,
    )
    assert resumed["trial_count"] == 6
    assert resumed["best_primary_objective"] == summary["best_primary_objective"]
