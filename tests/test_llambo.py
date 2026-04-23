from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib import error as urllib_error

import pytest

from bbo.algorithms import ALGORITHM_REGISTRY, LlamboAlgorithm
from bbo.algorithms.agentic.llambo import (
    CandidateGenerationRequest,
    ObservedPoint,
    OpenAICompatibleLlamboBackend,
    ScorePredictionRequest,
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
from bbo.run import _build_llambo_algorithm_kwargs, build_arg_parser, run_single_experiment


def _mixed_task_spec(*, max_evaluations: int = 7) -> TaskSpec:
    return TaskSpec(
        name="mixed_llambo_demo",
        search_space=SearchSpace(
            [
                FloatParam("lr", low=1e-4, high=1e-1, log=True, default=1e-2),
                IntParam("depth", low=2, high=8, default=4),
                CategoricalParam("activation", choices=("relu", "gelu", "tanh"), default="relu"),
            ]
        ),
        objectives=(ObjectiveSpec("loss", ObjectiveDirection.MINIMIZE),),
        max_evaluations=max_evaluations,
        metadata={"display_name": "Mixed LLAMBO Demo"},
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


def test_llambo_is_registered_and_cli_visible() -> None:
    parser = build_arg_parser()
    algorithm_action = next(action for action in parser._actions if action.dest == "algorithm")

    assert "llambo" in ALGORITHM_REGISTRY
    assert ALGORITHM_REGISTRY["llambo"].family == "agentic"
    assert ALGORITHM_REGISTRY["llambo"].numeric_only is False
    assert "llambo" in algorithm_action.choices
    assert parser.parse_args(["--algorithm", "llambo"]).algorithm == "llambo"
    parsed = parser.parse_args(
        [
            "--algorithm",
            "llambo",
            "--llambo-backend",
            "openai",
            "--llambo-openai-api-key-env",
            "LLAMBO_TEST_KEY",
            "--llambo-openai-base-url",
            "https://api.openai.com/v1",
            "--llambo-openai-timeout-seconds",
            "12.5",
        ]
    )
    assert parsed.llambo_backend == "openai"
    assert parsed.llambo_openai_api_key_env == "LLAMBO_TEST_KEY"
    assert parsed.llambo_openai_base_url == "https://api.openai.com/v1"
    assert parsed.llambo_openai_timeout_seconds == pytest.approx(12.5)


def test_llambo_openai_backend_must_be_injected_from_runner_layer() -> None:
    algorithm = LlamboAlgorithm(backend="openai")
    with pytest.raises(RuntimeError, match="runner/CLI layer"):
        algorithm.setup(_mixed_task_spec(max_evaluations=3), seed=7)


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


def test_llambo_runner_builds_online_backend_and_backend_uses_structured_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLAMBO_TEST_KEY", "sk-test")
    captured_requests: list[dict[str, Any]] = []
    queued_payloads = [
        {
            "choices": [
                {"message": {"content": json.dumps({"lr": 0.02, "depth": 5, "activation": "gelu"})}},
                {"message": {"content": json.dumps({"lr": 0.03, "depth": 6, "activation": "relu"})}},
            ]
        },
        {
            "choices": [
                {"message": {"content": json.dumps({"predicted_objective": 4.125})}},
                {"message": {"content": json.dumps({"predicted_objective": 4.0})}},
            ]
        },
    ]

    def _fake_urlopen(request, timeout: float):
        captured_requests.append(
            {
                "url": request.full_url,
                "headers": dict(request.header_items()),
                "body": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        return _FakeHttpResponse(queued_payloads.pop(0))

    monkeypatch.setattr("bbo.algorithms.agentic.llambo.urllib_request.urlopen", _fake_urlopen)

    kwargs = _build_llambo_algorithm_kwargs(
        llambo_backend="openai",
        llambo_model="gpt-4o-mini",
        llambo_initial_samples=3,
        llambo_candidates=6,
        llambo_templates=2,
        llambo_predictions=4,
        llambo_alpha=-0.1,
        llambo_openai_api_key_env="LLAMBO_TEST_KEY",
        llambo_openai_base_url="https://api.openai.com/v1",
        llambo_openai_organization="org-test",
        llambo_openai_project="proj-test",
        llambo_openai_timeout_seconds=12.5,
    )
    assert kwargs["backend"] == "openai"
    backend = kwargs["backend_impl"]
    assert isinstance(backend, OpenAICompatibleLlamboBackend)

    search_space = _mixed_task_spec().search_space
    candidate_request = CandidateGenerationRequest(
        prompt="Recommend a candidate configuration.",
        n_responses=2,
        seed=11,
        objective_direction=ObjectiveDirection.MINIMIZE,
        search_space=search_space,
        observed_points=(),
        desired_score=1.0,
        parameter_order=tuple(search_space.names()),
    )
    candidate_texts = backend.generate_candidate_texts(candidate_request)
    assert candidate_texts == [
        '<candidate>{"lr":0.02,"depth":5,"activation":"gelu"}</candidate>',
        '<candidate>{"lr":0.03,"depth":6,"activation":"relu"}</candidate>',
    ]

    score_request = ScorePredictionRequest(
        prompt="Predict the score for this candidate.",
        n_responses=2,
        seed=13,
        objective_direction=ObjectiveDirection.MINIMIZE,
        search_space=search_space,
        observed_points=(
            ObservedPoint(
                config={"lr": 0.01, "depth": 4, "activation": "relu"},
                score=4.2,
                trial_id=0,
            ),
        ),
        candidate_config={"lr": 0.02, "depth": 5, "activation": "gelu"},
        parameter_order=tuple(search_space.names()),
    )
    score_texts = backend.generate_score_texts(score_request)
    assert score_texts == [
        "<score>4.12500000</score>",
        "<score>4.00000000</score>",
    ]

    candidate_call = captured_requests[0]
    candidate_headers = {key.lower(): value for key, value in candidate_call["headers"].items()}
    assert candidate_call["url"] == "https://api.openai.com/v1/chat/completions"
    assert candidate_headers["authorization"] == "Bearer sk-test"
    assert candidate_headers["openai-organization"] == "org-test"
    assert candidate_headers["openai-project"] == "proj-test"
    assert candidate_call["body"]["response_format"]["json_schema"]["strict"] is True
    assert candidate_call["body"]["store"] is False

    score_call = captured_requests[1]
    assert score_call["body"]["response_format"]["json_schema"]["name"] == "llambo_score"


def test_llambo_openai_fallback_to_plain_text_when_json_schema_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the endpoint returns 4xx for json_schema, backend should fall back to plain text parsing."""
    monkeypatch.setenv("LLAMBO_TEST_KEY", "sk-test")
    captured_requests: list[dict[str, Any]] = []

    error_payload = {"error": {"message": "json_schema is not supported", "type": "unsupported"}}
    text_payload = {
        "choices": [
            {"message": {"content": '<candidate>{"lr": 0.02, "depth": 5, "activation": "gelu"}</candidate>'}},
        ]
    }

    call_count = 0
    def _fake_urlopen(request, timeout: float):
        nonlocal call_count
        call_count += 1
        captured_requests.append(
            {
                "url": request.full_url,
                "body": json.loads(request.data.decode("utf-8")),
            }
        )
        # First call (json_schema) fails; second call (plain text) succeeds.
        if call_count == 1:
            class _Err(urllib_error.HTTPError):
                def __init__(self) -> None:
                    self.code = 400
                def read(self) -> bytes:
                    return json.dumps(error_payload).encode("utf-8")
            raise _Err()
        return _FakeHttpResponse(text_payload)

    monkeypatch.setattr("bbo.algorithms.agentic.llambo.urllib_request.urlopen", _fake_urlopen)

    backend = OpenAICompatibleLlamboBackend(
        model="gpt-4o-mini",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
    )
    search_space = _mixed_task_spec().search_space
    candidate_request = CandidateGenerationRequest(
        prompt="Recommend a candidate configuration.",
        n_responses=1,
        seed=11,
        objective_direction=ObjectiveDirection.MINIMIZE,
        search_space=search_space,
        observed_points=(),
        desired_score=1.0,
        parameter_order=tuple(search_space.names()),
    )
    texts = backend.generate_candidate_texts(candidate_request)
    assert len(texts) == 1
    assert texts[0] == '<candidate>{"lr": 0.02, "depth": 5, "activation": "gelu"}</candidate>'
    # After the 4xx, structured outputs should be marked unavailable.
    assert backend._structured_outputs_unavailable is True
    # Second request should NOT include response_format.
    assert "response_format" not in captured_requests[1]["body"]


def test_llambo_replay_reconstructs_next_suggestion_for_mixed_space() -> None:
    task_spec = _mixed_task_spec(max_evaluations=6)
    algorithm = LlamboAlgorithm(
        backend="heuristic",
        n_initial_samples=2,
        n_candidates=5,
        n_templates=2,
        n_predictions=4,
    )
    algorithm.setup(task_spec, seed=17)

    history: list[TrialObservation] = []
    for trial_id in range(task_spec.max_evaluations):
        suggestion = algorithm.ask()
        observation = _make_observation(suggestion, trial_id)
        algorithm.tell(observation)
        history.append(observation)

    replayed = LlamboAlgorithm(
        backend="heuristic",
        n_initial_samples=2,
        n_candidates=5,
        n_templates=2,
        n_predictions=4,
    )
    replayed.setup(task_spec, seed=17)
    replayed.replay(history[:-1])
    next_suggestion = replayed.ask()

    assert next_suggestion.config == history[-1].suggestion.config
    replayed.tell(history[-1])
    assert replayed.incumbents() == algorithm.incumbents()


def test_llambo_branin_summary_and_resume_outputs(tmp_path: Path) -> None:
    summary = run_single_experiment(
        task_name="branin_demo",
        algorithm_name="llambo",
        seed=7,
        max_evaluations=6,
        results_root=tmp_path,
        resume=False,
        llambo_backend="heuristic",
        llambo_initial_samples=3,
        llambo_candidates=6,
        llambo_templates=2,
        llambo_predictions=4,
        llambo_alpha=-0.1,
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
        algorithm_name="llambo",
        seed=7,
        max_evaluations=6,
        results_root=tmp_path,
        resume=True,
        llambo_backend="heuristic",
        llambo_initial_samples=3,
        llambo_candidates=6,
        llambo_templates=2,
        llambo_predictions=4,
        llambo_alpha=-0.1,
    )
    assert resumed["trial_count"] == 6
    assert resumed["best_primary_objective"] == summary["best_primary_objective"]
