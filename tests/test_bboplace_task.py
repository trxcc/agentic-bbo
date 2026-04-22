from __future__ import annotations

import math
from typing import Any

import pytest

from bbo.core import TrialSuggestion
from bbo.tasks.bboplace import BBOPlaceTask, BBOPlaceTaskConfig


@pytest.mark.unit
def test_bboplace_evaluate_success_payload_and_metrics() -> None:
    capture: dict[str, Any] = {}

    from bbo.tasks.bboplace.task import default_bboplace_definition

    definition = default_bboplace_definition()

    def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        capture["url"] = url
        capture["payload"] = payload
        capture["timeout"] = timeout
        return {"hpwl": [123.0]}

    task = BBOPlaceTask(
        config=BBOPlaceTaskConfig(post_json=post_json, http_timeout_seconds=1.5),
        definition=definition,
    )

    config = task.spec.search_space.defaults()
    result = task.evaluate(TrialSuggestion(config=config, trial_id=0))

    assert result.success
    assert result.objectives["hpwl"] == 123.0
    assert capture["url"].endswith("/evaluate")
    assert capture["payload"]["benchmark"] == definition.benchmark
    assert capture["payload"]["placer"] == definition.placer
    assert capture["payload"]["n_macro"] == definition.n_macro
    assert capture["payload"]["seed"] == 0
    assert isinstance(capture["payload"]["x"], list)
    assert len(capture["payload"]["x"]) == 1
    assert len(capture["payload"]["x"][0]) == definition.dimension
    assert capture["timeout"] == 1.5

    assert math.isfinite(float(result.metrics["dimension"]))
    assert "coord::x_0" in result.metrics
    assert "coord::y_0" in result.metrics


@pytest.mark.unit
def test_bboplace_evaluate_invalid_response_shape() -> None:
    from bbo.tasks.bboplace.task import default_bboplace_definition

    definition = default_bboplace_definition()

    def post_json(_: str, __: dict[str, Any], ___: float) -> dict[str, Any]:
        return {"hpwl": []}

    task = BBOPlaceTask(config=BBOPlaceTaskConfig(post_json=post_json), definition=definition)
    result = task.evaluate(TrialSuggestion(config=task.spec.search_space.defaults(), trial_id=0))

    assert not result.success
    assert result.status.value == "failed"
    assert result.error_type == "InvalidResponse"


@pytest.mark.unit
def test_bboplace_evaluate_nonfinite_hpwl_is_failed() -> None:
    from bbo.tasks.bboplace.task import default_bboplace_definition

    definition = default_bboplace_definition()

    def post_json(_: str, __: dict[str, Any], ___: float) -> dict[str, Any]:
        return {"hpwl": [float("inf")]}

    task = BBOPlaceTask(config=BBOPlaceTaskConfig(post_json=post_json), definition=definition)
    result = task.evaluate(TrialSuggestion(config=task.spec.search_space.defaults(), trial_id=0))

    assert not result.success
    assert result.status.value == "failed"
    assert result.error_type == "InvalidResponse"

