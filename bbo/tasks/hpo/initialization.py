"""Published LLAMBO Bayesmark random initialization configurations."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .catalog import ASSET_ROOT, LLAMBO_SOURCE_COMMIT, BayesmarkModelDefinition


INITIALIZATION_ASSET = ASSET_ROOT / "llambo_initializations.json"
INITIALIZATION_ASSET_SHA256 = "e384825e3e40913ebeed01e5b614221f8fb56845c411af61cf2003bd1b6a09e4"
PAPER_RESULT_SEEDS = tuple(range(5))
PUBLISHED_CONFIG_SEEDS = tuple(range(10))
INITIALIZATION_COUNT = 5


@lru_cache(maxsize=1)
def _load_initialization_asset() -> dict[str, dict[str, list[dict[str, Any]]]]:
    raw = INITIALIZATION_ASSET.read_bytes()
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_sha256 != INITIALIZATION_ASSET_SHA256:
        raise ValueError(
            f"LLAMBO initialization asset checksum mismatch: {observed_sha256} "
            f"!= {INITIALIZATION_ASSET_SHA256}."
        )
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid LLAMBO initialization asset at {INITIALIZATION_ASSET}.")
    return payload


def published_initial_configurations(
    model: BayesmarkModelDefinition,
    *,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    """Return the five fixed configurations used for one published seed.

    The released MLP experiment omitted ``tol`` and ``validation_fraction``
    despite the paper's eight-dimensional space. At those historical points,
    scikit-learn therefore used its defaults. We materialize those effective
    values so the configurations are valid in the paper-faithful 8D space.
    """

    if int(seed) not in PUBLISHED_CONFIG_SEEDS:
        raise ValueError(
            "LLAMBO Bayesmark fixed initialization is available for seeds 0..9; "
            f"got seed={int(seed)}. Paper-result runs use seeds 0..4."
        )
    try:
        raw_configs = _load_initialization_asset()[model.key][str(int(seed))]
    except KeyError as exc:
        raise KeyError(f"Missing LLAMBO initialization for model={model.key!r}, seed={int(seed)}.") from exc
    if len(raw_configs) != INITIALIZATION_COUNT:
        raise ValueError(
            f"Expected {INITIALIZATION_COUNT} LLAMBO initial points for {model.key}/seed {seed}, "
            f"found {len(raw_configs)}."
        )

    configurations: list[dict[str, Any]] = []
    for raw in raw_configs:
        config = dict(raw)
        if model.key == "mlp_sgd":
            config.setdefault("tol", 1e-4)
            config.setdefault("validation_fraction", 0.1)
        configurations.append(model.search_space.coerce_config(config, use_defaults=False))
    return tuple(configurations)


def initialization_protocol_metadata(
    model: BayesmarkModelDefinition,
    *,
    seed: int,
) -> dict[str, Any]:
    configurations = published_initial_configurations(model, seed=seed)
    source_directories = {
        "random_forest": "RandomForest",
        "svm": "SVM",
        "decision_tree": "DecisionTree",
        "mlp_sgd": "MLP_SGD",
        "adaboost": "AdaBoost",
    }
    return {
        "name": "llambo_bayesmark_end_to_end_v1",
        "paper_result_seeds": list(PAPER_RESULT_SEEDS),
        "optimizer_evaluations": 25,
        "initialization": {
            "strategy": "fixed_configurations",
            "sampling": "random",
            "seed": int(seed),
            "count": len(configurations),
            "configurations": [dict(config) for config in configurations],
            "source": (
                "tennisonliu/LLAMBO@"
                f"{LLAMBO_SOURCE_COMMIT}:bayesmark/configs/{source_directories[model.key]}/<seed>.json:first5"
            ),
            "asset_sha256": INITIALIZATION_ASSET_SHA256,
            "scope": "shared_by_model_and_seed_across_datasets_and_algorithms",
            "mlp_missing_dimension_policy": "materialize_sklearn_defaults" if model.key == "mlp_sgd" else None,
        },
        "candidate_budget": {
            "policy": "min(5000,max(2048,200*d))",
            "value": min(5000, max(2048, 200 * model.dimension)),
            "applies_to": [
                "gp_ei.raw_samples",
                "botorch_turbo.n_candidates",
                "git_bo.n_candidates",
            ],
        },
    }


__all__ = [
    "INITIALIZATION_ASSET",
    "INITIALIZATION_ASSET_SHA256",
    "INITIALIZATION_COUNT",
    "PAPER_RESULT_SEEDS",
    "PUBLISHED_CONFIG_SEEDS",
    "initialization_protocol_metadata",
    "published_initial_configurations",
]
