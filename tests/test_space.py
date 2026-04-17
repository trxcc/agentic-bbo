from __future__ import annotations

import random

import numpy as np

from bbo.core import CategoricalParam, FloatParam, IntParam, SearchSpace


def test_search_space_sampling_and_numeric_roundtrip() -> None:
    space = SearchSpace(
        [
            FloatParam("lr", low=1e-3, high=1e-1, log=True, default=1e-2),
            IntParam("depth", low=2, high=8, default=4),
        ]
    )
    rng = random.Random(0)
    sample = space.sample(rng)
    vector = space.to_numeric_vector(sample)
    recovered = space.from_numeric_vector(vector)

    assert sample.keys() == recovered.keys()
    assert np.allclose(vector, space.to_numeric_vector(recovered))


def test_search_space_rejects_unknown_parameters() -> None:
    space = SearchSpace([CategoricalParam("mode", choices=("a", "b"), default="a")])
    try:
        space.validate_config({"mode": "a", "extra": 1})
    except ValueError as exc:
        assert "Unexpected parameters" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected a validation error for an unknown parameter.")
