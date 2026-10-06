from __future__ import annotations

import math

import numpy as np

from bbo.core import FloatParam, IntParam, SearchSpace, UnitCubeSearchSpaceConverter


def test_unit_cube_converter_applies_linear_log_and_logit_warps() -> None:
    logit_midpoint = 1.0 / (1.0 + math.exp(-(math.log(0.1 / 0.9) + math.log(0.9 / 0.1)) / 2.0))
    space = SearchSpace(
        [
            FloatParam("linear", low=-2.0, high=2.0, default=0.0),
            FloatParam("log", low=1e-4, high=1e2, log=True, default=1e-1),
            FloatParam("logit", low=0.1, high=0.9, default=logit_midpoint),
            IntParam("integer", low=1, high=9, default=5),
        ]
    )
    converter = UnitCubeSearchSpaceConverter(
        space,
        transforms={"linear": "linear", "log": "log", "logit": "logit", "integer": "linear"},
    )

    encoded = converter.encode_vector(space.defaults())
    assert np.allclose(encoded, np.full(4, 0.5), atol=1e-12)
    decoded = converter.decode_vector(encoded)
    assert decoded["integer"] == space.defaults()["integer"]
    assert np.allclose(
        [decoded["linear"], decoded["log"], decoded["logit"]],
        [space.defaults()["linear"], space.defaults()["log"], space.defaults()["logit"]],
        atol=1e-12,
    )
    assert converter.decode_vector([0.0, 0.0, 0.0, 0.0]) == {
        "linear": -2.0,
        "log": 1e-4,
        "logit": 0.1,
        "integer": 1,
    }
    assert converter.decode_vector([1.0, 1.0, 1.0, 1.0]) == {
        "linear": 2.0,
        "log": 1e2,
        "logit": 0.9,
        "integer": 9,
    }


def test_unit_cube_converter_rounds_and_clips_integer_values() -> None:
    converter = UnitCubeSearchSpaceConverter(SearchSpace([IntParam("depth", low=1, high=4, default=2)]))
    assert converter.decode_vector([-1.0])["depth"] == 1
    assert converter.decode_vector([0.6])["depth"] == 3
    assert converter.decode_vector([2.0])["depth"] == 4
