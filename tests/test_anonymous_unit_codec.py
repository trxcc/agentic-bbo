import math

import pytest

from bbo.core import SearchSpace, FloatParam, IntParam, CategoricalParam, search_space_to_schema
from bbo.algorithms.agentic.anonymous_space import AnonymousUnitCodec


def codec():
    return AnonymousUnitCodec(SearchSpace([
        FloatParam("regularization", low=1, high=1000, log=True, default=31),
        FloatParam("fraction", low=0.01, high=0.99),
        IntParam("depth", low=1, high=15),
        CategoricalParam("kernel", choices=("rbf", "linear", "poly")),
    ]), transforms={"fraction": "logit"}, salt="fixed-task")


def test_round_trip_and_no_semantics_in_schema():
    c = codec()
    original = dict(regularization=31.6, fraction=0.2, depth=8, kernel="rbf")
    encoded = c.encode(original)
    restored = c.decode(encoded)
    assert restored["depth"] == 8 and restored["kernel"] == "rbf"
    assert math.isclose(restored["regularization"], 31.6, rel_tol=1e-12)
    assert math.isclose(restored["fraction"], 0.2, rel_tol=1e-12)
    schema = str(search_space_to_schema(c.space))
    assert not any(secret in schema for secret in ("depth", "kernel", "rbf", "regularization", "choices", "1000"))
    assert all(p.low == 0 and p.high == 1 and not p.log and p.default == 0.5 for p in c.space)


def test_all_discrete_values_and_endpoint_decoding():
    c = codec()
    for depth in range(1, 16):
        for kernel in ("rbf", "linear", "poly"):
            config = dict(regularization=1000, fraction=0.99, depth=depth, kernel=kernel)
            assert c.decode(c.encode(config)) == config
    assert c.decode(dict(x1=0, x2=0, x3=0, x4=0))["depth"] == 1
    assert c.decode(dict(x1=1, x2=1, x3=1, x4=1))["depth"] == 15
    assert c.choices == codec().choices


@pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_unit_values_are_rejected_not_clipped(bad):
    with pytest.raises(ValueError):
        codec().decode(dict(x1=bad, x2=0.5, x3=0.5, x4=0.5))
