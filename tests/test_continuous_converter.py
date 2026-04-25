from __future__ import annotations

import numpy as np
import pytest

from bbo.core import CategoricalParam, FloatParam, IntParam, SearchSpace, build_continuous_converter


def _mixed_space() -> SearchSpace:
    return SearchSpace(
        [
            FloatParam("lr", low=1e-4, high=1e-1, log=True, default=1e-2),
            IntParam("depth", low=2, high=8, default=4),
            CategoricalParam("activation", choices=("relu", "gelu", "tanh"), default="relu"),
        ]
    )


def test_onehot_converter_roundtrips_mixed_configs() -> None:
    converter = build_continuous_converter(_mixed_space(), strategy="onehot")

    assert converter.feature_names == (
        "lr",
        "depth",
        "activation::relu",
        "activation::gelu",
        "activation::tanh",
    )

    encoded = converter.encode_vector({"lr": 0.02, "depth": 5, "activation": "gelu"})
    assert np.allclose(encoded, np.asarray([0.02, 5.0, 0.0, 1.0, 0.0]))

    decoded = converter.decode_vector(encoded)
    assert decoded == {"lr": pytest.approx(0.02), "depth": 5, "activation": "gelu"}


def test_onehot_converter_decodes_by_argmax_and_rounds_integers() -> None:
    converter = build_continuous_converter(_mixed_space(), strategy="onehot")

    decoded = converter.decode_vector([0.5, 6.6, 0.2, 0.7, 0.1], clip=True)
    assert decoded["lr"] == pytest.approx(0.1)
    assert decoded["depth"] == 7
    assert decoded["activation"] == "gelu"


def test_build_continuous_converter_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown categorical-to-continuous strategy"):
        build_continuous_converter(_mixed_space(), strategy="embedding")
